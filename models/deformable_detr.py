# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy
from .pose import DETRpose, PostProcessPose
from configs import cfg, config

import torchgeometry as tgm
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)




def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, focal_gamma=2):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = focal_alpha
        self.register_buffer('empty_weight', empty_weight)




    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        # ============ debug focal loss  =============
        fore_src = src_logits.argmax(-1)       
        pred_fore_num = (fore_src != self.num_classes).sum().item()
        matched_correct_num = (fore_src[idx] == target_classes_o).sum().item()
        
        matched_error = (fore_src[idx] != target_classes_o).nonzero(as_tuple=True)[0]
        matched_error_idx = (idx[0][matched_error], idx[1][matched_error])
        
        pred_fore_idx = (fore_src != self.num_classes).nonzero(as_tuple=True)
        fake_fore = [i for i, x in enumerate(pred_fore_idx[1]) if x not in idx[1].tolist()]
        fake_idx = (pred_fore_idx[0][fake_fore], pred_fore_idx[1][fake_fore])

        src_prob = src_logits.softmax(-1)
        fake_trace_avg_prob = src_prob[fake_idx][..., 0].mean().item()
        tgt_trace_avg_prob = src_prob[idx][..., 0].mean().item()
        missed_trace_avg_prob = src_prob[matched_error_idx][..., 0].mean().item()

        print(f"target_num: {target_classes_o.shape[0]}    pred_fore_num: {pred_fore_num}   matched_correct_num: {matched_correct_num}")

        # =============== finish debug ===============

        # =============== sigmod focal loss =============== 
        # target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
        #                                     dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        # target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # target_classes_onehot = target_classes_onehot[:,:,:-1]
        # loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.focal_gamma) * src_logits.shape[1]
        
        # =============== focal loss =======================
        src_logits = src_prob.log()
        src_prob = src_logits.softmax(-1)
        src_logits = src_logits *  ((1 - src_prob) ** self.focal_gamma)
        loss = nn.NLLLoss(weight=self.empty_weight, reduction='sum')
        loss_ce = loss(src_logits.transpose(1, 2), target_classes) / target_classes_o.shape[0]
        
        
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['bboxes_2d'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses


    def loss_model3d(self, outputs, targets, indices, num_boxes):
        assert "pred_model_scales" in outputs
        assert "pred_model_centers" in outputs
        assert "pred_model_points" in outputs
        losses = {}
        nm = outputs["pred_model_points"].shape[0]

        # 1. get predict scales
        src_scales = outputs["pred_model_scales"]
        tgt_scales = targets[0]["model_scales"]
        # 2. calcuate the scale loss
        losses["model3d_scales"] = nn.L1Loss()(src_scales, tgt_scales) / nm

        # 3. get predict centers
        src_centers = outputs["pred_model_centers"]
        tgt_centers = targets[0]["model_centers"]
        # 4. calcuate the center loss
        losses["model3d_centers"] = nn.L1Loss()(src_centers, tgt_centers) / nm

        # 5. get predict points
        src_points = outputs["pred_model_points"]
        tgt_points = targets[0]["model_points"]
        # 6. cacluat the points loss
        losses["model3d_points"] = []

        for i in range(nm):
            deform_verts = src_points[i]

            # Deform the mesh
            src_mesh = ico_sphere(4, deform_verts.device)
            new_src_mesh = src_mesh.offset_verts(deform_verts)

            # We sample 10k points from the surface of each mesh
            sample_trg = sample_points_from_meshes(tgt_points[i], config["model"]["model_num_samples"])
            sample_src = sample_points_from_meshes(new_src_mesh, config["model"]["model_num_samples"])

            # We compare the two sets of pointclouds by computing (a) the chamfer loss
            loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

            # and (b) the edge length of the predicted mesh
            loss_edge = mesh_edge_loss(new_src_mesh)

            # mesh normal consistency
            loss_normal = mesh_normal_consistency(new_src_mesh)

            # mesh laplacian smoothing
            loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

            # Weighted sum of the losses
            losses["model3d_points"].append(
                loss_chamfer * cfg["model3d_chamfer_weight"] + loss_edge * cfg["model3d_edge_weight"] + loss_normal *
                cfg["model3d_normal_weight"] + loss_laplacian * cfg["model3d_laplacian_weight"])

        losses["model3d_points"] = sum(losses["model3d_points"]) / nm

        return losses

    def loss_pose6dof(self, outputs, targets, indices, num_boxes):
        assert "pose_class" in outputs
        assert "pose_6dof" in outputs

        losses = {}

        src_idx = self._get_src_permutation_idx(indices)

        bs = outputs["pose_class"].shape[0]
        # 1. get model class loss
        # 1.1 get predict model class
        src_class = outputs["pose_class"]
        src_class = src_class[src_idx]
        # 1.2 get target model class
        tgt_class_o = torch.cat([t["model_ids"][J] for t, (_, J) in zip(targets, indices)])

        # 1.3 get the model class loss
        loss_mc = F.cross_entropy(src_class[None, :].permute(0, 2, 1), tgt_class_o[None, :])
        losses["pose_6dof_class"] = loss_mc

        # 2. get 6Dof loss
        # 2.1 get predict 6Dof
        src_pose = outputs["pose_6dof"]
        src_6dof_pose = src_pose[src_idx]
        model_index = src_class.argmax(dim=-1)
        src_6dof_pose = torch.stack([src_6dof_pose[i, j, :] for i, j in enumerate(model_index)], dim=0)

        src_6dof_t_pose = src_6dof_pose[:, :3]
        src_6dof_r_pose = src_6dof_pose[:, 3:]

        # 2.2 get target 6Dof pose
        tgt_6dof_t_pose_o = torch.cat([t["T_matrix_c2o"][J] for t, (_, J) in zip(targets, indices)])
        tgt_6dof_r_pose_o = torch.cat([t["R_quaternion_c2o"][J] for t, (_, J) in zip(targets, indices)])
        tgt_6dof_pose_o = torch.cat((tgt_6dof_t_pose_o, tgt_6dof_r_pose_o), dim=1)

        tgt_bboxes_2d = torch.cat([t["bboxes_2d"][J] for t, (_, J) in zip(targets, indices)])
        tgt_bboxes_3d_w = torch.cat([t["bboxes_3d_w"][J] for t, (_, J) in zip(targets, indices)])

        # 2.3 get the model add loss
        losses["pose_6dof_add"] = nn.L1Loss()(src_6dof_t_pose, tgt_6dof_t_pose_o)

        # TODO: 2.4 get the model l1 loss of 3d bbox

        # tgt_bboxes_3d_points = [[tgt_bboxes_3d_w[i][0], tgt_bboxes_3d_w[i][-1]]for i in range(tgt_bboxes_3d_w.shape[0])]

        # 2.5 get the model l1 loss of fps points
        # 2.5.1 get the fps points in object coordinate
        fps_points = targets[0]["fps_points"]
        fps_points = torch.stack([fps_points[i, ...] for i in model_index], dim=0)
        # fps_points = tgm.convert_points_to_homogeneous(fps_points)

        # 2.5.2 get the src matrix which can convert points from obj to camera

        src_matrix = tgm.angle_axis_to_rotation_matrix(tgm.quaternion_to_angle_axis(src_6dof_r_pose))

        src_matrix[:, :3, -1] = src_6dof_t_pose

        # 2.5.3 convert fps_points from obj coord to camera coordinate by src matrix
        src_fps_points_camera = tgm.transform_points(src_matrix, fps_points)

        # 2.5.4 get the target matrix which can convert points from obj to camera
        tgt_matrix = tgm.angle_axis_to_rotation_matrix(tgm.quaternion_to_angle_axis(tgt_6dof_r_pose_o))
        tgt_matrix[:, :3, -1] = tgt_6dof_t_pose_o

        # 2.5.5 convert fps_points from obj coord to camera coord by tgt matrix
        tgt_fps_points_camera = tgm.transform_points(tgt_matrix, fps_points)

        # 2.5.6 calcuate the l1 loss between target fps points and src fps points
        losses["pose_6dof_fps_points_3d"] = nn.L1Loss()(src_fps_points_camera, tgt_fps_points_camera)

        # TODO: 3 get the model bboxes_2d loss
        # losses["pose_6dof_bboxes_2d"] = 0

        # TODO: 4 get the 3d iou loss
        # losses["pose_6dof_bboxes_3d"] = 0

        return losses



    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'model3d': self.loss_model3d,
            "pose6dof": self.loss_pose6dof
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue

                    if loss == "model3d":
                        continue

                    if loss == "pose6dof":
                        continue

                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2


        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # prob = out_logits.sigmoid()
        # topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 300, dim=1)
        # scores = topk_values
        # topk_boxes = topk_indexes // out_logits.shape[2]
        # labels = topk_indexes % out_logits.shape[2]
        # boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        # img_h, img_w = target_sizes.unbind(1)
        # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'bboxes_2d': b} for s, l, b in zip(scores, labels, out_bbox)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):

    num_classes = args.class_num

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes + 1,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )

    if args.poses:
        model = DETRpose(model, freeze_detr=(args.frozen_weights is not None))

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_weight, 
                    'loss_bbox': args.bbox2d_loss_weight,
                    'loss_giou':args.giou_loss_weight}

    if args.poses:
        weight_dict["model3d_scales"] = args.model3d_scales_weight
        weight_dict["model3d_centers"] = args.model3d_centers_weight
        weight_dict["model3d_points"] = args.model3d_points_weight
        weight_dict["pose_6dof_class"] = args.model_6dof_class_weight
        weight_dict["pose_6dof_add"] = args.model_6dof_add_weight
        weight_dict["pose_6dof_fps_points_3d"] = args.model_6dof_fps_points_weight


    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    
    if args.poses:
        losses = losses + ["model3d", "pose6dof"]
    
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    if args.poses:
        postprocessors = {"pose": PostProcessPose()}


    return model, criterion, postprocessors
