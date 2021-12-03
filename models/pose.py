import io
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
from torch.nn.modules import padding

from models.Model3DNet import Model3DNet

import util.box_ops as box_ops
from util.misc import NestedTensor, interpolate, inverse_sigmoid, nested_tensor_from_tensor_list


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)

        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k):
        q = self.q_linear(q)
        k = self.k_linear(k)

        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], k.shape[1], self.num_heads, self.hidden_dim // self.num_heads)

        weights = torch.einsum("bqnc,bmnc->bqnm", qh * self.normalize_fact, kh)

        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights


class PoseHeadSmallLinear(nn.Module):
    """
    Simple linear head
    """

    def __init__(self, num_dims, num_heads):
        super().__init__()
        self.num_dims = num_dims
        self.num_heads = num_heads

        self.l1 = nn.Linear(num_dims, num_dims)
        self.g1 = nn.GroupNorm(8, num_dims)

        num_feat = num_dims + num_heads

        #### for model class ####
        l2 = []
        num_l2 = 4
        for i in range(num_l2):
            l2.append(nn.Linear(num_feat // (2 ** i), num_feat // (2 ** (i + 1))))
            l2.append(nn.ReLU())

        self.l2 = nn.Sequential(*l2)
        self.l3 = nn.Linear(num_feat // (2 ** num_l2), 1)

        ####  for 6 dof ###
        l4 = []
        num_l4 = 4
        for i in range(num_l4):
            l4.append(nn.Linear(num_feat // (2 ** i), num_feat // (2 ** (i + 1))))
            l4.append(nn.ReLU())
        self.l4 = nn.Sequential(*l4)
        self.l5 = nn.Linear(num_feat // (2 ** num_l4), 7)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x: Tensor, attention: Tensor):
        x = self.l1(x)
        x = self.g1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x = torch.cat([x.unsqueeze(1).repeat(1, attention.shape[1], 1, 1), attention.permute(0, 1, 3, 2)], dim=3)

        ## for model class
        x_c = self.l2(x)
        x_c = self.l3(x_c)
        x_c = F.softmax(x_c.squeeze(-1), dim=-1)

        ## for pose 6dof
        x_p = self.l4(x)
        x_p = self.l5(x_p)

        ## make the quaternion obey the rules as follows
        ### w**2 + x**2 + y**2 + z**2 = 1
        x_f, x_l = torch.split(x_p, [3, 4], dim=-1)
        x_l = x_l.softmax(dim=-1).sqrt()
        x_p = torch.cat((x_f, x_l), dim=-1)
        return x_c, x_p


class DETRpose(nn.Module):
    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        self.detr = detr

        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead

        self.model_3d_net = Model3DNet()

        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)

        self.pose_head = PoseHeadSmallLinear(hidden_dim, nheads)

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.detr.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.detr.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.detr.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.detr.num_feature_levels):
                if l == _len_srcs:
                    src = self.detr.input_proj[l](features[-1].tensors)
                else:
                    src = self.detr.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.detr.two_stage:
            query_embeds = self.detr.query_embed.weight

        bs = features[-1].tensors.shape[0]

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.detr.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.detr.class_embed[lvl](hs[lvl])
            tmp = self.detr.bbox_embed[lvl](hs[lvl])
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
        if self.detr.aux_loss:
            out['aux_outputs'] = self.detr._set_aux_loss(outputs_class, outputs_coord)

        # 3d model and query attention
        # 1. get the feature of 3d model
        model_3d_feat = self.model_3d_net.forward_encoder()

        pred_model_point = self.model_3d_net.forward_decoder(model_3d_feat)

        model_3d_feat = model_3d_feat[None].repeat(bs, 1, 1)

        # 2. cacluate the attention between the query bbox and the 3d model
        # input: hs[-1] (bs x n_q x n_f) model_3d_feat: (bs x n_m x n_f)
        # output: (bs x n_q x n_h x n_m)
        bbox_3dmodel_attention = self.bbox_attention(hs[-1], model_3d_feat)

        # 3. output the 3d model class and pose 6dof for each bbox
        # input: model_3d_feat (bs x n_m x n_f)
        # input: bbox_3dmodel_attention (bs x n_q x n_h x n_m)
        # output: pose_class (bs x n_q x n_m)
        # output: pose_6dof (bs x n_q x n_m x 6)
        pose_class, pose_6dof = self.pose_head(model_3d_feat, bbox_3dmodel_attention)

        out["pose_class"] = pose_class
        out["pose_6dof"] = pose_6dof
        out["pred_model_points"] = pred_model_point[0]
        out["pred_model_scales"] = pred_model_point[1]
        out["pred_model_centers"] = pred_model_point[2]

        return out


class PostProcessPose(nn.Module):
    """ This module converts the model's output into the display formate"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bboxes_2d, out_pose_class, out_pose_6dof, out_pred_model_points, out_pred_model_scales, out_pred_model_centers = \
            outputs['pred_logits'], outputs['pred_boxes'], outputs["pose_class"], outputs["pose_6dof"], outputs[
                "pred_model_points"], outputs["pred_model_scales"], outputs["pred_model_centers"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)

        # convert to [x0, y0, x1, y1] format
        # boxes = box_ops.box_cxcywh_to_xyxy(out_bboxes_2d)
        # and from relative [0, 1] to absolute [0, height] coordinates
        # img_h, img_w = target_sizes.unbind(1)
        # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # boxes = boxes * scale_fct[:, None, :]

        src_results = [
            {
                'scores': s, 'labels': l, 'bboxes_2d': b, "pose_class": p_s, "pose_6dof": p_6dof
            }
            for s, l, b, p_s, p_6dof
            in zip(scores, labels, out_bboxes_2d, out_pose_class, out_pose_6dof)
        ]

        model_results = [
            {
                "pred_model_points": m_p, "pred_model_scales": m_s, "pred_model_centers": m_c
            }
            for m_p, m_s, m_c
            in zip(out_pred_model_points, out_pred_model_scales, out_pred_model_centers)
        ]

        return src_results, model_results

