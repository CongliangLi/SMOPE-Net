"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path, PurePath
from datasets_labelimg3d import KITTI_CLASSES
import os
from configs import config
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from datasets_labelimg3d.transforms import NormalizeInverse
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_obj, save_obj
import json
from util.utils import get_distance, RQuaternion_2_RMatrix, Rotate_x_axis



# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
normalizeInverse = NormalizeInverse(config["pixel_mean"], config["pixel_std"])


class PoseResultPlotor(object):
    def __init__(self, save_path):
        self.save_path = Path(save_path)
        # tgt save path
        self.tgt_save_path = os.path.join(self.save_path, "tgt")

        self.tgt_img_save_path = os.path.join(self.tgt_save_path, "images")
        self.tgt_annotation_save_path = os.path.join(self.tgt_save_path, "annotations")
        self.tgt_model_path = os.path.join(self.tgt_save_path, "models")

        # pred save path
        self.pred_save_path = os.path.join(self.save_path, "pred")

        self.pred_img_save_path = os.path.join(self.pred_save_path, "images")
        self.pred_annotation_save_path = os.path.join(self.pred_save_path, "annotations")
        self.pred_model_path = os.path.join(self.pred_save_path, "models")

        if not os.path.exists(self.save_path):
            self.save_path.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(self.pred_save_path) or not os.path.exists(self.tgt_save_path):
            os.makedirs(self.pred_save_path)
            os.makedirs(self.tgt_save_path)

        if not os.path.exists(self.pred_img_save_path) or not os.path.exists(self.pred_annotation_save_path) \
                or not os.path.exists(self.pred_model_path):
            os.makedirs(self.pred_img_save_path)
            os.makedirs(self.pred_annotation_save_path)
            os.makedirs(self.pred_model_path)

        if not os.path.exists(self.tgt_img_save_path) or not os.path.exists(self.tgt_annotation_save_path) \
                or not os.path.exists(self.tgt_model_path):
            os.makedirs(self.tgt_img_save_path)
            os.makedirs(self.tgt_annotation_save_path)
            os.makedirs(self.tgt_model_path)

    def __call__(self, samples, predictions, targets, args, save_name: str = None):
        """Plot predictions

        Args:
            predictions: [src_predictions, model_predictions]
            save_name (str): plot image save file name. Defaults to None.
        """
        try:
            pre_src, pre_model = predictions

            # model display
            if args.poses and pre_model is not None:
                for i, model in enumerate(pre_model):
                    pre_model_points = model["pred_model_points"]
                    pred_model_scale = model["pred_model_scales"]
                    pred_model_center = model["pred_model_centers"]
                    self._save_model(i, pre_model_points, pred_model_scale, pred_model_center, args, self.pred_model_path)

            # image display
            num = 0
            for sample, pre_res, target in zip(samples, pre_src, targets):
                _, h, w = sample.size()

                # target
                tgt_labels = target["labels"]
                tgt_bboxes_2d = box_cxcywh_to_xyxy(target['bboxes_2d']).cpu() * torch.Tensor([w, h, w, h])
                tgt_orig_size = target["orig_size"]

                if args.poses:
                    tgt_bboxes_3d_w = target["bboxes_3d_w"]
                    tgt_fps_points = target["fps_points"]
                    
                    tgt_model_ids = target["model_ids"]

                    tgt_T_matrix_c2o = target["T_matrix_c2o"]
                    tgt_R_quaternion_c2o = target["R_quaternion_c2o"]
                    tgt_pose_6dof = torch.cat((tgt_T_matrix_c2o, tgt_R_quaternion_c2o),dim=-1).cpu().detach()

                tgt_scores = torch.ones_like(tgt_labels)

                # predicted value
                pred_scores = pre_res["scores"].cpu().detach()
                pred_classes = pre_res["labels"].cpu().detach()
                pred_bboxes_2d = box_cxcywh_to_xyxy(pre_res['bboxes_2d'].cpu().detach()) * torch.Tensor([w, h, w, h])

                if args.poses:
                    pred_model_ids = torch.max(pre_res['pose_class'].cpu().detach(), dim=1)[1]
                    pred_pose_6dof = pre_res["pose_6dof"].cpu().detach()

                ####  save pred and tgt ####
                # save tgt images
                this_epoch = None
                if "evl" in save_name:
                    this_epoch = int(save_name.split("e")[-1].split("_")[0])
                
                if this_epoch == 0 or this_epoch is None:
                    self._plot_bbox2d(sample.permute(1, 2, 0).cpu(), tgt_bboxes_2d, tgt_model_ids, tgt_scores,
                                        f'{self.tgt_img_save_path}/{save_name}_{num}.png')
                    if args.poses:
                        # save targets 6dof as annotations for labelImg3d
                        self._save_tgt_6dof_annotation(sample.permute(1, 2, 0).cpu(), tgt_pose_6dof, tgt_model_ids, 
                                                    f'{self.tgt_annotation_save_path}/{save_name}_{num}.json')
                        
                        
                threshold = args.plot_threshold
                pred_mask = pred_scores > threshold
                if pred_mask.any():
                    th_classes = pred_classes[pred_mask]
                    th_scores = pred_scores[pred_mask]
                    th_bboxes_2d = pred_bboxes_2d[pred_mask]
                    if args.poses:
                        th_model_ids = pred_model_ids[pred_mask]
                        th_pose_6dof = pred_pose_6dof[pred_mask]

                        # save pred images
                        self._plot_bbox2d(sample.permute(1, 2, 0).cpu(), th_bboxes_2d, th_model_ids, th_scores,
                                            f'{self.pred_img_save_path}/{save_name}_{num}.png')
                        # save pred 6dof as annotations for labelImg3d
                        self._save_pred_6dof_annotation(sample, th_pose_6dof, th_model_ids, f"{self.pred_annotation_save_path}/{save_name}_{num}.json")
                    else:
                        # save pred images
                        self._plot_bbox2d(sample.permute(1, 2, 0).cpu(), th_bboxes_2d, th_classes, th_scores,
                                            f'{self.pred_img_save_path}/{save_name}_{num}.png')

                num = num + 1
        except Exception as e:
            print(e)

    def _plot_bbox2d(self, image, boxes, labels, scores, save_name: str = None):
        fig = plt.figure(figsize=(16, 10))
        ax = plt.gca()
        ax.axis('off')
        ima = np.array(image, np.int32)
        im = ax.imshow(image)
        bbox = self._draw_bbox_2d(ax, boxes, labels, scores)

        ani = animation.ArtistAnimation(fig, [[im] + bbox], interval=200, blit=False)
        if save_name:
            ani.save(save_name, writer='pillow')
        plt.close(fig)

    def _draw_bbox_2d(self, ax, boxes, labels, scores):
        """Plot bounding boxes rectangle and each box's label and score, returen all plot ops as list"""
        bboxes_ops = []
        for (xmin, ymin, xmax, ymax), cl, s, c in zip(boxes.tolist(), labels.tolist(), scores.tolist(), COLORS * 100):
            # passed padding bbox
            if (xmax - xmin) == 0 and (ymax - ymin) == 0:
                continue
            bboxes_ops.append(ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                                         fill=False, color=c, linewidth=3)))
            text = f'{cl}_{KITTI_CLASSES[cl]}: {s:0.2f}'
            bboxes_ops.append(ax.text(xmin, ymin, text, fontsize=15,
                                      bbox=dict(facecolor='yellow', alpha=0.5)))
        return bboxes_ops



    def _save_model(self, index, model_points, model_scale, model_center, args, save_path):
        # Deform the mesh
        src_mesh = ico_sphere(4, args.device)
        new_src_mesh = src_mesh.offset_verts(model_points)

        # Fetch the verts and faces of the final predicted mesh
        verts_pre, faces_pre = new_src_mesh.get_mesh_verts_faces(0)

        # Scale normalize back to the original target size
        verts_pre = verts_pre * model_scale + model_center

        # Store the predicted mesh using save_obj
        obj_pre = os.path.join(save_path, "{}.obj".format(KITTI_CLASSES[index]))
        save_obj(obj_pre, verts_pre, faces_pre)

    def _save_pred_6dof_annotation(self, sample, pose_6dof, model_ids, json_save_path):
        save_name = json_save_path.split("/")[-1].split(".")[0]
        data = self.getEmptyJson(save_name)
        data["model"]["num"] = len(pose_6dof)

        for i, pose in enumerate(pose_6dof):
            model_save_path = f"models/{KITTI_CLASSES[model_ids[i]]}.obj"

            model_class = KITTI_CLASSES[model_ids[i]]
            model_class_id = model_ids.tolist()[i]

            pose = pose[model_class_id, :]
            pose_t = [pose[0], pose[1], -pose[2] + get_distance(config["camera_fov"])]
            pose_r = np.dot(RQuaternion_2_RMatrix(pose[3:]), Rotate_x_axis(-90)).tolist()
            pose_r = [pose_r[0], pose_r[2], pose_r[1]]

            matrix = np.column_stack([pose_r, pose_t])
            matrix = np.row_stack([matrix, np.array([0, 0, 0, 1])]).tolist()

            model = load_obj(os.path.join(self.pred_model_path, f"{KITTI_CLASSES[model_ids[i]]}.obj"))[0]

            model_size = (model.max(dim=0)[0] - model.min(dim=0)[0]).tolist()

            data["model"]["{}".format(i)] = {
                "model_file": model_save_path,
                "matrix": matrix,
                "class": model_class_id,
                "class_name": model_class_id,
                "size": model_size
            }

        with open(json_save_path, 'w+') as f:
            json.dump(data, f, indent=4)

    def _save_tgt_6dof_annotation(self, sample, pose_6dof, model_ids, json_save_path):
        save_name = json_save_path.split("/")[-1].split(".")[0]
        data = self.getEmptyJson(save_name)
        data["model"]["num"] = pose_6dof.shape[0]

        for i, pose in enumerate(pose_6dof):
            model_save_path = f"models/{KITTI_CLASSES[model_ids[i]]}.obj"

            model_class = KITTI_CLASSES[model_ids[i]]
            model_class_id = model_ids.tolist()[i]

            pose_t = [pose[0], pose[1], -pose[2] + get_distance(config["camera_fov"])]
            pose_r = np.dot(RQuaternion_2_RMatrix(pose[3:]), Rotate_x_axis(-90)).tolist()
            pose_r = [pose_r[0], pose_r[2], pose_r[1]]

            matrix = np.column_stack([pose_r, pose_t])
            matrix = np.row_stack([matrix, np.array([0, 0, 0, 1])]).tolist()

            model = load_obj(os.path.join(self.pred_model_path, f"{KITTI_CLASSES[model_ids[i]]}.obj"))[0]

            model_size = (model.max(dim=0)[0] - model.min(dim=0)[0]).tolist()

            data["model"]["{}".format(i)] = {
                "model_file": model_save_path,
                "matrix": matrix,
                "class": model_class_id,
                "class_name": model_class_id,
                "size": model_size
            }

    def getEmptyJson(self, save_name):
        return {
            "image_file": f"images/{save_name}.png",
            "model": {"num": 0},
            "camera": {
                "matrix": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, get_distance(config["camera_fov"]),
                           0.0, 0.0, 0.0, 1.0],
                "position": [0.0, 0.0, get_distance(config["camera_fov"])],
                "focalPoint": [0.0, 0.0, 0.0],
                "fov": config["camera_fov"],
                "viewup": [0.0, 1.0, 0.0],
                "distance": get_distance(config["camera_fov"])
            }
        }

