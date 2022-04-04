from numpy.ma.core import nomask
import pandas as pd
import torch
from pathlib import Path
import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torchvision
# from datasets.transforms import resize
from util.utils import get_camera_intrinsics, RMatrix_2_RQuaternion, RQuaternion_2_RMatrix, RMatrix_2_REuler, \
    REuler_2_RMatrix, get_all_path, parse_yaml
from PIL import Image
import datasets_labelimg3d.transforms as T
from pytorch3d.io import load_obj, save_obj
from torchvision import transforms
from configs import cfg, config
from util.box_ops import box_cxcywh_to_xyxy


# Set the device
if torch.cuda.is_available():
    device = torch.device(config["device"])
else:
    device = torch.device("cpu")
    print("WARNING: CPU only when reading models, this will be slow!")


CLASSES = config["model"]["classes"]


class SingleImgParser:
    def __init__(self, img_path, annotation_data):
        
        self.img_path = img_path
        self.img_num = int(img_path.split("/")[-1].split(".")[0])
        self.anno_data = annotation_data[self.img_num]

        self.orig_size = Image.open(img_path).size
        self.camera_matrix = get_camera_intrinsics(config["camera_fov"], self.orig_size)
        self.model_ids, self.class_names, self.bboxes_2d = [], [], []

        self.R_matrix_c2o, self.R_quaternion_c2o, self.R_euler_c2o = [], [], []
        self.T_matrix_c2o = []
        self.class_ids = []
        self.model_num = len(self.anno_data)

        for anno in self.anno_data:
            model_id = config["model"]["model_id_map"][str(anno["obj_id"])]
            self.model_ids.append(model_id)
            self.class_ids.append(0)
            self.class_names.append(config["model"]["classes"][model_id])
            self.bboxes_2d.append(anno["obj_bb"])  # (l, t, w, h)
            self.T_matrix_c2o.append(anno["cam_t_m2c"])

            self.R_matrix_c2o.append(anno["cam_R_m2c"])
            self.R_quaternion_c2o.append(RMatrix_2_RQuaternion(
                np.array(anno["cam_R_m2c"]).reshape(3, 3)).tolist())
            self.R_euler_c2o.append(
                RMatrix_2_REuler(np.array(anno["cam_R_m2c"]).reshape(3, 3)).tolist())

    def __getitem__(self):
        img = Image.open(self.img_path)
        boxes = torch.tensor(self.bboxes_2d)
        if len(boxes) == 0:
            print(self.img_path)

        targets = {  
            'bboxes_2d': torch.cat([boxes[:, :2] + boxes[:,2:] / 2, boxes[:,2:]], dim=-1) if len(boxes) != 0 else torch.empty(0, 4),  # convert (l,t,w,h) to (cxcywh)
            "model_ids": torch.tensor(self.model_ids).long() if len(boxes) != 0 else torch.empty(0).long(),
            "labels": torch.tensor(self.class_ids).long() if len(boxes) != 0 else torch.empty(0).long(),
            "T_matrix_c2o": torch.tensor(self.T_matrix_c2o)/1000 if len(boxes) != 0 else torch.empty(0, 3),
            "R_quaternion_c2o": torch.tensor(self.R_quaternion_c2o) if len(boxes) != 0 else torch.empty(0, 4),
            "R_euler_c2o": torch.tensor(self.R_euler_c2o) if len(boxes) != 0 else torch.empty(0, 3),
            'orig_size': torch.tensor(self.orig_size),
            "img_path": torch.tensor(int(self.img_path.split("/")[-1].split(".")[0]))
        }
        return img, targets


class ModelParser:
    def __init__(self, model_folder, model_name=config["model"]["model_name"]):
        self.model_path = []
        for model in model_name:
            self.model_path.append(os.path.join(model_folder, model))

    def __getitem__(self):
        model_target = []
        for p in self.model_path:
            verts, _, _ = load_obj(p)
            model_target.append(verts)

        return model_target


class Li3dDataset(Dataset):
    """
    the datasets Annotated using LabelImg3d
    """

    def __init__(self, img_folder, anno_path, choose_data,transform=None):
        self.anno_data = parse_yaml(anno_path)

        self.img_folder = img_folder
        self.img_path = [p[:4]+".png" for p in open(choose_data).readlines()]
        self.transform = transform

        self.data = []
        self.lens = 0

        self.load_data()

    def load_data(self):
        # analysis files in DETRAC-Train-Annotations-MOT
        assert self.img_folder and self.anno_data and self.img_path is not None
        
        all_img_path = [os.path.join(self.img_folder, p) for p in self.img_path]
        pbar = tqdm(all_img_path)
        self.lens = len(pbar)
        for path in pbar:
            pbar.set_description(f'reading: {path}')
            self.data = self.data + [SingleImgParser(path, self.anno_data).__getitem__()]


    def __len__(self):
        return self.lens

    def __getitem__(self, item):
        assert item < self.__len__(), f'the item of dataset must less than length {self.__len__()}, but get {item}'
        img, targets = self.data[item]
        if self.transform is not None:
            img, targets = self.transform(img, targets)

        return img, targets

    def get_orig_data(self, item):
        assert item < self.__len__(), f'the item of dataset must less than length {self.__len__()}, but get {item}'
        img, targets = self.data[item]
        return img, targets


def make_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(config["pixel_mean"], config["pixel_std"])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomResize([config["image_height"]], max_size=config["image_width"]),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([config["image_height"]], max_size=config["image_width"]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.dataset_path)
    assert root.exists(), f'provided labelImg3d scene folder path {root} does not exist'
    PATHS = {
        "train": (root / "rgb", root / "gt.yml", root / "train.txt"),
        "val": (root / "rgb", root / "gt.yml", root / "test.txt")
    }

    img_folder, ann_path, choose_data = PATHS[image_set]
    dataset = Li3dDataset(img_folder=img_folder,
                          anno_path=ann_path,
                          choose_data = choose_data,
                          transform=make_transforms(image_set)
                          )
    return dataset


if __name__ == '__main__':
    anno_path = "../../data/kitti/val/annotations/000002.json"
    img_path = "../../data/kitti/val/images/000002.png"
    model_folder = "../../data/kitti/val/model"
    img_folder = "../../data/kitti/val/annotations"
    anno_folder = "../../data/kitti/val/images"
    # ann = SingleAnnotationParser(img_path, anno_path)
    # ann.__getitem__()
    # m = ModelParser(model_folder, config["model"]["model_name"])
    # m.__getitem__()

    dataset = build("train", config)
    dataset.__getitem__(0)
    pass
