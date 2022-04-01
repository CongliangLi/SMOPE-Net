from .li3d_dataset import Li3dDataset, build
from .li3d_dataset import CLASSES as CLASSES
from .li3d_dataset import build as build_li3d
from .linemod_dataset import build as build_lm


def get_orig_data_from_dataset(dataset):
    def select_data(item):
        return dataset.get_orig_data(item)

    return select_data


def build_dataset(image_set, args):
    if args.dataset_name == "KITTI3D" or args.dataset_name == "UA-DETRAC3D":
        return build_li3d(image_set, args)
    elif args.dataset_name == "Linemod_preprocessed":
        return build_lm(image_set, args)
    raise ValueError(f'dataset {args.dataset_name} not supported')
