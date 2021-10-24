from .li3d_dataset import Li3dDataset
from .li3d_dataset import build as build_li3d


def get_orig_data_from_dataset(dataset):
    def select_data(item):
        return dataset.get_orig_data(item)

    return select_data


def build_dataset(image_set, config):
    if config["dataset_name"] == "kitti" or config["dataset_name"] == "UA-DETRAC":
        return build_li3d(image_set, config)
    raise ValueError(f'dataset {config["dataset_name"]} not supported')
