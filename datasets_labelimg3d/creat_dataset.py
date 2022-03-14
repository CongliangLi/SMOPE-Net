import random
import os
import shutil


def get_all_path(open_file_path):
    rootdir = open_file_path
    path_list = []
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        com_path = os.path.join(rootdir, list[i])
        if os.path.isfile(com_path):
            path_list.append(com_path)
        if os.path.isdir(com_path):
            path_list.extend(get_all_path(com_path))
    return path_list


def creat_datasets(dataset_path, ratio=[2, 1, 1]):
    """

    Args:
        dataset_path: path of the scene of labelImg3d
        Ratio: the ratios of [train, val, test]

    Returns:

    """
    source_img_path = os.path.join(dataset_path, "images")
    source_anno_path = os.path.join(dataset_path, "annotations")
    source_model_path = os.path.join(dataset_path, "models")

    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    test_path = os.path.join(dataset_path, "test")
    model_path = os.path.join(dataset_path, "model_{}".format(dataset_path.split("/")[-1]))

    dataset = [train_path, val_path, test_path, model_path]
    print("Start deleting the existing file!")
    for p in dataset:
        if os.path.exists(p):
            shutil.rmtree(p)
    print("Finish deleting the existing file!")

    train_img_path = os.path.join(train_path, "images")
    train_annotation_path = os.path.join(train_path, "annotations")

    val_img_path = os.path.join(val_path, "images")
    val_annotation_path = os.path.join(val_path, "annotations")

    test_img_path = os.path.join(test_path, "images")
    test_annotation_path = os.path.join(test_path, "annotations")

    path = [train_path, val_path, model_path,
            train_img_path, train_annotation_path,
            val_img_path, val_annotation_path,
            test_img_path, test_annotation_path]

    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)

    all_source_img_path = get_all_path(source_img_path)
    all_source_model_path = get_all_path(source_model_path)

    train_num = int(ratio[0] / (ratio[0] + ratio[1] + ratio[2]) * len(all_source_img_path))
    val_num = int(ratio[1] / (ratio[0] + ratio[1] + ratio[2]) * len(all_source_img_path))

    random_num = random.sample(range(0, len(all_source_img_path)), len(all_source_img_path))

    for p in all_source_model_path:
        if p.split(".")[-1] == "obj":
            shutil.copy(p, model_path)

    for i in range(train_num):
        shutil.copy(all_source_img_path[random_num[i]],
                    os.path.join(train_img_path, all_source_img_path[random_num[i]].split("images")[-1].split("/")[-1]))

        shutil.copy(source_anno_path + all_source_img_path[random_num[i]].split("images")[-1].split(".")[0] + ".json",
                    os.path.join(train_annotation_path,
                                 all_source_img_path[random_num[i]].split("images")[-1].split(".")[0].split("/")[
                                     -1] + ".json"))
        print(i)

    for i in range(train_num, train_num + val_num):
        shutil.copy(all_source_img_path[random_num[i]],
                    os.path.join(val_img_path, all_source_img_path[random_num[i]].split("images")[-1].split("/")[-1]))

        shutil.copy(source_anno_path + all_source_img_path[random_num[i]].split("images")[-1].split(".")[0] + ".json",
                    os.path.join(val_annotation_path,
                                 all_source_img_path[random_num[i]].split("images")[-1].split(".")[0].split("/")[
                                     -1] + ".json"))
        print(i)
    for i in range(train_num + val_num, len(all_source_img_path)):
        shutil.copy(all_source_img_path[random_num[i]],
                    os.path.join(test_img_path, all_source_img_path[random_num[i]].split("images")[-1].split("/")[-1]))

        shutil.copy(source_anno_path + all_source_img_path[random_num[i]].split("images")[-1].split(".")[0] + ".json",
                    os.path.join(test_annotation_path,
                                 all_source_img_path[random_num[i]].split("images")[-1].split(".")[0].split("/")[
                                     -1] + ".json"))
        print(i)


if __name__ == '__main__':
    dataset_path = "../data/KITTI3D"
    creat_datasets(dataset_path, [3, 1, 1])
    print("finish creating datasets")
