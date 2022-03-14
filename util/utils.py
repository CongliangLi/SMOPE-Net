import os
import numpy as np
from numpy.linalg import inv
import cv2
from math import atan, radians, degrees, cos, sin
import yaml
from scipy.spatial.transform import Rotation
from plyfile import PlyData


def cart2hom(pts_3d):
    """ Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    """
    assert len(pts_3d.shape) == 2
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


def hom2cart(pts_3d):
    assert len(pts_3d.shape) == 2
    return pts_3d[:, :-1]


def getAngle(x, y):
    return np.arctan2(
        (np.cross(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))).sum(),
        (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))).sum()
    )


def draw_projected_box3d(image, qs, color=[(255, 0, 0), (0, 0, 255), (0, 255, 0)],
                         thickness=2):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    if qs is None:
        return image
    qs = qs[[7, 5, 4, 6, 3, 1, 0, 2], :].astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color[0], thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color[1], thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color[2], max(1, thickness // 2))
    return image


def reconnect(signal, newhandler=None, oldhandler=None):
    try:
        if oldhandler is not None:
            while True:
                signal.disconnect(oldhandler)
        else:
            signal.disconnect()
    except TypeError:
        pass
    if newhandler is not None:
        signal.connect(newhandler)


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


def get_dirname(path):
    """

    Args:
        path: dir path

    Returns:
        dir_name: all dir names in the file path

    """
    dir_path = []
    for lists in os.listdir(path):
        sub_path = os.path.join(path, lists)
        if os.path.isdir(sub_path):
            dir_path.append(sub_path)

    return dir_path


# Solve the problem that opencv can't read Chinese path
def cv_imread(filepath):
    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    return img


# get cot of angle system
def cot(angle):
    angle = radians(angle)
    return cos(angle) / sin(angle)


# Calculating distance from fov(angle value)
def get_distance(fov):
    return round(1 / 2 * cot(fov / 2), 2)


# Calculating fov(angle value) from distance
def get_fov(distance):
    return round(2 * atan(1 / (2 * distance)), 2)


# parse yaml to dict
def parse_yaml(yaml_path):
    """
   Reads a yaml file
    Args:
        yaml_path: Path to the yaml file
    Returns:
        yaml_dic: Dictionary containing the yaml file content

    """

    if not os.path.isfile(yaml_path):
        print("Error: file {} does not exist!".format(yaml_path))
        return None

    with open(yaml_path) as fid:
        yaml_dic = yaml.safe_load(fid)

    return yaml_dic


# Calculating camera intrinsics from fov and image size
def get_camera_intrinsics(fov_h, img_size):
    """

    Args:
        fov_h: horizontal fov of camera
        img_size: image_size [w,h]

    Returns:
        camera intrinsics matrix（3*3）

    """
    w, h = img_size
    f = round(w / 2 * cot(fov_h / 2), 2)
    fov_v = 2 * atan(h / (2 * f))

    cx = w / 2
    cy = h / 2

    camera_intrinsics = [f, 0, cx,
                         0, f, cy,
                         0, 0, 1]

    return camera_intrinsics


def get_R_obj2w(model_matrix):
    """

    Args:
        model_matrix: model matrix of annotations  shape:(16,) or (4,4)

    Returns: Rotation matrix for object to world  (3,3)

    """
    if model_matrix.shape == (16,):
        model_matrix = model_matrix.reshape(4, 4)

    return model_matrix[:3, : 3]


def get_R_w2c():
    """

    Returns: Rotation matrix for world to camera (3,3)

    """
    return np.dot(np.dot(Rotate_y_axis(180), Rotate_z_axis(-90)), np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]))


def get_R_obj2c(model_matrix):
    """

    Args:
        model_matrix: model matrix of annotations  shape:(16,) or (4,4)

    Returns: Rotation matrix for object to camera  (3,3)

    """
    # return np.dot(get_R_obj2w(model_matrix), get_R_w2c())
    return np.dot(get_R_w2c(), get_R_obj2w(model_matrix))


def get_T_obj2w(model_matrix):
    """

    Args:
        model_matrix: model matrix of annotations  shape:(16,) or (4,4)

    Returns: Rotation matrix for object to world  (shape:(1,3) unit:meter)

    """
    if model_matrix.shape == (16,):
        model_matrix = model_matrix.reshape(4, 4)

    return np.concatenate((-model_matrix[:2, -1], [model_matrix[2, -1]]))


def get_T_w2c(fov):
    """

    Args:
        fov: camera fov (angle value)

    Returns: Translocation matrix of world to camera   (shape:(1,3) unit:meter)

    """
    return np.array([0, 0, -get_distance(fov)])


def get_T_obj2c(model_matrix, fov):
    """

    Args:
        model_matrix: model matrix of annotations  shape:(16,) or (4,4)
        fov: camera fov (angle value)

    Returns: Translocation matrix of world to camera (shape:(1,3) unit:meter)

    """
    return get_T_obj2w(model_matrix) + get_T_w2c(fov)


def get_T_obj_bottom2center(obj_size):
    """

    Args:
        obj_size: obj size (x,y,z) unit:meter

    Returns: Translocation matrix from bottom to center (shape:(1,3) unit:meter)

    """
    return np.array([0, 0, obj_size[2] / 2])


def load_model_ply(path_to_ply_file):
    """
   Loads a 3D model from a plyfile
    Args:
        path_to_ply_file: Path to the ply file containing the object's 3D model
    Returns:
        points_3d: numpy array with shape (num_3D_points, 3) containing the x-, y- and z-coordinates of all 3D model points

    """
    model_data = PlyData.read(path_to_ply_file)
    vertex = model_data['vertex']
    points_3d = np.stack([vertex[:]['x'], vertex[:]['y'], vertex[:]['z']], axis=-1)
    return points_3d


def Rotate_x_axis(theta):
    """

    Args:
        theta: angle value

    Returns: the matrix (3,3)

    """
    theta = radians(theta)
    return np.array([[1., 0., 0.], [0., cos(theta), -sin(theta)], [0., sin(theta), cos(theta)]])


def Rotate_y_axis(theta):
    """

    Args:
        theta: angle value

    Returns: the matrix (3,3)

    """
    theta = radians(theta)
    return np.array([[cos(theta), 0., sin(theta)], [0., 1., 0.], [-sin(theta), 0., cos(theta)]])


def Rotate_z_axis(theta):
    """

    Args:
        theta: angle value

    Returns: the matrix (3,3)

    """
    theta = radians(theta)
    return np.array([[cos(theta), -sin(theta), 0.], [sin(theta), cos(theta), 0.], [0., 0., 1.]])



def get_R_w2c():
    """

    Returns: Rotation matrix for world to camera (3,3)

    """
    return np.dot(np.dot(Rotate_y_axis(180), Rotate_z_axis(-90)), np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]))



def rotation_mat_to_axis_angle(rotation_matrix):
    """
    Computes an axis angle rotation vector from a rotation matrix
    Arguments:
        rotation_matrix: numpy array with shape (3, 3) containing the rotation
    Returns:
        axis_angle: numpy array with shape (3,) containing the rotation
    """
    axis_angle, jacobian = cv2.Rodrigues(rotation_matrix)

    return np.squeeze(axis_angle)


def axis_angle_to_rotation_mat(rotation_vector):
    """
    Computes a rotation matrix from an axis angle rotation vector
    Arguments:
        rotation_vector: numpy array with shape (3,) containing the rotation
    Returns:
        rotation_mat: numpy array with shape (3, 3) containing the rotation
    """
    rotation_mat, jacobian = cv2.Rodrigues(np.expand_dims(rotation_vector, axis=-1))

    return rotation_mat


def draw_box(image, box, color=(255, 255, 0), thickness=1):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)
    return image


def trans_3d_2_2d(points_3d, R_obj2c, T_obj2c, camera_intrinsics):
    """

    Args:
        points_3d: 3d points in object coordinate
        R_obj2c: the rotation matrix from object coordinate to camera coordinate   (numpy, shape(3, 3))
        T_obj2c: the trans matrix from object coordinate to camera coordinate   (numpy, shape(1, 3))
        camera_intrinsics: the camera intrinsics  (numpy, shape(3, 3))

    Returns:
        points_2d: 2d points in image coordinate
    """
    points_3d = np.dot(points_3d, R_obj2c) + np.squeeze(T_obj2c)
    points_2d, _ = cv2.projectPoints(points_3d, np.zeros((3,)), np.zeros((3,)), camera_intrinsics, None)

    points_2d = np.squeeze(points_2d)
    points_2d = np.copy(points_2d).astype(np.int32)
    return points_2d


def get_mask_img(img_size, model_3d_points, R_obj2c, T_obj2c, camera_intrinsics):
    """ get image mask of a object

    Args:
        img_size: image size of the origin image [ , ]
        model_3d_points: 3D points of model in object coordinate  (numpy)
        R_obj2c: the rotation matrix from object coordinate to camera coordinate   (numpy, shape(3, 3))
        T_obj2c: the trans matrix from object coordinate to camera coordinate   (numpy, shape(1, 3))
        camera_intrinsics: the camera intrinsics  (numpy, shape(3, 3))

    Returns:
        mask_img: the mask image (numpy)

    """
    mask_img = np.zeros([img_size[1], img_size[0], 3], np.uint8)

    model_2d_points = trans_3d_2_2d(model_3d_points, R_obj2c, T_obj2c, camera_intrinsics)
    tuple_points = tuple(map(tuple, model_2d_points))
    # Lines scan
    for y in range(img_size[1]):
        start = img_size[0]
        end = -1
        for point in tuple_points:
            if point[1] == y:
                if point[0] > end:
                    end = point[0]
                if point[0] < start:
                    start = point[0]

        if start == img_size[0] and end == -1:
            continue

        for x in range(start, end):
            cv2.circle(mask_img, (x, y), 1, (255, 255, 255), -1)

    mask_img_fin = np.zeros([img_size[1], img_size[0], 3], np.uint8)
    # Column scan fill hole area
    for x in range(img_size[0]):
        start = img_size[1]
        end = -1
        for y in range(img_size[1]):
            if mask_img[y][x].tolist() == [255, 255, 255] and y > end:
                end = y
            if mask_img[y][x].tolist() == [255, 255, 255] and y < start:
                start = y

        if start == img_size[1] and end == -1:
            continue

        for y in range(start, end):
            cv2.circle(mask_img_fin, (x, y), 1, (255, 255, 255), -1)

    return mask_img_fin


def get_fps_points(model_3d_points, model_center_3d_point, fps_num=8):
    """

    Args:
        model_3d_points:  3D points of model in object coordinate  (numpy)
        model_center_3d_point: 3D points of model center in object coordinate  (numpy)
        fps_num: default 8

    Returns:

    """
    fps_3d_points = [model_center_3d_point]

    for _ in range(fps_num):
        farthest_point = {"point": [], "distance": 0}
        for point in model_3d_points:
            distance = 0.
            for fps in fps_3d_points:
                distance = distance + ((fps[0] - point[0]) ** 2 + (fps[1] - point[1]) ** 2 +
                             (fps[2] - point[2]) ** 2) ** 0.5
            if distance > farthest_point["distance"]:
                farthest_point["point"] = point
                farthest_point["distance"] = distance

        fps_3d_points.append(farthest_point["point"])

    return np.array(fps_3d_points[1:])


def get_model_bbox_3d(model_path):
    """

    Args:
        model_path: the path of .ply model

    Returns:
       model_bbox_3d: 3d bbox of the model

    """
    model_3d_points = load_model_ply(model_path)
    min_x = model_3d_points.T[0].min()
    min_y = model_3d_points.T[1].min()
    min_z = model_3d_points.T[2].min()
    max_x = model_3d_points.T[0].max()
    max_y = model_3d_points.T[1].max()
    max_z = model_3d_points.T[2].max()

    model_bbox_3d = [[min_x, min_y, min_z], [min_x, min_y, max_z],
                     [min_x, max_y, min_z], [min_x, max_y, max_z],
                     [max_x, min_y, min_z], [max_x, min_y, max_z],
                     [max_x, max_y, min_z], [max_x, max_y, max_z]]

    return model_bbox_3d


def get_model_center_3d(model_path):
    """

    Args:
        model_path: the path of .ply model

    Returns:
        model_center_3d: 3d center point of model
    """
    model_3d_points = load_model_ply(model_path)
    model_center_3d = [0, 0, (model_3d_points.T[2].max() - model_3d_points.T[2].min()) / 2]

    return model_center_3d


def RMatrix_2_RQuaternion(R_matrix):
    """
    from Rotation matrix to Rotation quaternion
    Args:
        R_matrix: Rotation matrix, shape(3, 3)

    Returns: Rotation quaternion, shape(1, 4)

    """
    return Rotation.from_matrix(R_matrix).as_quat()


def RQuaternion_2_RMatrix(R_quaternion):
    """
    from Rotation quaternion to Rotation matrix
    Args:
        R_quaternion: Rotation quaternion, shape(1, 4)

    Returns: Rotation matrix, shape(3, 3)

    """
    return Rotation.from_quat(R_quaternion).as_matrix()


def RMatrix_2_REuler(R_matrix, R_order="xyz", degrees=True):
    """

    Args:
        R_order : string, length 3, optional
                3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic
                rotations, or {'x', 'y', 'z'} for extrinsic rotations [1]_.
                Adjacent axes cannot be the same.
                Extrinsic and intrinsic rotations cannot be mixed in one function
                call.
        degrees : boolean, optional
                Returned angles are in degrees if this flag is True, else they are
                in radians. Default is False.

    Returns:

    """
    return Rotation.from_matrix(R_matrix).as_euler(R_order, degrees)


def REuler_2_RMatrix(R_euler, R_order="xyz", degrees=True):
    """

    Args:
        R_euler: Euler angles
        R_order : string, length 3, optional
                3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic
                rotations, or {'x', 'y', 'z'} for extrinsic rotations [1]_.
                Adjacent axes cannot be the same.
                Extrinsic and intrinsic rotations cannot be mixed in one function
                call.
        degrees : boolean, optional
                Returned angles are in degrees if this flag is True, else they are
                in radians. Default is False.

    Returns:

    """

    if len(R_euler) != 3:
        return
    else:
        return Rotation.from_euler(R_order,R_euler,degrees).as_matrix()
