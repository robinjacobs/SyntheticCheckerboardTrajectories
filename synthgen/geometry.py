from typing import List
import numpy as np
from scipy.spatial.transform import Rotation as R


def crop_selection(points: np.array, image_boundaries: np.array, axis: int = 1):
    return np.all(
        (points >= image_boundaries[:, 0]) & (points <= image_boundaries[:, 1]),
        axis=axis,
        keepdims=True,
    )


def crop_image_points(points: np.array, image_boundaries: np.array, axis: int = 1):
    "Crop image points s.t., only one visible which are within image boundary"
    sel_pts = crop_selection(points, image_boundaries, axis)
    return points[sel_pts]


def transform_to_frame(points: np.array, transform: np.array):
    """Transform points in original frame
    w.r.t. new frame defined in transform as [old_to_new_rot_vect, old_to_new_translation]
    """
    old_to_new_rot_vect = transform[:3]
    old_to_new_translation = transform[3:]

    rotated_points = rotate(points, old_to_new_rot_vect[None, :])  # TODO check
    transformed_points = rotated_points + old_to_new_translation

    return transformed_points


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.

    See https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid="ignore"):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return (
        cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v
    )


def points2d_in_ideal_cam_coord(points, camera_params):
    f1 = camera_params[:, 6]
    f2 = camera_params[:, 7]
    k1 = camera_params[:, 8]
    k2 = camera_params[:, 9]

    points[:, 0] = (points[:, 0] - k1) / f1
    points[:, 1] = (points[:, 1] - k2) / f2

    return points


def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting them to the image plane

    Arguments:
        camera_params[:, 3:6] : translation from camera to world frame expressed in camera frame
        camera_params[:, :3] : rotation vector transforming points from world frame to camera frame

    """
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f1 = camera_params[:, 6]
    f2 = camera_params[:, 7]
    k1 = camera_params[:, 8]
    k2 = camera_params[:, 9]
    points_proj[:, 0] = points_proj[:, 0] * f1 + k1
    points_proj[:, 1] = points_proj[:, 1] * f2 + k2
    return points_proj


def get_cam_pos_world(camera_params):  # TODO make s.t., can have batch dim
    rot_vec = camera_params[:, :3]
    r = R.from_rotvec(rot_vec)
    rot_mat = r.inv().as_matrix().squeeze(0)
    cam_world_pos = -rot_mat @ camera_params[:, 3:6].squeeze(0)

    return cam_world_pos


def cam_extrinsic_to_homogeneos_tf_matrix(cam_extrinsic: np.array):
    # Extract rotation matrix from the rotation vector
    r = R.from_rotvec(cam_extrinsic[:3])
    r = r.as_matrix()
    t = cam_extrinsic[3:]

    tf = np.concatenate((r, t[:, None]), axis=-1)
    tf = np.vstack((tf, np.array([0, 0, 0, 1])))
    return tf


def cam_intrinsics_to_intrinsics_matrix(cam_intrinsic: np.array):
    return np.array(
        [
            [cam_intrinsic[0], 0, cam_intrinsic[2]],
            [0, cam_intrinsic[1], cam_intrinsic[3]],
            [0, 0, 1],
        ]
    )


def affine4d_to_rvectra(affine_matrix_list: List[np.array]):
    rvecttra_list = [
        np.concatenate([R.from_matrix(el[:3, :3]).as_rotvec(), el[:3, 3].flatten()])
        for el in affine_matrix_list
    ]
    return rvecttra_list
