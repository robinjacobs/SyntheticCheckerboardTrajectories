from typing import List, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R

from synthgen.plot import plot_camera_poses


def generate_rotations(n_points: int) -> np.array:
    """Generate n_points uniformly random orientation in SO(3) in upper half sphere"""
    rotations = []
    rot_ = []

    for _ in range(n_points):
        valid = False
        while not valid:
            theta = (
                2.0 * np.pi * np.random.random()
            )  # Uniformly distributed azimuthal angle
            phi = np.arccos(
                2.0 * np.random.random() - 1.0
            )  # Uniformly distributed polar angle

            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            vec = np.array([x, y, z])
            angle = np.random.random() * np.pi * 2

            rot_vec = vec * angle
            rot_mat = R.from_rotvec(rot_vec)

            # only accept if z axis still points upwards
            valid = np.dot(rot_mat.inv().as_matrix()[2, :], np.array([0, 0, 1])) >= 0

        rotations.append(rot_mat.as_matrix())
        rot_.append(rot_mat)

    tf_list = [np.concatenate([el.as_rotvec(), np.array([0, 0, 0])]) for el in rot_]
    plot_camera_poses(tf_list)

    return rotations


def get_random_points_in_working_cube(
    n_points: int, center_point: np.array, cube_boundaries: np.array
) -> np.array:
    """Sample n_points uniformly in working cube volume"""
    # Calculate the half-lengths of the cube along each axis
    half_lengths = (cube_boundaries[:, 1] - cube_boundaries[:, 0]) / 2

    # Generate random points within the unit cube
    points = np.random.rand(n_points, 3) - 0.5

    # Scale and shift points to the desired cube
    points = points * half_lengths + center_point

    return points


def get_random_sample_vectors(
    n_points: int = 30,
    start_point: np.array = np.array([0, 0, 0]),
    dist: float = 0.3,
    radius_bound: Optional[List] = None,
) -> np.array:
    """Sample n_points following  a random walk along the base vectors"""
    pts_list = [start_point]
    for i in range(1, n_points):
        pt = pts_list[i - 1]

        dir_ = np.random.randint(0, 3)
        sign = np.random.randint(0, 2)

        if dir_ == 0:
            dir_vec = np.array([1, 0, 0])
        if dir_ == 1:
            dir_vec = np.array([0, 1, 0])
        if dir_ == 2:
            dir_vec = np.array([0, 0, 1])

        sign_ = 1 if sign < 1 else -1

        dir_vec = dir_vec * sign_
        pt_new = pt + dist * dir_vec

        if radius_bound is not None:
            if np.linalg.norm(pt_new - start_point) > radius_bound:
                pt_new = pt

        pts_list.append(pt_new)
    pts_lis_np = np.array(pts_list)

    return pts_lis_np
