import numpy as np
from scipy.spatial.transform import Rotation as R
from trimesh import registration


def gcd(first: int, second: int):
    assert isinstance(first, int) and isinstance(second, int)
    if first < second:
        tmp = first
        first = second
        second = tmp
    divisor = 1
    for i in range(1, second + 1):
        if first % i == 0 and second % i == 0:
            divisor = i

    return divisor


def lcm(first: int, second: int):
    return np.abs(first * second) / gcd(first, second)


def sample_tracker_tf(center: np.array, boundary: np.array):
    # Calculate the half-lengths of the cube along each axis
    diff = boundary[:, 1] - boundary[:, 0]

    # Generate random points within the unit cube
    point = diff * np.random.rand(1, 3) + boundary[:, 0]
    # Scale and shift points to the desired cube
    pos = point + center
    pos = pos.flatten()
    # print(pos.flatten())

    ori = np.identity(3)
    r = R.from_matrix(ori)
    rot_vec = r.as_rotvec()

    pos = -ori @ pos  # needs to be in tracker frame

    return np.concatenate([rot_vec, pos.flatten()])


def sample_camera_extrinsics(
    num_cams: int, radius: float, center: np.array, z_boundary: np.array
):
    """Sample camera on cylindrical surface within certain height specified by z_boundary from center point
    returns: np.array([rot_vec_world_to_camera, translation_cam_to-world_in_cam_frame])
    """
    # angle_btw = 2 * np.pi / num_cams
    angles = np.random.rand(num_cams) * np.pi * 2
    diff_z = z_boundary[1] - z_boundary[0]
    height = z_boundary[0] + (np.random.rand(num_cams)) * diff_z

    rot_list = []
    for i, angle in enumerate(angles):
        cam_pos = np.array(
            [
                center[0] + radius * np.cos(angle),
                center[1] + radius * np.sin(angle),
                center[2] + height[i],
            ]
        )
        cam_to_center = center - cam_pos

        nz = cam_to_center / np.linalg.norm(cam_to_center)
        up = np.array([0, 0, 1])
        nx = np.cross(nz, up)
        ny = np.cross(nz, nx)
        rot_mat = np.stack([nx, ny, nz], axis=1)

        cam_to_world_cam_frame = rot_mat.T @ (-cam_pos)
        r = R.from_matrix(rot_mat.T)
        rot_list.append(np.concatenate([r.as_rotvec(), cam_to_world_cam_frame]))

    return rot_list


def cameras_around_origin_on_xy_plane(num_cams, radius=2.5, offset=1.5):
    angle_btw = 2 * np.pi / num_cams
    angles = np.arange(0, num_cams) * angle_btw

    rot_list = []

    for angle in angles:
        pos = np.array([0, offset, radius])  # cam to world

        nz = np.array([-np.cos(angle), -np.sin(angle), 0])
        nx = np.array([nz[1], -nz[0], 0])
        ny = -np.array([0, 0, 1])

        rot_mat = np.stack([nx, ny, nz], axis=1)

        # rot around x axis
        r_adj = R.from_euler("xyz", [20, 0, 0], degrees=True)
        rot_mat = rot_mat @ r_adj.as_matrix().T

        r = R.from_matrix(rot_mat.T)
        rot_list.append(np.concatenate([r.as_rotvec(), pos]))

    return rot_list


if __name__ == "__main__":
    cameras_around_origin_on_xy_plane(5)


def get_sim_transform(from_points: np.array, to_points: np.array):
    """Calculates transformation `from_points` to `to_points`"""
    s_procrustes_result, _, cost = registration.procrustes(
        from_points,
        to_points,
        reflection=False,
        scale=False,
        translation=True,
        return_cost=True,
    )

    t = s_procrustes_result[:3, 3].astype(np.float32)
    scaled_rot = s_procrustes_result[:3, :3].astype(np.float32)

    scale = np.linalg.norm(scaled_rot, axis=0)[0]
    rotation_matrix = scaled_rot / scale
    return rotation_matrix, t, scale, cost
