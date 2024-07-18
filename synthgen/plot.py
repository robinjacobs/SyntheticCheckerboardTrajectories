from typing import List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

from synthgen.constants import KINECT_HEIGHT_PX, KINECT_WIDTH_PX
from synthgen.geometry import get_cam_pos_world


def plot_camera_poses(
    camera_list: List[np.array],
    ax: Axes3D = None,
    name: str = "C",
    plot_frustum: bool = True,
    *args,
    **kwargs,
):
    """
    Plots camera poses in 3D.

    Args:
        rotation_vectors (list of np.ndarray): List of rotation vectors (6-dimensional).
            Each vector represents the position and orientation of a camera.

    Returns:
        None
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = None

    # Define camera frame axes
    if "scale" not in kwargs:
        scale = 1.0
    else:
        scale = kwargs["scale"]
        kwargs.pop("scale")

    axes = scale * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    for i, rvec in enumerate(camera_list):
        # Extract rotation matrix from the rotation vector
        r = R.from_rotvec(rvec[:3])
        R_matrix = r.as_matrix()

        # Compute camera frame axes in world coordinates
        camera_axes = R_matrix @ axes

        # Extract camera position
        cam_world_pos = get_cam_pos_world(rvec[None, :])

        # camera_position = rvec[3:6]
        camera_position = cam_world_pos.flatten()

        # Plot camera frame axes
        for j in range(3):
            assert np.isclose(np.linalg.norm(camera_axes[j, :]), scale)
            ax.quiver(
                camera_position[0],
                camera_position[1],
                camera_position[2],
                camera_axes[j, 0],
                camera_axes[j, 1],
                camera_axes[j, 2],
                color=["r", "g", "b"][j],
                label=f"Camera {i+1}" if j == 0 else None,
                *args,
                **kwargs,
            )

        if plot_frustum:
            cam_scale_factor = 2.5e-4
            focal_length = cam_scale_factor * rvec[7]
            x = KINECT_WIDTH_PX * cam_scale_factor
            y = KINECT_HEIGHT_PX * cam_scale_factor
            fovx = 2 * np.arctan(x / (2 * focal_length))
            fovy = 2 * np.arctan(y / (2 * focal_length))

            p1 = focal_length * (
                camera_axes[2, :]
                + np.tan(fovx / 2.0) * camera_axes[0, :]
                + np.tan(fovy / 2.0) * camera_axes[1, :]
            )
            p2 = focal_length * (
                camera_axes[2, :]
                + np.tan(fovx / 2.0) * camera_axes[0, :]
                - np.tan(fovy / 2.0) * camera_axes[1, :]
            )
            p3 = focal_length * (
                camera_axes[2, :]
                - np.tan(fovx / 2.0) * camera_axes[0, :]
                - np.tan(fovy / 2.0) * camera_axes[1, :]
            )
            p4 = focal_length * (
                camera_axes[2, :]
                - np.tan(fovx / 2.0) * camera_axes[0, :]
                + np.tan(fovy / 2.0) * camera_axes[1, :]
            )

            v = np.array([[0, 0, 0], p1, p2, p3, p4]) + camera_position
            verts = [
                [v[0], v[1], v[2]],
                [v[0], v[2], v[3]],
                [v[0], v[3], v[4]],
                [v[0], v[4], v[1]],
                [v[0], v[1], v[2], v[3]],
            ]
            ax.add_collection3d(
                Poly3DCollection(
                    verts,
                    facecolors="cyan",
                    linewidths=1,
                    edgecolors="black",
                    alpha=0.25,
                )
            )

        # Add floating text label indicating camera number
        ax.text(
            camera_position[0],
            camera_position[1],
            camera_position[2] + 0.1,
            f"{name} {i+1}",
            color="black",
            fontsize=10,
            ha="center",
            va="center",
        )

    # Set plot limits
    # ax.set_xlim(-10, 10)
    # ax.set_ylim(-10, 10)
    # ax.set_zlim(-10, 10)

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Add legend
    # ax.legend()

    # Show the plot
    if ax is None:
        plt.show()
    return fig, ax


def plot_trajectory_time_colored(pos_trajectory: np.array, ax=None, *args, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    else:
        fig = None

    t = np.linspace(0, 1, len(pos_trajectory))
    cmap = plt.get_cmap("viridis")

    for i in range(1, len(pos_trajectory)):
        ax.plot(
            pos_trajectory[i - 1 : i + 1, 0],
            pos_trajectory[i - 1 : i + 1, 1],
            pos_trajectory[i - 1 : i + 1, 2],
            color=cmap(t[i]),
            *args,
            **kwargs,
        )

    plt.colorbar(
        cm.ScalarMappable(norm=None, cmap=cmap), ax=ax, label="normalized time"
    )
    # ax.plot(pos_trajectory[0,:], pos_trajectory[1,:], pos_trajectory[2,:])
    return fig, ax


def plot_trajectory(pos_trajectory: np.array, ax=None, *args, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    else:
        fig = None
    ax.plot(
        pos_trajectory[:, 0],
        pos_trajectory[:, 1],
        pos_trajectory[:, 2],
        *args,
        **kwargs,
    )
    # ax.plot(pos_trajectory[0,:], pos_trajectory[1,:], pos_trajectory[2,:])
    return fig, ax


def plot_frame_trajectory(frame_tf: np.array, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = None

    # Subsample for performance reasons\
    n_points = 20
    frame_tf_sub = frame_tf[:: len(frame_tf) // n_points]

    plot_camera_poses(frame_tf_sub, ax=ax, alpha=0.3, name="T")

    # Euler Angle s plot
    fig, axs = plt.subplots(1, 3)
    eulers = []
    for t in range(len(frame_tf)):
        tf = frame_tf[t]
        rot_vec = tf[:3]
        euler = R.from_rotvec(rot_vec).as_euler("xzy")

        #
        # euler = euler + np.pi

        eulers.append(euler)

    eulers = np.array(eulers)
    axs[0].plot(eulers[:, 0])
    axs[1].plot(eulers[:, 1])
    axs[2].plot(eulers[:, 2])


def plot_scene(
    pos_trajectory: np.array,
    cameras: List,
    tracker_frame: np.array,
    checkerboard_pts_traj: np.array = None,
    checkerboard_tf_traj: np.array = None,
    name: str = "",
):
    # fig, ax = plot_trajectory(pos_trajectory, color="purple")
    fig, ax = plot_trajectory_time_colored(pos_trajectory)

    plot_camera_poses(cameras, ax=ax)
    plot_camera_poses([tracker_frame], ax=ax, name="T", plot_frustum=False)

    if checkerboard_pts_traj is not None:
        for j in range(checkerboard_pts_traj.shape[1]):
            plot_trajectory(checkerboard_pts_traj[:, j, :], ax, color="gray", alpha=0.7)

    ax.set_aspect("equal")
    fig.suptitle(name)
    plt.show(block=False)


if __name__ == "__main__":
    # Example usage
    rotation_vector1 = np.array([5, 3, 2, -2.9, 0, 0])
    rotation_vector2 = np.array([-2, 4, 1, -0.707, 0, 0])

    rotation_vectors = [rotation_vector1, rotation_vector2]
    plot_camera_poses(rotation_vectors)
