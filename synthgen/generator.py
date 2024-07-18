"""Functions for generating trajectories"""

import datetime
import math
import time
from typing import List, Optional, Tuple
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from synthgen.checkerboard import Checkerboard
from synthgen.config import GeneratorConfig
from synthgen.geometry import (
    crop_selection,
    points2d_in_ideal_cam_coord,
    project,
    rotate,
    transform_to_frame,
)
from synthgen.plot import plot_scene, plot_trajectory

time_now_s = time.time()


def remove_occluded_points(
    cameras: List[np.array],
    points_trajectory: np.array,
    checker_frame_traj: np.array,
) -> List[np.array]:
    """Removes checkerboard points if they are non visible from the camera"""
    trajectory_list = []

    for cam in cameras:
        points_trajectory_cp = points_trajectory.copy()
        for t in range(len(points_trajectory_cp)):
            rot_marker_to_world = (
                R.from_rotvec(checker_frame_traj[t, :3]).inv().as_matrix()
            )
            tra_checker_to_world_m = checker_frame_traj[t, 3:]
            z_check_vec = rot_marker_to_world[:, -1]

            rot_cam_to_world = R.from_rotvec(cam[:3]).inv().as_matrix()
            tra_cam_to_world_c = cam[3:6]

            # calculate line of sight from camera to checkerboard origin
            cam_to_checker = (
                rot_cam_to_world @ tra_cam_to_world_c
                - rot_marker_to_world @ tra_checker_to_world_m
            )
            is_visible = cam_to_checker.dot(z_check_vec) < 0
            if not is_visible:
                points_trajectory_cp[t, :, :] = np.nan

        trajectory_list.append(points_trajectory_cp)

    return trajectory_list


def project_all_markers_points(
    cameras: List[np.array],
    points_trajectory_list: List[np.array],
    camera_crop_projected: bool = False,
    camera_boundary_px: Optional[np.array] = None,
) -> Tuple:
    """Project 3d marker points to ideal and pixel coordinates of cameras defined in list"""
    proj_points_px_per_cam = []  # in pixel coordinates (dependent on cam intriniscs)
    proj_points_ideal_coord_per_cam = []  # in ideal camera coordinates

    trajectory_length = points_trajectory_list[0].shape[0]

    for j, (cam, points_trajectory) in enumerate(zip(cameras, points_trajectory_list)):
        proj_points_t = []
        proj_points_ideal_t = []
        for t in range(trajectory_length):
            proj_points = project(points_trajectory[t, :, :], cam[None, :])
            proj_points_t.append(proj_points)

            proj_points_ideal = points2d_in_ideal_cam_coord(proj_points.copy(), cam[None, :])
            proj_points_ideal_t.append(proj_points_ideal)

        proj_points_t = np.array(proj_points_t)
        proj_points_ideal = np.array(proj_points_ideal_t)

        proj_points_px_per_cam.append(proj_points_t)
        proj_points_ideal_coord_per_cam.append(proj_points_ideal)

    if camera_crop_projected:
        # Crop points s.t., only use the ones visible in image frame
        for proj in proj_points_px_per_cam:
            sel = crop_selection(proj, image_boundaries=camera_boundary_px, axis=-1)
            proj[:] = np.where(
                sel, proj, np.nan
            )  # TODO currently this leads that whole frame is ignored in io

    return proj_points_px_per_cam, proj_points_ideal_coord_per_cam


def generate(config: GeneratorConfig, plot: bool = True):
    """Generate checkerboard and tracker trajectory and computes projections for each camera"""
    print(">>Trajectory generation")

    # Checkerboard init
    checkerboard = Checkerboard(config=config.checkerboard)
    fiducials = checkerboard.fiducials
    checkerboard_geometry = checkerboard.points[:, :].T

    # Tracker frame transformation (world to tracker frame)
    if config.tracker is not None and config.tracker.world_to_tracker_tf is not None:
        world_to_tracker_pose = config.tracker.world_to_tracker_tf
    else:
        print("No optical tracker specified will use identity")
        world_to_tracker_pose = np.zeros((6,))

    # Caclulate sampling frequency to be a multiple of tracker and camera frequency
    assert isinstance(config.sample_freq_cameras_hz, int) and isinstance(
        config.sample_freq_tracker_hz, int
    ), "currently only support integer sampling frequencies"
    sampling_freq = math.lcm(
        int(config.sample_freq_cameras_hz), int(config.sample_freq_tracker_hz)
    )
    print("Sampling Frequency:", sampling_freq)
    num_time_stps = int((config.time_end_s - config.time_start_s) * sampling_freq)
    print("Number of time steps (total): ", num_time_stps)
    n_timestps_cameras = sampling_freq / config.sample_freq_cameras_hz
    print("Number of time steps btw camera samples: ", n_timestps_cameras)
    n_timestps_tracker = sampling_freq / config.sample_freq_tracker_hz
    print("Number of time steps btw tracker samples: ", n_timestps_tracker)
    # Get number of time steps for per camera offset
    n_timestps_offset_tracker = config.tracker_t_offset_s * sampling_freq
    assert len(config.cameras_t_offset_s) == len(
        config.cameras.intrinsics
    ), f" {len(config.cameras_t_offset_s)} neq {len(config.cameras.intrinsics)} "
    n_timestps_offset_cameras = [
        np.rint(t_offset * sampling_freq).astype(int)
        for t_offset in config.cameras_t_offset_s
    ]
    print(
        "Number of time steps used for offset of each camera:",
        n_timestps_offset_cameras,
    )

    # "Simulation" Time Vector with dt = 1/sampling_freq
    t_vec = np.linspace(config.time_start_s, config.time_end_s, num_time_stps)
    assert config.time_end_s > config.time_start_s
    t_normalized = (t_vec - config.time_start_s) / (
        config.time_end_s - config.time_start_s
    )
    assert np.all(t_normalized >= 0) and np.all(t_normalized <= 1)

    # Checkerboard origin position trajectory expressed in world frame
    traj_check_origin = config.checkerboard.position_trajectory.function(
        t_normalized, config.checkerboard.position_trajectory.config
    ).T

    # Orientation trajectory
    start_rot = R.from_rotvec(checkerboard.origin[:3]).as_matrix()
    if "start" not in config.checkerboard.orientation_trajectory.config:
        config.checkerboard.orientation_trajectory.config["start"] = start_rot
    if "end" not in config.checkerboard.orientation_trajectory.config:
        end_rot = np.eye(3)
        config.checkerboard.orientation_trajectory.config["end"] = end_rot
    traj_check_orient = config.checkerboard.orientation_trajectory.function(
        t_normalized, config.checkerboard.orientation_trajectory.config
    )  # Trajectory of rotation matrices from checkerboard to world

    #
    print(">Simulate")

    # Trajectory of all checkerboard points, frame transforms
    # and fiducial points in either world or tracker (atracsys) frame
    check_pts_traj_world = []
    check_pts_traj_tracker = []
    checkerframe_traj_world = (
        []
    )  # poses follow convention .._X_world use [rot_vec_world_to_X | X_to_world_in_X_frame]
    checkerframe_traj_tracker = []
    fiducial_pts_traj_world = []
    fiducial_pts_traj_tracker = []

    for idx in range(len(t_normalized)):
        # All checkerboard points position in world coord.
        all_check_pts_world = (
            traj_check_origin[idx][None, :]
            + (traj_check_orient[idx, :, :] @ (checkerboard.points).T).T
        )

        check_pts_traj_world.append(all_check_pts_world)

        # START Alternative calculation markers w.r.t. tracker frame using affine matrix, TODO remove again
        rot_world_to_atra = R.from_rotvec(world_to_tracker_pose[:3]).as_matrix()
        traj_tracker_to_world_tracker = world_to_tracker_pose[3:]
        affine_3d_tracker = np.concatenate(
            [
                rot_world_to_atra @ traj_check_orient[idx, :, :],
                rot_world_to_atra @ traj_check_origin[idx][:, None]
                + traj_tracker_to_world_tracker[:, None],
            ],
            axis=1,
        )

        affine_3d_tracker = np.concatenate(
            [affine_3d_tracker, np.array([0, 0, 0, 1])[None, :]], axis=0
        )
        all_check_pts_tracker = (affine_3d_tracker) @ np.vstack(
            [checkerboard_geometry, np.ones([1, checkerboard_geometry.shape[1]])]
        )
        all_check_pts_tracker = all_check_pts_tracker[:3, :].T
        calc_points_in_tracker_frame = transform_to_frame(
            all_check_pts_world, world_to_tracker_pose
        )
        assert np.all(
            np.isclose(calc_points_in_tracker_frame, all_check_pts_tracker)
        ), f"not the same"
        # END alternative calculation

        # check_pts_traj_tracker.append(
        #     transform_to_frame(all_check_pts_t, world_to_tracker_tf)
        # )
        check_pts_traj_tracker.append(all_check_pts_tracker)

        # Transformation world to new checkerboard origin
        r_world_to_check = R.from_matrix(traj_check_orient[idx, :, :]).inv()
        r_world_to_check_vec = r_world_to_check.as_rotvec()
        check_to_world_check = -r_world_to_check.as_matrix() @ traj_check_origin[idx]
        checkerboard_origin_tf = np.hstack([r_world_to_check_vec, check_to_world_check])
        checkerframe_traj_world.append(checkerboard_origin_tf)

        # Translation tracker to checkerboard origin
        tracker_to_check_tracker = (
            world_to_tracker_pose[3:]
            + rotate(
                traj_check_origin[None, idx], world_to_tracker_pose[None, :3]
            ).flatten()
        )  # = tracker_to_world_tracker + world_to_checker

        # Rotation from checkerboard frame to tracker frame
        rot_check_to_tracker = R.from_matrix(
            R.from_rotvec(world_to_tracker_pose[:3]).as_matrix()
            @ r_world_to_check.inv().as_matrix()
        ).as_rotvec()  # = rot_world_to_tracker @ rot_checker_to_world = rot_world_to_tracker @ rot_world_to_check**T

        checkerframe_tracker = np.hstack(
            [rot_check_to_tracker, tracker_to_check_tracker]
        )
        checkerframe_traj_tracker.append(checkerframe_tracker)

        # Fiducial points of checkerboard
        fiducial_pts_t = (
            traj_check_origin[idx][None, :]
            + (traj_check_orient[idx, :, :] @ (fiducials).T).T
        )
        fiducial_pts_traj_world.append(fiducial_pts_t)
        fiducial_pts_traj_tracker.append(
            transform_to_frame(all_check_pts_world, world_to_tracker_pose)
        )

    check_pts_traj_world = np.array(check_pts_traj_world)
    check_pts_traj_tracker = np.array(check_pts_traj_tracker)
    checkerframe_traj_world = np.array(checkerframe_traj_world)
    fiducial_pts_traj_world = np.array(fiducial_pts_traj_world)
    fiducial_pts_traj_tracker = np.array(fiducial_pts_traj_tracker)
    checkerframe_traj_tracker = np.array(checkerframe_traj_tracker)

    print(
        "Calculated checkerboard points trajectory: shape = ",
        check_pts_traj_world.shape,
    )

    # Camera projections of checkerboard points
    cam_exts = config.cameras.extrinsics
    cam_intrs = config.cameras.intrinsics
    cameras = [
        np.concatenate([extr, intr]) for (extr, intr) in zip(cam_exts, cam_intrs)
    ]

    if config.cameras.enable_physical_occlusion:
        check_pts_traj_world_list = remove_occluded_points(
            cameras, check_pts_traj_world, checkerframe_traj_world
        )
    else:
        check_pts_traj_world_list = len(cameras) * [check_pts_traj_world]

    proj_points_px_per_cam, proj_points_ideal_coord_per_cam = (
        project_all_markers_points(
            cameras,
            check_pts_traj_world_list,
            camera_crop_projected=config.cameras.camera_crop_projected,
            camera_boundary_px=config.cameras.camera_boundary_px,
        )
    )

    # Sampling with and without offset
    # Kinect cameras are shifted by an offset wheras tracker data is not
    cam_proj_px_sampled = [
        a[offset :: int(n_timestps_cameras), :, :]
        for a, offset in zip(proj_points_px_per_cam, n_timestps_offset_cameras)
    ]
    cam_proj_ideal_sampled = [
        a[offset :: int(n_timestps_cameras), :, :]
        for a, offset in zip(proj_points_ideal_coord_per_cam, n_timestps_offset_cameras)
    ]
    tracker_sampled = check_pts_traj_tracker[
        n_timestps_offset_tracker :: int(n_timestps_tracker), :, :
    ]
    checkerframe_traj_tracker_samples = checkerframe_traj_tracker[
        n_timestps_offset_tracker :: int(n_timestps_tracker), :
    ]

    # Choose only available times and projections for tracker and cameras + account for offset
    true_cam_time_wrt_tracker = [
        t_vec[offset :: int(n_timestps_cameras)] for offset in n_timestps_offset_cameras
    ]
    cam_dev_time = np.arange(0, len(true_cam_time_wrt_tracker[0])) * (
        1.0 / config.sample_freq_cameras_hz
    )
    tracker_dev_time = np.arange(0, len(checkerframe_traj_tracker_samples)) * (
        1.0 / config.sample_freq_tracker_hz
    )

    print("Shape of generated cam dev time vector: ", cam_dev_time.shape)
    print("Shape of generated tracker dev time vector: ", tracker_dev_time.shape)

    if plot:
        # Plots
        print(">> Plotting")

        # Checkerboard
        checkerboard.plot()
        # Projection
        fig, axs = plt.subplots(2, int(np.ceil(len(cameras) / 2)))
        fig_t0, axs_t0 = plt.subplots()
        fig_t0.suptitle("Time 0 projection ideal coordinates")

        t_plt = np.linspace(0, 1, len(proj_points_px_per_cam[0]))
        cmap = plt.get_cmap("viridis")

        for j, (cam, ax_) in enumerate(zip(cameras, axs.flat)):
            for k in range(len(proj_points_ideal_coord_per_cam[0])):
                ax_.scatter(
                    proj_points_ideal_coord_per_cam[j][k, :, 0].flatten(),
                    -proj_points_ideal_coord_per_cam[j][k, :, 1].flatten(),
                    s=0.1,
                    color=cmap(t_plt[k]),
                )
            ax_.set_aspect("equal")
            ax_.set_title(f"cam {j+1}")
            ax_.tick_params(labelbottom=False, labelleft=False)

            # 2nd Figure: plot only first time stamp
            axs_t0.scatter(
                proj_points_ideal_coord_per_cam[j][0, :, 0],
                proj_points_ideal_coord_per_cam[j][0, :, 1],
                s=1,
                label=f"cam {j}",
            )
            axs_t0.set_aspect("equal")

        if len(cameras) % 2 != 0:
            axs.flat[-1].axis("off")

        fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()
        plt.show(block=False)

        plot_scene(
            traj_check_origin,
            cameras,
            tracker_frame=world_to_tracker_pose,
            checkerboard_pts_traj=check_pts_traj_world,
            name="trajectory of all checkerboard points - world frame",
        )

        plot_scene(
            traj_check_origin,
            cameras,
            tracker_frame=world_to_tracker_pose,
            checkerboard_pts_traj=fiducial_pts_traj_world,
            name="trajectory of fiducial points - world frame",
        )

        ax = None
        num_pts = 4
        for i in range(num_pts):
            _, ax = plot_trajectory(
                pos_trajectory=check_pts_traj_tracker[:, i, :], ax=ax
            )
        plt.suptitle(
            f"trajectory of {num_pts} points on the checkerboard - tracker frame"
        )
        plt.show()

    # Get the current date and time
    now = datetime.datetime.now()
    # Format the date and time as a string in the format "YYYY-MM-DD_HH-MM"
    timestamp = now.strftime("%Y-%m-%d_%H-%M")

    return {
        "timestamp": timestamp,
        "sim_time_s": t_vec,
        "markers": cam_proj_ideal_sampled,
        "markers_px": cam_proj_px_sampled,
        "cam_time_gt_s": true_cam_time_wrt_tracker,
        "cam_device_time_s": cam_dev_time,
        "start_cam_device_time_s": 0,
        "tracker_device_time_s": tracker_dev_time,
        "start_tracker_device_time_s": 0,
        "start_host_time_s": time_now_s,
        "checkerboard_points_tracker": tracker_sampled,
        "fiducial_points_tracker": fiducial_pts_traj_tracker,
        "checkerframe_traj_world": checkerframe_traj_world,
        "checkerframe_traj_tracker": checkerframe_traj_tracker,
        "checkerframe_traj_tracker_sampled": checkerframe_traj_tracker_samples,
        "checkerboard": checkerboard,
        "gen_config": config,
        "cameras": cameras,
    }
