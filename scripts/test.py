import argparse
import csv
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from synthgen.checkerboard import Checkerboard
from synthgen.config import CheckerboardConfig
from synthgen.geometry import cam_extrinsic_to_homogeneos_tf_matrix

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="run",
        description="Test synthetic trajectory generation",
        epilog="",
    )
    parser.add_argument("-r", "--root_path")
    args = parser.parse_args()

    root_ = Path(str(args.root_path))
    cfg_file = root_ / "generation_config.json"
    recordings_folder = root_ / "recordings"

    matching_atracsys_files = [
        filename
        for filename in os.listdir(recordings_folder)
        if filename.startswith("atracsys_") and filename.endswith(".csv")
    ]

    assert matching_atracsys_files, "tracker file not found"
    atrasys_file = recordings_folder / matching_atracsys_files[0]
    print(atrasys_file)

    # Read camera files
    kinect_files = []
    for p in recordings_folder.iterdir():
        if os.path.isdir(p):
            f_name = p / "markers.csv"
            kinect_files.append(f_name)

    checkerboard_frame_tracker = []

    # Read tracker file
    with open(atrasys_file, "r") as f:
        a = csv.reader(f)
        for r in a:
            checkerboard_frame_tracker.append([float(el) for el in r[4:16]])
    checkerboard_frame_tracker_np = np.array(checkerboard_frame_tracker)

    all_cam_detection = []
    for f_name in kinect_files:
        print(f_name)
        detection = []

        # Iterate over available detection
        with open(f_name, "r") as f:
            a = csv.reader(f)
            for r in a:
                raw_row = [float(el) for el in r[-80:]]
                print(len(raw_row))
                if len(raw_row) == 80:  # only if not occluded
                    detection.append(np.array(raw_row).reshape((-1, 2)))
        detection = np.array(detection)
        all_cam_detection.append(detection)

    # Investigate scene at first recorded time step
    time_idx = 0
    first_rec_atra = checkerboard_frame_tracker_np[time_idx, :]
    first_rec_kinects = np.array([cap[time_idx] for cap in all_cam_detection])

    # Plot all projected points ideal coordinates
    for i in range(len(first_rec_kinects)):
        plt.scatter(
            first_rec_kinects[i, :, 0], first_rec_kinects[i, :, 1], label=f"{i}"
        )
        [
            plt.text(
                first_rec_kinects[i, el, 0], first_rec_kinects[i, el, 1], s=str(el)
            )
            for el in range(len(first_rec_kinects[i]))
        ]
        plt.legend()
    plt.title(f"Time {time_idx} projections for each camera ideal coordinates")

    # Reproject points
    checkerboard_cfg = CheckerboardConfig(
        num_points_x=6,
        num_points_y=9,
        square_length_m=0.03,
        fiducial_file="geometry102.ini",  # needs to be in data folder
        position_trajectory=None,
        orientation_trajectory=None,
    )
    checkerboard = Checkerboard(
        config=checkerboard_cfg, origin=np.array([0, 0, 0, 0, 0, 1])
    )  # TODO read from file instead

    pose_c_np = np.hstack(
        [first_rec_atra[3:].reshape(3, 3), first_rec_atra[:3][:, None] / 1e3]
    )
    pose_c_np = np.vstack([pose_c_np, np.array([0, 0, 0, 1])[None, :]])

    marker_geometry_r = checkerboard.points[:, :].T
    converted_points_cpp = (pose_c_np) @ np.vstack(
        [marker_geometry_r, np.ones([1, marker_geometry_r.shape[1]])]
    )
    converted_points_cpp = converted_points_cpp.T
    converted_points_cpp_proj = (
        converted_points_cpp[:, :2] / converted_points_cpp[:, 2][:, None]
    )

    ax = plt.figure()
    plt.scatter(x=converted_points_cpp_proj[:, 0], y=converted_points_cpp_proj[:, 1])
    for i in range(len(converted_points_cpp_proj)):
        plt.text(
            converted_points_cpp_proj[i, 0], converted_points_cpp_proj[i, 1], str(i)
        )
    plt.title("Atracsys Projection")

    # PNP test
    cam_proj = first_rec_kinects[0, :].copy()
    pose_c_np_mm = np.hstack(
        [first_rec_atra[3:].reshape(3, 3), first_rec_atra[:3][:, None]]
    )
    pose_c_np_mm = np.vstack([pose_c_np_mm, np.array([0, 0, 0, 1])[None, :]])
    marker_geometry_r_mm = 1e3 * checkerboard.points[:, :].T
    converted_points_cpp_mm = (pose_c_np_mm) @ np.vstack(
        [marker_geometry_r_mm, np.ones([1, marker_geometry_r_mm.shape[1]])]
    )
    converted_points_cpp_mm = converted_points_cpp_mm.T[:, :3]

    cameraMatrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    dist = np.array([0, 0, 0, 0])

    # Solve pnp from checkerboard points in tracker frame
    solve_method = cv2.SOLVEPNP_EPNP  # or cv2.SOLVEPNP_ITERATIVE
    res = cv2.solvePnP(
        converted_points_cpp_mm, cam_proj, cameraMatrix, dist, solve_method
    )
    suc, res_rot, res_tra = res

    res_rot_m = R.from_rotvec(res_rot.flatten()).as_matrix()
    res_rot_m_cv, _ = cv2.Rodrigues(res_rot)

    affine = cam_extrinsic_to_homogeneos_tf_matrix(
        np.hstack([res_rot.flatten(), res_tra.flatten()])
    )

    pose_atra_np = np.hstack(
        [first_rec_atra[3:].reshape(3, 3), first_rec_atra[:3][:, None]]
    )  # TODO check shape again
    pose_atra_np = np.vstack([pose_atra_np, np.array([0, 0, 0, 1])[None, :]])

    hom_converted_points_cpp_mm = np.concatenate(
        [converted_points_cpp_mm, np.ones([len(converted_points_cpp_mm), 1])], axis=1
    )
    # reproj_points_pnp_sol = (affine[None, :]) @ hom_converted_points_cpp_mm[:, :, None]
    # reproj_points_pnp_sol_2d = (
    #     reproj_points_pnp_sol[:, :2] / reproj_points_pnp_sol[:, 2][:, None]
    # )

    cam_matrix = cameraMatrix.reshape(3, 3)
    proj_res_2d, _ = cv2.projectPoints(
        hom_converted_points_cpp_mm[:, :3],
        R.from_matrix(affine[:3, :3]).as_rotvec(),
        affine[:3, 3],
        np.float32(cam_matrix),
        np.float32(dist),
    )
    proj_res_2d = proj_res_2d.reshape(-1, 2)

    ax2 = plt.figure()

    plt.scatter(
        first_rec_kinects[0, :, 0],
        first_rec_kinects[0, :, 1],
        color="red",
        alpha=0.4,
        linewidths=12,
        marker="+",
        label="saved projection in csv file",
    )
    plt.scatter(
        cam_proj[:, 0],
        cam_proj[:, 1],
        alpha=0.4,
        color="yellow",
        label="projection using generated camera matrix",
    )
    plt.scatter(
        proj_res_2d[:, 0],
        proj_res_2d[:, 1],
        color="blue",
        alpha=0.4,
        marker="*",
        label="projection using pnp camera matrix",
    )
    plt.legend()
    plt.title(f"Time {time_idx} projection")

    plt.show()
