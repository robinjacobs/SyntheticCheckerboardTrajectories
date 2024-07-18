""" Module for checking the camera calibration results coming from the multicamera calibration pipeline
Important: use overwrite argument in multicam calibration to get new calibration files so that can compare before after saved data,
"""

import argparse
import os
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from synthgen.geometry import get_cam_pos_world
from synthgen.geometry import affine4d_to_rvectra
from synthgen.helper import get_sim_transform
from synthgen.plot import plot_camera_poses


def get_file_names(recordings_folder):
    kinect_files = []
    kinect_calib_files = []
    calib_debug_files = []
    for p in recordings_folder.iterdir():
        if os.path.isdir(p):
            f_name = p / "markers.csv"
            f_calib_name = p / "calibration.json"
            f_debug_name = p / "debug_calibration.json"
            kinect_files.append(f_name)
            kinect_calib_files.append(f_calib_name)
            calib_debug_files.append(f_debug_name)

    return kinect_files, kinect_calib_files, calib_debug_files


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    parser = argparse.ArgumentParser(
        prog="run",
        description="Check multi camera calibration results",
        epilog="",
    )
    parser.add_argument("-r", "--root_before")
    parser.add_argument("-a", "--after_run")  #
    args = parser.parse_args()

    root_before = Path(str(args.root_before))
    cfg_file_before = root_before / "generation_config.json"
    recordings_folder_before = root_before / "recordings"

    root_after = Path(str(args.after_run))
    cfg_file_after = root_after / "generation_config.json"
    recordings_folder_after = root_after / "recordings"

    kinect_files_before, kinect_calib_files_before, calib_debug_files_before = (
        get_file_names(recordings_folder_before)
    )
    print(calib_debug_files_before)
    kinect_files_after, kinect_calib_files_after, _ = get_file_names(
        recordings_folder_after
    )  # debug files not needed since should be the same

    all_cams_after, all_cams_before, all_cams_before_debug = [], [], []

    for idx, (calib_before_f, calib_after_f) in enumerate(
        zip(kinect_calib_files_before, kinect_calib_files_after)
    ):
        detection = []
        print(100 * "=")

        # Iterate over available detection
        # with open(f_name, "r") as f:
        #     a = csv.reader(f)
        #     for r in a:
        #         raw_row = [float(el) for el in r[-80:]]
        #         # print(len(raw_row))
        #         if len(raw_row) == 80:  # only if not occluded
        #             detection.append(np.array(raw_row).reshape((-1, 2)))

        with open(calib_before_f, "r") as f:
            print("before ", calib_before_f)
            cal_before = json.load(f)

        with open(calib_debug_files_before[idx], "r") as f:
            cal_debug_before = json.load(f)

        with open(calib_after_f, "r") as f:
            cal_after = json.load(f)

        ref2devtfs_before = np.array(cal_before["ref2DevTransform"]["data"]).reshape(
            4, 4
        )

        ref2devtfs_before_debug = np.array(cal_debug_before["tf_cam_world_mm"]).reshape(
            4, 4
        )

        ref2devtfs_after = np.array(cal_after["ref2DevTransform"]["data"]).reshape(4, 4)

        print(ref2devtfs_after)

        all_cams_after.append(ref2devtfs_after)
        all_cams_before.append(ref2devtfs_before)
        all_cams_before_debug.append(ref2devtfs_before_debug)

    # ref2devtfs = [
    #     np.array(el["ref2DevTransform"]["data"]).reshape(4, 4) for el in all_cam_calib
    # ]

    all_cams_after_rottra = affine4d_to_rvectra(all_cams_after)
    all_cams_before_d_rottra = affine4d_to_rvectra(all_cams_before_debug)

    fig, ax = plot_camera_poses(all_cams_after_rottra, scale=1e3)
    fig, ax = plot_camera_poses(all_cams_before_d_rottra, scale=1e3, ax=ax)
    ax.set_aspect("equal")

    # Test procruste points from simulation camera coord to calculated
    before_pts = np.array(
        [get_cam_pos_world(el[None, :]) for el in all_cams_before_d_rottra]
    )
    after_pts = np.array(
        [get_cam_pos_world(el[None, :]) for el in all_cams_after_rottra]
    )
    rotation_matrix, t, scale, cost = get_sim_transform(before_pts, after_pts)

    print("Procruste cost: ", cost)
    print("Calculated Rotation matrix:\n", rotation_matrix)
    print("Calculated translation:\n", t)

    ax_procr = plt.figure().add_subplot(projection="3d")
    ax_procr.scatter(
        xs=before_pts[:, 0],
        ys=before_pts[:, 1],
        zs=before_pts[:, 2],
        color="red",
        label="Used in generation",
    )
    ax_procr.scatter(
        xs=after_pts[:, 0],
        ys=after_pts[:, 1],
        zs=after_pts[:, 2],
        color="blue",
        label="After multi camera calibration",
        alpha=0.4,
    )
    # print("Shape", rotation_matrix.shape, before_pts.shape, t.shape)
    before_tf_pts = (before_pts @ rotation_matrix.T + t[None, :]) * scale
    ax_procr.scatter(
        xs=before_tf_pts[:, 0],
        ys=before_tf_pts[:, 1],
        zs=before_tf_pts[:, 2],
        color="green",
        label="Procruste transformed points first to second",
    )
    ax_procr.set_title("Procruste tf result camera centers")
    ax_procr.legend()

    plt.show()
