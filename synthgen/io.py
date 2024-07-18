import configparser
import csv
import datetime
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from synthgen.constants import ATRACSYS_ID, CHECKER_BOARD_ID
from synthgen.geometry import (
    cam_extrinsic_to_homogeneos_tf_matrix,
    cam_intrinsics_to_intrinsics_matrix,
)
from synthgen.templates import template_pre_calibration_dict

# Paths
root_dir = Path(__file__).parent.parent.resolve()
data_dir = root_dir / "data"

now = datetime.datetime.now()
nr_dir = "gen_" + now.strftime("%Y_%m_%d_%H_%M_%S")
out_dir = root_dir / "output" / nr_dir
recordings_dir = out_dir / "recordings"


def load_ini_as_dict(file_path: Path) -> Dict:
    """Load .ini files as dictionary"""
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the .ini file
    config.read(file_path)

    # Convert the ConfigParser object to a dictionary
    config_dict = {section: dict(config[section]) for section in config.sections()}

    # Return the dictionary
    return config_dict


def get_chessboard_fiducial_point(file: Path) -> Tuple:
    file_name = data_dir / file
    config_dict = load_ini_as_dict(file_name)
    fiducial_cfg = {}
    geometry_cfg = {}
    for key in config_dict.keys():
        if key != "geometry":
            fiducial_cfg[key] = np.array(
                [
                    0.001 * float(config_dict[key]["x"]),
                    0.001 * float(config_dict[key]["y"]),
                    0.001 * float(config_dict[key]["z"]),
                ]
            )  # convert mm to m
        else:
            geometry_cfg = config_dict[key]

    return fiducial_cfg, geometry_cfg


def get_camera_folder_name(kinect_number: int, timestamp: str):
    # Generate the filename
    folder_name = f"kinect_{kinect_number}_{timestamp}"
    return folder_name


def write_csv(file_path: Path, dict_data: Dict, header=True):
    # List of keys for the CSV header
    if header:
        header = dict_data.keys()
    else:
        header = None
    # Open the file in write mode
    with open(file_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)

        # Write the header
        writer.writeheader()

        # Write the data
        for i in range(len(dict_data["Name"])):
            writer.writerow({key: value[i] for key, value in dict_data.items()})


def write_list_to_csv(file_path: Path, data: List):
    with open(file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        # Write the data
        for element in data:
            writer.writerow(element)


def load_csv_as_list(filename: Path):
    out_list = []
    with open(filename, "r") as f:
        reader_ = csv.reader(f)
        for r in reader_:
            out_list.append(r)
    return out_list


def write_dict_to_json(filename: Path, dictionary: Dict):
    with open(filename, "w") as f:
        json.dump(dictionary, f, indent=4)


def load_json_as_dict(filename: Path):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def load_cameras(filename: Path):
    cams_raw = load_csv_as_list(filename=filename)
    cams_raw = [[float(item) for item in sublist] for sublist in cams_raw]
    assert all(len(sublist) == 10 for sublist in cams_raw)
    # Note: This converts it from a non opencv convention!
    extrinsics = []
    for item in cams_raw:
        ext = item[:6]

        rot_mat = R.from_rotvec(ext[:3]).as_matrix()
        rot_z_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

        new_ext = R.from_matrix(rot_z_180 @ rot_mat).as_rotvec()
        new_pos = rot_z_180 @ np.array(item[3:6]).reshape(3, 1)

        extrinsics.append(np.hstack([new_ext, new_pos.flatten()]))

    # extrinsics = [item[:6] for item in cams_raw]
    intrinsics = [item[6:] for item in cams_raw]
    return extrinsics, intrinsics


def generation_log_save(out_dict: Dict):
    log_file_name = "generation_config.json"
    out_str_dict = {k: str(el) for k, el in out_dict.items()}
    complete_file_path = out_dir / log_file_name
    write_dict_to_json(complete_file_path, out_str_dict)


def meta_data_save(out_dict: Dict):
    print(">>Save meta data")

    recordings_file_name = "recordings.json"
    complete_file_path = out_dir / recordings_file_name

    save_dict = {}
    save_dict["recordings"] = [
        out_dict["tracker_save_paths"] + out_dict["camera_save_paths"]
    ]

    write_dict_to_json(complete_file_path, save_dict)


def tracker_recording_save(out_dict: Dict):
    print(">> Save tracker recording")

    save_paths = []

    # _original.csv not needed
    # print("Skipped generation of  .._original.csv")

    # .csv
    checker_marker_traj = out_dict["checkerframe_traj_tracker_sampled"]
    data_rows = []
    m_str = "m"
    registrationError = 0  # No error
    fiducial_indices = [
        1,
        2,
        3,
        4,
    ]  # No permutation

    for i, ts in enumerate(out_dict["tracker_device_time_s"]):
        timestamp_us = int((ts + out_dict["start_tracker_device_time_s"]) * 1e6)
        host_time_us = int(1e6 * out_dict["start_host_time_s"]) + timestamp_us
        device_time = timestamp_us  # TODO check

        marker_tf = checker_marker_traj[i, :]
        marker_pos = (1e3 * marker_tf[3:]).tolist()  # tracker uses mm
        marker_ori = R.from_rotvec(marker_tf[:3]).as_matrix().flatten().tolist()

        row = (
            [host_time_us, device_time, m_str, CHECKER_BOARD_ID]
            + marker_pos
            + marker_ori
            + [registrationError]
            + fiducial_indices
        )
        data_rows.append(row)

    rec_filename = ATRACSYS_ID + out_dict["timestamp"] + ".csv"
    save_path = recordings_dir / rec_filename
    write_list_to_csv(save_path, data_rows)
    save_paths.append(str(save_path.relative_to(out_dir)).replace("\\", "/"))

    # _calibration.json - not needed
    # calibration_dict = deepcopy(template_calibration_dict)
    # calibration_dict["ref2DevClock"] = str(
    #     int(out_dict["tracker_device_time_s"][0] - out_dict["start_host_time_s"])
    # calib_file_path = ATRACSYS_ID + out_dict["timestamp"] + "_calibration.json"
    # write_dict_to_json(recordings_dir / calib_file_path, calibration_dict)

    # _preCalibration.json
    pre_calibration_dict = deepcopy(template_pre_calibration_dict)
    pre_calibration_dict["ref2DevClock"] = str(
        int(out_dict["tracker_device_time_s"][0] - out_dict["start_host_time_s"])
    )  # Note assumes at start obtained tracker
    pre_calibration_dict["recording"] = f"recordings/{str(rec_filename)}"
    calib_file_path = ATRACSYS_ID + out_dict["timestamp"] + "_preCalibration.json"
    write_dict_to_json(recordings_dir / calib_file_path, pre_calibration_dict)

    out_dict["tracker_save_paths"] = save_paths


def camera_recording_save(out_dict: Dict, save_in_ideal_coordinates: bool = True):
    print(">>Save camera recording data")
    print(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    save_paths = []
    add_name = ""
    if save_in_ideal_coordinates:
        markers = out_dict["markers"]
    else:
        markers = out_dict["markers_px"]
        add_name = "_px"

    num_cameras = len(markers)
    kinect_ids = out_dict["gen_config"].cameras.serial_ids
    master_idx = out_dict["gen_config"].master_idx
    camera_extrinsics = out_dict["gen_config"].cameras.extrinsics
    camera_intrinsics = out_dict["gen_config"].cameras.intrinsics

    for cam_idx in range(num_cameras):
        # markers.csv - Saves projections of markers in ideal coordinates
        data_rows = []

        for i, t in enumerate(out_dict["cam_device_time_s"]):
            timestamp_us = int(
                (t + out_dict["start_cam_device_time_s"]) * 1e6
            )  # microseconds
            marker_id = CHECKER_BOARD_ID
            reprojection_error = 0.0
            marker_to_cam = np.eye(4).flatten()  # Set to affine identity
            ref_to_cam = np.eye(4).flatten()

            markers_flat = markers[cam_idx][i, :, :].flatten()
            is_not_nan = np.logical_not(np.isnan(markers_flat))
            markers_flat = markers_flat[is_not_nan]  # remove occlusion entries
            markers_flat = markers_flat.tolist()

            assert len(markers_flat) == 2 * out_dict[
                "checkerboard"
            ].num_points or np.any(np.logical_not(is_not_nan))

            data_row = (
                [timestamp_us, marker_id, reprojection_error]
                + marker_to_cam.tolist()
                + ref_to_cam.tolist()
            )
            assert len(data_row) == 35, len(data_row)
            data_row += markers_flat
            data_rows.append(data_row)

        folder_name = get_camera_folder_name(kinect_ids[cam_idx], out_dict["timestamp"])
        cam_dir = recordings_dir / folder_name
        os.makedirs(cam_dir, exist_ok=True)

        write_list_to_csv(cam_dir / f"markers{add_name}.csv", data_rows)

        # meta.json
        first_device_timestamp = out_dict[
            "start_cam_device_time_s"
        ]  # TODO extend to define for each camera individually
        meta_dict = {
            "firstDeviceTimestamp": str(first_device_timestamp),
            "isMaster": str(int(cam_idx == master_idx)),
        }
        # if cam_idx != master_idx:
        device_to_host_clock_offset = int(
            out_dict["start_host_time_s"] - out_dict["start_cam_device_time_s"]
        )
        device_to_host_clock_offset_max_var = (
            0  # doesn't seems to be used in calibrator
        )
        meta_dict = {
            **meta_dict,
            "deviceToHostClockOffset": str(device_to_host_clock_offset),
            "deviceToHostClockOffsetMaxVariance": str(
                device_to_host_clock_offset_max_var
            ),
        }
        write_dict_to_json(cam_dir / "meta.json", meta_dict)

        # calibration.json TODO this might be incorrect currently (not mm, not w.r.t, atracsys frame?)
        ref2DevClock = "0"
        inlier_ratio = 1.0
        inlier_rmse = 0.0
        tf_matrix = cam_extrinsic_to_homogeneos_tf_matrix(camera_extrinsics[cam_idx])
        # TODO problem here camera intrinsics are w.r.t, world coordinates and not in

        calibration_dict = {
            "ref2DevClock": ref2DevClock,
            "ref2DevTransform": {
                "type_id": "opencv-matrix",
                "rows": 4,
                "cols": 4,
                "dt": "d",
                "data": tf_matrix.flatten().tolist(),
            },
            "inlierRatio": inlier_ratio,
            "inlierRMSE": inlier_rmse,
        }
        write_dict_to_json(cam_dir / "calibration.json", calibration_dict)

        # Debug file camera extrinsics w.r.t. tracker frame
        tf_matrix_cp = tf_matrix.copy()
        tf_matrix_cp[:3, 3] = tf_matrix_cp[:3, 3] * 1e3  # to mm converesion
        calibration_debug_dict = {}
        calibration_debug_dict["tf_cam_world_mm"] = tf_matrix_cp.flatten().tolist()
        write_dict_to_json(cam_dir / "debug_calibration.json", calibration_debug_dict)

        # rot_gt_atra_to_cam = tf_matrix_cp[:3,:3] @ (pose_atra_np[:3, :3].T)
        # tra_gt = pose_atra_np[:3, 3] + (rot_gt_atra_to_cam[:3,:3].T) @ (-tf_matrix_cp[:3, 3])

        # factory_calibration.json
        # Skipped not needed

        # _custom_intrinsic.json
        distortion_coefficients = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        intrinsics = cam_intrinsics_to_intrinsics_matrix(camera_intrinsics[cam_idx])

        custom_intrinsic_dict = {
            "sensors": {
                "RGB": {
                    "intrinsics": {
                        "type_id": "opencv-matrix",
                        "rows": 3,
                        "cols": 3,
                        "dt": "d",
                        "data": intrinsics.flatten().tolist(),
                    },
                    "distortionCoefficients": {
                        "type_id": "opencv-matrix",
                        "rows": 1,
                        "cols": 8,
                        "dt": "d",
                        "data": distortion_coefficients,
                    },
                }
            }
        }
        file_name = f"kinect_{kinect_ids[cam_idx]}_custom_intrinsics.json"
        write_dict_to_json(recordings_dir / file_name, custom_intrinsic_dict)

        save_path = "recordings/" + str(folder_name) + ".mkv"
        save_paths.append(save_path)

    out_dict["camera_save_paths"] = save_paths


def save_np_trajectory(array, name="none"):
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{name}.csv"
    filename_path = out_dir / filename

    with open(filename_path, "w+") as f:
        np.savetxt(f, array, delimiter=",", fmt="%3f", comments="")


def save_np(array, name="none"):
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{name}.npy"
    filename_path = out_dir / filename

    with open(filename_path, "wb") as f:
        np.save(f, array)
