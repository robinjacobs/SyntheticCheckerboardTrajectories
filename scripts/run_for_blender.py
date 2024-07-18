"""Run and save the trajectory generation for used in blender"""

import argparse
from pathlib import Path

import numpy as np

from synthgen.config import (
    CamerasConfig,
    CheckerboardConfig,
    GeneratorConfig,
    TrajectoryConfig,
)
from synthgen.constants import (
    WORKING_VOLUME_CENTER,
)
from synthgen.generator import generate
from synthgen.io import (
    camera_recording_save,
    load_cameras,
    load_json_as_dict,
    save_np,
    save_np_trajectory,
)
from synthgen.trajectory import (
    bspline,
    slerp_wrapper,
)

np.random.seed(42)


position_trajectory_cfg = TrajectoryConfig(
    function=bspline,
    config={
        "randomize": True,
        "sampling_box_boundaries": np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]),
        "sampling_box_center": np.array([0, 0, 1.0]),
        "num_sampling_points": 15,
        "sampling_type": "random_uniform",
        "mode": "3d",
    },
)


position_planar_trajectory_cfg = TrajectoryConfig(
    function=bspline,
    config={
        "randomize": True,
        "sampling_box_boundaries": np.array([[-1.5, 1.5], [-1.5, 1.5], [-0.0, 0.01]]),
        "sampling_box_center": np.array([0, 0, 0.5]),
        "num_sampling_points": 20,
        "sampling_type": "random_uniform",
        "mode": "planar",
        "planar_height_m": 0.5,
    },
)


orientation_trajectory_cfg = TrajectoryConfig(
    slerp_wrapper,
    config={
        "start": np.identity(3),
        "end": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    },
)

blender_gen_cfg = GeneratorConfig(
    checkerboard=None,
    cameras=CamerasConfig(
        extrinsics=None,
        intrinsics=None,
        camera_boundary_px=None,
        camera_crop_projected=False,
        serial_ids=None,
        enable_physical_occlusion=True,
    ),
    working_volume_center_pos=WORKING_VOLUME_CENTER,
    tracker=None,
    time_start_s=0,
    time_end_s=0.1,
    sample_freq_cameras_hz=30,
    sample_freq_tracker_hz=400,
    cameras_t_offset_s=None,
    tracker_t_offset_s=0,
    master_idx=0,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run",
        description="Run synthetic trajectory generation for use in blender",
        epilog="Uses pre defined configuration to generate multi view trajectories",
    )

    parser.add_argument("-p", "--plot", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "-c",
        "--cameras",
        type=str,
        required=True,
        help="Path to .csv file specifiyng intrinsics and extrinsics for each camera",
    )
    parser.add_argument(
        "-b",
        "--board",
        type=str,
        required=True,
        help="Path to .json file specifiyng checkerboard geometry",
    )

    args = parser.parse_args()
    cam_path = Path(args.cameras)
    board_path = Path(args.board)

    # print(blender_gen_cfg)

    # Load extrinisics and intriniscs
    extrinsics, intrinsics = load_cameras(cam_path)
    num_cameras = len(extrinsics)
    print(f"Found {num_cameras} cameras")
    print("Extrinsics: ", extrinsics)
    print("Inntrinsics: ", intrinsics)

    # Update config
    blender_gen_cfg.cameras.extrinsics = np.array(extrinsics)
    blender_gen_cfg.cameras.intrinsics = np.array(intrinsics)
    blender_gen_cfg.cameras.serial_ids = [
        str(el) for el in range(num_cameras)
    ]  # TODO might change to load from csv
    blender_gen_cfg.cameras_t_offset_s = num_cameras * [0]
    blender_gen_cfg.cameras.camera_boundary_px = 2*blender_gen_cfg.cameras.intrinsics[0][-2:] # TODO support multiple camera resolution

    print("Image dimensions: ", blender_gen_cfg.cameras.camera_boundary_px)


    # Checkerboard geometry/object
    check_dict = load_json_as_dict(board_path)

    if check_dict["mode"] == "planar":
        check_position_trajectory_cfg = position_planar_trajectory_cfg
        orientation_trajectory_cfg.config["end"] = np.array(
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        )
        # or any other end orientation
        check_orientation_trajectory_cfg = orientation_trajectory_cfg

    else:
        check_position_trajectory_cfg = position_trajectory_cfg
        check_orientation_trajectory_cfg = orientation_trajectory_cfg

    check_dict.pop("mode")
    checkerboard_cfg = CheckerboardConfig(
        **check_dict,
        position_trajectory=check_position_trajectory_cfg,
        orientation_trajectory=check_orientation_trajectory_cfg,
    )
    print(check_dict)
    print(checkerboard_cfg)
    blender_gen_cfg.checkerboard = checkerboard_cfg

    # Generate
    out_dict = generate(blender_gen_cfg, plot=args.plot)

    # Save data
    combined_traj_and_time = np.hstack(
        [
            out_dict["sim_time_s"][:, None] * 1e6,  # convert to us
            out_dict["checkerframe_traj_world"], # note this follows camera convention [rot_vec_world_to_check, check_to_world_in_check_frame]
        ]
    )

    save_np_trajectory(combined_traj_and_time, name="trajectory_world")

    # Save for each camera each simulation sampling times in csv separatetly
    cam_sample_times = np.array(out_dict["cam_time_gt_s"]) * 1e6
    save_np_trajectory(cam_sample_times.T, name="timestamps_cameras")

    # Save camera recordings
    camera_recording_save(out_dict)
    camera_recording_save(out_dict, save_in_ideal_coordinates=False)    
    
    markers_px = out_dict["markers_px"]
    save_np(markers_px, "cam_recording_px")
    
    # 
    # tracker_recording_save(out_dict)
    # meta_data_save(out_dict)
    # generation_log_save(out_dict)
