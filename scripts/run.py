"""Run and save the trajectory generation"""

import argparse

import numpy as np

from synthgen.config import (
    CamerasConfig,
    CheckerboardConfig,
    GeneratorConfig,
    TrackerConfig,
    TrajectoryConfig,
)
from synthgen.constants import (
    KINECT_HEIGHT_PX,
    KINECT_INTRINSICS,
    KINECT_SERIAL_IDS,
    KINECT_WIDTH_PX,
    WORKING_VOLUME_CENTER,
)
from synthgen.generator import generate
from synthgen.helper import (
    sample_camera_extrinsics,
)
from synthgen.io import (
    camera_recording_save,
    generation_log_save,
    meta_data_save,
    tracker_recording_save,
)
from synthgen.trajectory import (
    bspline,
    slerp_wrapper,
)

# Fix random seed
np.random.seed(42)


run_gen_cfg = GeneratorConfig(
    checkerboard=CheckerboardConfig(
        num_points_x=6,
        num_points_y=9,
        square_length_m=0.03,
        fiducial_file="geometry102.ini",  # needs to be in data folder
        position_trajectory=TrajectoryConfig(
            function=bspline,
            config={
                "randomize": True,
                "sampling_box_boundaries": np.array(
                    [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]
                ),
                "sampling_box_center": np.array([0, 0, 1.0]),
                "num_sampling_points": 15,
                "sampling_type": "random_uniform",
                "mode": "3d",
            },
        ),
        # orientation_trajectory=TrajectoryConfig(identity, config={}),
        orientation_trajectory=TrajectoryConfig(
            slerp_wrapper,
            config={
                "start": np.identity(3),
                "end": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
            },
        ),
    ),
    cameras=CamerasConfig(
        extrinsics=sample_camera_extrinsics(
            5, radius=1.25, center=WORKING_VOLUME_CENTER, z_boundary=np.array([1, 1.5])
        ),  # or cameras_around_origin_on_xy_plane, [[rot_vec_world_to_cam, cam_to_world_in_cam_frame],...]
        intrinsics=5 * [KINECT_INTRINSICS],
        camera_boundary_px=np.array([[0, KINECT_WIDTH_PX], [0, KINECT_HEIGHT_PX]]),
        camera_crop_projected=False,
        serial_ids=KINECT_SERIAL_IDS,
        enable_physical_occlusion=True,
    ),
    working_volume_center_pos=WORKING_VOLUME_CENTER,
    tracker=TrackerConfig(
        world_to_tracker_tf=sample_camera_extrinsics(
            1,
            radius=1.5,
            center=WORKING_VOLUME_CENTER,
            z_boundary=np.array([2.5, 3]),
        )[0],
        # as with cameras extrinsics these are expressed as [rot_vec_world_to_cam, cam_to_world_in_cam_frame]
    ),
    time_start_s=0,
    time_end_s=3,
    sample_freq_cameras_hz=30,
    sample_freq_tracker_hz=400,
    cameras_t_offset_s=5
    * [
        0.01
    ],  # This is the per camera clockOffset to be estimated later, note all are with respect to the tracker
    tracker_t_offset_s=0,
    master_idx=0,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run",
        description="Run synthetic trajectory generation",
        epilog="Uses pre defined configuration to generate multi view trajectories",
    )
    parser.add_argument("-p", "--plot", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    out_dict = generate(run_gen_cfg, plot=args.plot)
    # Save data
    camera_recording_save(out_dict)
    tracker_recording_save(out_dict)
    meta_data_save(out_dict)
    generation_log_save(out_dict)
