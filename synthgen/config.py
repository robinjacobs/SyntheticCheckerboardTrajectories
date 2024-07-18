from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Dict, Optional
import numpy as np


class PrintableConfig:
    """Configuration specifing str output"""

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)


@dataclass
class TrajectoryConfig(PrintableConfig):
    """Configuration specifying the trajectory of the origin marker frame"""

    function: Callable
    config: Dict


@dataclass
class CheckerboardConfig(PrintableConfig):
    """Configuration specifying the checkerboard"""

    num_points_x: int
    num_points_y: int
    square_length_m: float
    fiducial_file: str
    position_trajectory: TrajectoryConfig
    orientation_trajectory: TrajectoryConfig


@dataclass
class CamerasConfig(PrintableConfig):
    """Configuration specifying the camera parameters"""

    intrinsics: List[np.array]  # List of [f1, f2, c1, c2]
    extrinsics: List[np.array]  # List of [rot_vec, translation]
    camera_crop_projected: bool
    camera_boundary_px: Optional[np.array]
    enable_physical_occlusion: bool
    serial_ids: List[str]  # i.e., serial nr. of kinects


@dataclass
class TrackerConfig(PrintableConfig):
    """Configuration specifying the tracker frame"""

    world_to_tracker_tf: np.array  # [rot_vec, translation]


@dataclass
class GeneratorConfig(PrintableConfig):
    """Configuration defining the scene and the simulation parameters"""

    checkerboard: CheckerboardConfig
    cameras: CamerasConfig
    tracker: Optional[TrackerConfig]
    working_volume_center_pos: np.array
    time_start_s: float
    time_end_s: float
    sample_freq_tracker_hz: int  # needs to be an integer
    sample_freq_cameras_hz: int  # needs to be an integer
    cameras_t_offset_s: List[float]  # This is the offset we want to estimate
    tracker_t_offset_s: float
    master_idx: int  # Camera idx which is set to be the master clock
