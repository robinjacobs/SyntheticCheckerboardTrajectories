from typing import Dict, Union

import numpy as np
from scipy.interpolate import splev, splprep

from synthgen.interpolation import slerp_interpolation
from synthgen.randomization import (
    generate_rotations,
    get_random_points_in_working_cube,
    get_random_sample_vectors,
)


def identity(t: Union[float, np.array], config: Dict = {"start": np.array([0, 0, 0])}):
    return np.stack(t.shape[0] * [config["start"]])


def helix_test(t: Union[float, np.array], config: Dict = {}):
    a = 1
    b = 0.4
    c = 0.2
    w = 3
    f = c * np.array([a * np.cos(w * t), a * np.sin(w * t), b * t])
    return f


def bspline(t: Union[float, np.array], config: Dict):
    if config["randomize"]:
        num_sample_points = config["num_sampling_points"]
        if config["sampling_type"] == "random_walk":
            x_samples = get_random_sample_vectors(
                n_points=num_sample_points, radius_bound=2
            )
        elif config["sampling_type"] == "random_uniform":
            box_boundaries = config["sampling_box_boundaries"]
            box_center = config["sampling_box_center"]
            x_samples = get_random_points_in_working_cube(
                n_points=num_sample_points,
                center_point=box_center,
                cube_boundaries=box_boundaries,
            )

        if config["mode"] == "planar":
            # Remove z values and set to constant value
            x_samples = np.hstack(
                [
                    x_samples[:, :2],
                    config["planar_height_m"] * np.ones_like(x_samples[:, 2][:, None]),
                ]
            )

        tckp, u = splprep(
            x=[x_samples[:, i] for i in range(x_samples.shape[-1])], s=0.0, k=3, nest=-1
        )
    else:
        # tckp, u = splprep([x_test, y_test, z_test], s=0.0, k=3, nest=-1)
        tckp, u = splprep(
            [
                [0, 0.1, 0.2, 1, 2],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            s=0.0,
            k=3,
            nest=-1,
        )
        # TODO decide whether use clampled
        # l, r = [(1, (0, 0, 0))],  [(2, (0, 0, 0))]
        # if want to force boundary points use this instead
        # clamped_spline = make_interp_spline(u, np.array([x_test, y_test, z_test]).T, bc_type=(l, r))

    curve_points = splev(t, tckp)
    curve_points = np.array(curve_points)
    return curve_points


def interpolate_orientation_points(
    t: Union[float, np.array], config={"n_knot_points": 20}
):
    n_knot_points = config["n_knot_points"]
    orientation_pts = generate_rotations(n_knot_points)
    concat_ori = []

    assert len(t) % n_knot_points == 0

    for i in range(n_knot_points - 1):
        start_ori = orientation_pts[i]
        end_ori = orientation_pts[i + 1]

        lower_b = i * (1.0 / (n_knot_points - 1))
        upper_b = (i + 1) * (1.0 / (n_knot_points - 1))

        if i == n_knot_points - 2:
            upper_b = np.inf

        s = t[np.logical_and(t >= lower_b, t < upper_b)]
        s_norm = (s - s[0]) / s[-1]

        oris = slerp_interpolation(
            s_norm, start_ori, end_ori
        )  # TODO might also use interpolate_rotation_spline
        concat_ori.append(oris)

    concat_ori = np.vstack(concat_ori)
    return concat_ori


def slerp_wrapper(t: Union[float, np.array], config={"start": [], "end": []}):
    return slerp_interpolation(t, rot1=config["start"], rot2=config["end"])
