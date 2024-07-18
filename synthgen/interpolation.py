import numpy as np
from scipy.spatial.transform import Rotation as R, RotationSpline, Slerp


def slerp_interpolation(
    s: np.array,
    rot1: np.array,
    rot2: np.array,
):
    """Slerp rotation interpolation between rot1 and rot2"""
    # Convert rotation matrices to Rotation objects
    r1 = R.from_matrix(rot1)
    r2 = R.from_matrix(rot2)

    rotations = R.concatenate([r1, r2])

    # Create the interpolator object
    slerp = Slerp([0, 1], rotations)

    # Interpolate the rotations at the given time t
    interp_rot = slerp(s)

    # Return the interpolated rotation matrix
    return interp_rot.as_matrix()


def spline_interpolation(rot1: np.array, rot2: np.array, s: np.array):
    """spline rotation interpolation between rot1 and rot2"""
    # Convert rotation matrices to Rotation objects
    r1 = R.from_matrix(rot1)
    r2 = R.from_matrix(rot2)

    multi_rot = R.concatenatep([r1, r2])
    # Create the RotationSpline object
    times = [0, 1]
    rotations = multi_rot
    # rotations = [r1, r2]
    spline = RotationSpline(times, rotations)

    # Interpolate the rotations at the given time t
    interp_rot = spline(s)

    # Return the interpolated rotation matrix
    return interp_rot.as_matrix()
