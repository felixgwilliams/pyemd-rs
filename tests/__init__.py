from __future__ import annotations

import numpy as np

zc_arrays = {
    "zeros": np.array([0, 0, 0, 0, 0], dtype=np.float64),
    "zeros_ends": np.array([0, 1, 2, -3, 0], dtype=np.float64),
    "normal": np.array([1, 2, -1, -2, 3, -1], dtype=np.float64),
    "zero_gap": np.array([1, 2, 0, 0, -1], dtype=np.float64),
    "zero_gap2": np.array([1, 2, 0, 0, 1], dtype=np.float64),
    "zero_tail": np.array([1, 2, 0, 0, 0], dtype=np.float64),
    "too_short": np.array([0], dtype=np.float64),
    "too_short2": np.array([1], dtype=np.float64),
    "empty": np.array([], dtype=np.float64),
    "zz": np.array([0, 0], dtype=np.float64),
    "zp": np.array([0, 1], dtype=np.float64),
    "zn": np.array([0, -1], dtype=np.float64),
    "pz": np.array([1, 0], dtype=np.float64),
    "pp": np.array([1, 1], dtype=np.float64),
    "on": np.array([1, -1], dtype=np.float64),
    "nz": np.array([-1, 0], dtype=np.float64),
    "np": np.array([-1, 1], dtype=np.float64),
    "nn": np.array([-1, -1], dtype=np.float64),
    "d2zero": np.array([2, 3, 3, 2, 4], dtype=np.float64),
    "d2zero2": np.array([2, 3, 3, 3, 3, 2, 4], dtype=np.float64),
    "d2zero3": np.array([2, 3, 3, 4, 2], dtype=np.float64),
    "d2zero4": np.array([2, 3, 3, 3, 3, 4, 2], dtype=np.float64),
    "repeats2": np.array([1, 2, 3, 3, 2, 2, 4], dtype=np.float64),
    "repeats3": np.array([1, 2, 3, 3, 3, 2, 2, 2, 4], dtype=np.float64),
    "original": np.array([-1, 0, 1, 0, -1, 0, 3, 0, -9, 0], dtype=np.float64),
    "repeats": np.array([-1, 0, 1, 1, 0, -1, 0, 3, 0, -9, 0], dtype=np.float64),
    "bound_extrapolation1": np.array(
        [0, -3, 1, 4, 3, 2, -2, 0, 1, 2, 1, 0, 1, 2, 5, 4, 0, -2, -1], dtype=np.float64
    ),
    "bound_extrapolation2": np.array(
        [-3, 1, 4, 3, 2, -2, 0, 1, 2, 1, 0, 1, 2, 5, 4, 0, -2], dtype=np.float64
    ),
    "bound_extrapolation3": np.array([1, 4, 3, 2, -2, 0, 1, 2, 1, 0, 1, 2, 5, 4], dtype=np.float64),
    "bound_extrapolation4": np.array([4, 3, 2, -2, 0, 1, 2, 1, 0, 1, 2, 5], dtype=np.float64),
    "oscillate": np.array([1, 0, -1, 0, 1, 0, -1, 0, 1], dtype=np.float64),
    "test": np.array([2, 1, 2, 3, 2, 3, 2, 3, 2, 2], dtype=np.float64),
    "mirror_mr1": np.array(
        [81, 89, 63, 96, 64, 13, 52, 54, 18, 11, 71, 88, 61, 78, 2, 90, 76, 64, 54, 23],
        dtype=np.float64,
    ),
    "mirror_ml1": np.array(
        [26, 66, 84, 92, 93, 11, 44, 17, 40, 7, 95, 25, 98, 13, 40, 78, 87, 78, 18, 40],
        dtype=np.float64,
    ),
    "mirror_mr2": np.array(
        [52, 20, 75, 56, 65, 65, 37, 79, 73, 66, 9, 48, 57, 44, 75, 3, 34, 36, 38, 73],
        dtype=np.float64,
    ),
    "mirror_ml2": np.array(
        [81, 78, 69, 65, 59, 21, 99, 88, 2, 98, 56, 77, 84, 28, 11, 52, 55, 27, 46, 11],
        dtype=np.float64,
    ),
    "unmatched": np.array(
        [-34, -34, 15, -33, 34, -29, 44, 0, -9, 1, 37, 44, 3, -48, -16, 25, -45, 12, 40, -9],
        dtype=np.float64,
    ),
    "unmatched2": np.array(
        [45, 26, -26, -22, -21, 48, 48, -35, -35, 12, 8, -49, -20, 43, -6, 9, 20, -29, 4, -36],
        dtype=np.float64,
    ),
    "level_ind_1": np.array(
        [35, 32, 32, -32, -32, -10, 7, -37, 46, -17, -38, 2, 31, -37, 12, 24, -40, -34, 6, 43],
        dtype=np.float64,
    ),
}
long_zc = {k: v for k, v in zc_arrays.items() if len(v) >= 6}
