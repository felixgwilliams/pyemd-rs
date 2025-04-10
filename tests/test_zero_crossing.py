from __future__ import annotations

import numpy as np
import pytest
from PyEMD.EMD import EMD
from pyemd_rs._pyemd_rs import find_extrema_simple

zc_arrays = {
    "zeros": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
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
    "original": np.array([-1, 0, 1, 0, -1, 0, 3, 0, -9, 0], dtype=np.float64),
    "repeats": np.array([-1, 0, 1, 1, 0, -1, 0, 3, 0, -9, 0], dtype=np.float64),
}


@pytest.mark.parametrize("arr_id", zc_arrays.items(), ids=zc_arrays.keys())
def test_find_extrema(arr_id):
    id_, arr = arr_id
    maxpos, _, minpos, _, zc = EMD._find_extrema_simple(np.arange(len(arr)), arr)  # noqa: SLF001
    feo = find_extrema_simple(np.arange(len(arr), dtype=np.float64), arr)
    assert np.allclose(feo.max_pos, maxpos)
    assert np.allclose(feo.min_pos, minpos)
    assert np.allclose(feo.zc_ind, zc)
