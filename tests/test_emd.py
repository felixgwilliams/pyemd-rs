from __future__ import annotations

import numpy as np
import pytest
from PyEMD.EMD import EMD
from tqdm import trange

from pyemd_rs import emd
from pyemd_rs._testing import cubic_spline, find_extrema_simple, prepare_points_simple

from . import long_zc, zc_arrays

FindExtremaOutput2 = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@pytest.mark.parametrize("arr_id", zc_arrays.items(), ids=zc_arrays.keys())
def test_find_extrema(arr_id):
    id_, arr = arr_id

    maxpos, _, minpos, _, zc = EMD._find_extrema_simple(np.arange(len(arr)), arr)  # noqa: SLF001
    feo = find_extrema_simple(arr)
    assert np.array_equal(feo.max_pos, maxpos)
    assert np.array_equal(feo.min_pos, minpos)
    assert np.array_equal(feo.zc_ind, zc)


@pytest.mark.parametrize("arr_id", long_zc.items(), ids=long_zc.keys())
def test_prepare_points(arr_id):
    id_, arr = arr_id
    # emd = EMD()
    pos = np.arange(len(arr))

    maxpos, maxval, minpos, minval, zc = EMD._find_extrema_simple(pos, arr)  # noqa: SLF001
    if len(maxpos) + len(minpos) < 3:
        pytest.skip()

    max_extrema, min_extrema = EMD(nbsym=2).prepare_points_simple(
        pos, arr, maxpos, None, minpos, None
    )

    print(id_, max_extrema, min_extrema)

    (tmin, zmin, tmax, zmax) = prepare_points_simple(
        arr, minpos.astype(np.uintp), maxpos.astype(np.uint64), 2
    )
    assert np.array_equal(max_extrema[0, :], tmax)
    assert np.array_equal(max_extrema[1, :], zmax)
    assert np.array_equal(min_extrema[0, :], tmin)
    assert np.array_equal(min_extrema[1, :], zmin)

    print(tmin, zmin, tmax, zmax)


def test_prepare_points2():
    pytest.skip("Use to generate test examples")
    gen = np.random.default_rng(12313)
    emd_obj = EMD()

    for _i in trange(10000):
        arr = np.round(gen.random(size=20) * 100) - 50
        pos = np.arange(len(arr))
        maxpos, maxval, minpos, minval, zc = EMD._find_extrema_simple(pos, arr)  # noqa: SLF001
        feo = find_extrema_simple(arr)
        assert np.array_equal(feo.max_pos, maxpos)
        assert np.array_equal(feo.min_pos, minpos)
        assert np.array_equal(feo.zc_ind, zc)
        if len(maxpos) + len(minpos) < 3:
            continue
        max_extrema, min_extrema = EMD(nbsym=2).prepare_points_simple(
            pos, arr, maxpos, None, minpos, None
        )
        max_spline_pos, max_spline_val = emd_obj.spline_points(pos, max_extrema)
        min_spline_pos, min_spline_val = emd_obj.spline_points(pos, min_extrema)
        max_spline_pos2, max_spline_val2 = cubic_spline(
            len(arr), max_extrema[0].astype(np.intp), max_extrema[1]
        )
        min_spline_pos2, min_spline_val2 = cubic_spline(
            len(arr), min_extrema[0].astype(np.intp), min_extrema[1]
        )

        assert np.allclose(max_spline_val, max_spline_val2)
        assert np.allclose(min_spline_val, min_spline_val2)
        assert np.allclose(max_spline_pos, max_spline_pos2)
        assert np.allclose(min_spline_pos, min_spline_pos2)
        emd_obj = EMD()
        emd_obj.emd(arr)
        imf, resid = emd(arr)
        assert np.allclose(imf, emd_obj.imfs)
        assert np.allclose(resid, emd_obj.residue)


@pytest.mark.parametrize("arr_id", long_zc.items(), ids=long_zc.keys())
def test_cubic_spline(arr_id):
    id_, arr = arr_id
    emd = EMD()
    pos = np.arange(len(arr), dtype=np.intp)

    maxpos, maxval, minpos, minval, zc = EMD._find_extrema_simple(pos, arr)  # noqa: SLF001
    if len(maxpos) + len(minpos) < 3:
        pytest.skip()

    max_extrema, min_extrema = EMD(nbsym=2).prepare_points_simple(
        pos, arr, maxpos, None, minpos, None
    )
    max_spline_pos, max_spline_val = emd.spline_points(pos, max_extrema)
    min_spline_pos, min_spline_val = emd.spline_points(pos, min_extrema)
    max_spline_pos2, max_spline_val2 = cubic_spline(
        len(arr), max_extrema[0].astype(np.intp), max_extrema[1]
    )
    min_spline_pos2, min_spline_val2 = cubic_spline(
        len(arr), min_extrema[0].astype(np.intp), min_extrema[1]
    )

    assert np.allclose(max_spline_val, max_spline_val2)
    assert np.allclose(min_spline_val, min_spline_val2)
    assert np.allclose(max_spline_pos, max_spline_pos2)
    assert np.allclose(min_spline_pos, min_spline_pos2)
    # t = pos[np.r_[pos >= min_extrema[0, 0]] & np.r_[pos <= min_extrema[0, -1]]]
    # t = pos[np.r_[pos >= max_extrema[0, 0]] & np.r_[pos <= max_extrema[0, -1]]]


spline_inputs = {
    "min_example": np.array(
        [[-3, 0, 2, 5, 9, 12, 14, 16, 18], [13.0, 63.0, 63.0, 13.0, 11.0, 61.0, 2.0, 2.0, 61.0]]
    ),
    "max_example": np.array(
        [[-5, -1, 1, 3, 7, 11, 13, 15, 17], [54.0, 96.0, 89.0, 96.0, 54.0, 88.0, 78.0, 90.0, 78.0]]
    ),
}


@pytest.mark.parametrize("cube_arr_id", spline_inputs.items(), ids=spline_inputs.keys())
def test_cubic_spline2(cube_arr_id):
    id_, arr = cube_arr_id
    emd = EMD()
    n = 20
    pos = np.arange(n, dtype=np.intp)

    spline_pos, spline_val = emd.spline_points(pos, arr)
    spline_pos2, spline_val2 = cubic_spline(n, arr[0].astype(np.intp), arr[1])

    assert np.allclose(spline_pos, spline_pos2)
    assert np.allclose(spline_val, spline_val2)


@pytest.mark.parametrize("arr_id", long_zc.items(), ids=long_zc.keys())
def test_emd(arr_id):
    id_, arr = arr_id
    emd_obj = EMD()
    emd_obj.emd(arr)
    imf, resid = emd(arr)
    assert np.allclose(imf, emd_obj.imfs)  # pyright: ignore[reportArgumentType]
    assert np.allclose(resid, emd_obj.residue)  # pyright: ignore[reportArgumentType]
