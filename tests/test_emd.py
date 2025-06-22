from __future__ import annotations

import numpy as np
import pytest
from PyEMD.EMD import EMD
from tqdm import trange

from pyemd_rs import emd
from pyemd_rs._testing import cubic_spline, find_extrema_simple, prepare_points_simple

from . import long_zc, zc_arrays

FindExtremaOutput2 = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def prepare_points_simple_orig(  # noqa: C901, PLR0912, PLR0915
    nbsym: int,
    T: np.ndarray,  # noqa: N803
    S: np.ndarray,  # noqa: N803
    max_pos: np.ndarray,
    max_val: np.ndarray | None,  # noqa: ARG001
    min_pos: np.ndarray,
    min_val: np.ndarray | None,  # noqa: ARG001
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs mirroring on signal which extrema can be indexed on
    the position array.

    See :meth:`EMD.prepare_points`.
    """

    # Find indexes of pass
    ind_min = min_pos.astype(np.intp)
    ind_max = max_pos.astype(np.intp)

    # Local variables

    end_min, end_max = len(min_pos), len(max_pos)

    ####################################
    # Left bound - mirror nbsym points to the left
    if ind_max[0] < ind_min[0]:
        if S[0] > S[ind_min[0]]:
            print("l1")
            lmax = ind_max[1 : min(end_max, nbsym + 1)][::-1]
            lmin = ind_min[0 : min(end_min, nbsym + 0)][::-1]
            lsym = ind_max[0]
        else:
            print("l2")

            lmax = ind_max[0 : min(end_max, nbsym)][::-1]
            lmin = np.append(ind_min[0 : min(end_min, nbsym - 1)][::-1], 0)
            lsym = 0
    else:  # noqa: PLR5501
        if S[0] < S[ind_max[0]]:
            print("l3")
            lmax = ind_max[0 : min(end_max, nbsym + 0)][::-1]
            lmin = ind_min[1 : min(end_min, nbsym + 1)][::-1]
            lsym = ind_min[0]
        else:
            print("l4")
            lmax = np.append(ind_max[0 : min(end_max, nbsym - 1)][::-1], 0)
            lmin = ind_min[0 : min(end_min, nbsym)][::-1]
            lsym = 0

    ####################################
    # Right bound - mirror nbsym points to the right
    if ind_max[-1] < ind_min[-1]:
        if S[-1] < S[ind_max[-1]]:
            print("r1")
            rmax = ind_max[max(end_max - nbsym, 0) :][::-1]
            rmin = ind_min[max(end_min - nbsym - 1, 0) : -1][::-1]
            rsym = ind_min[-1]
        else:
            print("r2")
            rmax = np.append(ind_max[max(end_max - nbsym + 1, 0) :], len(S) - 1)[::-1]
            rmin = ind_min[max(end_min - nbsym, 0) :][::-1]
            rsym = len(S) - 1
    else:  # noqa: PLR5501
        if S[-1] > S[ind_min[-1]]:
            print("r3")
            rmax = ind_max[max(end_max - nbsym - 1, 0) : -1][::-1]
            rmin = ind_min[max(end_min - nbsym, 0) :][::-1]
            rsym = ind_max[-1]
        else:
            print("r4")
            rmax = ind_max[max(end_max - nbsym, 0) :][::-1]
            rmin = np.append(ind_min[max(end_min - nbsym + 1, 0) :], len(S) - 1)[::-1]
            rsym = len(S) - 1

    # In case any array missing
    if not lmin.size:
        lmin = ind_min
    if not rmin.size:
        rmin = ind_min
    if not lmax.size:
        lmax = ind_max
    if not rmax.size:
        rmax = ind_max

    # Mirror points
    tlmin = 2 * T[lsym] - T[lmin]
    tlmax = 2 * T[lsym] - T[lmax]
    trmin = 2 * T[rsym] - T[rmin]
    trmax = 2 * T[rsym] - T[rmax]

    # If mirrored points are not outside passed time range.
    if tlmin[0] > T[0] or tlmax[0] > T[0]:
        if lsym == ind_max[0]:
            lmax = ind_max[0 : min(end_max, nbsym)][::-1]
        else:
            lmin = ind_min[0 : min(end_min, nbsym)][::-1]

        if lsym == 0:
            raise Exception("Left edge BUG")  # noqa: TRY002

        lsym = 0
        tlmin = 2 * T[lsym] - T[lmin]
        tlmax = 2 * T[lsym] - T[lmax]

    if trmin[-1] < T[-1] or trmax[-1] < T[-1]:
        if rsym == ind_max[-1]:
            rmax = ind_max[max(end_max - nbsym, 0) :][::-1]
        else:
            rmin = ind_min[max(end_min - nbsym, 0) :][::-1]

        if rsym == len(S) - 1:
            raise Exception("Right edge BUG")  # noqa: TRY002

        rsym = len(S) - 1
        trmin = 2 * T[rsym] - T[rmin]
        trmax = 2 * T[rsym] - T[rmax]

    zlmax = S[lmax]
    zlmin = S[lmin]
    zrmax = S[rmax]
    zrmin = S[rmin]

    tmin = np.append(tlmin, np.append(T[ind_min], trmin))
    tmax = np.append(tlmax, np.append(T[ind_max], trmax))
    zmin = np.append(zlmin, np.append(S[ind_min], zrmin))
    zmax = np.append(zlmax, np.append(S[ind_max], zrmax))

    max_extrema = np.array([tmax, zmax])
    min_extrema = np.array([tmin, zmin])

    # Make double sure, that each extremum is significant
    max_dup_idx = np.where(max_extrema[0, 1:] == max_extrema[0, :-1])
    max_extrema = np.delete(max_extrema, max_dup_idx, axis=1)
    min_dup_idx = np.where(min_extrema[0, 1:] == min_extrema[0, :-1])
    min_extrema = np.delete(min_extrema, min_dup_idx, axis=1)

    return max_extrema, min_extrema


def find_extrema_simple_orig(T: np.ndarray, S: np.ndarray) -> FindExtremaOutput2:  # noqa: C901, N803, PLR0912, PLR0915
    """
    Performs extrema detection, where extremum is defined as a point,
    that is above/below its neighbours.

    See :meth:`EMD.find_extrema`.
    """

    # Finds indexes of zero-crossings
    S1, S2 = S[:-1], S[1:]  # noqa: N806
    indzer = np.nonzero(S1 * S2 < 0)[0]
    if np.any(S == 0):
        indz = np.nonzero(S == 0)[0]
        if np.any(np.diff(indz) == 1):
            zer = S == 0
            dz = np.diff(np.append(np.append(0, zer), 0))
            debz = np.nonzero(dz == 1)[0]
            finz = np.nonzero(dz == -1)[0] - 1
            indz = np.round((debz + finz) / 2.0)

        indzer = np.sort(np.append(indzer, indz))

    # Finds local extrema
    d = np.diff(S)
    d1, d2 = d[:-1], d[1:]
    indmin = np.nonzero(np.r_[d1 * d2 < 0] & np.r_[d1 < 0])[0] + 1
    indmax = np.nonzero(np.r_[d1 * d2 < 0] & np.r_[d1 > 0])[0] + 1

    # When two or more points have the same value
    if np.any(d == 0):
        imax, imin = [], []

        bad = d == 0
        dd = np.diff(np.append(np.append(0, bad), 0))
        debs = np.nonzero(dd == 1)[0]
        fins = np.nonzero(dd == -1)[0]
        if debs[0] == 1:
            if len(debs) > 1:
                debs, fins = debs[1:], fins[1:]
            else:
                debs, fins = [], []

        if len(debs) > 0:  # noqa: SIM102
            if fins[-1] == len(S) - 1:
                if len(debs) > 1:
                    debs, fins = debs[:-1], fins[:-1]
                else:
                    debs, fins = [], []

        lc = len(debs)
        if lc > 0:
            for k in range(lc):
                if d[debs[k] - 1] > 0:
                    if d[fins[k]] < 0:
                        imax.append(np.round((fins[k] + debs[k]) / 2.0))
                elif d[fins[k]] > 0:
                    imin.append(np.round((fins[k] + debs[k]) / 2.0))

        if len(imax) > 0:
            indmax = indmax.tolist()
            for x in imax:
                indmax.append(int(x))
            indmax.sort()

        if len(imin) > 0:
            indmin = indmin.tolist()
            for x in imin:
                indmin.append(int(x))
            indmin.sort()

    local_max_pos = T[indmax]
    local_max_val = S[indmax]
    local_min_pos = T[indmin]
    local_min_val = S[indmin]

    return local_max_pos, local_max_val, local_min_pos, local_min_val, indzer


@pytest.mark.parametrize("arr_id", zc_arrays.items(), ids=zc_arrays.keys())
def test_find_extrema(arr_id):
    id_, arr = arr_id
    maxpos, _, minpos, _, zc = find_extrema_simple_orig(np.arange(len(arr)), arr)
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

    max_extrema, min_extrema = prepare_points_simple_orig(2, pos, arr, maxpos, None, minpos, None)

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
        maxpos, maxval, minpos, minval, zc = find_extrema_simple_orig(pos, arr)
        feo = find_extrema_simple(arr)
        assert np.array_equal(feo.max_pos, maxpos)
        assert np.array_equal(feo.min_pos, minpos)
        assert np.array_equal(feo.zc_ind, zc)
        if len(maxpos) + len(minpos) < 3:
            continue
        max_extrema, min_extrema = prepare_points_simple_orig(
            2, pos, arr, maxpos, None, minpos, None
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

    max_extrema, min_extrema = prepare_points_simple_orig(2, pos, arr, maxpos, None, minpos, None)
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
