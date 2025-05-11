from __future__ import annotations

import numpy as np
import pytest
from PyEMD.EMD import EMD
from pyemd_rs._pyemd_rs import cubic_spline, find_extrema_simple, prepare_points_simple

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
        if lsym == 0:
            raise Exception("Left edge BUG")  # noqa: TRY002
        if lsym == ind_max[0]:
            print("ml1")
            lmax = ind_max[0 : min(end_max, nbsym)][::-1]
            tlmax = 2 * T[lsym] - T[lmax]
        else:
            print("ml2")
            lmin = ind_min[0 : min(end_min, nbsym)][::-1]
            tlmin = 2 * T[lsym] - T[lmin]

        lsym = 0
    print(rmin, rmax, rsym)
    print(trmin, trmax)
    if trmin[-1] < T[-1] or trmax[-1] < T[-1]:
        if rsym == len(S) - 1:
            raise Exception("Right edge BUG")  # noqa: TRY002
        if rsym == ind_max[-1]:
            print("mr1")
            rmax = ind_max[max(end_max - nbsym, 0) :][::-1]
            trmax = 2 * T[rsym] - T[rmax]
        else:
            print("mr2")
            rmin = ind_min[max(end_min - nbsym, 0) :][::-1]
            trmin = 2 * T[rsym] - T[rmin]

        rsym = len(S) - 1

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
        arr, minpos.astype(np.uint64), maxpos.astype(np.uint64), 2
    )
    assert np.array_equal(max_extrema[0, :], tmax)
    assert np.array_equal(max_extrema[1, :], zmax)
    assert np.array_equal(min_extrema[0, :], tmin)
    assert np.array_equal(min_extrema[1, :], zmin)

    print(tmin, zmin, tmax, zmax)


def test_prepare_points2():
    pytest.skip("Use to generate test examples")
    gen = np.random.default_rng(12313)
    emd = EMD()

    for _i in range(10000):
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
