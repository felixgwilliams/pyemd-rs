from __future__ import annotations

import numpy as np
import pytest
from PyEMD.EMD import EMD
from pyemd_rs._pyemd_rs import find_extrema_simple, prepare_points_simple

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
}
long_zc = {k: v for k, v in zc_arrays.items() if len(v) >= 6}


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


@pytest.mark.parametrize("arr_id", zc_arrays.items(), ids=zc_arrays.keys())
def test_find_extrema(arr_id):
    id_, arr = arr_id
    maxpos, _, minpos, _, zc = EMD._find_extrema_simple(np.arange(len(arr)), arr)  # noqa: SLF001
    feo = find_extrema_simple(arr)
    assert np.allclose(feo.max_pos, maxpos)
    assert np.allclose(feo.min_pos, minpos)
    assert np.allclose(feo.zc_ind, zc)


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
