import numpy as np
from numpy import typing as npt

# functions for unit testing
class FindExtremaOutput:
    max_pos: npt.NDArray[np.float64]
    max_val: npt.NDArray[np.float64]
    min_pos: npt.NDArray[np.float64]
    min_val: npt.NDArray[np.float64]
    zc_ind: npt.NDArray[np.uintp]

def find_extrema_simple(val: npt.NDArray[np.float64]) -> FindExtremaOutput: ...
def prepare_points_simple(
    val: npt.NDArray[np.float64],
    min_pos: npt.NDArray[np.uintp],
    max_pos: npt.NDArray[np.uintp],
    nsymb: int,
) -> tuple[
    npt.NDArray[np.uintp],
    npt.NDArray[np.float64],
    npt.NDArray[np.uintp],
    npt.NDArray[np.float64],
]: ...
def find_extrema_simple_pos(
    val: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.uintp], npt.NDArray[np.uintp]]: ...
def cubic_spline(
    n: int,
    extrema_pos: npt.NDArray[np.intp],
    extrema_val: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]: ...
def normal_mt(seed: int | None, size: int, scale: float) -> npt.NDArray[np.float64]: ...

# public functions
def emd(
    val: npt.NDArray[np.float64], max_imf: int | None = None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate the IMFs and residual of an array via EMD.

    Args:
        val (npt.NDArray[np.float64]): Array to calculate IMFs
        max_imf (int | None, optional): maximum number of IMFs including the residual.
            Defaults to None.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: 2D array of IMFs with
            1D array of the residual
    """

def ceemdan(
    val: npt.NDArray[np.float64],
    trials: int = 100,
    max_imf: int | None = None,
    seed: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate the IMFs and residual of an array via CEEMDAN.

    Args:
        val (npt.NDArray[np.float64]): Array to calculate IMFs
        trials (int, optional): Number of trials in CEEMDAN. Defaults to 100.
        max_imf (int | None, optional): maximum number of IMFs including the residual.
            Defaults to None.
        seed (int | None, optional): Random seed for generating the noise. If not given, a seed
            will be generated using the getrandom crate. Defaults to None.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: 2D array of IMFs with
            1D array of the residual
    """
