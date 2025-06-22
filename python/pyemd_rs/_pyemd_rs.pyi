import numpy as np
from numpy import typing as npt

def sum_as_string(a: int, b: int) -> str:
    """Formats the sum of two numbers as string.

    Args:
        a (int): first number
        b (int): second number

    Returns:
        str: sum of numbers as a string
    """

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
def emd(
    val: npt.NDArray[np.float64], max_imf: int | None = None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def normal_mt(seed: int | None, size: int, scale: float) -> npt.NDArray[np.float64]: ...
def ceemdan(
    val: npt.NDArray[np.float64],
    trials: int = 100,
    max_imf: int | None = None,
    seed: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
