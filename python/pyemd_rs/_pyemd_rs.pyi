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
    zc_ind: npt.NDArray[np.intp]

def find_extrema_simple(
    pos: npt.NDArray[np.float64], val: npt.NDArray[np.float64]
) -> FindExtremaOutput: ...
