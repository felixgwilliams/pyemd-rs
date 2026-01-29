from typing import TypedDict

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
    val: npt.NDArray[np.float64],
    max_imf: int | None = None,
    imf_check: bool = True,
    *,
    s_number: int = 0,
    max_iteration: int = 1000,
    svar_thresh: float = 0.001,
    energy_ratio_thresh: float = 0.2,
    std_thresh: float = 0.2,
    range_thresh: float = 0.001,
    total_power_thresh: float = 0.05,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate the IMFs and residual of an array via EMD.

    Args:
        val (npt.NDArray[np.float64]): Array to calculate IMFs
        max_imf (int | None, optional): maximum number of IMFs including the residual.
            Defaults to None.
        imf_check (bool, optional): Whether to check the IMF condition and stopping criteria to
            terminate sifting. If False, rely on max_iteration. Defaults to True
        s_number (int, optional): Number of siftings to perform before performing IMF checks
            and stopping criteria. Defaults to 0.
        max_iteration (int, optional): Maximum number of iterations for EMD. Defaults to 1000
        svar_thresh (float, optional): Variance threshold for EMD. Defaults to 0.001
        energy_ratio_thresh (float, optional): Energy ratio threshold for EMD. Defaults to 0.2
        std_thresh (float, optional): Standard deviation threshold for EMD. Defaults to 0.2
        range_thresh (float, optional): Range threshold for EMD. Defaults to 0.001
        total_power_thresh (float, optional): Total power threshold for EMD. Defaults to 0.05

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: 2D array of IMFs with
            1D array of the residual
    """

def ceemdan(
    val: npt.NDArray[np.float64],
    trials: int = 100,
    max_imf: int | None = None,
    seed: int | None = None,
    epsilon: float = 0.005,
    imf_check: bool = True,
    *,
    s_number: int = 0,
    parallel: bool = True,
    noise_scale: float = 1.0,
    c_range_thresh: float = 0.01,
    c_total_power_thresh: float = 0.05,
    c_max_imf: int = 100,
    max_iteration: int = 1000,
    svar_thresh: float = 0.001,
    energy_ratio_thresh: float = 0.2,
    std_thresh: float = 0.2,
    range_thresh: float = 0.001,
    total_power_thresh: float = 0.05,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate the IMFs and residual of an array via CEEMDAN.

    Args:
        val (npt.NDArray[np.float64]): Array to calculate IMFs
        trials (int, optional): Number of trials in CEEMDAN. Defaults to 100.
        max_imf (int | None, optional): maximum number of IMFs including the residual.
            Defaults to None.
        seed (int | None, optional): Random seed for generating the noise. If not given, a seed
            will be generated using the getrandom crate. Defaults to None.
        epsilon (float, optional): Scale for random noise added to input. Defaults to 0.005
        imf_check (bool, optional): Whether to check the IMF condition and stopping criteria to
            terminate sifting. If False, rely on max_iteration. Defaults to True
        s_number (int, optional): Number of siftings to perform before performing IMF checks
            and stopping criteria. Defaults to 0.
        parallel (bool, optional): Whether to use rayon for parallelising code. Defaults to True
        noise_scale (float, optional): Scale for random noise added to input. Defaults to 1.0
        c_range_thresh (float, optional): Range threshold for CEEMDAN. Defaults to 0.01
        c_total_power_thresh (float, optional): Total power threshold for CEEMDAN. Defaults to 0.05
        c_max_imf (int, optional): Maximum number of IMFs to calculate. Defaults to 100
        max_iteration (int, optional): Maximum number of iterations for EMD. Defaults to 1000
        svar_thresh (float, optional): Variance threshold for EMD. Defaults to 0.001
        energy_ratio_thresh (float, optional): Energy ratio threshold for EMD. Defaults to 0.2
        std_thresh (float, optional): Standard deviation threshold for EMD. Defaults to 0.2
        range_thresh (float, optional): Range threshold for EMD. Defaults to 0.001
        total_power_thresh (float, optional): Total power threshold for EMD. Defaults to 0.05


    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: 2D array of IMFs with
            1D array of the residual
    """

class EmdOptsDict(TypedDict):
    """EMD Options as a dict.

    Attributes:
        imf_check (bool, optional): Whether to check the IMF condition to terminate sifting.
            If False,rely on max_iteration. Defaults to True
        max_iteration (int, optional): Maximum number of iterations for EMD. Defaults to 1000
        svar_thresh (float, optional): Variance threshold for EMD. Defaults to 0.001
        energy_ratio_thresh (float, optional): Energy ratio threshold for EMD. Defaults to 0.2
        std_thresh (float, optional): Standard deviation threshold for EMD. Defaults to 0.2
        range_thresh (float, optional): Range threshold for EMD. Defaults to 0.001
        total_power_thresh (float, optional): Total power threshold for EMD. Defaults to 0.05
    """

    imf_check: bool
    max_iteration: int
    svar_thresh: float
    energy_ratio_thresh: float
    std_thresh: float
    range_thresh: float
    total_power_thresh: float

class CeemdanOptsDict(TypedDict):
    """CEEMDAN Options as a dict.

    Attributes:
        noise_scale (float, optional): Scale for random noise added to input. Defaults to 1.0
        c_range_thresh (float, optional): Range threshold for CEEMDAN. Defaults to 0.01
        c_total_power_thresh (float, optional): Total power threshold for CEEMDAN. Defaults to 0.05
        c_max_imf (int, optional): Maximum number of IMFs to calculate. Defaults to 100
    """

    noise_scale: float
    c_range_thresh: float
    c_total_power_thresh: float
    c_max_imf: int

def default_emd_opts() -> EmdOptsDict:
    """Get Default EMD Options as dict."""

def default_ceemdan_opts() -> CeemdanOptsDict:
    """Get Default CEEMDAN Options as dict."""

__version__: str
