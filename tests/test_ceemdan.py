from __future__ import annotations

import numpy as np
import pytest
from PyEMD import CEEMDAN

from pyemd_rs import ceemdan
from pyemd_rs._testing import normal_mt


def test_generate_noise():
    ceemdan_obj = CEEMDAN(seed=123)
    rng = np.random.RandomState(123)
    ceemdan_noise = ceemdan_obj.generate_noise(1.0, 1000)
    numpy_noise = rng.normal(loc=0.0, scale=1.0, size=1000)
    assert np.array_equal(ceemdan_noise, numpy_noise)
    rs_noise = normal_mt(123, 1000, 1.0)
    assert np.allclose(ceemdan_noise, rs_noise)
    assert np.array_equal(ceemdan_noise, rs_noise)


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("trials", [5, 10, 100])
@pytest.mark.parametrize("epsilon", [0.005, 0.001, 0.01])
def test_ceemdan(epsilon, trials, parallel):
    ceemdan_obj = CEEMDAN(seed=123, trials=trials, epsilon=epsilon)
    rng = np.random.RandomState(123)
    x = rng.normal(size=20)
    out_imf, out_resid = ceemdan(x, trials=trials, seed=123, epsilon=epsilon, parallel=parallel)
    out = ceemdan_obj(x)
    print(out)
    assert np.allclose(out[:-1, :], out_imf)
    assert np.allclose(out[-1, :], out_resid)
