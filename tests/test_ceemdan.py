from __future__ import annotations

import numpy as np
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


def test_ceemdan():
    ceemdan_obj = CEEMDAN(seed=123, trials=10)
    rng = np.random.RandomState(123)
    x = rng.normal(size=20)
    out_imf, out_resid = ceemdan(x, 10, seed=123)
    out = ceemdan_obj(x)
    print(out)
    assert np.allclose(out[:-1, :], out_imf)
    assert np.allclose(out[-1, :], out_resid)
