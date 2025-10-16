# pyemd-rs

[![License:MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Version](https://img.shields.io/pypi/v/pyemd-rs)](https://pypi.org/project/pyemd-rs/)

This library attempts to reimplement the [PyEMD](https://github.com/laszukdawid/PyEMD) library in Rust.

The original library is licensed under [Apache-2.0 Copyright 2017 Dawid Laszuk](https://github.com/laszukdawid/PyEMD/blob/master/LICENSE.md)

## Usage

The library exports two functions, `emd` and `ceemdan`.
Each takes a 1D numpy array and returns a tuple.
The first element is a 2D array of the Intrinsic Mode Functions (IMFs) and the second is a 1D array of the non-periodic residual component.

The `ceemdan` method uses random noise as part of the procedure, so you may want to set the seed to ensure reproducibility.

```python
import numpy as np
from pyemd_rs import emd, ceemdan

x = np.sin(np.linspace(0, 2*np.pi, 101)) + np.random.normal(size=101, scale=0.1)

imfs_emd, residual_emd = emd(x)
imfs_ceemdan, residual_ceemdan = ceemdan(x, seed=123)
```
