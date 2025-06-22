from __future__ import annotations

import numpy as np
import polars as pl
from PyEMD import CEEMDAN

from pyemd_rs._pyemd_rs import ceemdan

# def main():
df = pl.read_parquet("example_data.pq")
arr = df.get_column("open").to_numpy()
offset = 12340
x = arr[offset : (14400 + offset)].copy()
# x = np.random.normal(size=14400).cumsum()
imfs, resid = ceemdan(x, trials=10, seed=123)

ceemdan_obj = CEEMDAN(seed=123, trials=10)
out = ceemdan_obj(x)
assert np.allclose(out[:-1, :], imfs)  # noqa: S101
assert np.allclose(out[-1, :], resid)  # noqa: S101


# if __name__ == "__main__":
#     main()
