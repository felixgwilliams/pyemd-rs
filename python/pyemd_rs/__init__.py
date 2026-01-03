"""PyEMD with Rust."""

from __future__ import annotations

from ._pyemd_rs import ceemdan, default_ceemdan_opts, default_emd_opts, emd

__version__ = "0.1.3"
__all__ = [
    "ceemdan",
    "default_ceemdan_opts",
    "default_emd_opts",
    "emd",
]
