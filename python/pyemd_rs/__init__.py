"""PyEMD with Rust."""

from __future__ import annotations

from ._pyemd_rs import ceemdan, emd

__version__ = "0.1.2"
__all__ = ["ceemdan", "emd"]
