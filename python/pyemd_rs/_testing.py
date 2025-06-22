"""Re-exports for functions used for testing."""

from __future__ import annotations

from ._pyemd_rs import (
    FindExtremaOutput,
    cubic_spline,
    find_extrema_simple,
    find_extrema_simple_pos,
    normal_mt,
    prepare_points_simple,
)

__all__ = [
    "FindExtremaOutput",
    "cubic_spline",
    "find_extrema_simple",
    "find_extrema_simple_pos",
    "normal_mt",
    "prepare_points_simple",
]
