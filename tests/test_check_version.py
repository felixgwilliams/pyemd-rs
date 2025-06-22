from __future__ import annotations

from importlib.metadata import version

import pyemd_rs


def test_version():
    assert pyemd_rs.__version__ == version("pyemd_rs")
