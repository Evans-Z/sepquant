from __future__ import annotations

from sepquant.formats.base import FP4Format
from sepquant.formats.mxfp import MXFP4Format
from sepquant.formats.nvfp import NVFP4Format


def get_fp4_format(name: str) -> FP4Format:
    """Return a named FP4 format configuration."""

    normalized = name.lower()
    if normalized == "mxfp4":
        return MXFP4Format()
    if normalized == "nvfp4":
        return NVFP4Format()
    raise ValueError(f"Unsupported FP4 format: {name}")

