from __future__ import annotations

from sepquant.formats.base import FP4Format
from sepquant.formats.mxfp import MXFP4Format, MXFP4ScaleSearchFormat
from sepquant.formats.nvfp import NVFP4Format, NVFP4ScaleSearchFormat


def get_fp4_format(name: str) -> FP4Format:
    """Return a named FP4 format configuration."""

    normalized = name.lower()
    if normalized == "mxfp4":
        return MXFP4Format()
    if normalized in {"mxfp4_search", "mxfp4-scale-search", "mxfp4_scale_search"}:
        return MXFP4ScaleSearchFormat()
    if normalized == "nvfp4":
        return NVFP4Format()
    if normalized in {"nvfp4_search", "nvfp4-scale-search", "nvfp4_scale_search"}:
        return NVFP4ScaleSearchFormat()
    raise ValueError(f"Unsupported FP4 format: {name}")

