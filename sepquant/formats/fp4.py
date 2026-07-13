from __future__ import annotations

from sepquant.formats.base import FP4Format
from sepquant.formats.hifp import HIF4Format, HIF4ScaleSearchFormat
from sepquant.formats.mxfp import (
    MXFP4Format,
    MXFP4PlusFormat,
    MXFP4PlusScaleSearchFormat,
    MXFP4ScaleSearchFormat,
)
from sepquant.formats.nvfp import NVFP4Format, NVFP4ScaleSearchFormat


def get_fp4_format(name: str) -> FP4Format:
    """Return a named FP4 format configuration."""

    normalized = name.lower()
    if normalized == "mxfp4":
        return MXFP4Format()
    if normalized in {"mxfp4+", "mxfp4_plus", "mxfp4-plus", "mxfp4plus"}:
        return MXFP4PlusFormat()
    if normalized in {
        "mxfp4+_search",
        "mxfp4_plus_search",
        "mxfp4-plus-search",
        "mxfp4_plus_scale_search",
        "mxfp4-plus-scale-search",
        "mxfp4plus_search",
    }:
        return MXFP4PlusScaleSearchFormat()
    if normalized in {"mxfp4_search", "mxfp4-scale-search", "mxfp4_scale_search"}:
        return MXFP4ScaleSearchFormat()
    if normalized == "nvfp4":
        return NVFP4Format()
    if normalized in {"nvfp4_search", "nvfp4-scale-search", "nvfp4_scale_search"}:
        return NVFP4ScaleSearchFormat()
    if normalized == "hif4":
        return HIF4Format()
    if normalized in {"hif4_search", "hif4-scale-search", "hif4_scale_search"}:
        return HIF4ScaleSearchFormat()
    raise ValueError(f"Unsupported FP4 format: {name}")

