from __future__ import annotations

from typing import Protocol

import torch


class FP4Format(Protocol):
    """Common interface for FP4 fake-quantization formats."""

    name: str
    block_size: int
    eps: float

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize-dequantize a tensor."""
        ...

