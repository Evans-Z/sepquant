from __future__ import annotations

from dataclasses import dataclass

import torch

from sepquant.formats.fp_ops import (
    from_blocks,
    quantize_e2m1,
    raw_e2m1_block_scale,
    to_blocks,
)


@dataclass(frozen=True)
class MXFP4Format:
    """MXFP4 fake quantization with E2M1 payload and power-of-two block scales."""

    name: str = "mxfp4"
    block_size: int = 32
    eps: float = 1e-12

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        blocks, metadata = to_blocks(tensor, self.block_size)
        scale = self._block_scale(blocks)
        quantized = quantize_e2m1(blocks / scale)
        dequantized = quantized * scale
        return from_blocks(dequantized, metadata)

    def _block_scale(self, blocks: torch.Tensor) -> torch.Tensor:
        raw_scale = raw_e2m1_block_scale(blocks, self.eps)
        return torch.pow(2.0, torch.ceil(torch.log2(raw_scale)))

