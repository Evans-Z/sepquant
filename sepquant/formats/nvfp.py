from __future__ import annotations

from dataclasses import dataclass

import torch

from sepquant.formats.fp_ops import (
    from_blocks,
    max_normal,
    quantize_e2m1,
    quantize_unsigned_fp,
    raw_e2m1_block_scale,
    to_blocks,
)


@dataclass(frozen=True)
class NVFP4Format:
    """NVFP4 fake quantization with E2M1 payload and two-level scaling.

    The simulated scale path is:

    final_scale = FP32 tensor scale * E4M3 block scale
    """

    name: str = "nvfp4"
    block_size: int = 16
    eps: float = 1e-12

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        blocks, metadata = to_blocks(tensor, self.block_size)
        scale = self._block_scale(blocks)
        quantized = quantize_e2m1(blocks / scale)
        dequantized = quantized * scale
        return from_blocks(dequantized, metadata)

    def _block_scale(self, blocks: torch.Tensor) -> torch.Tensor:
        raw_scale = raw_e2m1_block_scale(blocks, self.eps)
        return self._quantize_scale(raw_scale)

    def _quantize_scale(self, raw_scale: torch.Tensor) -> torch.Tensor:
        raw_scale_fp32 = raw_scale.float()
        e4m3_max = max_normal(exponent_bits=4, mantissa_bits=3, exponent_bias=7)
        tensor_scale = torch.clamp(raw_scale_fp32.amax() / e4m3_max, min=self.eps)
        block_scale = quantize_unsigned_fp(
            raw_scale_fp32 / tensor_scale,
            exponent_bits=4,
            mantissa_bits=3,
            exponent_bias=7,
        )
        return torch.clamp(tensor_scale * block_scale, min=self.eps).to(raw_scale.dtype)

