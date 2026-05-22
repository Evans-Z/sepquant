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


@dataclass(frozen=True)
class NVFP4ScaleSearchFormat:
    """NVFP4 fake quantization with online E4M3 block-scale search.

    The tensor scale is fixed from the default raw block scales. Each block starts
    from the nearest E4M3 block scale, then searches neighboring E4M3 scale codes.
    """

    name: str = "nvfp4_search"
    block_size: int = 16
    scale_code_offsets: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)
    eps: float = 1e-12

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        blocks, metadata = to_blocks(tensor, self.block_size)
        raw_scale = raw_e2m1_block_scale(blocks, self.eps)
        tensor_scale = self._tensor_scale(raw_scale)
        base_block_scale = quantize_unsigned_fp(
            raw_scale.float() / tensor_scale,
            exponent_bits=4,
            mantissa_bits=3,
            exponent_bias=7,
        )
        base_scale_code = _e4m3_code_from_value(base_block_scale)
        best_quantized: torch.Tensor | None = None
        best_scores: torch.Tensor | None = None

        for scale_code_offset in self.scale_code_offsets:
            scale_code = torch.clamp(base_scale_code + scale_code_offset, min=0, max=E4M3_MAX_CODE)
            block_scale = _e4m3_value_from_code(scale_code, dtype=raw_scale.dtype)
            scale = torch.clamp(tensor_scale * block_scale, min=self.eps).to(raw_scale.dtype)
            quantized = quantize_e2m1(blocks / scale) * scale
            scores = torch.sum((blocks - quantized).square(), dim=-1)
            if best_scores is None or best_quantized is None:
                best_scores = scores
                best_quantized = quantized
                continue

            improved = scores < best_scores
            best_scores = torch.where(improved, scores, best_scores)
            best_quantized = torch.where(improved.unsqueeze(-1), quantized, best_quantized)

        if best_quantized is None:
            raise RuntimeError("No NVFP4 activation scale candidates were evaluated")
        return from_blocks(best_quantized, metadata)

    def _tensor_scale(self, raw_scale: torch.Tensor) -> torch.Tensor:
        raw_scale_fp32 = raw_scale.float()
        e4m3_max = max_normal(exponent_bits=4, mantissa_bits=3, exponent_bias=7)
        return torch.clamp(raw_scale_fp32.amax() / e4m3_max, min=self.eps)


def _e4m3_code_from_value(value: torch.Tensor) -> torch.Tensor:
    value = value.float().clamp(min=0.0, max=E4M3_MAX)
    subnormal_code = torch.round(value / E4M3_SUBNORMAL_STEP)

    safe_value = torch.clamp(value, min=E4M3_MIN_NORMAL)
    exponent = torch.floor(torch.log2(safe_value)).clamp(
        min=1 - E4M3_EXPONENT_BIAS,
        max=((1 << E4M3_EXPONENT_BITS) - 1) - E4M3_EXPONENT_BIAS,
    )
    exponent_field = exponent + E4M3_EXPONENT_BIAS
    mantissa = torch.round((safe_value / torch.pow(2.0, exponent) - 1.0) * (1 << E4M3_MANTISSA_BITS))
    mantissa = mantissa.clamp(min=0, max=(1 << E4M3_MANTISSA_BITS) - 1)
    normal_code = exponent_field * (1 << E4M3_MANTISSA_BITS) + mantissa

    code = torch.where(value < E4M3_MIN_NORMAL, subnormal_code, normal_code)
    return code.round().long().clamp(min=0, max=E4M3_MAX_CODE)


def _e4m3_value_from_code(code: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    code = code.long().clamp(min=0, max=E4M3_MAX_CODE)
    exponent_field = torch.div(code, 1 << E4M3_MANTISSA_BITS, rounding_mode="floor")
    mantissa = code.remainder(1 << E4M3_MANTISSA_BITS).float()

    subnormal = mantissa * E4M3_SUBNORMAL_STEP
    normal = (1.0 + mantissa / (1 << E4M3_MANTISSA_BITS)) * torch.pow(
        2.0,
        exponent_field.float() - E4M3_EXPONENT_BIAS,
    )
    return torch.where(exponent_field == 0, subnormal, normal).to(dtype)


E4M3_EXPONENT_BITS = 4
E4M3_MANTISSA_BITS = 3
E4M3_EXPONENT_BIAS = 7
E4M3_MAX_CODE = (1 << (E4M3_EXPONENT_BITS + E4M3_MANTISSA_BITS)) - 1
E4M3_MAX = max_normal(
    exponent_bits=E4M3_EXPONENT_BITS,
    mantissa_bits=E4M3_MANTISSA_BITS,
    exponent_bias=E4M3_EXPONENT_BIAS,
)
E4M3_MIN_NORMAL = 2.0 ** (1 - E4M3_EXPONENT_BIAS)
E4M3_SUBNORMAL_STEP = 2.0 ** (
    1 - E4M3_EXPONENT_BIAS - E4M3_MANTISSA_BITS
)

