from __future__ import annotations

from dataclasses import dataclass

import torch

from sepquant.formats.fp_ops import (
    from_blocks,
    quantize_e2m1,
    quantize_unsigned_fp,
    raw_e2m1_block_scale,
    to_blocks,
    unsigned_fp_code_from_value,
    unsigned_fp_value_from_code,
)


@dataclass(frozen=True)
class MXFP4Format:
    """MXFP4 fake quantization with E2M1 payload and power-of-two block scales."""

    name: str = "mxfp4"
    block_size: int = 32
    eps: float = 1e-12
    min_scale_exponent: int = -127
    max_scale_exponent: int = 127

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        blocks, metadata = to_blocks(tensor, self.block_size)
        scale = self._block_scale(blocks)
        quantized = quantize_e2m1(blocks / scale)
        dequantized = quantized * scale
        return from_blocks(dequantized, metadata)

    def _block_scale(self, blocks: torch.Tensor) -> torch.Tensor:
        raw_scale = raw_e2m1_block_scale(blocks, self.eps)
        return _e8m0_scale_from_raw(
            raw_scale,
            min_exponent=self.min_scale_exponent,
            max_exponent=self.max_scale_exponent,
        )


@dataclass(frozen=True)
class MXFP4E4M3Format:
    """MXFP4 with E2M1 payload and unsigned E4M3 per-block scales.

    Same microscaling layout as MXFP4 (block size 32); only the block-scale
    encoding changes from E8M0 (power-of-two) to unsigned E4M3.
    """

    name: str = "mxfp4_e4m3"
    block_size: int = 32
    eps: float = 1e-12
    scale_exponent_bits: int = 4
    scale_mantissa_bits: int = 3
    scale_exponent_bias: int = 7

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        blocks, metadata = to_blocks(tensor, self.block_size)
        scale = self._block_scale(blocks)
        quantized = quantize_e2m1(blocks / scale)
        dequantized = quantized * scale
        return from_blocks(dequantized, metadata)

    def _block_scale(self, blocks: torch.Tensor) -> torch.Tensor:
        raw_scale = raw_e2m1_block_scale(blocks, self.eps)
        return _exmy_scale_from_raw(
            raw_scale,
            exponent_bits=self.scale_exponent_bits,
            mantissa_bits=self.scale_mantissa_bits,
            exponent_bias=self.scale_exponent_bias,
            eps=self.eps,
        )


@dataclass(frozen=True)
class MXFP4E5M3Format:
    """MXFP4 with E2M1 payload and unsigned E5M3 (UE5M3) per-block scales.

    Same microscaling layout as MXFP4 (block size 32); only the block-scale
    encoding changes from E8M0 (power-of-two) to unsigned E5M3. Relative to
    E4M3, the unused sign bit becomes an extra exponent bit (bias 15), so the
    minimum non-zero scale goes from ``2**-9`` to ``2**-17``.
    """

    name: str = "mxfp4_e5m3"
    block_size: int = 32
    eps: float = 1e-12
    scale_exponent_bits: int = 5
    scale_mantissa_bits: int = 3
    scale_exponent_bias: int = 15

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        blocks, metadata = to_blocks(tensor, self.block_size)
        scale = self._block_scale(blocks)
        quantized = quantize_e2m1(blocks / scale)
        dequantized = quantized * scale
        return from_blocks(dequantized, metadata)

    def _block_scale(self, blocks: torch.Tensor) -> torch.Tensor:
        raw_scale = raw_e2m1_block_scale(blocks, self.eps)
        return _exmy_scale_from_raw(
            raw_scale,
            exponent_bits=self.scale_exponent_bits,
            mantissa_bits=self.scale_mantissa_bits,
            exponent_bias=self.scale_exponent_bias,
            eps=self.eps,
        )


@dataclass(frozen=True)
class MXFP4E4M3ScaleSearchFormat:
    """MXFP4 E4M3 fake quantization with online per-block scale-code search."""

    name: str = "mxfp4_e4m3_search"
    block_size: int = 32
    scale_code_offsets: tuple[int, ...] = (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8)
    eps: float = 1e-12
    scale_exponent_bits: int = 4
    scale_mantissa_bits: int = 3
    scale_exponent_bias: int = 7

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        return _quantize_mxfp4_float_scale_search(
            tensor,
            block_size=self.block_size,
            scale_code_offsets=self.scale_code_offsets,
            exponent_bits=self.scale_exponent_bits,
            mantissa_bits=self.scale_mantissa_bits,
            exponent_bias=self.scale_exponent_bias,
            eps=self.eps,
        )

    def _block_scale(self, blocks: torch.Tensor) -> torch.Tensor:
        raw_scale = raw_e2m1_block_scale(blocks, self.eps)
        return _exmy_scale_from_raw(
            raw_scale,
            exponent_bits=self.scale_exponent_bits,
            mantissa_bits=self.scale_mantissa_bits,
            exponent_bias=self.scale_exponent_bias,
            eps=self.eps,
        )


@dataclass(frozen=True)
class MXFP4E5M3ScaleSearchFormat:
    """MXFP4 E5M3 fake quantization with online per-block scale-code search."""

    name: str = "mxfp4_e5m3_search"
    block_size: int = 32
    scale_code_offsets: tuple[int, ...] = (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8)
    eps: float = 1e-12
    scale_exponent_bits: int = 5
    scale_mantissa_bits: int = 3
    scale_exponent_bias: int = 15

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        return _quantize_mxfp4_float_scale_search(
            tensor,
            block_size=self.block_size,
            scale_code_offsets=self.scale_code_offsets,
            exponent_bits=self.scale_exponent_bits,
            mantissa_bits=self.scale_mantissa_bits,
            exponent_bias=self.scale_exponent_bias,
            eps=self.eps,
        )

    def _block_scale(self, blocks: torch.Tensor) -> torch.Tensor:
        raw_scale = raw_e2m1_block_scale(blocks, self.eps)
        return _exmy_scale_from_raw(
            raw_scale,
            exponent_bits=self.scale_exponent_bits,
            mantissa_bits=self.scale_mantissa_bits,
            exponent_bias=self.scale_exponent_bias,
            eps=self.eps,
        )


@dataclass(frozen=True)
class MXFP4PlusFormat:
    """MXFP4+ fake quantization with macro E0M8 and block E8M0 scales.

    Quantization runs along the last hidden dimension. Each 128-value macro
    block contains eight 16-value E2M1 payload blocks. The shared E0M8 macro
    scale carries the mantissa portion, while each payload block keeps an E8M0
    power-of-two scale.
    """

    name: str = "mxfp4_plus"
    block_size: int = 16
    macro_block_size: int = 128
    eps: float = 1e-12
    min_scale_exponent: int = -127
    max_scale_exponent: int = 127

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        original_dtype = tensor.dtype
        macro_blocks, metadata = to_blocks(tensor, self.macro_block_size)
        grouped = macro_blocks.float().reshape(
            -1,
            self.macro_block_size // self.block_size,
            self.block_size,
        )
        scale = self._block_scale(grouped)
        quantized = quantize_e2m1(grouped / scale)
        dequantized = (quantized * scale).reshape(-1, self.macro_block_size).to(original_dtype)
        return from_blocks(dequantized, metadata)

    def _block_scale(self, grouped: torch.Tensor) -> torch.Tensor:
        raw_scale = raw_e2m1_block_scale(grouped, self.eps)
        mbs_factor = _e0m8_macro_scale_from_raw(raw_scale)
        mbs_scaled_raw_scale = torch.clamp(raw_scale * mbs_factor, min=self.eps)
        block_scale = _e8m0_scale_from_raw(
            mbs_scaled_raw_scale,
            min_exponent=self.min_scale_exponent,
            max_exponent=self.max_scale_exponent,
        )
        return torch.clamp(block_scale / mbs_factor, min=self.eps)


@dataclass(frozen=True)
class MXFP4PlusScaleSearchFormat:
    """MXFP4+ fake quantization with online E0M8 macro-scale search."""

    name: str = "mxfp4_plus_search"
    block_size: int = 16
    macro_block_size: int = 128
    macro_scale_code_offsets: tuple[int, ...] = (-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8)
    eps: float = 1e-12
    min_scale_exponent: int = -127
    max_scale_exponent: int = 127

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        original_dtype = tensor.dtype
        macro_blocks, metadata = to_blocks(tensor, self.macro_block_size)
        grouped = macro_blocks.float().reshape(
            -1,
            self.macro_block_size // self.block_size,
            self.block_size,
        )
        raw_scale = raw_e2m1_block_scale(grouped, self.eps)
        base_macro_scale_code = _e0m8_macro_scale_code_from_raw(raw_scale)
        best_quantized: torch.Tensor | None = None
        best_scores: torch.Tensor | None = None

        for macro_scale_code_offset in self.macro_scale_code_offsets:
            macro_scale_code = torch.clamp(
                base_macro_scale_code + macro_scale_code_offset,
                min=0,
                max=E0M8_MAX_CODE,
            )
            mbs_factor = _e0m8_macro_scale_from_code(macro_scale_code, dtype=raw_scale.dtype)
            mbs_scaled_raw_scale = torch.clamp(raw_scale * mbs_factor, min=self.eps)
            block_scale = _e8m0_scale_from_raw(
                mbs_scaled_raw_scale,
                min_exponent=self.min_scale_exponent,
                max_exponent=self.max_scale_exponent,
            )
            scale = torch.clamp(block_scale / mbs_factor, min=self.eps)
            quantized = quantize_e2m1(grouped / scale) * scale
            scores = torch.sum((grouped - quantized).square(), dim=(-1, -2))
            if best_scores is None or best_quantized is None:
                best_scores = scores
                best_quantized = quantized
                continue

            improved = scores < best_scores
            best_scores = torch.where(improved, scores, best_scores)
            best_quantized = torch.where(
                improved.reshape(-1, 1, 1),
                quantized,
                best_quantized,
            )

        if best_quantized is None:
            raise RuntimeError("No MXFP4+ macro-scale candidates were evaluated")
        dequantized = best_quantized.reshape(-1, self.macro_block_size).to(original_dtype)
        return from_blocks(dequantized, metadata)


@dataclass(frozen=True)
class MXFP4ScaleSearchFormat:
    """MXFP4 fake quantization with online power-of-two block-scale search.

    Blocks are always formed along the last dimension, which corresponds to the
    hidden dimension for linear-layer activations.
    """

    name: str = "mxfp4_search"
    block_size: int = 32
    exponent_offsets: tuple[int, ...] = (-1, 0, 1)
    eps: float = 1e-12
    min_scale_exponent: int = -127
    max_scale_exponent: int = 127

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        blocks, metadata = to_blocks(tensor, self.block_size)
        raw_scale = raw_e2m1_block_scale(blocks, self.eps)
        base_exponent = _e8m0_exponent_from_raw(
            raw_scale,
            min_exponent=self.min_scale_exponent,
            max_exponent=self.max_scale_exponent,
        )
        best_quantized: torch.Tensor | None = None
        best_scores: torch.Tensor | None = None

        for exponent_offset in self.exponent_offsets:
            exponent = torch.clamp(
                base_exponent + exponent_offset,
                min=self.min_scale_exponent,
                max=self.max_scale_exponent,
            )
            scale = torch.pow(2.0, exponent)
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
            raise RuntimeError("No MXFP4 activation scale candidates were evaluated")
        return from_blocks(best_quantized, metadata)

    def _block_scale(self, blocks: torch.Tensor) -> torch.Tensor:
        raw_scale = raw_e2m1_block_scale(blocks, self.eps)
        return _e8m0_scale_from_raw(
            raw_scale,
            min_exponent=self.min_scale_exponent,
            max_exponent=self.max_scale_exponent,
        )


def _exmy_scale_from_raw(
    raw_scale: torch.Tensor,
    *,
    exponent_bits: int,
    mantissa_bits: int,
    exponent_bias: int,
    eps: float,
) -> torch.Tensor:
    block_scale = quantize_unsigned_fp(
        raw_scale.float(),
        exponent_bits=exponent_bits,
        mantissa_bits=mantissa_bits,
        exponent_bias=exponent_bias,
    )
    return torch.clamp(block_scale, min=eps).to(raw_scale.dtype)


def _quantize_mxfp4_float_scale_search(
    tensor: torch.Tensor,
    *,
    block_size: int,
    scale_code_offsets: tuple[int, ...],
    exponent_bits: int,
    mantissa_bits: int,
    exponent_bias: int,
    eps: float,
) -> torch.Tensor:
    blocks, metadata = to_blocks(tensor, block_size)
    raw_scale = raw_e2m1_block_scale(blocks, eps)
    base_block_scale = quantize_unsigned_fp(
        raw_scale.float(),
        exponent_bits=exponent_bits,
        mantissa_bits=mantissa_bits,
        exponent_bias=exponent_bias,
    )
    base_scale_code = unsigned_fp_code_from_value(
        base_block_scale,
        exponent_bits=exponent_bits,
        mantissa_bits=mantissa_bits,
        exponent_bias=exponent_bias,
    )
    max_code = (1 << (exponent_bits + mantissa_bits)) - 1
    best_quantized: torch.Tensor | None = None
    best_scores: torch.Tensor | None = None

    for scale_code_offset in scale_code_offsets:
        scale_code = torch.clamp(base_scale_code + scale_code_offset, min=0, max=max_code)
        scale = torch.clamp(
            unsigned_fp_value_from_code(
                scale_code,
                exponent_bits=exponent_bits,
                mantissa_bits=mantissa_bits,
                exponent_bias=exponent_bias,
                dtype=raw_scale.dtype,
            ),
            min=eps,
        )
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
        raise RuntimeError("No MXFP4 float-scale candidates were evaluated")
    return from_blocks(best_quantized, metadata)


def _e8m0_scale_from_raw(
    raw_scale: torch.Tensor,
    *,
    min_exponent: int,
    max_exponent: int,
) -> torch.Tensor:
    exponent = _e8m0_exponent_from_raw(
        raw_scale,
        min_exponent=min_exponent,
        max_exponent=max_exponent,
    )
    return torch.pow(2.0, exponent)


def _e8m0_exponent_from_raw(
    raw_scale: torch.Tensor,
    *,
    min_exponent: int,
    max_exponent: int,
) -> torch.Tensor:
    exponent = torch.ceil(torch.log2(raw_scale))
    return torch.clamp(exponent, min=min_exponent, max=max_exponent)


def _e0m8_macro_scale_from_raw(raw_scale: torch.Tensor) -> torch.Tensor:
    code = _e0m8_macro_scale_code_from_raw(raw_scale)
    return _e0m8_macro_scale_from_code(code, dtype=raw_scale.dtype)


def _e0m8_macro_scale_code_from_raw(raw_scale: torch.Tensor) -> torch.Tensor:
    macro_raw_scale = torch.clamp(raw_scale.amax(dim=1, keepdim=True), min=1e-12)
    reciprocal_scale = torch.reciprocal(macro_raw_scale)
    exponent = torch.floor(torch.log2(reciprocal_scale))
    mantissa = reciprocal_scale / torch.pow(2.0, exponent)
    return torch.floor((mantissa - 1.0) * E0M8_LEVELS).long().clamp(min=0, max=E0M8_MAX_CODE)


def _e0m8_macro_scale_from_code(code: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    code = code.long().clamp(min=0, max=E0M8_MAX_CODE)
    return (1.0 + code.float() / E0M8_LEVELS).to(dtype)


E0M8_LEVELS = 1 << 8
E0M8_MAX_CODE = E0M8_LEVELS - 1
E0M8_MAX = 2.0 - 1.0 / E0M8_LEVELS

