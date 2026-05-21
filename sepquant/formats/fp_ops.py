from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BlockMetadata:
    rows: int
    width: int
    padded_width: int
    original_shape: torch.Size


def to_blocks(tensor: torch.Tensor, block_size: int) -> tuple[torch.Tensor, BlockMetadata]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    original_shape = tensor.shape
    flat = tensor.reshape(-1, tensor.shape[-1])
    rows, width = flat.shape
    remainder = width % block_size
    pad = 0 if remainder == 0 else block_size - remainder
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    metadata = BlockMetadata(
        rows=rows,
        width=width,
        padded_width=width + pad,
        original_shape=original_shape,
    )
    return flat.reshape(-1, block_size), metadata


def from_blocks(blocks: torch.Tensor, metadata: BlockMetadata) -> torch.Tensor:
    flat = blocks.reshape(metadata.rows, metadata.padded_width)
    flat = flat[:, : metadata.width]
    return flat.reshape(metadata.original_shape)


def raw_e2m1_block_scale(blocks: torch.Tensor, eps: float) -> torch.Tensor:
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    return torch.clamp(amax / E2M1_MAX, min=eps)


def quantize_e2m1(tensor: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(tensor)
    abs_tensor = tensor.abs().clamp(max=E2M1_MAX)

    subnormal = torch.round(abs_tensor / E2M1_SUBNORMAL_STEP) * E2M1_SUBNORMAL_STEP

    safe_abs = torch.clamp(abs_tensor, min=E2M1_MIN_NORMAL)
    exponent = torch.floor(torch.log2(safe_abs))
    exponent = torch.clamp(
        exponent,
        min=1 - E2M1_EXPONENT_BIAS,
        max=((1 << E2M1_EXPONENT_BITS) - 1) - E2M1_EXPONENT_BIAS,
    )
    mantissa_step = 2.0 ** (exponent - E2M1_MANTISSA_BITS)
    normal = torch.round(safe_abs / mantissa_step) * mantissa_step
    normal = normal.clamp(max=E2M1_MAX)

    quantized_abs = torch.where(abs_tensor < E2M1_MIN_NORMAL, subnormal, normal)
    return sign * quantized_abs


def quantize_unsigned_fp(
    tensor: torch.Tensor,
    *,
    exponent_bits: int,
    mantissa_bits: int,
    exponent_bias: int,
) -> torch.Tensor:
    max_value = max_normal(
        exponent_bits=exponent_bits,
        mantissa_bits=mantissa_bits,
        exponent_bias=exponent_bias,
    )
    min_normal = 2.0 ** (1 - exponent_bias)
    subnormal_step = 2.0 ** (1 - exponent_bias - mantissa_bits)

    abs_tensor = tensor.abs().clamp(max=max_value)
    subnormal = torch.round(abs_tensor / subnormal_step) * subnormal_step

    safe_abs = torch.clamp(abs_tensor, min=min_normal)
    exponent = torch.floor(torch.log2(safe_abs))
    exponent = torch.clamp(
        exponent,
        min=1 - exponent_bias,
        max=((1 << exponent_bits) - 1) - exponent_bias,
    )
    mantissa_step = 2.0 ** (exponent - mantissa_bits)
    normal = torch.round(safe_abs / mantissa_step) * mantissa_step
    normal = normal.clamp(max=max_value)

    return torch.where(abs_tensor < min_normal, subnormal, normal)


def max_normal(*, exponent_bits: int, mantissa_bits: int, exponent_bias: int) -> float:
    max_exp_field = (1 << exponent_bits) - 1
    max_exp = max_exp_field - exponent_bias
    max_mant = 2.0 - 2.0 ** (-mantissa_bits)
    return max_mant * (2.0**max_exp)


E2M1_EXPONENT_BITS = 2
E2M1_MANTISSA_BITS = 1
E2M1_EXPONENT_BIAS = 1
E2M1_MAX = max_normal(
    exponent_bits=E2M1_EXPONENT_BITS,
    mantissa_bits=E2M1_MANTISSA_BITS,
    exponent_bias=E2M1_EXPONENT_BIAS,
)
E2M1_MIN_NORMAL = 2.0 ** (1 - E2M1_EXPONENT_BIAS)
E2M1_SUBNORMAL_STEP = 2.0 ** (
    1 - E2M1_EXPONENT_BIAS - E2M1_MANTISSA_BITS
)

