from __future__ import annotations

from dataclasses import dataclass

import torch

from sepquant.formats.fp_ops import from_blocks, to_blocks


@dataclass(frozen=True)
class HIF4Format:
    """HiF4 fake quantization with three-level hierarchical block scales.

    Quantization always runs along the last hidden dimension. Each 64-value
    block is viewed as `(8, 2, 4)`.
    """

    name: str = "hif4"
    block_size: int = 64

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        original_dtype = tensor.dtype
        blocks, metadata = to_blocks(tensor, self.block_size)
        grouped = blocks.float().reshape(-1, 8, 2, 4)

        a_3 = torch.max(torch.abs(grouped), dim=-1, keepdim=True)[0]
        a_2 = torch.max(a_3, dim=-2, keepdim=True)[0]
        a_1 = torch.max(a_2, dim=-3, keepdim=True)[0]

        e_1, m_1 = _level1_e6m2_components(a_1)
        dequantized = _quantize_hif4_grouped(
            grouped=grouped,
            a_2=a_2,
            a_3=a_3,
            e_1=e_1,
            m_1=m_1,
        )
        dequantized = dequantized.reshape(-1, self.block_size).to(original_dtype)
        return from_blocks(dequantized, metadata)


@dataclass(frozen=True)
class HIF4ScaleSearchFormat:
    """HiF4 fake quantization with online level-1 E6M2 scale search."""

    name: str = "hif4_search"
    block_size: int = 64
    level1_code_offsets: tuple[int, ...] = (-2, -1, 0, 1, 2)

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        original_dtype = tensor.dtype
        blocks, metadata = to_blocks(tensor, self.block_size)
        grouped = blocks.float().reshape(-1, 8, 2, 4)

        a_3 = torch.max(torch.abs(grouped), dim=-1, keepdim=True)[0]
        a_2 = torch.max(a_3, dim=-2, keepdim=True)[0]
        a_1 = torch.max(a_2, dim=-3, keepdim=True)[0]
        base_code = _level1_code_from_a1(a_1)

        best_quantized: torch.Tensor | None = None
        best_scores: torch.Tensor | None = None
        for level1_code_offset in self.level1_code_offsets:
            code = torch.clamp(base_code + level1_code_offset, min=0, max=E6M2_MAX_CODE)
            e_1, m_1 = _level1_components_from_code(code)
            quantized = _quantize_hif4_grouped(
                grouped=grouped,
                a_2=a_2,
                a_3=a_3,
                e_1=e_1,
                m_1=m_1,
            )
            scores = torch.sum((grouped - quantized).square(), dim=(-1, -2, -3))
            if best_scores is None or best_quantized is None:
                best_scores = scores
                best_quantized = quantized
                continue

            improved = scores < best_scores
            best_scores = torch.where(improved, scores, best_scores)
            best_quantized = torch.where(
                improved.reshape(-1, 1, 1, 1),
                quantized,
                best_quantized,
            )

        if best_quantized is None:
            raise RuntimeError("No HiF4 level-1 scale candidates were evaluated")
        dequantized = best_quantized.reshape(-1, self.block_size).to(original_dtype)
        return from_blocks(dequantized, metadata)


def _quantize_hif4_grouped(
    *,
    grouped: torch.Tensor,
    a_2: torch.Tensor,
    a_3: torch.Tensor,
    e_1: torch.Tensor,
    m_1: torch.Tensor,
) -> torch.Tensor:
    s_1 = m_1 * 2 ** (e_1 - 2)

    # level 2 scale: 1 bit
    a_2_tilde = (a_2 / s_1).clip(0, 4)
    e_2 = torch.floor(a_2_tilde / 4.0)
    s_2 = 2 ** e_2

    # level 3 scale: 1 bit
    a_3_tilde = (a_3 / (s_1 * s_2)).clip(0, 2)
    e_3 = torch.floor(a_3_tilde / 2.0)
    s_3 = 2 ** e_3

    # element E0M3
    x_tilde = (torch.abs(grouped) / (s_1 * s_2 * s_3)).clip(0, 1.75)
    x_hat = torch.floor(x_tilde * 4 + 0.5)
    return torch.sign(grouped) * m_1 * 2 ** (e_1 + e_2 + e_3 - 4) * x_hat


def _level1_e6m2_components(a_1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a_1_tilde = (a_1 / 7.0).clip(2 ** (-48), 49152)
    e_1 = torch.floor(torch.log2(a_1_tilde))
    m_1 = torch.round(a_1_tilde / 2 ** (e_1 - 2))
    return e_1, m_1


def _level1_code_from_a1(a_1: torch.Tensor) -> torch.Tensor:
    e_1, m_1 = _level1_e6m2_components(a_1)
    return _level1_code_from_components(e_1=e_1, m_1=m_1)


def _level1_code_from_components(*, e_1: torch.Tensor, m_1: torch.Tensor) -> torch.Tensor:
    exponent_field = e_1 + E6M2_EXPONENT_BIAS
    mantissa_field = m_1 - (1 << E6M2_MANTISSA_BITS)
    code = exponent_field * (1 << E6M2_MANTISSA_BITS) + mantissa_field
    return code.round().long().clamp(min=0, max=E6M2_MAX_CODE)


def _level1_components_from_code(code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    code = code.long().clamp(min=0, max=E6M2_MAX_CODE)
    exponent_field = torch.div(code, 1 << E6M2_MANTISSA_BITS, rounding_mode="floor")
    mantissa_field = code.remainder(1 << E6M2_MANTISSA_BITS)
    e_1 = exponent_field.float() - E6M2_EXPONENT_BIAS
    m_1 = mantissa_field.float() + (1 << E6M2_MANTISSA_BITS)
    return e_1, m_1


E6M2_EXPONENT_BITS = 6
E6M2_MANTISSA_BITS = 2
E6M2_EXPONENT_BIAS = 31
E6M2_MAX_CODE = (1 << (E6M2_EXPONENT_BITS + E6M2_MANTISSA_BITS)) - 1
