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

