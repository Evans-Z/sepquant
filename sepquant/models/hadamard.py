from __future__ import annotations

import math

import torch


def block_hadamard_last_dim(tensor: torch.Tensor, *, block_size: int = 32) -> torch.Tensor:
    """Apply normalized block-wise FWHT along the last dimension."""

    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    if tensor.shape[-1] % block_size != 0:
        raise ValueError(
            f"Last dimension {tensor.shape[-1]} must be divisible by Hadamard block size {block_size}"
        )

    original_shape = tensor.shape
    blocks = tensor.reshape(-1, block_size)
    transformed = _fwht(blocks) / math.sqrt(block_size)
    return transformed.reshape(original_shape)


def rotate_gram_block_hadamard(gram: torch.Tensor, *, block_size: int = 32) -> torch.Tensor:
    """Return R^T G R for block-diagonal normalized Hadamard R."""

    rotated = block_hadamard_last_dim(gram, block_size=block_size)
    return block_hadamard_last_dim(rotated.transpose(0, 1), block_size=block_size).transpose(0, 1)


def _fwht(blocks: torch.Tensor) -> torch.Tensor:
    output = blocks.clone()
    width = output.shape[-1]
    step = 1
    while step < width:
        output = output.reshape(-1, width // (step * 2), step * 2)
        left = output[..., :step].clone()
        right = output[..., step:]
        output[..., :step] = left + right
        output[..., step:] = left - right
        output = output.reshape(-1, width)
        step *= 2
    return output
