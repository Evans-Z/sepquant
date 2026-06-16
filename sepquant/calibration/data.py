from __future__ import annotations

import random

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def build_calibration_batches(
    *,
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    nsamples: int,
    seed: int,
    sequence_length: int,
    batch_size: int = 1,
) -> list[torch.Tensor]:
    """Build fixed-length calibration token batches from a text dataset."""

    if nsamples <= 0:
        raise ValueError("nsamples must be positive")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    dataset_args = [dataset_name]
    if dataset_config:
        dataset_args.append(dataset_config)
    dataset = load_dataset(*dataset_args, split=split)
    encoded = tokenizer(" ".join(dataset[text_column]), return_tensors="pt").input_ids

    if encoded.shape[1] <= sequence_length:
        raise ValueError(
            f"Calibration data has {encoded.shape[1]} tokens, "
            f"which is not enough for sequence_length={sequence_length}"
        )

    rng = random.Random(seed)
    samples: list[torch.Tensor] = []
    for _ in range(nsamples):
        begin = rng.randint(0, encoded.shape[1] - sequence_length - 1)
        end = begin + sequence_length
        samples.append(encoded[:, begin:end])

    return [
        torch.cat(samples[begin : begin + batch_size], dim=0)
        for begin in range(0, len(samples), batch_size)
    ]

