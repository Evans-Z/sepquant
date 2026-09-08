from __future__ import annotations

from typing import Any

import torch

try:
    from datasets import load_dataset
except ModuleNotFoundError:
    def load_dataset(*args, **kwargs):
        from datasets import load_dataset as _load_dataset

        return _load_dataset(*args, **kwargs)


def build_multimodal_calibration_batches(
    *,
    processor: Any,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    image_column: str,
    text_column: str | None,
    prompt: str,
    nsamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build deterministic single-image Qwen3-VL calibration batches."""

    if nsamples <= 0:
        raise ValueError("nsamples must be positive")
    dataset_args = [dataset_name]
    if dataset_config:
        dataset_args.append(dataset_config)
    dataset = load_dataset(*dataset_args, split=split)
    if image_column not in dataset.column_names:
        raise ValueError(f"Image column {image_column!r} is not present in {dataset_name}")
    if text_column is not None and text_column not in dataset.column_names:
        raise ValueError(f"Text column {text_column!r} is not present in {dataset_name}")

    sample_count = min(nsamples, len(dataset))
    selected = dataset.shuffle(seed=seed).select(range(sample_count))
    return [
        build_qwen3_vl_calibration_batch(
            processor=processor,
            image=sample[image_column],
            prompt=_sample_prompt(sample, text_column=text_column, fallback=prompt),
        )
        for sample in selected
    ]


def build_qwen3_vl_calibration_batch(
    *, processor: Any, image: Any, prompt: str
) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    batch = dict(encoded)
    # Some processor versions emit this unused key while Qwen3-VL rejects it.
    batch.pop("token_type_ids", None)
    if not any(isinstance(value, torch.Tensor) for value in batch.values()):
        raise ValueError("Processor did not return tensor model inputs")
    return batch


def _sample_prompt(sample: dict[str, Any], *, text_column: str | None, fallback: str) -> str:
    if text_column is not None:
        value = sample[text_column]
        if isinstance(value, str) and value.strip():
            return value
    return fallback
