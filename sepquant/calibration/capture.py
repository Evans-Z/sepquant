from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from sepquant.models import TargetLinear


CaptureMode = Literal["inputs", "gram", "both"]


@dataclass
class CalibrationCapture:
    inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    grams: dict[str, torch.Tensor] = field(default_factory=dict)
    token_counts: dict[str, int] = field(default_factory=dict)


def collect_linear_inputs(
    *,
    model: nn.Module,
    targets: list[TargetLinear],
    batches: list[torch.Tensor],
    max_tokens_per_layer: int,
) -> dict[str, torch.Tensor]:
    """Collect flattened input activations for target linear layers."""

    capture = collect_linear_calibration(
        model=model,
        targets=targets,
        batches=batches,
        max_tokens_per_layer=max_tokens_per_layer,
        capture_mode="inputs",
    )
    return capture.inputs


def collect_linear_calibration(
    *,
    model: nn.Module,
    targets: list[TargetLinear],
    batches: list[torch.Tensor],
    max_tokens_per_layer: int,
    capture_mode: CaptureMode = "gram",
    input_layer_patterns: list[str] | None = None,
) -> CalibrationCapture:
    """Collect calibration statistics for target linear layers.

    `gram` stores X^T X for each layer. `inputs` stores raw flattened X only for
    layers matching `input_layer_patterns`; if no patterns are provided, all
    target layers are stored.
    """

    if max_tokens_per_layer <= 0:
        raise ValueError("max_tokens_per_layer must be positive")
    if capture_mode not in {"inputs", "gram", "both"}:
        raise ValueError(f"Unsupported capture_mode: {capture_mode}")

    captured_inputs: dict[str, list[torch.Tensor]] = {target.name: [] for target in targets}
    grams: dict[str, torch.Tensor] = {}
    input_counts: dict[str, int] = {target.name: 0 for target in targets}
    gram_counts: dict[str, int] = {target.name: 0 for target in targets}
    hooks = []
    qwen3_moe_expert_targets: dict[int, tuple[nn.Module, list[TargetLinear]]] = {}

    for target in targets:
        if target.source == "qwen3_moe_expert":
            experts = getattr(target.module, "experts", None)
            if isinstance(experts, nn.Module):
                key = id(experts)
                if key not in qwen3_moe_expert_targets:
                    qwen3_moe_expert_targets[key] = (experts, [])
                qwen3_moe_expert_targets[key][1].append(target)
            continue

        hooks.append(
            target.module.register_forward_hook(
                _make_capture_hook(
                    name=target.name,
                    captured_inputs=captured_inputs,
                    grams=grams,
                    input_counts=input_counts,
                    gram_counts=gram_counts,
                    max_tokens_per_layer=max_tokens_per_layer,
                    capture_mode=capture_mode,
                    store_inputs=_matches_any(target.name, input_layer_patterns),
                )
            )
        )
    for experts, expert_targets in qwen3_moe_expert_targets.values():
        hooks.append(
            experts.register_forward_hook(
                _make_qwen3_moe_expert_capture_hook(
                    targets=expert_targets,
                    captured_inputs=captured_inputs,
                    grams=grams,
                    input_counts=input_counts,
                    gram_counts=gram_counts,
                    max_tokens_per_layer=max_tokens_per_layer,
                    capture_mode=capture_mode,
                    input_layer_patterns=input_layer_patterns,
                )
            )
        )

    try:
        with torch.inference_mode():
            for batch in tqdm(batches, desc="Collecting calibration activations", unit="batch"):
                model(batch.to(model.device))
    finally:
        for hook in hooks:
            hook.remove()

    inputs = {
        name: torch.cat(chunks, dim=0)
        for name, chunks in captured_inputs.items()
        if chunks
    }
    return CalibrationCapture(inputs=inputs, grams=grams, token_counts=gram_counts)


def _make_capture_hook(
    *,
    name: str,
    captured_inputs: dict[str, list[torch.Tensor]],
    grams: dict[str, torch.Tensor],
    input_counts: dict[str, int],
    gram_counts: dict[str, int],
    max_tokens_per_layer: int,
    capture_mode: CaptureMode,
    store_inputs: bool,
):
    def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        activation = inputs[0].detach()
        flattened = activation.reshape(-1, activation.shape[-1]).float().cpu()
        _record_flattened_activation(
            name=name,
            flattened=flattened,
            captured_inputs=captured_inputs,
            grams=grams,
            input_counts=input_counts,
            gram_counts=gram_counts,
            max_tokens_per_layer=max_tokens_per_layer,
            capture_mode=capture_mode,
            store_inputs=store_inputs,
        )

    return hook


def _make_qwen3_moe_expert_capture_hook(
    *,
    targets: list[TargetLinear],
    captured_inputs: dict[str, list[torch.Tensor]],
    grams: dict[str, torch.Tensor],
    input_counts: dict[str, int],
    gram_counts: dict[str, int],
    max_tokens_per_layer: int,
    capture_mode: CaptureMode,
    input_layer_patterns: list[str] | None,
):
    targets_by_expert: dict[int, dict[str, TargetLinear]] = {}
    for target in targets:
        if target.expert_idx is None or target.projection is None:
            continue
        targets_by_expert.setdefault(target.expert_idx, {})[target.projection] = target

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        if len(inputs) < 2:
            return
        hidden_states = inputs[0].detach()
        top_k_index = inputs[1].detach()
        if hidden_states.ndim != 2 or top_k_index.ndim != 2:
            return

        gate_up_proj = getattr(module, "gate_up_proj", None)
        if gate_up_proj is None:
            return

        with torch.no_grad():
            for expert_idx, projection_targets in targets_by_expert.items():
                token_mask = (top_k_index == expert_idx).any(dim=-1)
                if not token_mask.any():
                    continue

                expert_inputs = hidden_states[token_mask]
                target = projection_targets.get("gate_up_proj")
                if target is not None:
                    _record_flattened_activation(
                        name=target.name,
                        flattened=expert_inputs.reshape(-1, expert_inputs.shape[-1]).float().cpu(),
                        captured_inputs=captured_inputs,
                        grams=grams,
                        input_counts=input_counts,
                        gram_counts=gram_counts,
                        max_tokens_per_layer=max_tokens_per_layer,
                        capture_mode=capture_mode,
                        store_inputs=_matches_any(target.name, input_layer_patterns),
                    )

                target = projection_targets.get("down_proj")
                if target is None:
                    continue
                gate, up = F.linear(expert_inputs, gate_up_proj[expert_idx]).chunk(2, dim=-1)
                down_inputs = module.act_fn(gate) * up
                _record_flattened_activation(
                    name=target.name,
                    flattened=down_inputs.reshape(-1, down_inputs.shape[-1]).float().cpu(),
                    captured_inputs=captured_inputs,
                    grams=grams,
                    input_counts=input_counts,
                    gram_counts=gram_counts,
                    max_tokens_per_layer=max_tokens_per_layer,
                    capture_mode=capture_mode,
                    store_inputs=_matches_any(target.name, input_layer_patterns),
                )

    return hook


def _record_flattened_activation(
    *,
    name: str,
    flattened: torch.Tensor,
    captured_inputs: dict[str, list[torch.Tensor]],
    grams: dict[str, torch.Tensor],
    input_counts: dict[str, int],
    gram_counts: dict[str, int],
    max_tokens_per_layer: int,
    capture_mode: CaptureMode,
    store_inputs: bool,
) -> None:
    if flattened.numel() == 0:
        return

    if capture_mode in {"gram", "both"}:
        gram = flattened.t().matmul(flattened)
        if name in grams:
            grams[name] += gram
        else:
            grams[name] = gram
        gram_counts[name] += flattened.shape[0]

    if capture_mode in {"inputs", "both"} and store_inputs:
        if input_counts[name] >= max_tokens_per_layer:
            return
        remaining = max_tokens_per_layer - input_counts[name]
        if flattened.shape[0] > remaining:
            flattened = flattened[:remaining]
        captured_inputs[name].append(flattened)
        input_counts[name] += flattened.shape[0]


def _matches_any(layer_name: str, patterns: list[str] | None) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(layer_name, pattern) for pattern in patterns)

