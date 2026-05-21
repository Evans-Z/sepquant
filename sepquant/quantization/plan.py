from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LayerQuantizationSpec:
    """Per-layer quantization decision.

    Fields left as `None` inherit from the global/default configuration.
    Setting `enabled=false` keeps the matched layer in floating point.
    """

    weight_format: str | None = None
    activation_format: str | None = None
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayerQuantizationSpec":
        metadata = {
            key: value
            for key, value in data.items()
            if key not in {"weight_format", "activation_format", "enabled"}
        }
        return cls(
            weight_format=data.get("weight_format"),
            activation_format=data.get("activation_format"),
            enabled=data.get("enabled", True),
            metadata=metadata,
        )


@dataclass(frozen=True)
class PatternQuantizationSpec:
    pattern: str
    spec: LayerQuantizationSpec

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatternQuantizationSpec":
        if "pattern" not in data:
            raise ValueError("Pattern quantization entries must include a 'pattern' field")
        return cls(pattern=data["pattern"], spec=LayerQuantizationSpec.from_dict(data))


@dataclass(frozen=True)
class QuantizationPlan:
    """Layer-wise quantization plan.

    Supported JSON shape:

    {
      "version": 1,
      "default": {"weight_format": "mxfp4", "activation_format": "none"},
      "layers": {
        "model.layers.0.self_attn.q_proj": {"weight_format": "nvfp4"}
      },
      "patterns": [
        {"pattern": "*.fc1", "weight_format": "nvfp4"}
      ]
    }
    """

    default: LayerQuantizationSpec | None = None
    layers: dict[str, LayerQuantizationSpec] = field(default_factory=dict)
    patterns: tuple[PatternQuantizationSpec, ...] = ()
    version: int = 1

    @classmethod
    def from_file(cls, path: str | Path) -> "QuantizationPlan":
        plan_path = Path(path)
        with plan_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Quantization plan must be a JSON object: {plan_path}")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantizationPlan":
        default_data = data.get("default")
        default = (
            LayerQuantizationSpec.from_dict(default_data)
            if isinstance(default_data, dict)
            else None
        )

        layers_data = data.get("layers", {})
        if not isinstance(layers_data, dict):
            raise ValueError("Quantization plan 'layers' field must be an object")
        layers = {
            layer_name: LayerQuantizationSpec.from_dict(spec)
            for layer_name, spec in layers_data.items()
        }

        patterns_data = data.get("patterns", [])
        if not isinstance(patterns_data, list):
            raise ValueError("Quantization plan 'patterns' field must be a list")
        patterns = tuple(PatternQuantizationSpec.from_dict(item) for item in patterns_data)

        return cls(
            default=default,
            layers=layers,
            patterns=patterns,
            version=data.get("version", 1),
        )

    def get(self, layer_name: str) -> LayerQuantizationSpec | None:
        if layer_name in self.layers:
            return self.layers[layer_name]

        for pattern_spec in self.patterns:
            if fnmatch.fnmatch(layer_name, pattern_spec.pattern):
                return pattern_spec.spec

        return self.default

