from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from sepquant.formats import FP4Format


class QuantLinear(nn.Module):
    """Linear layer with pre-quantized weights and optional online activation quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        weight_format: FP4Format,
        activation_format: FP4Format | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_format = weight_format
        self.activation_format = activation_format

        factory_kwargs = {"device": device, "dtype": dtype}
        self.register_buffer("weight", torch.empty(out_features, in_features, **factory_kwargs))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, **factory_kwargs))
        else:
            self.bias = None

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        module: nn.Linear,
        *,
        weight_format: FP4Format,
        activation_format: FP4Format | None = None,
    ) -> "QuantLinear":
        quant_linear = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            weight_format=weight_format,
            activation_format=activation_format,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        quant_linear.weight.copy_(weight_format.quantize(module.weight.detach()))
        if module.bias is not None:
            quant_linear.bias.copy_(module.bias.detach())
        return quant_linear

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.activation_format is not None:
            inputs = self.activation_format.quantize(inputs)
        return F.linear(inputs, self.weight, self.bias)

    def extra_repr(self) -> str:
        act_name = self.activation_format.name if self.activation_format else "none"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"weight_format={self.weight_format.name}, activation_format={act_name}"
        )

