"""Model patching utilities."""

from sepquant.models.hadamard import block_hadamard_last_dim, rotate_gram_block_hadamard
from sepquant.models.load import LoadedCausalLM, load_quantized_causal_lm
from sepquant.models.patch import PatchReport, TargetLinear, get_target_linears, patch_causal_lm_linears
from sepquant.models.quant_linear import QuantLinear

__all__ = [
    "LoadedCausalLM",
    "PatchReport",
    "QuantLinear",
    "TargetLinear",
    "block_hadamard_last_dim",
    "get_target_linears",
    "load_quantized_causal_lm",
    "patch_causal_lm_linears",
    "rotate_gram_block_hadamard",
]

