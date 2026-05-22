# SepQuant

SepQuant is a research framework for **DAFQ: Decoupled Accuracy-Efficiency Floating-Point Quantization**.

The core idea is to separate LLM weight-activation quantization by deployment constraint:

- **Weights** are static and can be quantized offline with accuracy-first methods.
- **Activations** are dynamic and should be quantized online with efficiency-first methods.

Current implementation focuses on fake quantization for:

- MXFP4
- NVFP4
- Qwen/Qwen3-style decoder models
- OPT-style decoder models
- Wikitext2 perplexity evaluation
- Downstream task evaluation through `lm-eval-harness`

For the research motivation and algorithm sketch, see `DAFQ.md`.

## Installation

Create and activate your environment, then install the package in editable mode:

```bash
pip install -e .
```

For downstream task evaluation with `lm-eval-harness`, install the optional eval dependencies:

```bash
pip install -e ".[eval]"
```

For development tools:

```bash
pip install -e ".[dev]"
```

## Project Layout

```text
sepquant/
  formats/
    mxfp.py          # MXFP4 fake quantization
    nvfp.py          # NVFP4 fake quantization
    fp_ops.py        # shared FP/block quantization ops
    benchmark.py     # format correctness/error/timing benchmark
  models/
    patch.py         # generic model-family patching
    quant_linear.py  # QuantLinear module
    load.py          # shared quantized HF model loader
  evaluation/
    ppl.py           # Wikitext-style perplexity evaluation
    tasks.py         # lm-eval-harness downstream evaluation

configs/eval/
  ppl_qwen3_wikitext2.json
  ppl_opt125m_wikitext2.json
  tasks_qwen3.json
  tasks_opt125m.json

scripts/
  run_eval_ppl.sh
  run_eval_tasks.sh
  quantize_generate.py
```

## Supported Formats

### MXFP4

`MXFP4Format` uses:

- E2M1 FP4 payload
- block size 32
- power-of-two block scale

### NVFP4

`NVFP4Format` uses:

- E2M1 FP4 payload
- block size 16
- two-level scaling:

```text
final_scale = FP32 tensor scale * E4M3 block scale
```

Both formats currently perform fake quantization and return dequantized PyTorch tensors for algorithm research.

## Format Benchmark

Run the format benchmark:

```bash
sepquant-format-benchmark
```

Compare MXFP4 and NVFP4 on an outlier-heavy tensor:

```bash
sepquant-format-benchmark \
  --formats mxfp4 nvfp4 \
  --shape 16 4096 \
  --distribution outlier \
  --device auto \
  --dtype float32 \
  --iters 50
```

Supported devices:

```bash
--device cpu
--device cuda
--device cuda:0
--device mps
--device apple
--device auto
```

The benchmark reports shape/finite/zero correctness checks, reconstruction error, SQNR, and average runtime.

## Perplexity Evaluation

PPL evaluation is config-driven.

Run Qwen3 Wikitext2 PPL:

```bash
scripts/run_eval_ppl.sh configs/eval/ppl_qwen3_wikitext2.json
```

Run OPT-125M Wikitext2 PPL:

```bash
scripts/run_eval_ppl.sh configs/eval/ppl_opt125m_wikitext2.json
```

Equivalent direct command:

```bash
sepquant-eval-ppl --config configs/eval/ppl_opt125m_wikitext2.json
```

You can override config fields from the command line:

```bash
scripts/run_eval_ppl.sh configs/eval/ppl_opt125m_wikitext2.json \
  --weight-format nvfp4 \
  --activation-format none \
  --max-samples 64
```

By default, `max_samples` is `null`, so the full test split is used. Set `max_samples` only for quick debugging.

### PPL Config Example

```json
{
  "model": "facebook/opt-125m",
  "model_type": "opt",
  "dataset": "wikitext",
  "dataset_config": "wikitext-2-raw-v1",
  "split": "test",
  "text_column": "text",
  "weight_format": "mxfp4",
  "activation_format": "none",
  "include_lm_head": false,
  "device": "auto",
  "dtype": "auto",
  "max_samples": null,
  "sequence_length": 1024
}
```

## Downstream Task Evaluation

Downstream evaluation uses the Python API of `lm-eval-harness`.

Install eval dependencies first:

```bash
pip install -e ".[eval]"
```

Run Qwen3 tasks:

```bash
scripts/run_eval_tasks.sh configs/eval/tasks_qwen3.json
```

Run OPT-125M tasks:

```bash
scripts/run_eval_tasks.sh configs/eval/tasks_opt125m.json
```

Equivalent direct command:

```bash
sepquant-eval-tasks --config configs/eval/tasks_opt125m.json
```

Quick smoke test with a small limit:

```bash
scripts/run_eval_tasks.sh configs/eval/tasks_opt125m.json \
  --tasks hellaswag \
  --limit 10
```

Example task names:

```text
hellaswag
arc_challenge
mmlu
gsm8k
```

Task names are resolved by `lm-eval-harness`, so exact availability can depend on the installed `lm-eval` version.

### Task Config Example

```json
{
  "model": "facebook/opt-125m",
  "model_type": "opt",
  "tasks": ["hellaswag", "arc_challenge"],
  "num_fewshot": 0,
  "limit": 10,
  "batch_size": "1",
  "max_length": null,
  "weight_format": "mxfp4",
  "activation_format": "none",
  "include_lm_head": false,
  "device": "auto",
  "dtype": "auto",
  "output_path": "outputs/eval/tasks_opt125m_mxfp4.json",
  "log_samples": false
}
```

## Generation Smoke Test

Use `scripts/quantize_generate.py` to quickly check that a quantized model can generate text.

OPT example:

```bash
scripts/quantize_generate.py \
  --model facebook/opt-125m \
  --model-type opt \
  --weight-format mxfp4 \
  --activation-format none
```

Qwen3 example:

```bash
scripts/quantize_generate.py \
  --model /Users/phoenix/Works/models/Qwen3-0.6B \
  --model-type qwen3 \
  --weight-format mxfp4
```

## Model Patching

SepQuant replaces supported `nn.Linear` modules with `QuantLinear`.

Use the generic patching API:

```python
from sepquant.formats import get_fp4_format
from sepquant.models import patch_causal_lm_linears

report = patch_causal_lm_linears(
    model,
    weight_format=get_fp4_format("mxfp4"),
    activation_format=None,
    model_type="opt",
    include_lm_head=False,
)
```

Supported `model_type` values:

```text
auto
qwen
qwen2
qwen3
llama
mistral
gemma
opt
generic
```

For `auto`, SepQuant reads `model.config.model_type` and falls back to `generic` if the model family is unknown.

## Quantization Plans

For research algorithms, SepQuant supports an optional JSON quantization plan. A plan can override the global format per exact layer name or wildcard pattern.

Example:

```json
{
  "version": 1,
  "default": {
    "weight_format": "mxfp4",
    "activation_format": "none"
  },
  "layers": {
    "model.decoder.layers.0.fc2": {
      "weight_format": "nvfp4"
    }
  },
  "patterns": [
    {
      "pattern": "*.fc1",
      "weight_format": "nvfp4"
    }
  ]
}
```

Use a plan during PPL evaluation:

```bash
sepquant-eval-ppl --config configs/eval/ppl_opt125m_wikitext2.json \
  --weight-format none \
  --quantization-plan configs/quantization/example_mixed_fp4_plan.json
```

Use a plan during downstream task evaluation:

```bash
sepquant-eval-tasks --config configs/eval/tasks_opt125m.json \
  --weight-format none \
  --quantization-plan configs/quantization/example_mixed_fp4_plan.json
```

Plan precedence is:

```text
exact layer name > wildcard pattern > plan default > global CLI/config format
```

This is the intended bridge between future search algorithms and evaluation: a search script can emit a plan JSON, and the existing PPL/task evaluators can consume it directly.

## Calibration Data

Calibration is available as an independent pipeline so multiple research methods can reuse the same sampled data.

Collect Qwen3 calibration from Wikitext2:

```bash
scripts/run_collect_calib.sh configs/calib/qwen3_wikitext2.json
```

Collect OPT-125M calibration from Wikitext2:

```bash
scripts/run_collect_calib.sh configs/calib/opt125m_wikitext2.json
```

Direct command:

```bash
sepquant-collect-calib --config configs/calib/qwen3_wikitext2.json
```

Important calibration controls:

```json
{
  "dataset": "wikitext",
  "dataset_config": "wikitext-2-raw-v1",
  "split": "train",
  "text_column": "text",
  "nsamples": 16,
  "seed": 0,
  "sequence_length": 512,
  "max_tokens_per_layer": 2048,
  "capture_mode": "gram",
  "input_layer_patterns": null
}
```

These correspond to:

- `seed`: random seed for sampling train chunks.
- `sequence_length`: token length of each calibration sample.
- `nsamples`: number of sampled calibration chunks.
- `capture_mode`: one of `gram`, `inputs`, or `both`.
- `max_tokens_per_layer`: maximum raw input activation rows per linear layer.
- `input_layer_patterns`: optional wildcard list controlling which layers store raw inputs.

Capture modes:

```text
gram   -> store X^T X for each target linear layer
inputs -> store raw flattened X for selected target layers
both   -> store both Gram matrices and selected raw inputs
```

For weight-stage methods, `gram` is the recommended default because it stores only `[hidden_dim, hidden_dim]` per layer and is enough for reconstruction objectives. Use `inputs` or `both` when future activation-stage methods need token-level activation samples.

Store Gram matrices for all layers and raw inputs only for OPT `fc1`/`fc2` layers:

```bash
scripts/run_collect_calib.sh configs/calib/opt125m_wikitext2.json \
  --capture-mode both \
  --input-layer-patterns '*fc1' '*fc2'
```

Use C4 instead of Wikitext2:

```bash
scripts/run_collect_calib.sh configs/calib/qwen3_wikitext2.json \
  --dataset allenai/c4 \
  --dataset-config en \
  --split train \
  --text-column text \
  --output-dir outputs/calib/qwen3_c4
```

Calibration artifacts are saved as:

```text
outputs/calib/qwen3_wikitext2/
  metadata.json
  grams/
    <escaped-layer-name>.pt
  inputs/
    <escaped-layer-name>.pt
```

## Research Method: Weight Format Search

The first research-level method is **accuracy-first per-layer weight format search**.

For each target `nn.Linear` layer:

1. Load saved calibration inputs or Gram matrices.
2. Quantize the weight with each candidate format.
3. Compute relative reconstruction error.
4. Select the format with the lowest error.
5. Save the selected formats as a quantization plan.

Run OPT-125M search from saved calibration artifacts:

```bash
scripts/run_optimize_layers.sh configs/optimize/opt125m_weight_format_search.json
```

Run Qwen3 search from saved calibration artifacts:

```bash
scripts/run_optimize_layers.sh configs/optimize/qwen3_weight_format_search.json
```

Direct command:

```bash
sepquant-optimize-layers \
  --config configs/optimize/opt125m_weight_format_search.json
```

Example search config:

```json
{
  "model": "/Users/phoenix/Works/models/opt-125m",
  "model_type": "opt",
  "method": "weight_format_search",
  "calibration_dir": "outputs/calib/opt125m_wikitext2",
  "candidates": ["mxfp4", "nvfp4"],
  "activation_format": "none",
  "output_plan_path": "outputs/plans/opt125m_layerwise_optimized.json"
}
```

Evaluate the searched plan:

```bash
sepquant-eval-ppl --config configs/eval/ppl_opt125m_wikitext2.json \
  --weight-format none \
  --quantization-plan outputs/plans/opt125m_weight_format_search.json
```

```bash
sepquant-eval-tasks --config configs/eval/tasks_opt125m.json \
  --weight-format none \
  --quantization-plan outputs/plans/opt125m_weight_format_search.json
```

## Research Method: GPTQ

SepQuant also includes a GPTQ-style layer optimizer. It consumes saved Gram matrices `X^T X`, performs Hessian-inverse error compensation column by column, and returns optimized dequantized weights.

Run OPT-125M GPTQ with MXFP4:

```bash
scripts/run_optimize_layers.sh configs/optimize/opt125m_gptq_mxfp4.json
```

Run Qwen3 GPTQ with MXFP4:

```bash
scripts/run_optimize_layers.sh configs/optimize/qwen3_gptq_mxfp4.json
```

GPTQ config example:

```json
{
  "method": "gptq",
  "calibration_dir": "outputs/calib/opt125m_wikitext2",
  "weight_format": "mxfp4",
  "activation_format": "none",
  "gptq_damp_percent": 0.01,
  "output_plan_path": "outputs/plans/opt125m_gptq_mxfp4.json"
}
```

Optionally save a SepQuant checkpoint containing the GPTQ-corrected weights:

```bash
scripts/run_optimize_layers.sh configs/optimize/opt125m_gptq_mxfp4.json \
  --save-quantized-checkpoint outputs/checkpoints/opt125m_gptq_mxfp4
```

Run MXFP4 Hessian-aware scale-code search:

```bash
scripts/run_optimize_layers.sh configs/optimize/opt125m_mxfp4_hessian_scale_search.json
```

This method keeps the Hessian block diagonal with 32-column MXFP4 blocks and searches power-of-two scale offsets such as `[-2, -1, 0, 1, 2]`.

Run the integrated MXFP4 scale search + GPTQ method:

```bash
scripts/run_optimize_layers.sh configs/optimize/opt125m_mxfp4_hessian_scale_search_gptq.json
```

This first searches deployment-valid MXFP4 block scales, then runs GPTQ compensation while reusing those searched scales during column quantization.

Run the dynamic MXFP4 scale search + GPTQ method:

```bash
scripts/run_optimize_layers.sh configs/optimize/opt125m_mxfp4_dynamic_scale_search_gptq.json
```

This searches MXFP4 block scales inside the GPTQ loop from the current compensated block before quantizing that block. The scale-search objective can be set with `mxfp4_scale_objective`: `identity`, `diag`, or `block`.

Run the same method with block-Hadamard rotation and dynamic MXFP4 activation scale search:

```bash
scripts/run_optimize_layers.sh configs/optimize/qwen3_mxfp4_dynamic_scale_search_gptq_hadamard.json
```

With `rotation=block_hadamard`, weights and Gram matrices are rotated during optimization, and `QuantLinear` applies the same FWHT block rotation along the hidden dimension before activation quantization.

Run the dynamic method with greedy per-layer Hadamard selection:

```bash
scripts/run_optimize_layers.sh configs/optimize/qwen3_mxfp4_dynamic_scale_search_gptq_rotation_select.json
```

This compares full GPTQ+MXFP4 candidates with and without block-Hadamard rotation on saved calibration inputs, then writes the selected per-layer rotation into the quantization plan. Collect calibration with `capture_mode=both` for this method so both `X^T X` and layer inputs are available.

Evaluate a checkpoint that contains a mixed rotation plan:

```bash
sepquant-eval-ppl --config configs/eval/ppl_qwen3_rotation_select_wikitext2.json
sepquant-eval-tasks --config configs/eval/tasks_qwen3_rotation_select.json
```

When `pre_quant_model` contains `quantization_plan.json`, eval automatically uses that plan. Global `rotation=block_hadamard` is still supported for blind rotation; per-layer plan entries override the global fallback when a plan is present.

## Common Workflows

Evaluate FP baseline:

```bash
sepquant-eval-ppl --config configs/eval/ppl_opt125m_wikitext2.json \
  --weight-format none
```

Evaluate MXFP4 weight-only quantization:

```bash
sepquant-eval-ppl --config configs/eval/ppl_opt125m_wikitext2.json \
  --weight-format mxfp4 \
  --activation-format none
```

Evaluate MXFP4 weights with NVFP4 activations:

```bash
sepquant-eval-ppl --config configs/eval/ppl_opt125m_wikitext2.json \
  --weight-format mxfp4 \
  --activation-format nvfp4
```

Run downstream tasks with NVFP4:

```bash
sepquant-eval-tasks --config configs/eval/tasks_opt125m.json \
  --weight-format nvfp4 \
  --activation-format none
```

Evaluate a checkpoint that already contains fake-quantized weights:

```bash
sepquant-eval-ppl --model /Users/phoenix/Works/models/opt-125m \
  --pre-quant-model outputs/quant/opt125m_gptq_mxfp4 \
  --weight-format mxfp4 \
  --activation-format none
```

When `pre_quant_model` is set, `model` stays as the base model identity and tokenizer fallback, while the model weights are loaded from the pre-quantized checkpoint. Weight quantization is skipped to avoid quantizing those weights again.

## Experiment Records

Evaluation commands can save structured experiment runs for later comparison:

```bash
sepquant-eval-ppl --config configs/eval/ppl_opt125m_wikitext2.json \
  --experiment-dir outputs/experiments
```

Each run writes `config.json`, `metrics.json`, `metadata.json`, and `artifacts.json`.
Generate a searchable HTML summary from those records:

```bash
sepquant-report-experiments outputs/experiments \
  --output outputs/experiments/report.html
```

## Notes

- Current quantization is fake quantization for research validation, not packed-kernel inference.
- `include_lm_head` defaults to `false` because most quantization baselines leave the LM head unquantized.
- `activation_format` can be set to `none`, `mxfp4`, `mxfp4_search`, or `nvfp4`.
- Downstream task results are saved to `output_path` when provided.
- Use small `limit` values for task smoke tests before running full evaluations.

