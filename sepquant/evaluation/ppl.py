from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm

from sepquant.experiments import save_experiment_run, write_json
from sepquant.models import load_quantized_causal_lm


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, help="Path to a JSON config file.")
    config_args, remaining_argv = config_parser.parse_known_args()
    config = _load_config(config_args.config) if config_args.config else {}

    parser = argparse.ArgumentParser(
        description="Evaluate causal LM perplexity with optional FP4 fake quantization.",
        parents=[config_parser],
    )
    parser.add_argument("--model", default=None, help="Hugging Face model id or local path.")
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "qwen", "qwen2", "qwen3", "llama", "mistral", "gemma", "opt", "generic"],
        help="Model family used to select linear layers for quantization.",
    )
    parser.add_argument("--dataset", default="wikitext", help="Dataset name.")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--weight-format", default="mxfp4", choices=["none", "mxfp4", "nvfp4", "hif4"])
    parser.add_argument(
        "--activation-format",
        default="none",
        choices=["none", "mxfp4", "mxfp4_search", "nvfp4", "nvfp4_search", "hif4", "hif4_search"],
    )
    parser.add_argument("--quantization-plan", default=None)
    parser.add_argument("--pre-quant-model", "--pre-quantized-model", dest="pre_quant_model", default=None)
    parser.add_argument("--rotation", default="none", choices=["none", "block_hadamard"])
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--output-metrics-path", type=Path, default=None)
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.set_defaults(**config)
    args = parser.parse_args(remaining_argv)
    if args.model is None:
        parser.error("--model is required unless provided by --config")
    if isinstance(args.output_metrics_path, str):
        args.output_metrics_path = Path(args.output_metrics_path)
    if isinstance(args.experiment_dir, str):
        args.experiment_dir = Path(args.experiment_dir)
    return args


def main() -> None:
    args = parse_args()
    loaded = load_quantized_causal_lm(
        model_name_or_path=args.model,
        model_type=args.model_type,
        weight_format=args.weight_format,
        activation_format=args.activation_format,
        device=args.device,
        dtype=args.dtype,
        include_lm_head=args.include_lm_head,
        quantization_plan=args.quantization_plan,
        pre_quant_model=args.pre_quant_model,
        rotation=args.rotation,
    )
    if args.pre_quant_model is not None:
        print(f"Loaded pre-quantized model from {args.pre_quant_model}; skipped weight quantization.")
    if loaded.patch_report is not None:
        print(
            f"Replaced {loaded.patch_report.replaced} linear layers with QuantLinear "
            f"(model_type={loaded.patch_report.model_type})."
        )

    text = _load_text(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        max_samples=args.max_samples,
    )
    encoded = loaded.tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(loaded.model.device)

    loss, ppl, tokens = evaluate_perplexity(
        loaded.model,
        input_ids=input_ids,
        sequence_length=args.sequence_length,
    )
    print(f"tokens={tokens}")
    print(f"loss={loss:.4f}")
    print(f"ppl={ppl:.4f}")

    metrics = {
        "task": "perplexity",
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "loss": loss,
        "ppl": ppl,
        "tokens": tokens,
        "sequence_length": args.sequence_length,
        "max_samples": args.max_samples,
        "weight_source": args.pre_quant_model or args.model,
        "base_model": args.model,
        "pre_quant_model": args.pre_quant_model,
        "quantization_mode": "pre_quantized" if args.pre_quant_model is not None else "runtime",
    }
    if args.output_metrics_path is not None:
        write_json(args.output_metrics_path, metrics)
        print(f"Saved metrics to {args.output_metrics_path}")
    if args.experiment_dir is not None:
        run_dir = save_experiment_run(
            experiment_dir=args.experiment_dir,
            run_type="ppl",
            config=vars(args),
            metrics=metrics,
            artifacts={
                "quantization_plan": args.quantization_plan,
                "pre_quant_model": args.pre_quant_model,
                "output_metrics_path": args.output_metrics_path,
            },
        )
        print(f"Saved experiment run to {run_dir}")


def evaluate_perplexity(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    sequence_length: int,
) -> tuple[float, float, int]:
    if sequence_length <= 1:
        raise ValueError("sequence_length must be greater than 1")

    total_nll = 0.0
    total_tokens = 0
    nsamples = input_ids.numel() // sequence_length

    with torch.inference_mode():
        for sample_idx in tqdm(range(nsamples), desc="Evaluating PPL", unit="chunk"):
            begin = sample_idx * sequence_length
            end = begin + sequence_length

            batch = input_ids[:, begin:end]
            labels = batch.clone()
            outputs = model(batch, labels=labels)
            token_count = sequence_length
            total_nll += outputs.loss.item() * token_count
            total_tokens += token_count

    if total_tokens == 0:
        raise ValueError("Not enough tokens to evaluate perplexity")

    avg_loss = total_nll / total_tokens
    return avg_loss, math.exp(avg_loss), total_tokens


def _load_text(
    *,
    dataset_name: str,
    dataset_config: str,
    split: str,
    text_column: str,
    max_samples: int | None,
) -> str:
    dataset_args = [dataset_name]
    if dataset_config:
        dataset_args.append(dataset_config)
    dataset = load_dataset(*dataset_args, split=split)
    rows = dataset
    if max_samples is not None:
        rows = dataset.select(range(min(max_samples, len(dataset))))
    return "\n\n".join(row[text_column] for row in rows if row[text_column].strip())


def _load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        raise ValueError(f"Only JSON config files are supported for now: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a JSON object: {path}")
    return config


if __name__ == "__main__":
    main()

