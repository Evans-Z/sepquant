from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from sepquant.formats import get_fp4_format


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    shape_ok: bool
    finite_ok: bool
    zero_ok: bool
    mse: float
    mae: float
    max_abs_error: float
    sqnr_db: float
    avg_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FP4 fake-quantization formats.")
    parser.add_argument("--formats", nargs="+", default=["mxfp4", "nvfp4"])
    parser.add_argument("--shape", nargs="+", type=int, default=[16, 4096])
    parser.add_argument("--distribution", choices=["normal", "uniform", "outlier"], default="normal")
    parser.add_argument(
        "--device",
        default="mps",
        help="Device to benchmark on: cpu, cuda, cuda:0, mps, apple, apple-silicon, or auto.",
    )
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = _parse_dtype(args.dtype)
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    print(f"device={device}, dtype={dtype}, shape={tuple(args.shape)}")

    tensor = make_tensor(
        shape=tuple(args.shape),
        distribution=args.distribution,
        device=device,
        dtype=dtype,
    )

    results = [
        benchmark_format(name, tensor=tensor, warmup=args.warmup, iters=args.iters)
        for name in args.formats
    ]
    print_results(results)


def make_tensor(
    *,
    shape: tuple[int, ...],
    distribution: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if distribution == "normal":
        return torch.randn(shape, device=device, dtype=dtype)
    if distribution == "uniform":
        return torch.empty(shape, device=device, dtype=dtype).uniform_(-1.0, 1.0)
    if distribution == "outlier":
        tensor = torch.randn(shape, device=device, dtype=dtype)
        flat = tensor.flatten()
        stride = max(1, flat.numel() // 128)
        flat[::stride] *= 16.0
        return tensor
    raise ValueError(f"Unsupported distribution: {distribution}")


def benchmark_format(
    name: str,
    *,
    tensor: torch.Tensor,
    warmup: int,
    iters: int,
) -> BenchmarkResult:
    fmt = get_fp4_format(name)
    zero = torch.zeros_like(tensor)

    quantized = fmt.quantize(tensor)
    zero_quantized = fmt.quantize(zero)

    for _ in range(warmup):
        fmt.quantize(tensor)
    _sync_if_needed(tensor)

    start = time.perf_counter()
    for _ in range(iters):
        fmt.quantize(tensor)
    _sync_if_needed(tensor)
    avg_ms = (time.perf_counter() - start) * 1000.0 / max(1, iters)

    error = (quantized.float() - tensor.float()).abs()
    mse = torch.mean(error.square()).item()
    signal = torch.mean(tensor.float().square()).item()
    sqnr_db = 10.0 * torch.log10(torch.tensor(signal / max(mse, 1e-30))).item()

    return BenchmarkResult(
        name=name,
        shape_ok=quantized.shape == tensor.shape,
        finite_ok=torch.isfinite(quantized).all().item(),
        zero_ok=torch.equal(zero_quantized, zero),
        mse=mse,
        mae=error.mean().item(),
        max_abs_error=error.max().item(),
        sqnr_db=sqnr_db,
        avg_ms=avg_ms,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    headers = [
        "format",
        "shape",
        "finite",
        "zero",
        "mse",
        "mae",
        "max_abs",
        "sqnr_db",
        "avg_ms",
    ]
    rows = [
        [
            result.name,
            str(result.shape_ok),
            str(result.finite_ok),
            str(result.zero_ok),
            f"{result.mse:.6e}",
            f"{result.mae:.6e}",
            f"{result.max_abs_error:.6e}",
            f"{result.sqnr_db:.2f}",
            f"{result.avg_ms:.3f}",
        ]
        for result in results
    ]
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]

    print(" | ".join(header.ljust(widths[col]) for col, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[col]) for col, value in enumerate(row)))


def _parse_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _resolve_device(device: str) -> torch.device:
    normalized = device.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized == "cpu":
        return torch.device("cpu")

    if normalized == "cuda" or normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available in this PyTorch environment.")
        return torch.device(device)

    if normalized in {"mps", "apple", "apple-silicon", "apple_silicon"}:
        if not _mps_is_available():
            raise SystemExit(
                "Apple Silicon MPS was requested but is not available in this PyTorch environment."
            )
        return torch.device("mps")

    raise SystemExit(
        f"Unsupported device '{device}'. Use cpu, cuda, cuda:0, mps, apple, apple-silicon, or auto."
    )


def _mps_is_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _sync_if_needed(tensor: torch.Tensor) -> None:
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)
    elif tensor.device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


if __name__ == "__main__":
    main()

