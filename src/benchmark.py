"""Controlled per-graph training-hardware benchmark.

Measures what one training step (forward + backward + optimizer step) costs
for a given reduction config — step time, peak VRAM, activation VRAM — on a
small seeded sample of graphs, **one graph per batch**.

Why one graph per batch (not real training's node-budget dynamic batching):
node-budget batching packs graphs until it hits a fixed node budget, so every
batch has ~the same node count regardless of reduction, which holds peak VRAM
~constant across methods *by construction* and hides the whole point of
reduction (lower per-graph memory footprint → avoids OOM). Measuring one graph
at a time exposes each graph's true footprint, lets the same graph be compared
full vs. reduced (pairing by graph_id downstream), and is fully reproducible.

Not a substitute for real training throughput — it is a controlled, relative
comparison of per-graph memory/latency across reduction methods.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch_geometric.data import Batch

import config
from config import HEAD_DROPOUT, NORMALIZE_EDGES
from data.datamodule import AIGDataModule
from models.lightning_model import AIGRegressionLightningModule
from train import _select_precision

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_commit_hash() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPO_ROOT,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPO_ROOT,
        ).stdout.strip()
        return f"{commit}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def _torch_geometric_version() -> str:
    try:
        import torch_geometric

        return torch_geometric.__version__
    except Exception:
        return "unknown"


def reproducibility_metadata(seed: int) -> dict:
    return {
        "git_commit_hash": _git_commit_hash(),
        "seed": seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "torch_version": torch.__version__,
        "torch_geometric_version": _torch_geometric_version(),
        "cli_invocation": " ".join(sys.argv),
    }


def run_label_for(algorithm: str, reduction_type: str, reduction_method: str | None) -> str:
    if reduction_type == "none":
        return algorithm
    return f"{algorithm}_{reduction_method}"


def resolve_reduction_kwargs(reduction_type: str, reduction_method: str | None) -> dict:
    """Maps the generic reduction axis onto AIGDataModule's concrete kwargs."""
    if reduction_type == "none":
        return {"sparsification": None, "partition": None}
    if reduction_type == "sparsification":
        return {"sparsification": reduction_method, "partition": None}
    if reduction_type == "partition":
        return {"sparsification": None, "partition": reduction_method}
    if reduction_type == "summarization":
        raise NotImplementedError(
            "summarization is not wired into AIGDataModule yet — add "
            "dataset-level support first."
        )
    raise ValueError(f"Unknown reduction_type: {reduction_type!r}")


def _safe_nanmean(values) -> float:
    """np.nanmean but returns NaN (no warning) when every value is NaN — the
    CPU case, where per-graph VRAM is unmeasured."""
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def write_single_row_csv(path: str | Path, row: dict) -> None:
    """Write one config's result as its own header+row file (overwrite).

    The SLURM array runs up to 9 tasks concurrently, so appending to one
    shared CSV would race on the header write and could interleave/corrupt
    rows, and a stale file from an older column schema would silently
    misalign. One file per config sidesteps all of that: each task owns its
    file, and re-running a config cleanly overwrites its own result instead of
    appending a duplicate. results_to_latex.py globs the directory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_per_graph_csv(path: str | Path, per_graph: list[dict], max_rows: int) -> None:
    """One row per measured graph, capped with seeded sampling (same pattern as
    test.py's per-graph predictions) so scratch never accumulates huge files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = per_graph
    if len(rows) > max_rows:
        idx = sorted(random.Random(42).sample(range(len(rows)), k=max_rows))
        rows = [per_graph[i] for i in idx]
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_model(compile_model: bool) -> AIGRegressionLightningModule:
    """Fresh model from config.py defaults — identical architecture across all
    reduction types, so no checkpoint is needed to get representative timing."""
    encoder_kwargs = config.ENCODER_KWARGS_DEFAULTS.copy()
    encoder_kwargs.update(
        {
            "num_layers": config.NUM_LAYERS,
            "hid_dim": config.HIDDEN_DIM,
            "dropout": config.DROPOUT,
            "norm_type": config.NORM_TYPE,
            "jk_mode": config.JK_MODE,
            "normalize_edges": NORMALIZE_EDGES,
        }
    )
    return AIGRegressionLightningModule(
        encoder_name=config.ENCODER_NAME,
        hidden_dim=config.HIDDEN_DIM,
        pe_type=config.PE_TYPE,
        pos_enc_dim=config.POS_ENC_DIM if config.PE_TYPE != "none" else 0,
        pooling_type=config.POOLING_TYPE,
        encoder_kwargs=encoder_kwargs,
        head_dropout=HEAD_DROPOUT,
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        compile_model=compile_model,
        loss_fn=nn.SmoothL1Loss(beta=0.01),
    )


def build_datamodule(args: argparse.Namespace) -> AIGDataModule:
    reduction_kwargs = resolve_reduction_kwargs(args.reduction_type, args.reduction_method)
    positional_encoding = config.PE_TYPE if config.PE_TYPE != "none" else None
    # batch_size is nominal — the benchmark iterates the dataset one graph at a
    # time (see run_benchmark), so no DataLoader batching happens here. Seeded
    # subsetting keeps the same graphs across every config for graph_id pairing.
    return AIGDataModule(
        csv_paths=args.csv_paths,
        positional_encoding=positional_encoding,
        batch_size=1,
        split_ratios=(0.8, 0.1, 0.1),
        seed=args.seed,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        hp_tuning_splits_path=args.hp_tuning_splits_path,
        tier0_cache_dir=args.tier0_cache_dir,
        tier1_cache_dir=args.tier1_cache_dir,
        train_num_samples=args.num_warmup_graphs + args.num_measure_graphs,
        **reduction_kwargs,
    )


def run_benchmark(
    model: AIGRegressionLightningModule,
    dataset,
    device: torch.device,
    *,
    num_warmup: int,
    precision: str,
    num_repeats: int = 3,
) -> tuple[dict, list[dict]]:
    """One-graph-per-batch training-step benchmark.

    Iterates the dataset directly (no DataLoader) so batch i maps to
    ``dataset.samples[i]`` deterministically and graph_id is known. The first
    ``num_warmup`` graphs are run but excluded from all aggregates — they cover
    CUDA/cuDNN init and AdamW's first-step optimizer-state allocation. A memory
    floor (model + optimizer state + CUDA context) is captured once after
    warmup so activation-only VRAM can be reported.

    Each measured graph's step is timed ``num_repeats`` times; the per-graph
    step time is the **median** of those repeats (robust to system noise), and
    the within-graph std is recorded so measurement stability is visible. VRAM
    is deterministic for a given graph, so its peak is taken across repeats
    (they agree) rather than needing repetition.
    """
    loss_fn = nn.SmoothL1Loss(beta=0.01)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    model.to(device)
    model.train()

    is_cuda = device.type == "cuda"
    use_bf16 = precision == "bf16-mixed" and is_cuda
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )

    proc = psutil.Process()
    peak_rss = 0
    per_graph: list[dict] = []
    n = len(dataset)
    num_warmup = min(num_warmup, n)

    def _one_step(batch) -> tuple[float, float | None]:
        """One training step with CUDA-synchronized timing bracketing exactly
        the compute. Returns (step_time_s, peak_bytes | None)."""
        targets = batch.y.view(-1)
        optimizer.zero_grad(set_to_none=True)
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with autocast_ctx:
            preds = model.forward(batch).squeeze(-1)
            loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        if is_cuda:
            torch.cuda.synchronize(device)
        step_time = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated(device) if is_cuda else None
        return step_time, peak

    # Warmup — run full steps (this is what allocates AdamW's optimizer state)
    # but record nothing. `del batch` frees each warmup graph's input so the
    # floor below is measured with no graph resident.
    for i in range(num_warmup):
        batch = Batch.from_data_list([dataset[i]]).to(device)
        _one_step(batch)
        del batch

    # Memory floor: model + optimizer state + CUDA context only — no input, no
    # gradients (just zeroed), no activations (freed after the last backward).
    # Measuring it once with nothing resident means activation_vram = peak −
    # floor is a clean per-graph increment that cannot be biased by, or driven
    # negative by, whichever graph happened to be on the device.
    optimizer.zero_grad(set_to_none=True)
    floor_mb = float("nan")
    if is_cuda:
        torch.cuda.synchronize(device)
        floor_mb = torch.cuda.memory_allocated(device) / (1024**2)

    for i in range(num_warmup, n):
        batch = Batch.from_data_list([dataset[i]]).to(device)

        rep_times: list[float] = []
        peak = None
        for _ in range(max(1, num_repeats)):
            st, pk = _one_step(batch)
            rep_times.append(st)
            if pk is not None:
                peak = pk if peak is None else max(peak, pk)
        peak_rss = max(peak_rss, proc.memory_info().rss)

        step_time = float(np.median(rep_times))
        step_time_std = float(np.std(rep_times))

        if is_cuda:
            peak_mb = peak / (1024**2)
            activation_mb = peak_mb - floor_mb
        else:
            peak_mb = float("nan")
            activation_mb = float("nan")

        per_graph.append(
            {
                "graph_id": dataset.samples[i].graph_path,
                "num_nodes": int(batch.x.size(0)),
                "num_edges": int(batch.edge_index.size(1)),
                "step_time_s": step_time,
                "step_time_std_s": step_time_std,
                "peak_vram_mb": peak_mb,
                "activation_vram_mb": activation_mb,
            }
        )
        del batch

    if per_graph:
        step_times = np.array([g["step_time_s"] for g in per_graph])
        within_stds = np.array([g["step_time_std_s"] for g in per_graph])
        peaks = np.array([g["peak_vram_mb"] for g in per_graph])
        activations = np.array([g["activation_vram_mb"] for g in per_graph])
        nodes = np.array([g["num_nodes"] for g in per_graph])
        edges = np.array([g["num_edges"] for g in per_graph])
        total_time = float(step_times.sum())
        aggregate = {
            "num_measured_graphs": len(per_graph),
            "num_repeats": max(1, num_repeats),
            # std_step_time_s is the spread ACROSS graphs (expected — graphs
            # vary in size); avg_within_graph_step_std_s is the repeat-to-repeat
            # measurement noise on the SAME graph (should be small — this is
            # the "are the timings stable" check).
            "avg_step_time_s": float(step_times.mean()),
            "std_step_time_s": float(step_times.std()),
            "avg_within_graph_step_std_s": float(within_stds.mean()),
            "throughput_graphs_per_s": (
                len(per_graph) / total_time if total_time > 0 else float("nan")
            ),
            "peak_vram_mb": _safe_nanmean(peaks),
            "activation_vram_mb": _safe_nanmean(activations),
            "memory_floor_mb": floor_mb,
            "peak_process_rss_mb": peak_rss / (1024**2),
            "avg_nodes_per_graph": float(nodes.mean()),
            "avg_edges_per_graph": float(edges.mean()),
        }
    else:
        aggregate = {
            "num_measured_graphs": 0,
            "num_repeats": max(1, num_repeats),
            "avg_step_time_s": float("nan"),
            "std_step_time_s": float("nan"),
            "avg_within_graph_step_std_s": float("nan"),
            "throughput_graphs_per_s": float("nan"),
            "peak_vram_mb": float("nan"),
            "activation_vram_mb": float("nan"),
            "memory_floor_mb": floor_mb,
            "peak_process_rss_mb": peak_rss / (1024**2),
            "avg_nodes_per_graph": float("nan"),
            "avg_edges_per_graph": float("nan"),
        }

    return aggregate, per_graph


def main(args: argparse.Namespace) -> None:
    if args.algorithm not in config.VALID_ALGORITHMS:
        raise ValueError(
            f"Algorithm '{args.algorithm}' must be one of {config.VALID_ALGORITHMS}"
        )
    if args.reduction_type != "none" and not args.reduction_method:
        raise ValueError("--reduction_method is required when --reduction_type != none")
    if args.num_warmup_graphs < 1:
        raise ValueError(
            "--num_warmup_graphs must be >= 1: the memory floor is measured "
            "after warmup, and AdamW only allocates its optimizer state on the "
            "first optimizer.step(). With 0 warmup the floor would exclude that "
            "state and inflate every activation_vram_mb reading."
        )

    # Same backend settings train.py uses, so the benchmark reflects real
    # training numerics rather than an artificially different configuration.
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = _select_precision()
    print(f"[benchmark] device={device} precision={precision}", flush=True)

    datamodule = build_datamodule(args)
    datamodule.setup("fit")
    model = build_model(compile_model=args.torch_compile)

    aggregate, per_graph = run_benchmark(
        model,
        datamodule.train_ds,
        device,
        num_warmup=args.num_warmup_graphs,
        precision=precision,
        num_repeats=args.num_repeats,
    )

    run_label = run_label_for(args.algorithm, args.reduction_type, args.reduction_method)
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reduction_type": args.reduction_type,
        "reduction_method": args.reduction_method or "",
        "algorithm": args.algorithm,
        "run_label": run_label,
        "device": device.type,
        "batch_size": 1,
        "num_workers": args.num_workers,
        "num_warmup_graphs": args.num_warmup_graphs,
        **aggregate,
        **reproducibility_metadata(args.seed),
    }
    results_path = Path(args.results_dir) / f"{run_label}_{device.type}.csv"
    write_single_row_csv(results_path, row)

    per_graph_path = Path(args.per_graph_dir) / f"{run_label}.csv"
    write_per_graph_csv(per_graph_path, per_graph, max_rows=args.max_per_graph_rows)

    print(
        f"[benchmark] {run_label}: avg_step_time_s={aggregate['avg_step_time_s']:.4f} "
        f"throughput={aggregate['throughput_graphs_per_s']:.1f} graphs/s "
        f"peak_vram_mb={aggregate['peak_vram_mb']:.1f} "
        f"activation_vram_mb={aggregate['activation_vram_mb']:.1f}",
        flush=True,
    )
    print(f"[benchmark] Wrote per-graph rows to {per_graph_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Controlled per-graph (one graph per batch) training-hardware benchmark."
    )
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--csv_paths", nargs="+", required=True)
    parser.add_argument(
        "--reduction_type",
        type=str,
        choices=["none", "sparsification", "partition", "summarization"],
        default="none",
    )
    parser.add_argument("--reduction_method", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--tier0_cache_dir", type=str, default=None)
    parser.add_argument("--tier1_cache_dir", type=str, default=None)
    parser.add_argument("--hp_tuning_splits_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_warmup_graphs",
        type=int,
        default=5,
        help="Graphs run but excluded from all aggregates (CUDA/cuDNN + optimizer-state warmup).",
    )
    parser.add_argument("--num_measure_graphs", type=int, default=100)
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=3,
        help="Times each graph's step is re-timed; per-graph time is the median "
        "(robust to system noise). VRAM is deterministic so it isn't repeated.",
    )
    parser.add_argument(
        "--torch_compile",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=False,
        help=(
            "Default off: at batch_size=1 every graph is a distinct shape, so "
            "torch.compile recompilation spikes would pollute per-graph timing. "
            "The benchmark measures eager-mode relative cost."
        ),
    )
    parser.add_argument("--max_per_graph_rows", type=int, default=20_000)
    parser.add_argument("--results_dir", type=str, default="results/training_benchmark")
    parser.add_argument("--per_graph_dir", type=str, default="results/benchmark_per_graph")

    main(parser.parse_args())
