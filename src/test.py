"""Evaluate a trained checkpoint on the complete test split.

Runs up to two passes per checkpoint:
    - "full_graph":       unchanged test graphs (baseline eval / RQ4 cross-state)
    - "matched_reduction": test graphs under the checkpoint's own reduction
                            (RQ1/RQ3 matched-state eval; skipped for baseline
                            checkpoints, since it would duplicate full_graph)

Each pass reports accuracy (Smooth L1, RMSE, R^2, Spearman) and inference
hardware stats (throughput, peak VRAM, GPU utilization, host memory) in one
forward-only sweep, and writes one per-config file into --results_dir plus
(optionally) a bounded per-graph predictions CSV for later error-vs-graph-size
breakdowns.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
from scipy.stats import spearmanr

import config
from data.datamodule import AIGDataModule
from models.lightning_model import AIGRegressionLightningModule

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


def _batch_per_graph_counts(batch) -> tuple[list[int], list[int]]:
    num_graphs = int(batch.num_graphs)
    node_counts = torch.bincount(batch.batch, minlength=num_graphs)
    src_graph = batch.batch[batch.edge_index[0]]
    edge_counts = torch.bincount(src_graph, minlength=num_graphs)
    return node_counts.tolist(), edge_counts.tolist()


def compute_accuracy_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """Smooth L1 / RMSE / R^2 / Spearman over a full set of predictions.

    Pulled out as a pure function (rather than inlined in run_eval_pass) so
    the metric math is directly unit-testable on synthetic tensors, without
    needing a dataloader, checkpoint, or GPU.
    """
    n = int(preds.numel())
    if n == 0:
        return {
            "smooth_l1": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "spearman": float("nan"),
        }

    loss_fn = nn.SmoothL1Loss(beta=0.01)
    smooth_l1 = float(loss_fn(preds, targets))
    rmse = float(torch.sqrt(torch.mean((preds - targets) ** 2)))

    ss_res = float(torch.sum((preds - targets) ** 2))
    ss_tot = float(torch.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    spearman = float(spearmanr(preds.numpy(), targets.numpy()).correlation) if n > 1 else float("nan")

    return {"smooth_l1": smooth_l1, "rmse": rmse, "r2": r2, "spearman": spearman}


def run_eval_pass(
    model: AIGRegressionLightningModule,
    dm_kwargs: dict,
    *,
    device: torch.device,
    gpu_util_sample_every: int = 5,
) -> tuple[dict, dict]:
    """Runs one forward-only sweep over the test split; returns (metrics, per_graph)."""
    datamodule = AIGDataModule(**dm_kwargs)
    datamodule.setup("test")
    loader = datamodule.test_dataloader()
    samples = datamodule.test_ds.samples

    model.eval()
    model.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    proc = psutil.Process()
    peak_rss = 0
    sys_mem_pcts: list[float] = []
    gpu_utils: list[float] = []

    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_node_counts: list[int] = []
    all_edge_counts: list[int] = []

    t_start = time.perf_counter()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            preds = model.forward(batch).squeeze(-1).cpu()
            targets = batch.y.view(-1).cpu()
            node_counts, edge_counts = _batch_per_graph_counts(batch)

            all_preds.append(preds)
            all_targets.append(targets)
            all_node_counts.extend(node_counts)
            all_edge_counts.extend(edge_counts)

            rss = proc.memory_info().rss
            if rss > peak_rss:
                peak_rss = rss
            sys_mem_pcts.append(psutil.virtual_memory().percent)

            if device.type == "cuda" and batch_idx % max(1, gpu_util_sample_every) == 0:
                try:
                    gpu_utils.append(float(torch.cuda.utilization(device)))
                except Exception:
                    pass

    total_time = time.perf_counter() - t_start

    preds_t = torch.cat(all_preds) if all_preds else torch.empty(0)
    targets_t = torch.cat(all_targets) if all_targets else torch.empty(0)
    num_graphs_total = int(preds_t.numel())

    accuracy_metrics = compute_accuracy_metrics(preds_t, targets_t)

    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else float("nan")
    )

    metrics = {
        "num_graphs": num_graphs_total,
        **accuracy_metrics,
        "total_time_s": total_time,
        "throughput_graphs_per_s": (
            num_graphs_total / total_time if total_time > 0 else float("nan")
        ),
        "peak_vram_mb": peak_vram_mb,
        "avg_gpu_utilization_pct": float(np.mean(gpu_utils)) if gpu_utils else float("nan"),
        "peak_process_rss_mb": peak_rss / (1024**2),
        "avg_system_memory_pct": (
            float(np.mean(sys_mem_pcts)) if sys_mem_pcts else float("nan")
        ),
        "avg_nodes_per_graph": (
            float(np.mean(all_node_counts)) if all_node_counts else float("nan")
        ),
        "avg_edges_per_graph": (
            float(np.mean(all_edge_counts)) if all_edge_counts else float("nan")
        ),
    }

    per_graph = {
        "graph_id": [s.graph_path for s in samples],
        "num_nodes": all_node_counts,
        "num_edges": all_edge_counts,
        "target": targets_t.tolist(),
        "prediction": preds_t.tolist(),
    }
    return metrics, per_graph


def write_single_row_csv(path: str | Path, row: dict) -> None:
    """Write one (config, eval_mode, device) result as its own header+row file
    (overwrite).

    test.sh and test_cpu.sh together run up to 18 array tasks concurrently, so
    appending to one shared CSV would race on the header write and could
    interleave/corrupt rows, and a stale file from an older column schema would
    silently misalign. One file per (config, eval_mode, device) sidesteps all
    of that; re-running a config cleanly overwrites its own result.
    results_to_latex.py globs the directory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_predictions_csv(path: str | Path, per_graph: dict, max_rows: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(per_graph["graph_id"])
    indices = list(range(n))
    if n > max_rows:
        indices = sorted(random.Random(42).sample(indices, k=max_rows))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["graph_id", "num_nodes", "num_edges", "target", "prediction", "abs_error"])
        for i in indices:
            target = per_graph["target"][i]
            pred = per_graph["prediction"][i]
            writer.writerow(
                [
                    per_graph["graph_id"][i],
                    per_graph["num_nodes"][i],
                    per_graph["num_edges"][i],
                    target,
                    pred,
                    abs(pred - target),
                ]
            )


def main(args: argparse.Namespace) -> None:
    if args.algorithm not in config.VALID_ALGORITHMS:
        raise ValueError(
            f"Algorithm '{args.algorithm}' must be one of {config.VALID_ALGORITHMS}"
        )
    if args.reduction_type != "none" and not args.reduction_method:
        raise ValueError("--reduction_method is required when --reduction_type != none")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    reduction_kwargs = resolve_reduction_kwargs(args.reduction_type, args.reduction_method)
    run_label = run_label_for(args.algorithm, args.reduction_type, args.reduction_method)
    checkpoint_path = Path(args.checkpoint_dir) / run_label / args.checkpoint_filename
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[test] device={device} checkpoint={checkpoint_path}", flush=True)
    model = AIGRegressionLightningModule.load_from_checkpoint(
        str(checkpoint_path), map_location="cpu"
    )

    # Derived from the checkpoint's own hparams (not a separate CLI flag) so the
    # data-loading PE config can never drift from what the model was trained with.
    positional_encoding = (
        model.hparams.pe_type if model.hparams.pe_type != "none" else None
    )

    base_dm_kwargs = dict(
        csv_paths=args.csv_paths,
        positional_encoding=positional_encoding,
        batch_size=args.batch_size,
        split_ratios=(0.8, 0.1, 0.1),
        seed=args.seed,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        hp_tuning_splits_path=args.hp_tuning_splits_path,
        tier0_cache_dir=args.tier0_cache_dir,
        tier1_cache_dir=args.tier1_cache_dir,
    )

    repro = reproducibility_metadata(args.seed)

    passes: list[tuple[str, dict]] = [("full_graph", {"sparsification": None, "partition": None})]
    if args.reduction_type != "none":
        passes.append(("matched_reduction", reduction_kwargs))

    for eval_mode, red_kwargs in passes:
        print(f"[test] Running {eval_mode!r} pass ...", flush=True)
        dm_kwargs = {**base_dm_kwargs, **red_kwargs}
        metrics, per_graph = run_eval_pass(
            model,
            dm_kwargs,
            device=device,
            gpu_util_sample_every=args.gpu_util_sample_every,
        )
        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "checkpoint_path": str(checkpoint_path),
            "run_label": run_label,
            "reduction_type": args.reduction_type,
            "reduction_method": args.reduction_method or "",
            "eval_mode": eval_mode,
            "device": device.type,
            **metrics,
            **repro,
        }
        results_path = Path(args.results_dir) / f"{run_label}_{eval_mode}_{device.type}.csv"
        write_single_row_csv(results_path, row)
        print(
            f"[test] {eval_mode}: rmse={metrics['rmse']:.4f} r2={metrics['r2']:.4f} "
            f"spearman={metrics['spearman']:.4f} "
            f"throughput={metrics['throughput_graphs_per_s']:.1f} graphs/s",
            flush=True,
        )

        if args.dump_predictions:
            pred_path = Path(args.predictions_dir) / f"{run_label}_{eval_mode}.csv"
            write_predictions_csv(pred_path, per_graph, max_rows=args.max_prediction_rows)
            print(f"[test] Wrote per-graph predictions to {pred_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained AIG regression checkpoint on the complete test split."
    )
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--csv_paths", nargs="+", required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--checkpoint_filename", type=str, default="last.ckpt")
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
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None)
    parser.add_argument("--gpu_util_sample_every", type=int, default=5)
    parser.add_argument(
        "--dump_predictions",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=True,
    )
    parser.add_argument("--max_prediction_rows", type=int, default=20_000)
    parser.add_argument("--results_dir", type=str, default="results/inference_results")
    parser.add_argument("--predictions_dir", type=str, default="results/predictions")

    main(parser.parse_args())
