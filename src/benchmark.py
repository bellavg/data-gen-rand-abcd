"""Controlled training-hardware benchmark on a small seeded sample of graphs.

Measures what one training step (forward + backward) costs for a given
reduction config — step time, throughput, peak VRAM, GPU utilization, host
memory — using fixed, identical settings across configs so the numbers are
actually comparable. This replaces the real-training W&B numbers, which
drifted across runs due to incidental settings changes (worker count, etc.),
per the reproducibility notes in the thesis experiments plan.

Not a substitute for real training: this only runs `--num_warmup_steps +
--num_measure_steps` batches on `--num_sample_graphs` graphs, discarding the
model afterward. It exists purely to produce apples-to-apples hardware
numbers per reduction config.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import pytorch_lightning as pl
import torch
import torch.nn as nn

import config
from config import HEAD_DROPOUT, NORMALIZE_EDGES
from data.datamodule import AIGDataModule
from models.lightning_model import AIGRegressionLightningModule
from train import _select_accelerator_and_devices, _select_precision

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


def append_csv_row(csv_path: str | Path, row: dict) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _batch_stats(batch) -> tuple[int, int, int]:
    num_graphs = getattr(batch, "num_graphs", None)
    if num_graphs is None:
        targets = getattr(batch, "y", None)
        num_graphs = int(targets.size(0)) if targets is not None and targets.dim() > 0 else 1
    else:
        num_graphs = int(num_graphs)
    num_nodes = int(batch.x.size(0)) if getattr(batch, "x", None) is not None else 0
    edge_index = getattr(batch, "edge_index", None)
    num_edges = int(edge_index.size(1)) if edge_index is not None else 0
    return num_graphs, num_nodes, num_edges


class BenchmarkCallback(pl.Callback):
    """Self-contained timing/VRAM/GPU-util/host-memory bookkeeping.

    Not added to the shared ``train_utils.py`` — no other script needs this,
    and it deliberately differs from ``TrainingStartupCallback`` by excluding
    a configurable warmup window from every aggregate so CUDA/cuDNN/compile
    warmup doesn't skew the reported numbers.
    """

    def __init__(
        self,
        num_warmup_steps: int,
        num_measure_steps: int,
        gpu_util_sample_every: int = 1,
    ) -> None:
        self.num_warmup_steps = num_warmup_steps
        self.num_measure_steps = num_measure_steps
        self.gpu_util_sample_every = gpu_util_sample_every
        self._step_times: list[float] = []
        self._graph_counts: list[int] = []
        self._node_counts: list[int] = []
        self._edge_counts: list[int] = []
        self._gpu_utils: list[float] = []
        self._batch_start: float | None = None
        self._proc = psutil.Process()
        self._peak_rss = 0
        self._sys_mem_pcts: list[float] = []

    def on_fit_start(self, trainer, pl_module) -> None:
        device = pl_module.device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self._batch_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        elapsed = time.perf_counter() - (self._batch_start or time.perf_counter())
        if batch_idx < self.num_warmup_steps:
            return

        self._step_times.append(elapsed)
        num_graphs, num_nodes, num_edges = _batch_stats(batch)
        self._graph_counts.append(num_graphs)
        self._node_counts.append(num_nodes)
        self._edge_counts.append(num_edges)

        rss = self._proc.memory_info().rss
        self._peak_rss = max(self._peak_rss, rss)
        self._sys_mem_pcts.append(psutil.virtual_memory().percent)

        device = pl_module.device
        if device.type == "cuda" and batch_idx % max(1, self.gpu_util_sample_every) == 0:
            try:
                self._gpu_utils.append(float(torch.cuda.utilization(device)))
            except Exception:
                pass

    def summary(self, device: torch.device) -> dict:
        step_times = np.array(self._step_times) if self._step_times else np.array([float("nan")])
        total_graphs = sum(self._graph_counts)
        total_time = float(np.nansum(step_times))
        peak_vram_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else float("nan")
        )
        return {
            "avg_step_time_s": float(np.nanmean(step_times)),
            "std_step_time_s": float(np.nanstd(step_times)),
            "throughput_graphs_per_s": (
                total_graphs / total_time if total_time > 0 else float("nan")
            ),
            "peak_vram_mb": peak_vram_mb,
            "avg_gpu_utilization_pct": (
                float(np.mean(self._gpu_utils)) if self._gpu_utils else float("nan")
            ),
            "peak_process_rss_mb": self._peak_rss / (1024**2),
            "avg_system_memory_pct": (
                float(np.mean(self._sys_mem_pcts)) if self._sys_mem_pcts else float("nan")
            ),
            "avg_nodes_per_batch": (
                float(np.mean(self._node_counts)) if self._node_counts else float("nan")
            ),
            "avg_edges_per_batch": (
                float(np.mean(self._edge_counts)) if self._edge_counts else float("nan")
            ),
        }


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
    return AIGDataModule(
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
        train_num_samples=args.num_sample_graphs,
        # Matches real training's batching *strategy* (node-budget batching,
        # since AIGs vary wildly in size) rather than defaulting to plain
        # fixed-count batching — otherwise this "controlled" benchmark would
        # measure a fundamentally different batching regime than production
        # training actually uses. --batch_size/--max_total_nodes_per_batch
        # are still held fixed identically across every config by
        # benchmark.sh, so the comparison stays controlled.
        dynamic_batching=args.dynamic_batching,
        max_total_nodes=args.max_total_nodes_per_batch,
        **reduction_kwargs,
    )


def main(args: argparse.Namespace) -> None:
    if args.algorithm not in config.VALID_ALGORITHMS:
        raise ValueError(
            f"Algorithm '{args.algorithm}' must be one of {config.VALID_ALGORITHMS}"
        )
    if args.reduction_type != "none" and not args.reduction_method:
        raise ValueError("--reduction_method is required when --reduction_type != none")

    # Same backend settings train.py uses, so the benchmark reflects real
    # training numerics rather than an artificially different configuration.
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    datamodule = build_datamodule(args)
    datamodule.setup("fit")

    model = build_model(compile_model=args.torch_compile)

    precision = _select_precision()
    accelerator, devices = _select_accelerator_and_devices()
    print(f"[benchmark] accelerator={accelerator} precision={precision}", flush=True)

    callback = BenchmarkCallback(
        num_warmup_steps=args.num_warmup_steps,
        num_measure_steps=args.num_measure_steps,
        gpu_util_sample_every=args.gpu_util_sample_every,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        limit_train_batches=args.num_warmup_steps + args.num_measure_steps,
        limit_val_batches=0,  # only training-step cost is being measured
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        callbacks=[callback],
    )

    trainer.fit(model, datamodule=datamodule)

    summary = callback.summary(model.device)
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reduction_type": args.reduction_type,
        "reduction_method": args.reduction_method or "",
        "algorithm": args.algorithm,
        "device": accelerator,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_sample_graphs": args.num_sample_graphs,
        "num_measure_steps": args.num_measure_steps,
        **summary,
        **reproducibility_metadata(args.seed),
    }
    append_csv_row(args.results_csv, row)
    print(
        f"[benchmark] avg_step_time_s={summary['avg_step_time_s']:.4f} "
        f"throughput={summary['throughput_graphs_per_s']:.1f} graphs/s "
        f"peak_vram_mb={summary['peak_vram_mb']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Controlled training-hardware benchmark on a seeded sample of graphs."
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
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument(
        "--dynamic_batching",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=config.DYNAMIC_BATCHING,
        help="Match real training's node-budget batching strategy (default: same as train.py).",
    )
    parser.add_argument(
        "--max_total_nodes_per_batch",
        type=int,
        default=config.MAX_TOTAL_NODES_PER_BATCH,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_sample_graphs", type=int, default=100)
    parser.add_argument("--num_warmup_steps", type=int, default=5)
    parser.add_argument("--num_measure_steps", type=int, default=30)
    parser.add_argument("--gpu_util_sample_every", type=int, default=1)
    parser.add_argument(
        "--torch_compile",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=config.TORCH_COMPILE,
        help=(
            "Match real training's torch.compile setting for fidelity. If "
            "enabled, consider raising --num_warmup_steps since compilation "
            "cost needs to be excluded from steady-state timing."
        ),
    )
    parser.add_argument("--results_csv", type=str, default="results/training_benchmark.csv")

    main(parser.parse_args())
