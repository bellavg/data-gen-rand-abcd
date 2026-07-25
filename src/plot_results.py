"""Static publication figures for the thesis Results chapter.

W&B stays useful for live dashboards during training runs, but its charts
aren't meant to be pulled into a LaTeX document as final figures. This
script reads the same CSVs as results_to_latex.py and writes PDF/PNG figures
under results/figures/, ready for \\includegraphics.

Reuses results_to_latex.py's CSV loaders and NaN-safe method-matching helper
rather than re-implementing the same joins here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display needed on a cluster or CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results_to_latex import (
    _method_matches,
    load_benchmark_per_graph,
    load_inference_results,
    load_offline_stats,
    load_training_benchmark,
)


def _config_label(reduction_type: str, reduction_method) -> str:
    if reduction_type == "none":
        return "baseline"
    return f"{reduction_method}\n({reduction_type})"


def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    # bbox_inches="tight" makes sure a legend placed outside the axes
    # (e.g. the Pareto front's side legend) isn't clipped off the saved image.
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_results] Wrote {path}")


def _bar_chart(labels: list[str], values: list[float], ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4))
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, path)


def plot_parity(predictions_dir: Path, figures_dir: Path) -> None:
    """Predicted-vs-actual scatter per (checkpoint, eval_mode) — a regression
    diagnostic a table of RMSE/R2 can't show: where and how predictions
    go wrong (e.g. systematic over/under-prediction at the extremes)."""
    csv_paths = sorted(predictions_dir.glob("*.csv"))
    if not csv_paths:
        print(f"[plot_results] No prediction files found in {predictions_dir} — skipping parity plots.")
        return
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if df.empty or "target" not in df.columns or "prediction" not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(df["target"], df["prediction"], s=8, alpha=0.4)
        lo = min(df["target"].min(), df["prediction"].min())
        hi = max(df["target"].max(), df["prediction"].max())
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Actual optimizability")
        ax.set_ylabel("Predicted optimizability")
        ax.set_title(csv_path.stem)
        _savefig(fig, figures_dir / f"{csv_path.stem}_parity.png")


def plot_rq2_hardware_bars(
    training_df: pd.DataFrame, inference_df: pd.DataFrame, figures_dir: Path
) -> None:
    """Peak VRAM / throughput bar charts, grouped by reduction config —
    the outline's "Bar charts comparing peak GPU Memory Allocated" and
    "Temporal Gains" subsections."""
    if training_df.empty:
        print("[plot_results] No training_benchmark.csv rows — skipping RQ2 hardware bar charts.")
        return

    configs = training_df[["reduction_type", "reduction_method"]].drop_duplicates().reset_index(drop=True)
    labels = [_config_label(r.reduction_type, r.reduction_method) for r in configs.itertuples()]

    train_vram, train_step_time, train_throughput = [], [], []
    infer_throughput, infer_vram = [], []

    for row in configs.itertuples():
        t_row = training_df[
            (training_df["reduction_type"] == row.reduction_type)
            & _method_matches(training_df["reduction_method"], row.reduction_method)
        ]
        train_vram.append(float(t_row["peak_vram_mb"].mean()) if not t_row.empty else np.nan)
        train_step_time.append(float(t_row["avg_step_time_s"].mean()) if not t_row.empty else np.nan)
        train_throughput.append(float(t_row["throughput_graphs_per_s"].mean()) if not t_row.empty else np.nan)

        infer_mode = "full_graph" if row.reduction_type == "none" else "matched_reduction"
        i_row = pd.DataFrame()
        if not inference_df.empty:
            i_row = inference_df[
                (inference_df["reduction_type"] == row.reduction_type)
                & _method_matches(inference_df["reduction_method"], row.reduction_method)
                & (inference_df["eval_mode"] == infer_mode)
                & (inference_df["device"] == "cuda")
            ]
        infer_throughput.append(float(i_row["throughput_graphs_per_s"].mean()) if not i_row.empty else np.nan)
        infer_vram.append(float(i_row["peak_vram_mb"].mean()) if not i_row.empty else np.nan)

    _bar_chart(labels, train_vram, "Peak VRAM (MB) — training", figures_dir / "rq2_train_peak_vram.png")
    _bar_chart(labels, train_step_time, "Avg training step time (s)", figures_dir / "rq2_train_step_time.png")
    _bar_chart(labels, train_throughput, "Training throughput (graphs/s)", figures_dir / "rq2_train_throughput.png")
    _bar_chart(labels, infer_throughput, "Inference throughput (graphs/s)", figures_dir / "rq2_infer_throughput.png")
    _bar_chart(labels, infer_vram, "Peak VRAM (MB) — inference", figures_dir / "rq2_infer_peak_vram.png")


def plot_rq2_offline_bars(
    sparsification_stats: pd.DataFrame, partition_stats: pd.DataFrame, figures_dir: Path
) -> None:
    """Offline reduction-algorithm wall-clock cost per method — the "Graph
    Reduction Offline Profile" numbers, distinct from GNN training/inference
    cost."""
    if sparsification_stats.empty and partition_stats.empty:
        print("[plot_results] No offline stats CSVs found — skipping RQ2 offline bar chart.")
        return

    labels: list[str] = []
    times_ms: list[float] = []
    if not sparsification_stats.empty:
        for method, grp in sparsification_stats.groupby("reduction_method"):
            labels.append(f"{method}\n(sparsification)")
            times_ms.append(float(grp["time_s"].mean()) * 1000)
    if not partition_stats.empty:
        for method, grp in partition_stats.groupby("reduction_method"):
            labels.append(f"{method}\n(partition)")
            times_ms.append(float(grp["time_s"].mean()) * 1000)

    _bar_chart(labels, times_ms, "Offline reduction wall-clock time (ms/graph)", figures_dir / "rq2_offline_time.png")


def plot_rq3_accuracy_bars(inference_df: pd.DataFrame, figures_dir: Path) -> None:
    """RMSE/R2 per reduced config vs. the full-graph baseline — the
    "Accuracy Degradation" chart (RQ3)."""
    if inference_df.empty:
        print("[plot_results] No inference_results.csv rows — skipping RQ3 accuracy bar chart.")
        return

    baseline = inference_df[
        (inference_df["reduction_type"] == "none")
        & (inference_df["eval_mode"] == "full_graph")
        & (inference_df["device"] == "cuda")
    ]
    baseline_rmse = float(baseline["rmse"].mean()) if not baseline.empty else np.nan
    baseline_r2 = float(baseline["r2"].mean()) if not baseline.empty else np.nan

    matched = inference_df[
        (inference_df["eval_mode"] == "matched_reduction") & (inference_df["device"] == "cuda")
    ]
    if matched.empty:
        print("[plot_results] No matched_reduction rows — skipping RQ3 accuracy bar chart.")
        return

    labels = [_config_label(r.reduction_type, r.reduction_method) for r in matched.itertuples()]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, matched["rmse"].to_numpy(), width, label="RMSE")
    ax.bar(x + width / 2, matched["r2"].to_numpy(), width, label="R2")
    if not np.isnan(baseline_rmse):
        ax.axhline(baseline_rmse, color="C0", linestyle="--", linewidth=1, label="Baseline RMSE")
    if not np.isnan(baseline_r2):
        ax.axhline(baseline_r2, color="C1", linestyle="--", linewidth=1, label="Baseline R2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Metric value")
    ax.set_title("Predictive retention vs. full-graph baseline (RQ3)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, figures_dir / "rq3_accuracy_degradation.png")


# Cycled per point instead of shared, so each config is identifiable by shape
# alone — the legend (placed outside the axes) then just maps shape -> name,
# instead of text labels crowding the plot near the axes.
_PARETO_MARKERS = ["*", "o", "s", "^", "D", "v", "P", "X", "h", "<", ">", "p"]


def plot_pareto_front(pareto_df: pd.DataFrame, figures_dir: Path) -> None:
    """Accuracy (RMSE) vs. training VRAM cost per config — RQ3's own
    suggested Pareto-front scatter, bridging RQ2 and RQ3."""
    if pareto_df.empty:
        print("[plot_results] No pareto_front.csv rows — skipping Pareto scatter.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, row in enumerate(pareto_df.itertuples()):
        label = _config_label(row.reduction_type, row.reduction_method).replace("\n", " ")
        is_baseline = row.reduction_type == "none"
        marker = _PARETO_MARKERS[i % len(_PARETO_MARKERS)]
        ax.scatter(
            row.train_peak_vram_mb,
            row.rmse,
            s=160 if is_baseline else 80,
            marker=marker,
            label=label,
        )
    ax.set_xlabel("Peak Training VRAM (MB)")
    ax.set_ylabel("RMSE")
    ax.set_title("Accuracy vs. Memory Cost — Pareto Front (RQ3)")
    ax.minorticks_on()
    ax.grid(which="major", alpha=0.3)
    ax.grid(which="minor", alpha=0.15, linestyle=":")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    _savefig(fig, figures_dir / "rq3_pareto_front.png")


def plot_benchmark_vram_vs_size(
    per_graph_df: pd.DataFrame, training_df: pd.DataFrame, figures_dir: Path
) -> None:
    """Peak training VRAM vs. graph size, one series per config. Node-reducing
    methods shift points down-and-left; the spread shows how per-graph memory
    scales with size — the quantity that determines whether a graph OOMs."""
    if per_graph_df.empty or per_graph_df["peak_vram_mb"].isna().all():
        print("[plot_results] No GPU per-graph VRAM data — skipping VRAM-vs-size scatter.")
        return

    label_map = {}
    if not training_df.empty:
        label_map = (
            training_df.drop_duplicates("run_label")
            .set_index("run_label")[["reduction_type", "reduction_method"]]
            .to_dict("index")
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, run_label in enumerate(sorted(per_graph_df["run_label"].unique())):
        grp = per_graph_df[per_graph_df["run_label"] == run_label]
        meta = label_map.get(run_label, {"reduction_type": "", "reduction_method": run_label})
        label = _config_label(meta["reduction_type"], meta["reduction_method"]).replace("\n", " ")
        ax.scatter(
            grp["num_nodes"], grp["peak_vram_mb"],
            s=18, alpha=0.5, marker=_PARETO_MARKERS[i % len(_PARETO_MARKERS)], label=label,
        )
    ax.set_xlabel("Graph size (nodes)")
    ax.set_ylabel("Peak training VRAM (MB)")
    ax.set_title("Per-graph training VRAM vs. graph size (RQ2)")
    ax.minorticks_on()
    ax.grid(which="major", alpha=0.3)
    ax.grid(which="minor", alpha=0.15, linestyle=":")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    _savefig(fig, figures_dir / "rq2_vram_vs_size.png")


def plot_benchmark_savings_vs_size(
    per_graph_df: pd.DataFrame, training_df: pd.DataFrame, figures_dir: Path
) -> None:
    """Per-graph VRAM saving vs. full-graph size, paired against baseline by
    graph_id. The headline plot: reduction saves most on the largest graphs —
    exactly where OOM happens."""
    if per_graph_df.empty or training_df.empty:
        print("[plot_results] No per-graph/benchmark data — skipping savings-vs-size scatter.")
        return

    baseline_labels = training_df[training_df["reduction_type"] == "none"]["run_label"].unique()
    if len(baseline_labels) == 0:
        print("[plot_results] No baseline run_label — skipping savings-vs-size scatter.")
        return
    baseline_label = baseline_labels[0]
    base = per_graph_df[per_graph_df["run_label"] == baseline_label][
        ["graph_id", "num_nodes", "peak_vram_mb"]
    ].rename(columns={"num_nodes": "base_nodes", "peak_vram_mb": "base_vram"})
    if base.empty or base["base_vram"].isna().all():
        print("[plot_results] No GPU baseline VRAM — skipping savings-vs-size scatter.")
        return

    label_map = (
        training_df.drop_duplicates("run_label")
        .set_index("run_label")[["reduction_type", "reduction_method"]]
        .to_dict("index")
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for i, run_label in enumerate(sorted(per_graph_df["run_label"].unique())):
        if run_label == baseline_label:
            continue
        grp = per_graph_df[per_graph_df["run_label"] == run_label][["graph_id", "peak_vram_mb"]]
        merged = grp.merge(base, on="graph_id")
        if merged.empty or merged["base_vram"].isna().all():
            continue
        saving_pct = (1 - merged["peak_vram_mb"] / merged["base_vram"]) * 100
        meta = label_map.get(run_label, {"reduction_type": "", "reduction_method": run_label})
        label = _config_label(meta["reduction_type"], meta["reduction_method"]).replace("\n", " ")
        ax.scatter(
            merged["base_nodes"], saving_pct,
            s=18, alpha=0.5, marker=_PARETO_MARKERS[(i + 1) % len(_PARETO_MARKERS)], label=label,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        print("[plot_results] No reduced configs paired with baseline — skipping savings-vs-size scatter.")
        return

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Full-graph size (nodes)")
    ax.set_ylabel("Training VRAM saving vs. baseline (%)")
    ax.set_title("Per-graph VRAM saving vs. graph size (RQ2/RQ3)")
    ax.minorticks_on()
    ax.grid(which="major", alpha=0.3)
    ax.grid(which="minor", alpha=0.15, linestyle=":")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    _savefig(fig, figures_dir / "rq2_vram_saving_vs_size.png")


def plot_rq4_cross_state_bars(inference_df: pd.DataFrame, figures_dir: Path) -> None:
    """Matched-state vs. cross-state RMSE per reduced config — makes the
    "performance drop-off" comparison (RQ4) visual rather than only tabular."""
    if inference_df.empty:
        print("[plot_results] No inference_results.csv rows — skipping RQ4 bar chart.")
        return

    reduced = inference_df[(inference_df["reduction_type"] != "none") & (inference_df["device"] == "cuda")]
    matched = reduced[reduced["eval_mode"] == "matched_reduction"].set_index(
        ["reduction_type", "reduction_method"]
    )
    cross = reduced[reduced["eval_mode"] == "full_graph"].set_index(
        ["reduction_type", "reduction_method"]
    )

    keys = [k for k in matched.index if k in cross.index]
    if not keys:
        print("[plot_results] No matched configs with both eval modes — skipping RQ4 bar chart.")
        return

    labels = [_config_label(k[0], k[1]) for k in keys]
    matched_rmse = [matched.loc[k, "rmse"] for k in keys]
    cross_rmse = [cross.loc[k, "rmse"] for k in keys]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, matched_rmse, width, label="Matched-state RMSE")
    ax.bar(x + width / 2, cross_rmse, width, label="Cross-state (full-graph) RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_title("Matched-state vs. cross-state generalization (RQ4)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, figures_dir / "rq4_cross_state.png")


def main(args: argparse.Namespace) -> None:
    inference_df = load_inference_results(Path(args.inference_dir))
    training_df = load_training_benchmark(Path(args.training_dir))
    sparsification_stats = load_offline_stats(Path(args.logs_dir), "sparsification_stats")
    partition_stats = load_offline_stats(Path(args.logs_dir), "partition_stats")
    per_graph_df = load_benchmark_per_graph(Path(args.per_graph_dir))

    figures_dir = Path(args.figures_dir)

    plot_parity(Path(args.predictions_dir), figures_dir)
    plot_rq2_hardware_bars(training_df, inference_df, figures_dir)
    plot_rq2_offline_bars(sparsification_stats, partition_stats, figures_dir)
    plot_benchmark_vram_vs_size(per_graph_df, training_df, figures_dir)
    plot_benchmark_savings_vs_size(per_graph_df, training_df, figures_dir)
    plot_rq3_accuracy_bars(inference_df, figures_dir)
    plot_rq4_cross_state_bars(inference_df, figures_dir)

    pareto_path = Path(args.tables_dir) / "pareto_front.csv"
    pareto_df = pd.read_csv(pareto_path) if pareto_path.is_file() else pd.DataFrame()
    plot_pareto_front(pareto_df, figures_dir)

    print(f"[plot_results] Done. Figures in {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate static publication figures from results/ + logs/ CSVs."
    )
    parser.add_argument("--inference_dir", type=str, default="results/inference_results")
    parser.add_argument("--training_dir", type=str, default="results/training_benchmark")
    parser.add_argument("--logs_dir", type=str, default="logs")
    parser.add_argument("--predictions_dir", type=str, default="results/predictions")
    parser.add_argument("--per_graph_dir", type=str, default="results/benchmark_per_graph")
    parser.add_argument("--tables_dir", type=str, default="results/tables")
    parser.add_argument("--figures_dir", type=str, default="results/figures")

    main(parser.parse_args())
