"""RQ5 figures: train on reduced graphs, query on full ones.

Each trained configuration is evaluated twice on the same test set -- once
under the reduction it was trained with (matched state) and once on unreduced
graphs (full graph). The difference between the two passes isolates the cost of
the structural shift from the cost of the reduction itself, which is the whole
of RQ5.

One asymmetry belongs next to every figure here. For the standard-track methods
a full graph needs no conversion: it already is a valid graph of size-1
super-nodes. The exact colour-refinement track uses a different input schema and
would need full graphs converted before it could be queried on them -- so the
two tracks are not the same experiment. That track has not been run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from analysis.fake_data import CPU_INFERENCE, TODO_CPU_INFERENCE, SUMMARIZATION
from analysis.style import (
    COL,
    FAKE_COLOR,
    FAMILY_COLORS,
    INK_MUTED,
    INK_SECONDARY,
    WIDE,
    color_for,
    hatch_for,
    label_for,
    mark_fake,
    meta,
    row_height,
    savefig,
    sort_key,
    style_axes,
)

METRICS = [("r2", "$R^2$"), ("rmse", "RMSE"), ("spearman", "Spearman $\\rho$")]


def cross_state(inference: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration with both evaluation passes side by side."""
    test = inference[inference["split"] == "test"].copy()
    test["reduction_method"] = test["reduction_method"].fillna("none")
    wide = test.pivot_table(
        index="reduction_method",
        columns="eval_mode",
        values=["r2", "rmse", "spearman", "throughput_graphs_per_s", "peak_vram_mb",
                "avg_nodes_per_graph", "avg_edges_per_graph"],
    )
    wide.columns = [f"{mode}_{metric}" for metric, mode in wide.columns]
    wide = wide.reset_index()
    return wide.sort_values("reduction_method", key=lambda s: s.map(sort_key))


def matched_vs_full(cross: pd.DataFrame, out: Path) -> None:
    """The headline RQ5 exhibit: each configuration's matched-state score and
    its full-graph score, joined."""
    df = cross[cross["reduction_method"] != "none"].copy()
    baseline = cross[cross["reduction_method"] == "none"]

    fig, axes = plt.subplots(1, 3, figsize=(6.9, row_height(len(df), base=1.8)), sharey=True)
    y = np.arange(len(df))

    for ax, (col, name) in zip(axes, METRICS):
        m_col, f_col = f"matched_reduction_{col}", f"full_graph_{col}"
        for yi, (_, row) in enumerate(df.iterrows()):
            method = row["reduction_method"]
            worse = row[f_col] < row[m_col] if col != "rmse" else row[f_col] > row[m_col]
            ax.plot([row[m_col], row[f_col]], [yi, yi],
                    color=FAKE_COLOR if worse else INK_MUTED, lw=1.5, alpha=0.7)
            ax.plot(row[m_col], yi, "o", ms=6, color=color_for(method),
                    markerfacecolor="white", markeredgewidth=1.5)
            ax.plot(row[f_col], yi, "o", ms=6, color=color_for(method))
        if not baseline.empty:
            ax.axvline(baseline[f_col].iloc[0], color=INK_SECONDARY, lw=1.0, ls="--")
        ax.axvline(0, color=INK_SECONDARY, lw=0.7)
        ax.set_title(name)
        style_axes(ax, grid_axis="x")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([label_for(m) for m in df["reduction_method"]], fontsize=7)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Matched-state (open) vs. full-graph (filled) evaluation of the same weights.\n"
        "Dashed: the full-graph baseline. A red connector means the shift hurt.",
        y=1.11,
    )
    savefig(fig, out, "rq5_matched_vs_full")


def dropoff_by_family(cross: pd.DataFrame, out: Path) -> None:
    """The drop-off itself, grouped by reduction family.

    Node-count-preserving methods (edge dropout, spanning forest, every
    partitioner) change the graph in a different way from node-removing ones
    (PageRank, AND-gate-only), so the family is the first place to look for a
    pattern in who survives the shift.

    $\\Delta R^2$ and $\\Delta$ Spearman get separate axes rather than one
    shared scale, since a bar chart of both on the same axis invites reading
    them as though they were the same quantity.
    """
    df = cross[cross["reduction_method"] != "none"].copy()
    df["delta_r2"] = df["full_graph_r2"] - df["matched_reduction_r2"]
    df["delta_spearman"] = df["full_graph_spearman"] - df["matched_reduction_spearman"]
    df = df.sort_values("delta_r2")
    methods = df["reduction_method"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    x = np.arange(len(df))
    for ax, (col, name) in zip(
        axes, [("delta_r2", "$\\Delta R^2$"), ("delta_spearman", "$\\Delta$ Spearman")]
    ):
        bars = ax.bar(x, df[col], width=0.55)
        for bar, method in zip(bars, methods):
            bar.set_facecolor(color_for(method))
            hatch = hatch_for(method)
            if hatch:
                bar.set_hatch(hatch)
                bar.set_edgecolor("white")
        ax.axhline(0, color=INK_SECONDARY, lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([label_for(m, short=True) for m in methods], fontsize=6.5, rotation=20)
        ax.set_title(name)
        style_axes(ax)
    axes[0].set_ylabel("Full-graph score $-$ matched-state score")
    fig.suptitle(
        "Cost of the structural shift (positive = the model does BETTER on full graphs)",
        y=1.04,
    )
    savefig(fig, out, "rq5_dropoff")


def transfer_quadrant(cross: pd.DataFrame, out: Path) -> None:
    """Matched-state score against full-graph score, one point per method.

    The diagonal is perfect transfer. Points below it lost accuracy in the
    shift; points above it gained, which would mean the reduction was hurting
    the model rather than the evaluation.
    """
    df = cross.copy()
    fig, ax = plt.subplots(figsize=COL)
    lo = min(df["matched_reduction_r2"].min(), df["full_graph_r2"].min(), 0) - 0.2
    hi = max(df["matched_reduction_r2"].max(), df["full_graph_r2"].max()) + 0.2
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, lw=1.0, ls="--", label="perfect transfer")
    ax.fill_between([lo, hi], [lo, hi], hi, color=FAMILY_COLORS["summarization"], alpha=0.07)
    ax.fill_between([lo, hi], lo, [lo, hi], color=FAKE_COLOR, alpha=0.05)

    for _, row in df.iterrows():
        method = row["reduction_method"]
        if pd.isna(row["matched_reduction_r2"]):
            continue
        info = meta(method)
        ax.plot(
            row["matched_reduction_r2"], row["full_graph_r2"],
            marker="o", ms=7, color=color_for(method),
            markerfacecolor=color_for(method) if info["domain"] else "white",
            markeredgewidth=1.4,
        )
        ax.annotate(label_for(method, short=True),
                    (row["matched_reduction_r2"], row["full_graph_r2"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=6, color=INK_SECONDARY)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Matched-state $R^2$ (trained and tested reduced)")
    ax.set_ylabel("Full-graph $R^2$ (zero-shot on unreduced)")
    ax.set_title("Transfer from reduced training to full-graph inference")
    ax.legend(loc="lower right", fontsize=6.5)
    style_axes(ax, grid_axis="both")
    savefig(fig, out, "rq5_transfer_quadrant")


def inference_cost(cross: pd.DataFrame, out: Path) -> None:
    """The practical pay-off: inference stores no gradients or activations, so
    the memory argument that forced the reduction at training time may not bind
    at inference time at all."""
    df = cross.copy()
    methods = df["reduction_method"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    x = np.arange(len(df))
    width = 0.38

    for ax, stem, ylabel, title in [
        (axes[0], "throughput_graphs_per_s", "Graphs per second",
         "Inference throughput"),
        (axes[1], "peak_vram_mb", "Peak memory (MB)", "Inference peak memory"),
    ]:
        for i, (mode, name) in enumerate(
            [("matched_reduction", "on reduced graphs"), ("full_graph", "on full graphs")]
        ):
            col = f"{mode}_{stem}"
            bars = ax.bar(x + (i - 0.5) * width, df[col], width=width * 0.9,
                          alpha=1.0 - 0.3 * i, label=name)
            for bar, method in zip(bars, methods):
                bar.set_facecolor(color_for(method))
                hatch = hatch_for(method)
                if hatch:
                    bar.set_hatch(hatch)
                    bar.set_edgecolor("white")
        ax.set_xticks(x)
        ax.set_xticklabels([label_for(m, short=True) for m in methods], fontsize=6.5, rotation=20)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=6.5)
        style_axes(ax)

    fig.suptitle("Cost of serving a query, by evaluation state", y=1.04)
    savefig(fig, out, "rq5_inference_cost")


def cpu_inference(out: Path) -> None:
    """The "infer full on modest hardware" half of RQ5, which has no data.

    Every surviving inference CSV records device=cuda, so the CPU column that
    the practical claim rests on does not exist yet.
    """
    df = CPU_INFERENCE.copy()
    # The raw device column names an implementation (cuda); the figure names
    # the device class instead, matching the abstraction used everywhere else.
    device_label = {"cuda": "accelerator", "cpu": "CPU"}
    fig, ax = plt.subplots(figsize=COL)
    x = np.arange(len(df))
    colors = [
        color_for(m) if real else FAKE_COLOR
        for m, real in zip(df["reduction_method"], df["measured"])
    ]
    bars = ax.bar(x, df["throughput_graphs_per_s"], color=colors, width=0.66)
    for bar, real in zip(bars, df["measured"]):
        if not real:
            bar.set_hatch("xxx")
            bar.set_edgecolor("white")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{label_for(m, short=True)}\n{device_label.get(d, d)}"
            + ("" if real else "\n[FAKE]")
            for m, d, real in zip(df["reduction_method"], df["device"], df["measured"])
        ],
        fontsize=6,
    )
    ax.set_ylabel("Graphs per second")
    ax.set_title("Full-graph inference throughput, accelerator vs. CPU")
    style_axes(ax)
    # Five of the six bars are invented, so this gets the full watermark
    # rather than the per-row marking used on mostly-measured figures.
    mark_fake(fig, ax, note=TODO_CPU_INFERENCE)
    savefig(fig, out, "rq5_cpu_inference")


def positive_control(out: Path) -> None:
    """The RQ5 positive control, which has not been run.

    Colour refinement at count-cap infinity merges only nodes the encoder
    provably cannot distinguish, so a model trained on those graphs MUST score
    identically to the full-graph baseline when queried on full graphs. Without
    it, a poor transfer elsewhere cannot be told apart from a broken evaluation
    path -- which is why this figure is a placeholder rather than an omission.
    """
    fake = SUMMARIZATION.set_index("reduction_method")
    methods = ["wl", "mffc", "cone", "convmatch", "random_merge"]
    fig, ax = plt.subplots(figsize=COL)
    y = np.arange(len(methods))
    matched = [fake.loc[m, "r2"] for m in methods]
    full = [fake.loc[m, "cross_state_r2"] for m in methods]
    for yi, m in enumerate(methods):
        ax.plot([matched[yi], full[yi]], [yi, yi], color=INK_MUTED, lw=1.4)
        ax.plot(matched[yi], yi, "o", ms=6, color=FAKE_COLOR, markerfacecolor="white",
                markeredgewidth=1.5)
        ax.plot(full[yi], yi, "o", ms=6, color=FAKE_COLOR)
    ax.axvline(0.342838, color=INK_SECONDARY, lw=1.2, ls="--")
    ax.text(0.342838, len(methods) - 0.4, " measured full-graph baseline",
            fontsize=6, color=INK_SECONDARY, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels([label_for(m) for m in methods], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("$R^2$ (open = matched state, filled = full graph)")
    ax.set_title("Summarization cross-state transfer")
    style_axes(ax, grid_axis="x")
    mark_fake(
        fig, ax,
        note="Colour refinement at count-cap infinity MUST land on the dashed line. "
             "Nothing here has been run.",
    )
    savefig(fig, out, "rq5_positive_control")


def build(inference: pd.DataFrame, out: Path) -> pd.DataFrame:
    cross = cross_state(inference)
    matched_vs_full(cross, out)
    dropoff_by_family(cross, out)
    transfer_quadrant(cross, out)
    inference_cost(cross, out)
    cpu_inference(out)
    positive_control(out)
    return cross
