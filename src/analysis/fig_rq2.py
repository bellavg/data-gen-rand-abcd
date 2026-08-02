"""RQ2 figures: what each reduction costs and what it buys.

Three groups. Offline: how much the reduction shrinks the graph and what it
costs to compute. Online: peak memory and step time under training, both as
per-config aggregates and as a function of graph size. Ranking: the same data
cut one row per method.

The summarization family appears in every method-level figure here as
fabricated rows, because none of its methods has been run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from analysis.fake_data import SUMMARIZATION, TODO_SUMMARIZATION
from analysis.results_to_latex import load_offline_stats
from analysis.style import (
    COL,
    FAKE_COLOR,
    FAMILY_COLORS,
    INK_MUTED,
    INK_SECONDARY,
    WIDE,
    color_for,
    family_legend,
    hatch_for,
    label_for,
    mark_fake,
    meta,
    row_height,
    savefig,
    sort_key,
    style_axes,
)

BASE = FAMILY_COLORS["baseline"]


def _paint(bars, methods) -> None:
    """Colour by family, hatch by domain-informed / fabricated."""
    for bar, method in zip(bars, methods):
        bar.set_facecolor(color_for(method))
        hatch = hatch_for(method)
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor("white")
        if not meta(method)["measured"]:
            bar.set_facecolor(FAKE_COLOR)


def with_summarization(offline: pd.DataFrame) -> pd.DataFrame:
    """Measured offline stats plus the fabricated summarization rows."""
    fake = SUMMARIZATION[
        ["reduction_method", "reduction_type", "node_retention", "edge_retention", "offline_s"]
    ].assign(node_retention_std=np.nan, edge_retention_std=np.nan, offline_s_std=np.nan)
    combined = pd.concat([offline.assign(measured=True), fake.assign(measured=False)])
    return combined.sort_values("reduction_method", key=lambda s: s.map(sort_key))


# --- Offline profile ---------------------------------------------------------
def retention_bars(offline: pd.DataFrame, out: Path) -> None:
    """Node and edge retention side by side, never as one compression number.

    Partitioning keeps every node by construction, so its node bar is pinned at
    1.0; reading a single "compression" figure across the two families would
    hide exactly that.
    """
    df = with_summarization(offline)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, row_height(len(df))), sharey=True)
    y = np.arange(len(df))
    methods = df["reduction_method"].tolist()

    for ax, col, std_col, name in [
        (axes[0], "node_retention", "node_retention_std", "Node retention"),
        (axes[1], "edge_retention", "edge_retention_std", "Edge retention"),
    ]:
        bars = ax.barh(y, df[col], xerr=df[std_col], height=0.68, error_kw={"lw": 0.8, "ecolor": INK_MUTED})
        _paint(bars, methods)
        ax.axvline(1.0, color=INK_SECONDARY, lw=1.0, ls="--")
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Fraction retained (mean $\\pm$ sd over 10,000 graphs)")
        ax.set_title(name)
        style_axes(ax, grid_axis="x")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [label_for(m) + ("  [FAKE]" if not meta(m)["measured"] else "") for m in methods],
        fontsize=7,
    )
    axes[0].invert_yaxis()
    family_legend(
        axes[1],
        ["partition", "sparsification"],
        include_fake=True,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    fig.suptitle("Offline compression: node and edge retention are separate quantities", y=1.04)
    mark_fake(fig, note="summarization rows only — " + TODO_SUMMARIZATION, watermark=False)
    savefig(fig, out, "rq2_retention")


def retention_distribution(measurements_dir: Path, out: Path) -> None:
    """Per-graph spread, not just the mean.

    A parameter-free method's compression is a property of each circuit rather
    than a setting, so its spread is the honest description. PageRank's node
    retention is nearly a point mass at its configured 0.8; AND-gate-only's is
    not, and that is what makes the pairing between them approximate.
    """
    sparse = load_offline_stats(measurements_dir, "sparsification_stats")
    part = load_offline_stats(measurements_dir, "partition_stats")
    if not part.empty:
        part = part.assign(edge_retention=1.0 - part["edge_cut_ratio"], node_retention=1.0)
    rows = pd.concat([sparse, part], ignore_index=True)
    methods = sorted(set(rows["reduction_method"]), key=sort_key)

    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    for ax, col, name in [
        (axes[0], "edge_retention", "Edge retention"),
        (axes[1], "node_retention", "Node retention"),
    ]:
        for method in methods:
            values = np.sort(rows.loc[rows["reduction_method"] == method, col].to_numpy())
            ax.plot(
                values,
                np.linspace(0, 1, len(values)),
                color=color_for(method),
                ls="--" if meta(method)["domain"] else "-",
                lw=1.3,
            )
        ax.set_xlabel(f"{name} per graph")
        ax.set_ylabel("Cumulative fraction of graphs")
        ax.set_title(f"{name}: distribution over 10,000 graphs")
        style_axes(ax, grid_axis="both")

    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            color=color_for(m),
            ls="--" if meta(m)["domain"] else "-",
            label=label_for(m),
        )
        for m in methods
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=6)
    fig.suptitle("Compression is a distribution, not a single ratio", y=1.04)
    savefig(fig, out, "rq2_retention_distribution")


def offline_cost(offline: pd.DataFrame, out: Path) -> None:
    df = with_summarization(offline)
    fig, ax = plt.subplots(figsize=COL)
    y = np.arange(len(df))
    methods = df["reduction_method"].tolist()
    bars = ax.barh(y, df["offline_s"], height=0.68)
    _paint(bars, methods)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [label_for(m, short=True) + ("*" if not meta(m)["measured"] else "") for m in methods],
        fontsize=6.5,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Offline wall-clock per graph (s, log)")
    ax.set_title("Cost of computing the reduction")
    style_axes(ax, grid_axis="x")
    mark_fake(fig, ax, note="* summarization rows fabricated", watermark=False)
    savefig(fig, out, "rq2_offline_cost")


def amortisation(offline: pd.DataFrame, savings: pd.DataFrame, out: Path) -> None:
    """How many epochs over the corpus a cached reduction must serve before it
    pays for itself.

    The offline cost is paid once per graph; the saving is per epoch. A method
    that costs seconds per graph and saves milliseconds per step is only worth
    it because the artifact is cached and reused, and this figure is what makes
    that argument quantitative rather than rhetorical.
    """
    merged = offline.merge(savings, on="reduction_method", how="inner")
    merged = merged[merged["mean_time_saving_pct"] > 0].copy()
    if merged.empty:
        print("[rq2] no positive time savings — skipping amortisation figure.")
        return

    baseline_step_s = savings.attrs.get("baseline_step_s", 0.0408)
    merged["saving_per_graph_s"] = baseline_step_s * merged["mean_time_saving_pct"] / 100.0
    merged["runs_to_break_even"] = merged["offline_s"] / merged["saving_per_graph_s"]
    merged = merged.sort_values("runs_to_break_even")

    fig, ax = plt.subplots(figsize=COL)
    y = np.arange(len(merged))
    methods = merged["reduction_method"].tolist()
    bars = ax.barh(y, merged["runs_to_break_even"], height=0.68)
    _paint(bars, methods)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([label_for(m, short=True) for m in methods], fontsize=6.5)
    ax.invert_yaxis()
    for yi, v in enumerate(merged["runs_to_break_even"]):
        ax.text(v, yi, f" {v:,.0f}", va="center", fontsize=6.5, color=INK_SECONDARY)
    ax.set_xlim(right=merged["runs_to_break_even"].max() * 4)
    ax.set_xlabel("Epochs over the corpus before the reduction pays for itself (log)")
    ax.set_title("Amortisation threshold")
    style_axes(ax, grid_axis="x")
    savefig(fig, out, "rq2_amortisation")


# --- Online profile ----------------------------------------------------------
def vram_bars(bench: pd.DataFrame, out: Path) -> None:
    """Peak allocated VRAM per configuration, at three order statistics.

    The mean, p95 and max disagree by an order of magnitude, and the reason is
    the point of the figure: a graph larger than the batch budget forms a batch
    of its own, so the maximum is floored by the largest circuit in the corpus
    no matter what the reduction did to the average one.
    """
    df = bench.sort_values("reduction_method", key=lambda s: s.map(sort_key)).copy()
    methods = [m if isinstance(m, str) else "none" for m in df["reduction_method"]]
    base = df[df["reduction_type"] == "none"].iloc[0]

    fig, ax = plt.subplots(figsize=WIDE)
    x = np.arange(len(df))
    width = 0.26
    for i, (col, name, alpha) in enumerate(
        [
            ("peak_vram_allocated_mean_mb", "mean", 1.0),
            ("peak_vram_allocated_p95_mb", "p95", 0.66),
            ("peak_vram_allocated_max_mb", "max", 0.36),
        ]
    ):
        bars = ax.bar(x + (i - 1) * width, df[col], width=width * 0.92, label=name)
        for bar, method in zip(bars, methods):
            bar.set_facecolor(color_for(method))
            bar.set_alpha(alpha)
            hatch = hatch_for(method)
            if hatch:
                bar.set_hatch(hatch)
                bar.set_edgecolor("white")
        ax.axhline(base[col], color=INK_SECONDARY, lw=0.8, ls=":" if i else "--")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([label_for(m, short=True) for m in methods], fontsize=6.5, rotation=20)
    ax.set_ylabel("Peak allocated VRAM (MB, log)")
    ax.set_title(
        "Peak training memory: the maximum is set by the largest circuit, not by the mean"
    )
    ax.legend(
        title="order statistic (lighter = higher)",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncols=3,
        title_fontsize=7,
    )
    style_axes(ax)
    savefig(fig, out, "rq2_vram")


def throughput_bars(bench: pd.DataFrame, out: Path) -> None:
    df = bench.sort_values("reduction_method", key=lambda s: s.map(sort_key)).copy()
    methods = [m if isinstance(m, str) else "none" for m in df["reduction_method"]]
    base = df[df["reduction_type"] == "none"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=WIDE)

    ax = axes[0]
    x = np.arange(len(df))
    bars = ax.bar(x, df["avg_step_time_s"], yerr=df["std_step_time_s"], width=0.66,
                  error_kw={"lw": 0.8, "ecolor": INK_MUTED})
    _paint(bars, methods)
    ax.axhline(base["avg_step_time_s"], color=INK_SECONDARY, lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([label_for(m, short=True) for m in methods], fontsize=6.5, rotation=25)
    ax.set_ylabel("Mean step time (s)")
    ax.set_title("Training step time (dashed: full-graph baseline)")
    style_axes(ax)

    ax = axes[1]
    speedup = base["avg_step_time_s"] / df["avg_step_time_s"]
    bars = ax.bar(x, speedup, width=0.66)
    _paint(bars, methods)
    ax.axhline(1.0, color=INK_SECONDARY, lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([label_for(m, short=True) for m in methods], fontsize=6.5, rotation=25)
    ax.set_ylabel("Speedup vs. full graph ($\\times$)")
    ax.set_title("Speedup under-delivers relative to compression")
    style_axes(ax)

    fig.suptitle("Training throughput across reduction methods", y=1.04)
    savefig(fig, out, "rq2_throughput")


def cost_vs_size(per_graph: pd.DataFrame, out: Path) -> None:
    """Peak memory and step time as a function of graph size.

    Reporting a single peak per configuration conflates the reduction's effect
    with which graphs happened to land in the peak batch; binning by node count
    separates the two.
    """
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.7))
    fig.subplots_adjust(wspace=0.42)
    labels = per_graph["run_label"].unique()

    # Bin on the graph's UNREDUCED size, joined by graph_id, not on the size it
    # happens to have after this method ran. The node-removing methods
    # (PageRank, AND-gate-only) shrink num_nodes, so binning on the recorded
    # value would put the same circuit in different bins for different methods
    # and the ratio panels below would compare different circuits.
    baseline_label = next(lbl for lbl in labels if _label_to_method(lbl) == "none")
    base_size = (
        per_graph[per_graph["run_label"] == baseline_label]
        .groupby("graph_id")["num_nodes"]
        .median()
        .rename("base_nodes")
    )
    bins = np.logspace(np.log10(max(base_size.min(), 1)), np.log10(base_size.max()), 14)

    binned = {}
    for label in labels:
        method = _label_to_method(label)
        sub = per_graph[per_graph["run_label"] == label].join(base_size, on="graph_id")
        sub = sub.assign(bin=pd.cut(sub["base_nodes"], bins))
        # observed=False keeps every configuration on the same bin index, so an
        # empty bin becomes NaN — a gap in the line — rather than a shorter
        # series that silently misaligns against the baseline in the ratio
        # panels below.
        binned[method] = sub.groupby("bin", observed=False).agg(
            nodes=("base_nodes", "median"),
            vram=("peak_vram_allocated_mb", "median"),
            time=("step_time_s", "median"),
        )
    base = binned["none"]

    handles = []
    from matplotlib.lines import Line2D

    for method in sorted(binned, key=sort_key):
        grouped = binned[method]
        kw = {
            "color": color_for(method),
            "lw": 1.9 if method == "none" else 1.1,
            "ls": "--" if meta(method)["domain"] else "-",
        }
        axes[0].plot(grouped["nodes"], grouped["vram"], **kw)
        axes[1].plot(grouped["nodes"], grouped["vram"] / base["vram"], **kw)
        axes[2].plot(grouped["nodes"], grouped["time"] / base["time"], **kw)
        handles.append(Line2D([], [], **kw, label=label_for(method)))

    axes[0].set_yscale("log")
    axes[0].set_ylabel("Peak allocated VRAM (MB, log)")
    axes[0].set_title("Absolute memory:\nevery method traces one curve")
    axes[1].set_ylabel("Peak VRAM $\\div$ baseline")
    axes[1].set_title("Memory relative\nto the full graph")
    axes[2].set_ylabel("Step time $\\div$ baseline")
    axes[2].set_title("Step time relative\nto the full graph")
    for ax in axes[1:]:
        ax.axhline(1.0, color=INK_SECONDARY, lw=0.9, ls="--")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Unreduced nodes per graph (log)")
        style_axes(ax, grid_axis="both")

    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncols=5, fontsize=6.5
    )
    fig.suptitle(
        "Per-graph cost profile (median within log-spaced size bins): "
        "memory tracks graph size, not the reduction",
        y=1.06,
    )
    savefig(fig, out, "rq2_cost_vs_size")


def _label_to_method(run_label: str) -> str:
    from analysis.style import METHODS

    for key in sorted(METHODS, key=len, reverse=True):
        if key != "none" and run_label.endswith(key):
            return key
    return "none"


def paired_savings(savings: pd.DataFrame, out: Path) -> None:
    """Per-graph savings against the baseline, with bootstrap CIs.

    The benchmark measures one graph per batch on the same seeded sample for
    every configuration, so each reduced graph pairs with its own full-size
    version. Bare means would invite the question of whether a saving is inside
    the noise; the interval and the Wilcoxon p answer it.
    """
    df = savings.sort_values("mean_vram_saving_pct")
    methods = df["reduction_method"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True)

    for ax, mean_col, lo_col, hi_col, p_col, name in [
        (axes[0], "mean_vram_saving_pct", "vram_saving_ci_low_pct",
         "vram_saving_ci_high_pct", "vram_saving_wilcoxon_p", "Peak VRAM saving"),
        (axes[1], "mean_time_saving_pct", "time_saving_ci_low_pct",
         "time_saving_ci_high_pct", "time_saving_wilcoxon_p", "Step-time saving"),
    ]:
        y = np.arange(len(df))
        err = np.vstack([df[mean_col] - df[lo_col], df[hi_col] - df[mean_col]])
        for yi, method in zip(y, methods):
            ax.plot([0, df[mean_col].iloc[yi]], [yi, yi], color=color_for(method), lw=1.0, alpha=0.4)
        ax.errorbar(
            df[mean_col], y, xerr=err, fmt="none", ecolor=INK_SECONDARY, elinewidth=1.0, capsize=2
        )
        for yi, method in zip(y, methods):
            ax.plot(
                df[mean_col].iloc[yi], yi, marker="o", ms=6,
                color=color_for(method),
                markeredgecolor="white", markeredgewidth=0.8,
            )
            p = df[p_col].iloc[yi]
            if p > 0.05:
                ax.text(df[mean_col].iloc[yi], yi + 0.28, f"p={p:.2f}", fontsize=6, color=FAKE_COLOR)
        ax.axvline(0, color=INK_SECONDARY, lw=0.9, ls="--")
        ax.set_xlabel(f"{name} vs. full graph (%)")
        ax.set_title(name)
        style_axes(ax, grid_axis="x")

    axes[0].set_yticks(np.arange(len(df)))
    axes[0].set_yticklabels([label_for(m) for m in methods], fontsize=7)
    family_legend(axes[1], ["partition", "sparsification"], include_domain=False, loc="lower right")
    fig.suptitle(
        "Paired per-graph savings (mean, 95% bootstrap CI); "
        "a red $p$ marks a saving inside the noise",
        y=1.04,
    )
    savefig(fig, out, "rq2_paired_savings")


def partition_balance(measurements_dir: Path, out: Path) -> None:
    """The four partitioners differ only in which edges they cut, so cut ratio
    against partition balance is the whole comparison in one plane."""
    part = load_offline_stats(measurements_dir, "partition_stats")
    if part.empty:
        return
    agg = part.groupby("reduction_method").agg(
        cut=("edge_cut_ratio", "mean"),
        cut_sd=("edge_cut_ratio", "std"),
        balance=("std_nodes_per_partition", "mean"),
        k=("num_partitions", "mean"),
        seconds=("time_s", "mean"),
    )

    fig, ax = plt.subplots(figsize=COL)
    for method, row in agg.iterrows():
        ax.errorbar(
            row["balance"], row["cut"], yerr=row["cut_sd"],
            fmt="o", ms=8, color=color_for(method), ecolor=INK_MUTED, elinewidth=0.8,
            markeredgecolor="white", markeredgewidth=0.8,
            markerfacecolor="white" if not meta(method)["domain"] else color_for(method),
        )
        ax.annotate(
            label_for(method),
            (row["balance"], row["cut"]),
            textcoords="offset points", xytext=(7, 3), fontsize=6.5, color=INK_SECONDARY,
        )
    ax.set_xlabel("Partition imbalance (sd of nodes per partition)")
    ax.set_ylabel("Edge cut ratio")
    ax.set_title(f"Partitioner trade-off (mean $k$ = {agg['k'].iloc[0]:.1f})")
    ax.set_xlim(-4, agg["balance"].max() * 1.45)
    style_axes(ax, grid_axis="both")
    savefig(fig, out, "rq2_partition_balance")


def build(offline, bench, per_graph, savings, measurements_dir, out) -> None:
    retention_bars(offline, out)
    retention_distribution(measurements_dir, out)
    offline_cost(offline, out)
    vram_bars(bench, out)
    throughput_bars(bench, out)
    cost_vs_size(per_graph, out)
    paired_savings(savings, out)
    partition_balance(measurements_dir, out)
    amortisation(offline, savings, out)
