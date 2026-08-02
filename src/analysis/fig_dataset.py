"""Corpus-level figures: what the model is actually being asked to predict.

Everything here is derived from the persisted per-graph predictions of the
full-graph baseline, which carry the true target, node count and edge count for
every graph in the split. That covers the evaluation corpus only; statistics over
the full stored corpus (tier 0 plus tier 1) need a separate pass over the graph
cache and are not available from the exported results.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.style import (
    COL,
    CORPUS_SAMPLE_COLOR,
    DATASET_FLAG_COLOR,
    FAMILY_COLORS,
    INK_MUTED,
    INK_SECONDARY,
    SOURCE_COLORS,
    SPLIT_COLORS,
    TIER_COLORS,
    WIDE,
    savefig,
    style_axes,
)

import matplotlib.pyplot as plt

# Two stored tiers. The labelling runs on tier-1 graphs are not retained as a
# third tier, so "tier2" never appears in a graph path.


def label_distribution(preds: pd.DataFrame, out: Path) -> None:
    """The target's own distribution, pooled and per tier.

    This is the denominator of every $R^2$ in the thesis: a narrow label
    distribution makes a small RMSE compatible with a near-zero $R^2$, so the
    spread here has to be read before any accuracy number is interpreted.
    """
    fig, axes = plt.subplots(1, 2, figsize=WIDE)

    ax = axes[0]
    # Log counts: the distribution is dominated by a spike at y ~ 0, and on a
    # linear axis every other bin disappears into the baseline.
    ax.hist(preds["target"], bins=60, color=FAMILY_COLORS["baseline"], edgecolor="none")
    ax.set_yscale("log")
    ax.axvline(preds["target"].mean(), color=DATASET_FLAG_COLOR, lw=1.4, ls="--")
    near_zero = float((preds["target"] < 0.005).mean())
    ax.text(
        0.97,
        0.94,
        f"mean {preds['target'].mean():.3f}   sd {preds['target'].std():.3f}\n"
        f"median {preds['target'].median():.3f}\n"
        f"{near_zero:.0%} of graphs have $y < 0.005$",
        transform=ax.transAxes,
        color=INK_SECONDARY,
        fontsize=7,
        va="top",
        ha="right",
    )
    ax.set_xlabel("Optimizability $y$ (relative node reduction)")
    ax.set_ylabel("Graphs (log)")
    ax.set_title("Pooled label distribution")
    style_axes(ax)

    ax = axes[1]
    tiers = [t for t in TIER_COLORS if t in set(preds["tier"].dropna())]
    for tier in tiers:
        sub = preds.loc[preds["tier"] == tier, "target"].to_numpy()
        sub = np.sort(sub)
        ax.plot(
            sub,
            np.linspace(0, 1, len(sub)),
            color=TIER_COLORS[tier],
            label=f"{tier} (n={len(sub):,}, med {np.median(sub):.3f})",
        )
    ax.set_xlabel("Optimizability $y$")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Label distribution by tier")
    ax.legend(loc="lower right")
    style_axes(ax)

    fig.suptitle("Optimizability targets on the held-out test corpus", y=1.04)
    savefig(fig, out, "dataset_label_distribution")


def size_distribution(preds: pd.DataFrame, out: Path) -> None:
    """Node and edge count distributions — the quantity the memory bottleneck
    is a function of, and the reason the corpus is bounded where it is."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE)

    for ax, col, name in zip(axes, ["num_nodes", "num_edges"], ["Nodes", "Edges"]):
        for tier in sorted(set(preds["tier"].dropna())):
            sub = np.sort(preds.loc[preds["tier"] == tier, col].to_numpy())
            ax.plot(sub, np.linspace(0, 1, len(sub)), color=TIER_COLORS[tier], label=tier)
        ax.set_xscale("log")
        ax.set_xlabel(f"{name} per graph (log)")
        ax.set_ylabel("Cumulative fraction")
        stats = preds[col]
        ax.set_title(
            f"{name}: median {stats.median():,.0f}, max {stats.max():,.0f}"
        )
        ax.legend(loc="lower right")
        style_axes(ax)

    fig.suptitle("Graph scale across the test corpus", y=1.04)
    savefig(fig, out, "dataset_size_distribution")


def size_vs_label(preds: pd.DataFrame, out: Path) -> None:
    """Is optimizability predictable from size alone? The size-only baseline in
    RQ1 is exactly this relationship fitted; the density plot shows what a
    size-only model has to work with."""
    fig, ax = plt.subplots(figsize=COL)
    hb = ax.hexbin(
        preds["num_nodes"],
        preds["target"],
        gridsize=45,
        xscale="log",
        bins="log",
        cmap="Blues",
        mincnt=1,
        linewidths=0,
    )
    fig.colorbar(hb, ax=ax, label="graphs (log)")
    ax.set_xlabel("Nodes per graph (log)")
    ax.set_ylabel("Optimizability $y$")
    rho = preds["num_nodes"].corr(preds["target"], method="spearman")
    ax.set_title(f"Size vs. label (Spearman $\\rho$ = {rho:.3f})")
    style_axes(ax, grid_axis="both")
    savefig(fig, out, "dataset_size_vs_label")


def design_composition(preds: pd.DataFrame, out: Path) -> None:
    """How the design-disjoint test set is composed.

    A design-level split means the test set is a handful of whole circuits, not
    a random sample. If one design dominates the row count, every pooled test
    metric in the thesis is largely that design's metric.
    """
    counts = preds.groupby("design").agg(
        graphs=("target", "size"),
        mean_y=("target", "mean"),
        med_nodes=("num_nodes", "median"),
    )
    counts = counts.sort_values("graphs", ascending=False)

    fig, ax = plt.subplots(figsize=(6.9, max(2.4, 0.22 * len(counts) + 0.9)))
    y = np.arange(len(counts))
    ax.barh(y, counts["graphs"], color=FAMILY_COLORS["baseline"], height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Test graphs")
    share = counts["graphs"] / counts["graphs"].sum()
    ax.set_title(
        f"Test-set composition: {len(counts)} designs, "
        f"largest holds {share.iloc[0]:.0%} of rows"
    )
    for yi, (n, m) in enumerate(zip(counts["graphs"], counts["mean_y"])):
        ax.text(n, yi, f"  mean $y$={m:.3f}", va="center", fontsize=6, color=INK_MUTED)
    ax.set_xlim(0, counts["graphs"].max() * 1.35)
    style_axes(ax, grid_axis="x")
    savefig(fig, out, "dataset_design_composition")


def label_by_source_algorithm(preds: pd.DataFrame, out: Path) -> None:
    """What the label mostly measures: which script already touched the graph.

    A tier-1 graph is a base circuit already optimized by one of four synthesis
    scripts, and the Orchestrate target is what is left to remove after that.
    Orchestrate finds almost nothing on a C2RS- or Deepsyn-optimized graph and
    a great deal on a Syn4 one, so the source script -- not the circuit -- is
    the strongest single determinant of the target.
    """
    order = ["tier0 base", "Syn4", "Deepsyn", "C2RS"]
    present = [s for s in order if s in set(preds["source_algorithm"])]
    colors = [SOURCE_COLORS[s] for s in present]

    fig, axes = plt.subplots(1, 2, figsize=WIDE)

    ax = axes[0]
    for source, color in zip(present, colors):
        y = np.sort(preds.loc[preds["source_algorithm"] == source, "target"].to_numpy())
        ax.plot(
            y, np.linspace(0, 1, len(y)), color=color,
            label=f"{source} (n={len(y):,}, mean {y.mean():.3f})",
        )
    ax.set_xlabel("Optimizability $y$")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Label distribution by source script")
    ax.legend(loc="lower right")
    style_axes(ax, grid_axis="both")

    ax = axes[1]
    shares = [
        float((preds.loc[preds["source_algorithm"] == s, "target"] == 0).mean())
        for s in present
    ]
    x = np.arange(len(present))
    ax.bar(x, shares, color=colors[: len(present)], width=0.6)
    for xi, share in zip(x, shares):
        ax.text(xi, share, f"{share:.0%}", ha="center", va="bottom", fontsize=7,
                color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(present, fontsize=7)
    ax.set_ylim(0, max(shares) * 1.25)
    ax.set_ylabel("Fraction with $y$ exactly 0")
    ax.set_title("Orchestrate finds nothing left to remove")
    style_axes(ax)

    fig.suptitle(
        "The strongest predictor of optimizability is which script ran before", y=1.04
    )
    savefig(fig, out, "dataset_label_by_source")


def zero_inflation(preds: pd.DataFrame, out: Path) -> None:
    """Where the point mass at exactly zero sits.

    Half the test split has an optimizability of exactly zero, and that mass is
    not spread evenly: it is concentrated in particular designs and in the
    smallest circuits, both of which are already at a fixed point for
    Orchestrate. This is the figure the size-only baseline result of RQ1 has to
    be read against -- graph size is largely a proxy for "is this optimizable at
    all".
    """
    fig, axes = plt.subplots(1, 2, figsize=WIDE)

    ax = axes[0]
    by_design = preds.groupby("design")["target"].agg(
        frac_zero=lambda s: (s == 0).mean(), max_y="max"
    ).sort_values("frac_zero")
    y = np.arange(len(by_design))
    colors = [
        DATASET_FLAG_COLOR if mx <= 0.001 else FAMILY_COLORS["baseline"]
        for mx in by_design["max_y"]
    ]
    ax.barh(y, by_design["frac_zero"], color=colors, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(by_design.index, fontsize=7)
    for yi, (frac, mx) in enumerate(zip(by_design["frac_zero"], by_design["max_y"])):
        ax.text(frac, yi, f"  max $y$={mx:.4f}", va="center", fontsize=6, color=INK_MUTED)
    ax.set_xlim(0, 1.35)
    ax.set_xlabel("Fraction of graphs with $y$ exactly 0")
    ax.set_title(textwrap.fill("By design (red: Orchestrate is at a fixed point)", width=30))
    style_axes(ax, grid_axis="x")

    ax = axes[1]
    bins = np.logspace(
        np.log10(preds["num_nodes"].min()), np.log10(preds["num_nodes"].max()), 16
    )
    binned = preds.assign(bin=pd.cut(preds["num_nodes"], bins)).groupby("bin", observed=True)
    grouped = binned.agg(
        centre=("num_nodes", "median"), frac_zero=("target", lambda s: (s == 0).mean())
    )
    ax.plot(grouped["centre"], grouped["frac_zero"], marker="o", ms=3,
            color=FAMILY_COLORS["baseline"])
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Nodes per graph (log)")
    ax.set_ylabel("Fraction with $y$ exactly 0")
    ax.set_title(textwrap.fill("By circuit scale: small circuits are already minimal", width=30))
    style_axes(ax, grid_axis="both")

    frac = float((preds["target"] == 0).mean())
    fig.suptitle(
        f"Zero inflation: {frac:.1%} of the test split has an optimizability of exactly zero",
        y=1.04,
    )
    savefig(fig, out, "dataset_zero_inflation")


def structural_statistics(preds: pd.DataFrame, measurements_dir: Path, out: Path) -> None:
    """Structural corpus statistics that the evaluation exports can still reach.

    The AND-gate fraction is recovered from the AND-gate-only node masks rather
    than from the graphs themselves: that method drops exactly the primary
    inputs and outputs, so its node retention *is* the AND-and-constant share of
    the corpus. That closes one of the open items in the methodology --- the
    method's compression is explained by a dataset statistic rather than merely
    accompanied by one.
    """
    from analysis.results_to_latex import load_offline_stats

    sparse = load_offline_stats(measurements_dir, "sparsification_stats")
    ago = sparse[sparse["reduction_method"] == "and_gate_only"]
    if ago.empty:
        print("[dataset] no and_gate_only stats — skipping structural figure.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.6))

    ax = axes[0]
    keep = ago["node_retention"].to_numpy()
    ax.hist(keep, bins=40, color=FAMILY_COLORS["baseline"], edgecolor="none")
    ax.axvline(keep.mean(), color=DATASET_FLAG_COLOR, lw=1.4, ls="--")
    ax.text(0.03, 0.95,
            f"AND + const: {keep.mean():.1%}\ninterface (PI/PO): {1 - keep.mean():.1%}",
            transform=ax.transAxes, fontsize=6.5, va="top", color=INK_SECONDARY)
    ax.set_xlabel("AND-and-constant share of nodes")
    ax.set_ylabel("Graphs")
    ax.set_title("Gate composition")
    style_axes(ax)

    ax = axes[1]
    density = (preds["num_edges"] / preds["num_nodes"]).to_numpy()
    ax.hist(density, bins=50, color=FAMILY_COLORS["baseline"], edgecolor="none")
    ax.set_xlabel("Edges per node")
    ax.set_ylabel("Graphs")
    ax.set_title(f"Density (median {np.median(density):.2f})")
    style_axes(ax)

    ax = axes[2]
    for values, color, name in [
        (np.sort(preds["num_nodes"].to_numpy()), FAMILY_COLORS["baseline"],
         f"evaluation splits (n={len(preds):,})"),
        (np.sort(ago["n_nodes"].to_numpy()), CORPUS_SAMPLE_COLOR,
         f"corpus sample (n={len(ago):,})"),
    ]:
        ax.plot(values, np.linspace(0, 1, len(values)), color=color, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Nodes per graph (log)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Are the eval splits representative?")
    ax.legend(loc="upper left", fontsize=6)
    style_axes(ax, grid_axis="both")

    fig.suptitle("Structural statistics of the corpus", y=1.06)
    savefig(fig, out, "dataset_structure")


def split_comparison(test: pd.DataFrame, val: pd.DataFrame, out: Path) -> None:
    """Validation and test are different circuits, not two samples of one pool.

    Under a design-level split this is expected, but the size of the difference
    bounds how much validation-based model selection transfers, and it is the
    direct explanation for the validation-to-test gap in
    \\ref{sec:results:rq3}.
    """
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    pairs = [("val", val, SPLIT_COLORS["val"]), ("test", test, SPLIT_COLORS["test"])]

    ax = axes[0]
    for name, d, color in pairs:
        y = np.sort(d["target"].to_numpy())
        ax.plot(y, np.linspace(0, 1, len(y)), color=color,
                label=f"{name}: {d['design'].nunique()} designs, "
                      f"{float((d['target'] == 0).mean()):.0%} at $y=0$")
    ax.set_xlabel("Optimizability $y$")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Label distribution differs between splits")
    ax.legend(loc="lower right")
    style_axes(ax, grid_axis="both")

    ax = axes[1]
    for name, d, color in pairs:
        n = np.sort(d["num_nodes"].to_numpy())
        ax.plot(n, np.linspace(0, 1, len(n)), color=color,
                label=f"{name}: median {np.median(n):,.0f} nodes")
    ax.set_xscale("log")
    ax.set_xlabel("Nodes per graph (log)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("So does the size distribution")
    ax.legend(loc="upper left")
    style_axes(ax, grid_axis="both")

    fig.suptitle(
        "Validation and test are disjoint circuit sets, not exchangeable samples", y=1.04
    )
    savefig(fig, out, "dataset_split_comparison")


def build(preds: pd.DataFrame, out: Path, measurements_dir: Path | None = None,
          preds_val: pd.DataFrame | None = None) -> None:
    label_distribution(preds, out)
    size_distribution(preds, out)
    size_vs_label(preds, out)
    design_composition(preds, out)
    label_by_source_algorithm(preds, out)
    zero_inflation(preds, out)
    if measurements_dir is not None:
        structural_statistics(preds, measurements_dir, out)
    if preds_val is not None and not preds_val.empty:
        split_comparison(preds, preds_val, out)
