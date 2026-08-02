"""Figures with no data behind them at all.

Each of these occupies a slot the Results chapter argues for and cannot yet
fill. They exist so the argument can be laid out and reviewed now, and so that
the outstanding run is visible as a figure rather than buried in a comment.

Every figure in this module is watermarked, red-framed and footnoted with the
command that would replace it. None of them may be read as a result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from analysis.fake_data import (
    RECEPTIVE_FIELD,
    SPLIT_PROTOCOL,
    TODO_RECEPTIVE_FIELD,
    TODO_SPLIT_PROTOCOL,
    TODO_WL_DEPTH,
    WL_DEPTH,
)
from analysis.style import (
    COL,
    FAKE_COLOR,
    INK_MUTED,
    INK_SECONDARY,
    WIDE,
    color_for,
    label_for,
    mark_fake,
    savefig,
    style_axes,
)


def split_protocol(out: Path) -> None:
    """RQ1a: how much of the score is design recognition?

    Three protocols, identical everything else. Only the design-disjoint row is
    measured; the other two are the runs that would tell you whether the
    published numbers this thesis is compared against are measuring the same
    thing.
    """
    df = SPLIT_PROTOCOL
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5), sharey=True)
    y = np.arange(len(df))
    for ax, col, name in [
        (axes[0], "rmse", "RMSE"),
        (axes[1], "r2", "$R^2$"),
        (axes[2], "spearman", "Spearman $\\rho$"),
    ]:
        colors = [INK_MUTED if real else FAKE_COLOR for real in df["measured"]]
        bars = ax.barh(y, df[col], color=colors, height=0.6)
        for bar, real in zip(bars, df["measured"]):
            if not real:
                bar.set_hatch("xxx")
                bar.set_edgecolor("white")
        ax.set_title(name)
        style_axes(ax, grid_axis="x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [p + ("" if real else "  [FAKE]") for p, real in zip(df["protocol"], df["measured"])],
        fontsize=6.5,
    )
    axes[0].invert_yaxis()
    fig.suptitle(
        "RQ1a: leakier splits should score higher. Only the bottom row exists.", y=1.06
    )
    mark_fake(fig, note=TODO_SPLIT_PROTOCOL, watermark=False)
    savefig(fig, out, "rq1a_split_protocol")


def receptive_field(out: Path) -> None:
    """H1: does coarsening contract paths?

    The metric -- mean $k$-hop fanin-cone size with $k$ = encoder depth,
    measured before and after reduction -- is specified in the methodology and
    not implemented. Until it is, H1 is asserted, not tested.
    """
    df = RECEPTIVE_FIELD
    fig, ax = plt.subplots(figsize=COL)
    y = np.arange(len(df))
    ratio = df["cone_after"] / df["cone_before"]
    colors = [color_for(m) if real else FAKE_COLOR
              for m, real in zip(df["reduction_method"], df["measured"])]
    bars = ax.barh(y, ratio, color=colors, height=0.66)
    for bar, real in zip(bars, df["measured"]):
        if not real:
            bar.set_hatch("xxx")
            bar.set_edgecolor("white")
    ax.axvline(1.0, color=INK_SECONDARY, lw=1.1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([label_for(m, short=True) for m in df["reduction_method"]], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("4-hop fanin cone size after $\\div$ before")
    ax.set_title("H1: only coarsening can grow the receptive field")
    style_axes(ax, grid_axis="x")
    mark_fake(fig, ax, note=TODO_RECEPTIVE_FIELD)
    savefig(fig, out, "h1_receptive_field")


def wl_depth_probe(out: Path) -> None:
    """Residual structural redundancy after strashing, by refinement depth.

    Every corpus graph is structurally hashed, so depth 1 should find almost
    nothing. How much survives beyond one hop is a reportable statistic about
    AIGs in its own right, and it bounds what the exact track can compress.
    """
    df = WL_DEPTH
    fig, ax = plt.subplots(figsize=COL)
    ax.plot(df["depth"], df["node_retention"], marker="o", color=FAKE_COLOR, lw=1.8)
    ax.axhline(1.0, color=INK_SECONDARY, lw=1.0, ls="--")
    ax.text(1.0, 1.0, " no compression", fontsize=6, color=INK_SECONDARY, va="bottom")
    ax.set_xticks(df["depth"])
    ax.set_xlabel("Colour-refinement depth $d$ (encoder depth = 4)")
    ax.set_ylabel("Node retention")
    ax.set_ylim(0.5, 1.08)
    ax.set_title("Residual redundancy beyond structural hashing")
    style_axes(ax, grid_axis="both")
    mark_fake(fig, ax, note=TODO_WL_DEPTH)
    savefig(fig, out, "summ_wl_depth_probe")


def summarization_landscape(out: Path) -> None:
    """The summarization family in the compression/accuracy plane.

    This is the figure RQ4's third pairing kind needs: no two real
    summarization methods can be made to meet at the same ratio, so each is
    read against a random within-type control run at that method's own achieved
    compression. Nothing here has been run, and random within-type merging is
    not implemented at all.
    """
    from analysis.fake_data import SUMMARIZATION, TODO_SUMMARIZATION

    df = SUMMARIZATION
    fig, axes = plt.subplots(1, 2, figsize=WIDE)

    ax = axes[0]
    for _, row in df.iterrows():
        ax.plot(row["node_retention"], row["r2"], "o", ms=8, color=FAKE_COLOR)
        ax.annotate(label_for(row["reduction_method"]),
                    (row["node_retention"], row["r2"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=6, color=INK_SECONDARY)
    ax.axhline(0.342838, color=INK_SECONDARY, lw=1.1, ls="--")
    ax.set_xlim(0.32, 1.10)
    ax.text(0.33, 0.342838, "measured full-graph baseline", fontsize=6,
            color=INK_SECONDARY, va="top")
    ax.set_xlabel("Node retention")
    ax.set_ylabel("Matched-state $R^2$")
    ax.set_title("Compression against accuracy")
    style_axes(ax, grid_axis="both")

    ax = axes[1]
    x = np.arange(len(df))
    ax.bar(x, df["offline_s"], color=FAKE_COLOR, hatch="xxx", edgecolor="white", width=0.66)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([label_for(m, short=True) for m in df["reduction_method"]],
                       fontsize=6.5, rotation=20)
    ax.set_ylabel("Offline cost per graph (s, log)")
    ax.set_title("Cost of computing the merge")
    style_axes(ax)

    fig.suptitle("The summarization family: every number below is invented", y=1.04)
    mark_fake(fig, note=TODO_SUMMARIZATION)
    savefig(fig, out, "summ_landscape")


def h1_path_contraction(out: Path) -> None:
    """H1 stated as a prediction the data could falsify.

    Sparsification removes edges and can only impede message propagation;
    coarsening contracts paths and shortens the distance a signal must travel.
    On a graph as deep as an AIG relative to four message-passing layers, that
    gives coarsening a mechanism by which it could raise accuracy rather than
    trade it away. The measured half of this plot is the sparsification and
    partition arm; the coarsening arm is invented.
    """
    from analysis.fake_data import SUMMARIZATION

    fig, ax = plt.subplots(figsize=COL)
    ratio = RECEPTIVE_FIELD.set_index("reduction_method")
    summ = SUMMARIZATION.set_index("reduction_method")

    for method in ratio.index:
        if method == "none":
            continue
        rf = ratio.loc[method, "cone_after"] / ratio.loc[method, "cone_before"]
        r2 = summ.loc[method, "r2"] if method in summ.index else np.nan
        if np.isnan(r2):
            continue
        ax.plot(rf, r2, "o", ms=8, color=FAKE_COLOR)
        ax.annotate(label_for(method, short=True), (rf, r2),
                    textcoords="offset points", xytext=(6, 3), fontsize=6, color=INK_SECONDARY)
    ax.axvline(1.0, color=INK_SECONDARY, lw=1.0, ls="--")
    ax.axhline(0.342838, color=INK_SECONDARY, lw=1.0, ls=":")
    ax.set_xlabel("Receptive field after $\\div$ before")
    ax.set_ylabel("Matched-state $R^2$")
    ax.set_title("H1: does a larger receptive field buy accuracy?")
    style_axes(ax, grid_axis="both")
    mark_fake(fig, ax, note=TODO_RECEPTIVE_FIELD)
    savefig(fig, out, "h1_path_contraction")


def build(out: Path) -> None:
    split_protocol(out)
    receptive_field(out)
    wl_depth_probe(out)
    summarization_landscape(out)
    h1_path_contraction(out)
