"""RQ3 figures: how much predictive accuracy survives reduction.

Every accuracy figure here reports RMSE, $R^2$ and Spearman together and never
one alone. A reduction can hold mean error steady while destroying explained
variance -- which is what CTS-Bench observed on a coarsened EDA graph -- so a
figure showing only RMSE would miss the failure this chapter exists to detect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from analysis.fake_data import SUMMARIZATION, TODO_SUMMARIZATION
from analysis.style import (
    COL,
    FAKE_COLOR,
    FAMILY_COLORS,
    FAMILY_MARKERS,
    INK_MUTED,
    INK_SECONDARY,
    WIDE,
    clip_bar_axis,
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
METRICS = [
    ("rmse", "RMSE", "lower is better"),
    ("r2", "$R^2$", "higher is better"),
    ("spearman", "Spearman $\\rho$", "higher is better"),
]


def matched_state(inference: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration under the reduction it was trained with.

    The baseline has no matched-reduction pass -- its matched state *is* the
    full graph -- so its full_graph row stands in.
    """
    test = inference[inference["split"] == "test"].copy()
    test["reduction_method"] = test["reduction_method"].fillna("none")
    matched = test[
        (test["eval_mode"] == "matched_reduction")
        | (test["reduction_type"] == "none")
    ]
    return matched.sort_values("reduction_method", key=lambda s: s.map(sort_key))


def _paint(bars, methods) -> None:
    for bar, method in zip(bars, methods):
        bar.set_facecolor(color_for(method))
        hatch = hatch_for(method)
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor("white")
        if not meta(method)["measured"]:
            bar.set_facecolor(FAKE_COLOR)


def with_summarization(matched: pd.DataFrame) -> pd.DataFrame:
    fake = SUMMARIZATION[["reduction_method", "reduction_type", "rmse", "r2", "spearman"]]
    combined = pd.concat([matched.assign(measured=True), fake.assign(measured=False)])
    return combined.sort_values("reduction_method", key=lambda s: s.map(sort_key))


def accuracy_bars(matched: pd.DataFrame, out: Path) -> None:
    """The three metrics side by side, against the full-graph baseline."""
    df = with_summarization(matched)
    methods = df["reduction_method"].tolist()
    base = df[df["reduction_method"] == "none"].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(6.9, row_height(len(df))), sharey=True)
    y = np.arange(len(df))

    for ax, (col, name, direction) in zip(axes, METRICS):
        bars = ax.barh(y, df[col], height=0.68)
        _paint(bars, methods)
        ax.axvline(base[col], color=INK_SECONDARY, lw=1.1, ls="--")
        ax.axvline(0, color=INK_SECONDARY, lw=0.7)
        title = f"{name}\n({direction})"
        if col == "r2":
            for i, value in clip_bar_axis(ax, df[col].tolist(), -1.0):
                ax.text(-1.0, i, f"  {value:.2f} <", fontsize=6, color="white",
                        va="center", ha="left", fontweight="bold", zorder=5)
            title += "\naxis clipped at $-1$"
        ax.set_title(title)
        style_axes(ax, grid_axis="x")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [label_for(m) + ("  [FAKE]" if not meta(m)["measured"] else "") for m in methods],
        fontsize=7,
    )
    axes[0].invert_yaxis()
    fig.suptitle(
        "Matched-state accuracy (dashed: full-graph baseline). "
        "Error and explained variance disagree.",
        y=1.05,
    )
    mark_fake(fig, note="summarization rows only: " + TODO_SUMMARIZATION, watermark=False)
    savefig(fig, out, "rq3_accuracy")


def metric_divergence(matched: pd.DataFrame, out: Path) -> None:
    """RMSE against $R^2$, one point per configuration.

    If the two metrics agreed, the points would fall on a monotone curve. They
    do not, and that is the single strongest argument in the chapter for
    reporting both: several configurations hold RMSE close to the baseline
    while $R^2$ collapses below zero.
    """
    df = matched.copy()
    floor = -1.0
    fig, ax = plt.subplots(figsize=COL)
    offsets = [(7, 4), (7, -9), (-8, 7), (-8, -11)]
    for i, (_, row) in enumerate(df.iterrows()):
        method = row["reduction_method"]
        info = meta(method)
        off_scale = row["r2"] < floor
        ypos = floor if off_scale else row["r2"]
        ax.plot(
            row["rmse"],
            ypos,
            marker="v" if off_scale else FAMILY_MARKERS[info["family"]],
            ms=9 if method == "none" else 7,
            color=color_for(method),
            markerfacecolor=color_for(method) if info["domain"] or method == "none" else "white",
            markeredgewidth=1.4,
            clip_on=False,
        )
        ax.annotate(
            label_for(method, short=True) + (f" ({row['r2']:.2f})" if off_scale else ""),
            (row["rmse"], ypos),
            textcoords="offset points",
            xytext=offsets[i % len(offsets)],
            fontsize=6,
            color=FAKE_COLOR if off_scale else INK_SECONDARY,
        )
    ax.set_ylim(floor - 0.08, max(0.55, df["r2"].max() + 0.2))
    base = df[df["reduction_method"] == "none"].iloc[0]
    ax.axhline(base["r2"], color=INK_SECONDARY, lw=0.9, ls="--")
    ax.axvline(base["rmse"], color=INK_SECONDARY, lw=0.9, ls="--")
    ax.axhline(0, color=FAKE_COLOR, lw=0.9, ls=":")
    ax.text(
        ax.get_xlim()[1],
        0,
        "$R^2 = 0$: no better\nthan the label mean ",
        fontsize=6,
        color=FAKE_COLOR,
        ha="right",
        va="bottom",
    )
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_ylabel("$R^2$ (higher is better)")
    ax.set_title(f"A small RMSE does not imply explained variance\n($R^2$ clipped at {floor:g})")
    style_axes(ax, grid_axis="both")
    savefig(fig, out, "rq3_metric_divergence")


def retention_vs_compression(matched: pd.DataFrame, offline: pd.DataFrame, out: Path) -> None:
    """Accuracy against how much was actually removed.

    Comparing "which method is best" without this plane confounds a genuinely
    better method with one that simply reduced less.
    """
    df = matched.merge(offline, on="reduction_method", how="left")
    base = matched[matched["reduction_method"] == "none"].iloc[0]

    floor = -1.0
    offsets = [(7, 4), (7, -9), (-8, 7), (-8, -11)]
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    for ax, xcol, xname in [
        (axes[0], "edge_retention", "Edge retention"),
        (axes[1], "node_retention", "Node retention"),
    ]:
        for i, (_, row) in enumerate(df.iterrows()):
            method = row["reduction_method"]
            if method == "none":
                continue
            info = meta(method)
            off_scale = row["r2"] < floor
            ypos = floor if off_scale else row["r2"]
            ax.plot(
                row[xcol],
                ypos,
                marker="v" if off_scale else FAMILY_MARKERS[info["family"]],
                ms=8,
                color=color_for(method),
                markerfacecolor=color_for(method) if info["domain"] else "white",
                markeredgewidth=1.4,
                clip_on=False,
            )
            ax.annotate(
                label_for(method, short=True) + (f" ({row['r2']:.2f})" if off_scale else ""),
                (row[xcol], ypos),
                textcoords="offset points",
                xytext=offsets[i % len(offsets)],
                fontsize=6,
                color=FAKE_COLOR if off_scale else INK_SECONDARY,
            )
        ax.set_ylim(floor - 0.08, max(0.55, df["r2"].max() + 0.2))
        ax.axhline(base["r2"], color=INK_SECONDARY, lw=1.0, ls="--")
        ax.text(
            0.02, base["r2"], " full-graph baseline", transform=ax.get_yaxis_transform(),
            fontsize=6, color=INK_SECONDARY, va="bottom",
        )
        ax.axhline(0, color=FAKE_COLOR, lw=0.9, ls=":")
        ax.set_xlabel(f"{xcol.replace('_', ' ').title()} (1.0 = nothing removed)")
        ax.set_ylabel("Matched-state $R^2$")
        ax.set_title(f"$R^2$ vs. {xname.lower()}")
        style_axes(ax, grid_axis="both")

    family_legend(axes[0], ["partition", "sparsification"], loc="lower left")
    fig.suptitle("Accuracy against achieved compression, the only fair comparison plane", y=1.04)
    savefig(fig, out, "rq3_retention_vs_compression")


def pareto(matched: pd.DataFrame, savings: pd.DataFrame, out: Path) -> None:
    """The trade-off the whole thesis is about: what accuracy costs what saving.

    The dominance frontier is drawn rather than described, and the baseline is
    a point on the plane at zero saving, so "no reduction" competes on the same
    axes as every reduction.
    """
    df = matched.merge(savings, on="reduction_method", how="left")
    df.loc[df["reduction_method"] == "none", ["mean_vram_saving_pct", "mean_time_saving_pct"]] = 0.0

    floor = -1.0  # METIS reaches -3.06 and would flatten every other point
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    for ax, xcol, xname in [
        (axes[0], "mean_vram_saving_pct", "Peak memory saving (%)"),
        (axes[1], "mean_time_saving_pct", "Step-time saving (%)"),
    ]:
        sub = df.dropna(subset=[xcol]).sort_values(xcol)
        # Alternate the label offset so that clustered points do not overprint.
        offsets = [(7, 4), (7, -9), (-8, 7), (-8, -11)]
        for i, (_, row) in enumerate(sub.iterrows()):
            method = row["reduction_method"]
            info = meta(method)
            off_scale = row["r2"] < floor
            ypos = floor if off_scale else row["r2"]
            ax.plot(
                row[xcol], ypos,
                marker="v" if off_scale else FAMILY_MARKERS[info["family"]],
                ms=9 if method == "none" else 7,
                color=color_for(method),
                markerfacecolor=color_for(method) if info["domain"] or method == "none" else "white",
                markeredgewidth=1.4,
                clip_on=False,
            )
            text = label_for(method, short=True)
            if off_scale:
                text += f" ({row['r2']:.2f})"
            ax.annotate(
                text, (row[xcol], ypos), textcoords="offset points",
                xytext=offsets[i % len(offsets)], fontsize=6,
                color=FAKE_COLOR if off_scale else INK_SECONDARY,
            )
        # Pareto frontier: maximise saving and R^2 simultaneously.
        frontier, best = [], -np.inf
        for _, row in sub.sort_values(xcol, ascending=False).iterrows():
            if row["r2"] > best:
                best = row["r2"]
                frontier.append((row[xcol], row["r2"]))
        frontier = sorted(frontier)
        ax.plot(
            [p[0] for p in frontier], [p[1] for p in frontier],
            color=INK_MUTED, lw=1.2, ls="--", zorder=0, label="Pareto frontier",
        )
        ax.axhline(0, color=FAKE_COLOR, lw=0.9, ls=":")
        ax.set_ylim(floor - 0.08, max(0.55, df["r2"].max() + 0.2))
        ax.set_xlabel(xname)
        ax.set_ylabel("Matched-state $R^2$")
        ax.set_title(f"$R^2$ vs. {xname}")
        ax.legend(loc="center right", fontsize=6.5)
        style_axes(ax, grid_axis="both")

    fig.suptitle(
        f"Pareto front: accuracy against efficiency ($R^2$ axis clipped at {floor:g})", y=1.04
    )
    savefig(fig, out, "rq3_pareto")


def accuracy_retained(matched: pd.DataFrame, out: Path) -> None:
    """Accuracy retained as a fraction of the baseline, so that each reduction
    method lines up directly against the efficiency ranking of RQ2.

    RMSE, $R^2$ and Spearman are three different quantities that happen to
    share a "fraction of baseline" unit; one shared axis invites reading them
    as if they moved together, so each gets its own axes here instead.
    """
    df = matched.copy()
    base = df[df["reduction_method"] == "none"].iloc[0]
    df = df[df["reduction_method"] != "none"]
    methods = df["reduction_method"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.9))
    x = np.arange(len(df))
    for ax, (col, name) in zip(
        axes,
        [("rmse", "RMSE ratio\n(baseline $\\div$ method)"),
         ("r2", "$R^2$ retained"),
         ("spearman", "Spearman retained")],
    ):
        values = base[col] / df[col] if col == "rmse" else df[col] / base[col]
        bars = ax.bar(x, values, width=0.6)
        _paint(bars, methods)
        ax.axhline(1.0, color=INK_SECONDARY, lw=1.1, ls="--")
        ax.axhline(0.0, color=INK_SECONDARY, lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([label_for(m, short=True) for m in methods], fontsize=6.5,
                           rotation=40, ha="right")
        ax.set_title(name, fontsize=7.5)
        style_axes(ax)
    axes[0].set_ylabel("Fraction of the full-graph baseline")
    fig.suptitle(
        "Accuracy retained under reduction (1.0 = no loss; below 0 = worse than the mean)",
        y=1.05,
    )
    savefig(fig, out, "rq3_accuracy_retained")


def val_test_gap(inference: pd.DataFrame, out: Path) -> None:
    """The same configurations on validation and on test.

    Under a design-level split these are different circuits, not two samples of
    one population, so a large gap is expected -- but its size bounds how much
    of the validation-selected model choice transfers.
    """
    df = inference[inference["eval_mode"].isin(["matched_reduction", "full_graph"])].copy()
    df["reduction_method"] = df["reduction_method"].fillna("none")
    matched = df[(df["eval_mode"] == "matched_reduction") | (df["reduction_type"] == "none")]
    pivot = matched.pivot_table(index="reduction_method", columns="split", values="r2")
    pivot = pivot.dropna().sort_values("test")

    fig, ax = plt.subplots(figsize=COL)
    y = np.arange(len(pivot))
    for yi, (method, row) in enumerate(pivot.iterrows()):
        ax.plot([row["val"], row["test"]], [yi, yi], color=color_for(method), lw=1.4, alpha=0.6)
        ax.plot(row["val"], yi, marker="o", ms=5, color=color_for(method), markerfacecolor="white",
                markeredgewidth=1.4)
        ax.plot(row["test"], yi, marker="o", ms=6, color=color_for(method))
    ax.set_yticks(y)
    ax.set_yticklabels([label_for(m, short=True) for m in pivot.index], fontsize=6.5)
    ax.axvline(0, color=INK_SECONDARY, lw=0.8)
    ax.set_xlabel("$R^2$   (open = validation, filled = test)")
    ax.set_title("Validation-to-test gap under a design-level split")
    style_axes(ax, grid_axis="x")
    savefig(fig, out, "rq3_val_test_gap")


def stratified_robustness(strata: pd.DataFrame, out: Path) -> None:
    """Is the full-vs-reduced conclusion a property of the model or of the label?

    The pooled label is ~49% exactly zero and ~73% below 0.005, so a pooled
    $R^2$ is dominated by a minority of graphs. The same persisted predictions
    are re-scored here on four subsets. A conclusion that survives all four is
    a property of the reduction; one that only holds on the pooled column is a
    property of the label distribution.
    """
    from analysis.loaders import STRATUM_LABELS

    for mode, tag in [("matched_reduction", "matched"), ("full_graph", "cross")]:
        # The baseline has no matched-reduction pass -- its matched state IS the
        # full graph -- so its full_graph row stands in. Without this the
        # matched panel has no reference to read the reductions against.
        keep = strata["eval_mode"] == mode
        if mode == "matched_reduction":
            keep |= strata["reduction_method"] == "none"
        df = strata[keep].copy()
        if df.empty:
            continue
        df = df.sort_values("all_r2", ascending=False)
        methods = df["reduction_method"].tolist()
        names = list(STRATUM_LABELS)

        fig, axes = plt.subplots(1, 2, figsize=(6.9, row_height(len(df), base=1.9)), sharey=True)
        y = np.arange(len(df))

        for ax, metric, mname, floor in [
            (axes[0], "r2", "$R^2$", -1.5),
            (axes[1], "spearman", "Spearman $\\rho$", None),
        ]:
            for offset, stratum in enumerate(names):
                values = df[f"{stratum}_{metric}"].to_numpy()
                shown = np.clip(values, floor, None) if floor is not None else values
                ax.scatter(
                    shown, y + (offset - 1.5) * 0.16,
                    s=26, marker=["o", "s", "^", "D"][offset],
                    color=[color_for(m) for m in methods],
                    edgecolor="white", linewidth=0.6, zorder=3,
                )
            for yi in y:
                row = df.iloc[yi]
                vals = [row[f"{s}_{metric}"] for s in names]
                lo, hi = min(vals), max(vals)
                if floor is not None:
                    lo, hi = max(lo, floor), max(hi, floor)
                ax.plot([lo, hi], [yi, yi], color=INK_MUTED, lw=0.8, zorder=1, alpha=0.6)
            ax.axvline(0, color=INK_SECONDARY, lw=0.9)
            if floor is not None:
                ax.set_xlim(left=floor - 0.05)
            ax.set_xlabel(mname)
            ax.set_title(f"{mname} across label strata")
            style_axes(ax, grid_axis="x")

        axes[0].set_yticks(y)
        axes[0].set_yticklabels([label_for(m) for m in methods], fontsize=7)
        axes[0].invert_yaxis()

        from matplotlib.lines import Line2D

        fig.legend(
            handles=[
                Line2D([], [], marker=["o", "s", "^", "D"][i], ls="none",
                       color=INK_SECONDARY, ms=5, label=STRATUM_LABELS[s])
                for i, s in enumerate(names)
            ],
            loc="upper center", bbox_to_anchor=(0.5, -0.02), ncols=2, fontsize=6.5,
        )
        title = "matched-state" if mode == "matched_reduction" else "cross-state (full graphs)"
        fig.suptitle(
            f"Robustness of the {title} ranking to the label skew\n"
            "(a spread along a row means the conclusion depends on which graphs are scored)",
            y=1.10,
        )
        savefig(fig, out, f"rq3_stratified_{tag}")


def by_subgroup(grouped: pd.DataFrame, by: str, order: list[str], out: Path,
                name: str, title: str, subtitle: str) -> None:
    """Matched-state accuracy for every configuration, split by a partition of
    the test set.

    This asks a question the pooled tables cannot: does a reduction cost more on
    one part of the corpus than another? Groups are drawn as separate bars per
    method rather than as separate figures, so the within-method spread --- which
    is the answer --- is read directly.
    """
    df = grouped[
        (grouped["eval_mode"] == "matched_reduction")
        | (grouped["reduction_method"] == "none")
    ].copy()
    if df.empty:
        return
    levels = [g for g in order if g in set(df[by])]
    methods = sorted(set(df["reduction_method"]), key=sort_key)
    shades = [1.0, 0.72, 0.5, 0.32]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, row_height(len(methods), base=2.0)),
                             sharey=True)
    y = np.arange(len(methods))
    height = 0.8 / len(levels)

    for ax, metric, mname, floor in [
        (axes[0], "r2", "$R^2$", -1.0),
        (axes[1], "spearman", "Spearman $\\rho$", None),
    ]:
        for gi, level in enumerate(levels):
            sub = df[df[by] == level].set_index("reduction_method")
            values = [sub.loc[m, metric] if m in sub.index else np.nan for m in methods]
            drawn = [max(v, floor) if (floor is not None and v == v) else v for v in values]
            offset = (gi - (len(levels) - 1) / 2) * height
            bars = ax.barh(y + offset, drawn, height=height * 0.88)
            for bar, method in zip(bars, methods):
                bar.set_facecolor(color_for(method))
                bar.set_alpha(shades[gi % len(shades)])
            if floor is not None:
                for yi, v in enumerate(values):
                    if v == v and v < floor:
                        ax.text(floor, yi + offset, f"  {v:.2f} <", fontsize=5.5,
                                color="white", va="center", fontweight="bold", zorder=5)
        ax.axvline(0, color=INK_SECONDARY, lw=0.9)
        if floor is not None:
            on_scale = [v for v in df[metric] if v == v and v >= floor]
            ax.set_xlim(floor, max(on_scale, default=0.5) * 1.15)
            mname += f" (clipped at {floor:g})"
        ax.set_xlabel(mname)
        ax.set_title(mname)
        style_axes(ax, grid_axis="x")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([label_for(m) for m in methods], fontsize=7)
    axes[0].invert_yaxis()

    from matplotlib.patches import Patch

    counts = df.groupby(by)["n"].max()
    fig.legend(
        handles=[
            Patch(facecolor=INK_SECONDARY, alpha=shades[i % len(shades)],
                  label=f"{lvl} (n={counts.get(lvl, 0):,})")
            for i, lvl in enumerate(levels)
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.01), ncols=min(4, len(levels)),
        fontsize=6.5,
    )
    fig.suptitle(f"{title}\n{subtitle}", y=1.09)
    savefig(fig, out, name)


def by_target_bin(binned: pd.DataFrame, out: Path) -> None:
    """Every configuration's behaviour as a function of how optimizable the
    circuit is.

    Inside a narrow band of the target, $R^2$ has almost no variance to divide
    by and is meaningless, so the two quantities plotted are the mean prediction
    against the band's mean truth --- a flat line is a model that has collapsed
    to a constant --- and the mean absolute error. This is the view that
    separates a model which has learned the task from one that has learned the
    label's marginal distribution.
    """
    df = binned[
        (binned["eval_mode"] == "matched_reduction")
        | (binned["reduction_method"] == "none")
    ].copy()
    if df.empty:
        return
    bands = [b for b in df["bin"].cat.categories if b in set(df["bin"])] \
        if hasattr(df["bin"], "cat") else sorted(set(df["bin"]))
    methods = sorted(set(df["reduction_method"]), key=sort_key)
    x = np.arange(len(bands))

    fig, axes = plt.subplots(1, 2, figsize=WIDE)

    truth = df.groupby("bin", observed=True)["true"].mean().reindex(bands)
    axes[0].plot(x, truth, color=INK_SECONDARY, lw=2.2, ls="--", label="true mean",
                 zorder=5)

    for method in methods:
        sub = df[df["reduction_method"] == method].set_index("bin").reindex(bands)
        kw = {
            "color": color_for(method),
            "lw": 2.0 if method == "none" else 1.1,
            "ls": "--" if meta(method)["domain"] else "-",
        }
        axes[0].plot(x, sub["pred"], **kw, label=label_for(method))
        axes[1].plot(x, sub["mae"], **kw)

    axes[0].set_ylabel("Mean prediction in band")
    axes[0].set_title("A flat line is a model predicting a constant")
    axes[1].set_ylabel("Mean absolute error")
    axes[1].set_yscale("log")
    axes[1].set_title("Error grows with how much there is to predict")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(bands, fontsize=6, rotation=30, ha="right")
        ax.set_xlabel("True optimizability band")
        style_axes(ax, grid_axis="both")
    axes[0].legend(loc="upper left", fontsize=5.5, ncols=2)

    fig.suptitle(
        "Every configuration by how much the circuit actually needs optimizing", y=1.04
    )
    savefig(fig, out, "rq3_by_target_bin")


def build(matched, inference, offline, savings, strata, by_tier, by_source,
          by_band, out) -> None:
    accuracy_bars(matched, out)
    stratified_robustness(strata, out)
    by_subgroup(
        by_tier, "tier", ["tier0", "tier1"], out, "rq3_by_tier",
        "Matched-state accuracy by dataset tier",
        "tier 0 is a base graph; tier 1 has already been through one synthesis script",
    )
    by_subgroup(
        by_source, "source_algorithm", ["tier0 base", "Syn4", "Deepsyn", "C2RS"], out,
        "rq3_by_source",
        "Matched-state accuracy by the script that produced the graph",
        "the cut that separates the corpus far more sharply than the tier does",
    )
    by_target_bin(by_band, out)
    metric_divergence(matched, out)
    retention_vs_compression(matched, offline, out)
    pareto(matched, savings, out)
    accuracy_retained(matched, out)
    val_test_gap(inference, out)
