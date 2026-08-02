"""RQ1 figures: how well the encoder predicts optimizability from the full AIG.

Splits into three groups: what the model does on the full-graph test set
(parity, calibration, residuals), where it fails (by size, by tier, by design),
and what it cost to get there (training curves, the Optuna sweep).

The trivial baselines in :func:`trivial_baselines` are fitted on the validation
predictions and scored on test, never fitted on test — a size-only regressor
fitted and scored on the same split is not a baseline, it is an upper bound.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib.pyplot as plt

from analysis.loaders import wandb_run_method
from analysis.style import (
    COL,
    FAKE_COLOR,
    FAMILY_COLORS,
    INK_MUTED,
    INK_SECONDARY,
    WIDE,
    clip_bar_axis,
    color_for,
    mark_fake,
    row_height,
    savefig,
    style_axes,
)

BASE = FAMILY_COLORS["baseline"]


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    resid = prediction - target
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "spearman": float(spearmanr(target, prediction).statistic),
    }


# --- What the model does -----------------------------------------------------
def parity(preds: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=COL)
    lim = max(preds["target"].max(), preds["prediction"].max()) * 1.05
    hb = ax.hexbin(
        preds["target"],
        preds["prediction"],
        gridsize=50,
        bins="log",
        cmap="Blues",
        mincnt=1,
        linewidths=0,
        extent=(0, lim, 0, lim),
    )
    fig.colorbar(hb, ax=ax, label="graphs (log)")
    ax.plot([0, lim], [0, lim], color=FAKE_COLOR, lw=1.2, ls="--", label="$\\hat y = y$")
    m = _metrics(preds["target"].to_numpy(), preds["prediction"].to_numpy())
    ax.text(
        0.04,
        0.96,
        f"RMSE {m['rmse']:.4f}\n$R^2$ {m['r2']:.3f}\n$\\rho$ {m['spearman']:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        color=INK_SECONDARY,
    )
    ax.set_xlabel("True optimizability $y$")
    ax.set_ylabel("Predicted $\\hat y$")
    ax.set_title("Full-graph baseline: predictions vs. truth")
    ax.legend(loc="lower right")
    style_axes(ax, grid_axis="both")
    savefig(fig, out, "rq1_parity")


def calibration(preds: pd.DataFrame, out: Path) -> None:
    """Binned reliability: within each decile of predicted value, is the mean
    prediction the mean truth? A regressor that has collapsed toward the label
    mean shows as a flat curve regardless of its RMSE."""
    fig, ax = plt.subplots(figsize=COL)
    q = pd.qcut(preds["prediction"], 20, duplicates="drop")
    grouped = preds.groupby(q, observed=True).agg(
        pred=("prediction", "mean"), true=("target", "mean"), n=("target", "size")
    )
    ax.plot(
        grouped["pred"],
        grouped["true"],
        marker="o",
        color=BASE,
        label="observed (20 quantile bins)",
    )
    lim = max(grouped["pred"].max(), grouped["true"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color=FAKE_COLOR, lw=1.2, ls="--", label="perfect")
    ax.set_xlabel("Mean predicted $\\hat y$ in bin")
    ax.set_ylabel("Mean true $y$ in bin")
    ax.set_title("Calibration of the full-graph baseline")
    ax.legend(loc="upper left")
    style_axes(ax, grid_axis="both")
    savefig(fig, out, "rq1_calibration")


# --- Where it fails ----------------------------------------------------------
def residuals_by_size(preds: pd.DataFrame, out: Path) -> None:
    """Error against graph size. This sets the expectation for the reduction
    chapters: if error is already concentrated in the largest graphs, a
    reduction that acts hardest on large graphs starts from a worse position."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE)

    bins = np.logspace(
        np.log10(preds["num_nodes"].min()), np.log10(preds["num_nodes"].max()), 22
    )
    preds = preds.assign(size_bin=pd.cut(preds["num_nodes"], bins))
    grouped = preds.groupby("size_bin", observed=True).agg(
        med=("abs_error", "median"),
        q25=("abs_error", lambda s: s.quantile(0.25)),
        q75=("abs_error", lambda s: s.quantile(0.75)),
        centre=("num_nodes", "median"),
        n=("abs_error", "size"),
    )
    ax = axes[0]
    ax.fill_between(
        grouped["centre"], grouped["q25"], grouped["q75"], color=BASE, alpha=0.18
    )
    ax.plot(grouped["centre"], grouped["med"], color=BASE, marker="o", ms=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Nodes per graph (log)")
    ax.set_ylabel("Absolute error (log)")
    ax.set_title("Error vs. graph size (median, IQR band)")
    style_axes(ax, grid_axis="both")

    ax = axes[1]
    ax.plot(grouped["centre"], grouped["n"], color=INK_MUTED, marker="o", ms=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Nodes per graph (log)")
    ax.set_ylabel("Graphs in bin (log)")
    ax.set_title("How many graphs each size bin holds")
    style_axes(ax, grid_axis="both")

    fig.suptitle("Residual analysis by circuit scale", y=1.04)
    savefig(fig, out, "rq1_residuals_by_size")


def error_by_tier(preds: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=COL)
    tiers = sorted(set(preds["tier"].dropna()))
    data = [preds.loc[preds["tier"] == t, "abs_error"].to_numpy() for t in tiers]
    parts = ax.boxplot(
        data,
        tick_labels=[f"{t}\n(n={len(d):,})" for t, d in zip(tiers, data)],
        showfliers=False,
        patch_artist=True,
        widths=0.5,
    )
    for patch in parts["boxes"]:
        patch.set_facecolor(BASE)
        patch.set_alpha(0.35)
        patch.set_edgecolor(BASE)
    for key in ("whiskers", "caps", "medians"):
        for line in parts[key]:
            line.set_color(BASE)
    ax.set_yscale("log")
    ax.set_ylabel("Absolute error (log)")
    ax.set_title("Error by optimization tier")
    style_axes(ax)
    savefig(fig, out, "rq1_error_by_tier")


def error_by_design(preds: pd.DataFrame, out: Path) -> None:
    """Per-design metrics on the design-disjoint test set.

    A pooled test metric over a design-level split is an average over a handful
    of whole circuits. If the spread across designs is wide, the pooled number
    describes the mix rather than the model.
    """
    rows = []
    for design, sub in preds.groupby("design"):
        m = _metrics(sub["target"].to_numpy(), sub["prediction"].to_numpy())
        m.update(design=design, n=len(sub), mean_y=sub["target"].mean())
        rows.append(m)
    per_design = pd.DataFrame(rows).sort_values("rmse")

    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    y = np.arange(len(per_design))

    ax = axes[0]
    ax.barh(y, per_design["rmse"], color=BASE, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{d} (n={n:,})" for d, n in zip(per_design["design"], per_design["n"])],
        fontsize=7,
    )
    pooled = _metrics(preds["target"].to_numpy(), preds["prediction"].to_numpy())
    ax.axvline(pooled["rmse"], color=FAKE_COLOR, ls="--", lw=1.2)
    ax.text(
        pooled["rmse"],
        len(per_design) - 0.4,
        f" pooled {pooled['rmse']:.4f}",
        color=FAKE_COLOR,
        fontsize=7,
        va="top",
    )
    ax.set_xlabel("RMSE")
    ax.set_title("Per-design RMSE")
    style_axes(ax, grid_axis="x")

    ax = axes[1]
    # One design scores R^2 ~ -748; drawn to scale every other design vanishes.
    floor = -2.0
    colors = [FAKE_COLOR if v < 0 else BASE for v in per_design["r2"]]
    ax.barh(y, per_design["r2"], color=colors, height=0.65)
    ax.axvline(0, color=INK_SECONDARY, lw=0.8)
    clipped = clip_bar_axis(ax, per_design["r2"].tolist(), floor)
    # White-on-red, drawn over the clipped bar: at the axis edge in FAKE_COLOR
    # the label sits on a bar of the same colour and disappears. The trailing
    # "<" reads as "continues past here", not as the axis's own extent.
    for i, value in clipped:
        ax.text(floor, i, f"  {value:,.0f} <", fontsize=6, color="white",
                va="center", ha="left", fontweight="bold", zorder=5)
    ax.set_xlabel("$R^2$  (negative = worse than that design's own mean)")
    ax.set_title(
        "Per-design $R^2$" + (f"\naxis clipped at ${floor:g}$" if clipped else "")
    )
    style_axes(ax, grid_axis="x")

    fig.suptitle(
        "The pooled test score is an average over "
        f"{len(per_design)} unseen designs",
        y=1.04,
    )
    savefig(fig, out, "rq1_error_by_design")
    return per_design


# --- What it cost ------------------------------------------------------------
def training_curves(history: pd.DataFrame, out: Path) -> None:
    """Validation loss and $R^2$ per epoch for every training run.

    Colour is by reduction family, so the baseline's trajectory can be compared
    against the reduced-graph runs without nine distinct hues.
    """
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    # train_C2RS targets a different synthesis algorithm and is out of scope
    # (\ref{sec:intro:scope:algorithm}); it is dropped rather than plotted in a
    # colour that would read as an Orchestrate configuration.
    runs = sorted(n for n in set(history["name"]) if not n.startswith("train_C2RS"))
    r2_floor = -2.0
    clipped = 0

    for name in runs:
        sub = history[history["name"] == name].sort_values("epoch")
        is_baseline = name == "train_Orchestrate"
        published = name.startswith("train_baseline_")
        color = INK_MUTED if published else color_for(wandb_run_method(name))
        lw = 2.2 if is_baseline else 1.1
        ls = ":" if published else "-"
        alpha = 1.0 if is_baseline else 0.8
        axes[0].plot(
            sub["epoch"], sub["val_loss"], color=color, lw=lw, ls=ls, alpha=alpha
        )
        r2 = sub["val_r2"]
        clipped += int((r2 < r2_floor).sum())
        axes[1].plot(
            sub["epoch"], r2.clip(lower=r2_floor), color=color, lw=lw, ls=ls, alpha=alpha
        )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Smooth L1 loss")
    axes[0].set_yscale("log")
    axes[0].set_title("Validation loss")
    style_axes(axes[0], grid_axis="both")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation $R^2$")
    axes[1].axhline(0, color=INK_SECONDARY, lw=0.8, ls=":")
    axes[1].set_ylim(r2_floor - 0.15, 1.05)
    axes[1].set_title(
        "Validation $R^2$" + (f" ({clipped} points clipped at {r2_floor:g})" if clipped else "")
    )
    style_axes(axes[1], grid_axis="both")

    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=FAMILY_COLORS["baseline"], lw=2.2, label="Full graph"),
        Line2D([], [], color=FAMILY_COLORS["partition"], lw=1.4, label="Partition"),
        Line2D([], [], color=FAMILY_COLORS["sparsification"], lw=1.4, label="Sparsification"),
        Line2D([], [], color=INK_MUTED, lw=1.4, ls=":", label="Published baseline"),
    ]
    axes[1].legend(handles=handles, loc="lower right", ncols=2)

    fig.suptitle(
        f"Training trajectories, {len(runs)} Orchestrate runs "
        "(early stopping on validation loss)",
        y=1.04,
    )
    savefig(fig, out, "rq1_training_curves")


def training_cost(history: pd.DataFrame, out: Path) -> None:
    """Epoch wall-clock and epochs-to-best, per run. Two runs of the same
    nominal budget are not the same amount of compute if one stopped at epoch 4
    and the other at 10."""
    rows = []
    for name, sub in history.groupby("name"):
        # train_C2RS targets a different synthesis algorithm and is out of
        # scope; wandb_run_method maps it to "none", so leaving it in would
        # paint it in the full-graph baseline's colour.
        if name.startswith("train_C2RS") or sub["val_loss"].notna().sum() == 0:
            continue
        best = sub.loc[sub["val_loss"].idxmin()]
        rows.append(
            {
                "name": name,
                "method": wandb_run_method(name),
                "epochs": int(sub["epoch"].max()) + 1,
                "best_epoch": int(best["epoch"]),
                "epoch_time_s": sub["epoch_time_seconds"].median(),
                "total_h": sub["epoch_time_seconds"].sum() / 3600.0,
            }
        )
    cost = pd.DataFrame(rows).sort_values("total_h", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    y = np.arange(len(cost))
    labels = [n.replace("train_", "").replace("_", " ") for n in cost["name"]]

    ax = axes[0]
    ax.barh(y, cost["total_h"], color=[color_for(m) for m in cost["method"]], height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("Total training time (h)")
    ax.set_title("Wall-clock per run")
    style_axes(ax, grid_axis="x")

    ax = axes[1]
    ax.barh(y, cost["epochs"], color=INK_MUTED, height=0.65, label="epochs run")
    ax.barh(
        y,
        cost["best_epoch"] + 1,
        color=[color_for(m) for m in cost["method"]],
        height=0.65,
        label="epochs to best val loss",
    )
    ax.set_xlabel("Epochs")
    ax.set_title("Epochs run vs. epochs to best")
    ax.legend(loc="lower right")
    style_axes(ax, grid_axis="x")

    fig.suptitle("Training budget actually consumed per configuration", y=1.04)
    savefig(fig, out, "rq1_training_cost")


def hp_sweep(trials: pd.DataFrame, out: Path) -> None:
    """The Optuna sweep that fixed the architecture.

    The study databases were lost with the scratch workspaces, so this is
    reconstructed from the worker logs: trial ids restart per worker and pruned
    trials carry no value.
    """
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.7))
    done = trials[trials["state"] == "COMPLETE"].copy()
    # A handful of catastrophic trials (objective up to ~28) compress everything
    # else onto one line; the axis is clipped and the clipping is stated.
    shown = done[done["value"] <= 0.18]
    clipped = len(done) - len(shown)

    worker_colors = [
        FAMILY_COLORS["partition"],
        FAMILY_COLORS["sparsification"],
        FAMILY_COLORS["summarization"],
        FAMILY_COLORS["baseline"],
    ]
    ax = axes[0]
    for i, (worker, sub) in enumerate(shown.groupby("worker")):
        sub = sub.sort_values("trial")
        color = worker_colors[i % len(worker_colors)]
        short = worker.replace("final_hp_", "").replace("_worker", "w")
        ax.plot(sub["trial"], sub["value"], "o", ms=2.5, alpha=0.45, color=color)
        ax.plot(sub["trial"], sub["value"].cummin(), lw=1.4, color=color, label=short)
    ax.set_xlabel("Trial index within worker")
    ax.set_ylabel("Objective (val loss)")
    ax.set_title(f"Sweep progress ({clipped} outliers clipped)")
    ax.legend(loc="upper right", ncols=2, fontsize=6)
    style_axes(ax, grid_axis="both")

    ax = axes[1]
    by_encoder = [
        (enc, sub["value"].to_numpy())
        for enc, sub in shown.groupby("encoder_name")
        if len(sub) >= 2
    ]
    by_encoder.sort(key=lambda kv: np.median(kv[1]), reverse=True)
    parts = ax.boxplot(
        [v for _, v in by_encoder],
        tick_labels=[f"{k} (n={len(v)})" for k, v in by_encoder],
        showfliers=False,
        patch_artist=True,
        widths=0.55,
        vert=False,
    )
    for patch in parts["boxes"]:
        patch.set_facecolor(FAMILY_COLORS["partition"])
        patch.set_alpha(0.35)
        patch.set_edgecolor(FAMILY_COLORS["partition"])
    for key in ("whiskers", "caps", "medians"):
        for line in parts[key]:
            line.set_color(FAMILY_COLORS["partition"])
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_xlabel("Objective (val loss)")
    ax.set_title("Objective by encoder")
    style_axes(ax, grid_axis="x")

    ax = axes[2]
    counts = {
        "completed": int(((trials["state"] == "COMPLETE")).sum()),
        "pruned\n(memory)": int((trials["oom_like"]).sum()),
        "pruned\n(other)": int(
            ((trials["state"] != "COMPLETE") & (~trials["oom_like"])).sum()
        ),
    }
    colors = [INK_MUTED, FAKE_COLOR, INK_MUTED]
    ax.bar(list(counts), list(counts.values()), color=colors, width=0.6)
    for i, v in enumerate(counts.values()):
        ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_ylabel("Trials")
    ax.set_ylim(0, max(counts.values()) * 1.18)
    ax.set_title(f"Trial outcomes ({len(trials)} total)")
    style_axes(ax)

    fig.suptitle(
        "Hyperparameter search: memory pressure is the dominant failure mode",
        y=1.06,
    )
    savefig(fig, out, "rq1_hp_sweep")


# --- Baselines ---------------------------------------------------------------
def trivial_baselines(val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Tier-1 of the RQ1 comparison: what $R^2 = 0$ looks like.

    Each predictor is fitted on the validation split and scored on test, so the
    comparison against the encoder is like-for-like.
    """
    y_test = test["target"].to_numpy()
    rows = []

    for name, const in [
        ("Predict validation mean", float(val["target"].mean())),
        ("Predict validation median", float(val["target"].median())),
    ]:
        m = _metrics(y_test, np.full_like(y_test, const))
        m["model"] = name
        rows.append(m)

    # Size-only ordinary least squares on log node and edge counts.
    def design(df: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                np.ones(len(df)),
                np.log10(df["num_nodes"].to_numpy()),
                np.log10(df["num_edges"].to_numpy()),
            ]
        )

    coef, *_ = np.linalg.lstsq(design(val), val["target"].to_numpy(), rcond=None)
    pred = np.clip(design(test) @ coef, 0.0, 1.0)
    m = _metrics(y_test, pred)
    m["model"] = "Graph size only (OLS)"
    rows.append(m)

    for row in rows:
        row["tier"] = "1: trivial"

    m = _metrics(y_test, test["prediction"].to_numpy())
    m["model"] = "Edge-aware GCN+ (ours)"
    # Not a tier: the encoder is the thing every tier is a baseline FOR, and
    # filing it under "trivial" in the generated table said the opposite.
    m["tier"] = "This work"
    rows.append(m)

    return pd.DataFrame(rows)[["model", "tier", "rmse", "mae", "r2", "spearman"]]


def baseline_tiers(tier1: pd.DataFrame, tier23: pd.DataFrame, out: Path) -> None:
    """The three-tier RQ1 comparison in one figure.

    Tier 1 is computed from the persisted predictions. Tier 2 (standard GNN
    encoders under identical training) has never been run, and tier 3 exists
    only for SynthNet — so both are drawn from placeholder numbers and the
    figure is marked accordingly.
    """
    combined = pd.concat([tier1, tier23], ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(6.9, row_height(len(combined))), sharey=True)
    y = np.arange(len(combined))
    labels = [
        f"{m}{'' if real else '  [FAKE]'}"
        for m, real in zip(combined["model"], combined["measured"].fillna(True))
    ]

    for ax, col, name in zip(
        axes, ["rmse", "r2", "spearman"], ["RMSE", "$R^2$", "Spearman $\\rho$"]
    ):
        colors = [
            BASE if real else FAKE_COLOR
            for real in combined["measured"].fillna(True)
        ]
        hatches = [
            None if real else "xxx" for real in combined["measured"].fillna(True)
        ]
        bars = ax.barh(y, combined[col], color=colors, height=0.65)
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
                bar.set_edgecolor("white")
        ax.axvline(0, color=INK_SECONDARY, lw=0.8)
        ax.set_title(name)
        style_axes(ax, grid_axis="x")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=6.5)
    axes[0].invert_yaxis()
    fig.suptitle("RQ1 three-tier baseline comparison (test split)", y=1.05)
    mark_fake(
        fig,
        note="tier 2 and DeepGate4/HOGA rows are invented; only tier 1 and SynthNet are measured",
    )
    savefig(fig, out, "rq1_baseline_tiers")


def hp_parameters(trials: pd.DataFrame, out: Path) -> None:
    """Objective against each searched hyperparameter.

    52 completed trials over 12 well-covered parameters, so this is a coarse
    sensitivity read rather than an importance analysis: Optuna's sampler
    concentrates on promising regions, and the sampler state was lost with the
    scratch workspaces. Read it as "which settings never produced a good trial",
    not as a controlled sweep.
    """
    done = trials[(trials["state"] == "COMPLETE") & (trials["value"] <= 0.18)]
    if done.empty:
        return
    categorical = ["encoder_name", "pe_type", "pooling_type", "norm_type", "jk_mode"]
    numeric = ["batch_size", "hidden_dim", "num_layers", "lr", "dropout", "weight_decay"]
    best = done["value"].min()

    fig, axes = plt.subplots(2, 6, figsize=(6.9, 3.6))

    for ax, col in zip(axes[0], numeric):
        ax.plot(done[col], done["value"], "o", ms=3, alpha=0.55,
                color=FAMILY_COLORS["partition"])
        if col in {"lr", "weight_decay"}:
            ax.set_xscale("log")
        ax.axhline(best, color=FAKE_COLOR, lw=0.9, ls="--")
        ax.set_title(col.replace("_", " "), fontsize=7)
        ax.tick_params(labelsize=5.5)
        style_axes(ax, grid_axis="both")
    axes[0][0].set_ylabel("objective", fontsize=7)

    for ax, col in zip(axes[1], categorical + [None]):
        if col is None:
            ax.axis("off")
            continue
        groups = [(k, g["value"].to_numpy()) for k, g in done.groupby(col) if len(g) >= 2]
        groups.sort(key=lambda kv: np.median(kv[1]))
        for i, (_, values) in enumerate(groups):
            ax.plot(np.full(len(values), i) + np.linspace(-0.12, 0.12, len(values)),
                    values, "o", ms=3, alpha=0.55, color=FAMILY_COLORS["sparsification"])
            ax.plot([i - 0.25, i + 0.25], [np.median(values)] * 2,
                    color=INK_SECONDARY, lw=1.4)
        ax.axhline(best, color=FAKE_COLOR, lw=0.9, ls="--")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([k for k, _ in groups], fontsize=5, rotation=40, ha="right")
        ax.set_title(col.replace("_", " "), fontsize=7)
        ax.tick_params(axis="y", labelsize=5.5)
        style_axes(ax)
    axes[1][0].set_ylabel("objective", fontsize=7)

    fig.suptitle(
        f"Hyperparameter sensitivity over {len(done)} completed trials "
        "(dashed: best objective; bar: group median)",
        y=1.02,
    )
    savefig(fig, out, "rq1_hp_parameters")


# --- Performance as a function of how optimizable the circuit is -------------
TARGET_BINS = [-0.001, 1e-9, 0.01, 0.05, 0.10, 0.20, 1.0]
TARGET_LABELS = ["$y = 0$", "(0, .01]", "(.01, .05]", "(.05, .10]",
                 "(.10, .20]", "$> .20$"]


def _bin_targets(df: pd.DataFrame) -> pd.Series:
    return pd.cut(df["target"], TARGET_BINS, labels=TARGET_LABELS)


def error_by_target_bin(preds: pd.DataFrame, out: Path) -> None:
    """Baseline performance as a function of how optimizable the circuit is.

    $R^2$ is undefined inside a narrow bin, so the diagnostic quantities here
    are the mean prediction against the bin's mean truth (regression toward the
    label mean shows as a flat line), the absolute error, and the signed bias.
    """
    df = preds.assign(bin=_bin_targets(preds))
    grouped = df.groupby("bin", observed=True).agg(
        n=("target", "size"), true=("target", "mean"), pred=("prediction", "mean"),
        mae=("abs_error", "mean"),
        bias=("prediction", lambda s: s.mean()),
    )
    grouped["bias"] = grouped["pred"] - grouped["true"]
    x = np.arange(len(grouped))

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.6))

    ax = axes[0]
    ax.bar(x, grouped["n"], color=INK_MUTED, width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Graphs (log)")
    ax.set_title("How many circuits\nin each band")
    style_axes(ax)

    ax = axes[1]
    ax.plot(x, grouped["true"], marker="o", color=INK_SECONDARY, label="true mean")
    ax.plot(x, grouped["pred"], marker="s", color=BASE, label="predicted mean")
    ax.set_ylabel("Optimizability")
    ax.set_title("Prediction against truth\nby band")
    ax.legend(loc="upper left", fontsize=6)
    style_axes(ax, grid_axis="both")

    ax = axes[2]
    colors = [FAKE_COLOR if b < 0 else FAMILY_COLORS["partition"] for b in grouped["bias"]]
    ax.bar(x, grouped["bias"], color=colors, width=0.6)
    ax.axhline(0, color=INK_SECONDARY, lw=0.9)
    ax.set_ylabel("Predicted $-$ true")
    ax.set_title("Signed bias\n(red = under-predicts)")
    style_axes(ax)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index, fontsize=5.5, rotation=35, ha="right")

    fig.suptitle(
        "Full-graph baseline by how much the circuit actually needs optimizing", y=1.08
    )
    savefig(fig, out, "rq1_error_by_target_bin")


def variance_decomposition(preds: pd.DataFrame, out: Path) -> None:
    """How much of the headline $R^2$ is reachable by group identity alone.

    $R^2$ is a ratio against the pooled label variance, and that variance has a
    between-group and a within-group part. If a grouping carries a large share
    of it, a model that only learns to recognise the group scores well without
    having learned anything about individual circuits. This figure puts the
    model's score next to what a group-mean lookup achieves, for two groupings.
    """
    y, p = preds["target"].to_numpy(), preds["prediction"].to_numpy()
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = ((p - y) ** 2).sum()
    pooled_r2 = 1 - ss_res / ss_tot

    groupings = [("tier", "Dataset tier"), ("source_algorithm", "Source script")]
    rows = []
    for col, name in groupings:
        means = preds.groupby(col)["target"].transform("mean").to_numpy()
        ss_within = ((y - means) ** 2).sum()
        rows.append({
            "name": name,
            "between_share": ((means - y.mean()) ** 2).sum() / ss_tot,
            "group_mean_r2": 1 - ((means - y) ** 2).sum() / ss_tot,
            "within_r2": 1 - ss_res / ss_within,
        })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=WIDE)

    ax = axes[0]
    x = np.arange(len(df))
    ax.bar(x, df["between_share"], width=0.55, color=FAMILY_COLORS["partition"])
    for xi, v in zip(x, df["between_share"]):
        ax.text(xi, v, f"{v:.1%}", ha="center", va="bottom", fontsize=7,
                color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(df["name"], fontsize=7)
    ax.set_ylim(0, max(df["between_share"]) * 1.35)
    ax.set_ylabel("Share of label variance\nbetween groups")
    ax.set_title("Variance carried\nby each grouping")
    style_axes(ax)

    ax = axes[1]
    width = 0.26
    ax.bar(x - width, df["group_mean_r2"], width=width * 0.9, color=INK_MUTED,
           label="predict the group mean only")
    ax.bar(x, [pooled_r2] * len(df), width=width * 0.9, color=BASE,
           label="the model, pooled")
    ax.bar(x + width, df["within_r2"], width=width * 0.9,
           color=FAMILY_COLORS["sparsification"], label="the model, within groups")
    ax.axhline(0, color=INK_SECONDARY, lw=0.8)
    for xi, row in zip(x, df.itertuples()):
        for dx, v in [(-width, row.group_mean_r2), (0, pooled_r2), (width, row.within_r2)]:
            ax.text(xi + dx, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6,
                    color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(df["name"], fontsize=7)
    ax.set_ylabel("$R^2$")
    ax.set_ylim(top=max(pooled_r2, df["group_mean_r2"].max()) * 1.45)
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_title("What survives when\nthe group is held fixed")
    style_axes(ax)

    fig.suptitle(
        "Decomposing the full-graph baseline's $R^2$ against two groupings", y=1.06
    )
    savefig(fig, out, "rq1_variance_decomposition")


def build(
    preds_test: pd.DataFrame,
    preds_val: pd.DataFrame,
    history: pd.DataFrame,
    trials: pd.DataFrame,
    tier23: pd.DataFrame,
    out: Path,
) -> pd.DataFrame:
    parity(preds_test, out)
    variance_decomposition(preds_test, out)
    calibration(preds_test, out)
    residuals_by_size(preds_test, out)
    error_by_tier(preds_test, out)
    error_by_target_bin(preds_test, out)
    per_design = error_by_design(preds_test, out)
    training_curves(history, out)
    training_cost(history, out)
    hp_sweep(trials, out)
    hp_parameters(trials, out)
    tier1 = trivial_baselines(preds_val, preds_test)
    baseline_tiers(tier1, tier23, out)
    return tier1, per_design
