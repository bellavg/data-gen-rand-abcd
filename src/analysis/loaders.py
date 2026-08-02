"""Loading and light derivation over the exported result CSVs.

The inference / benchmark / offline-stats loaders already exist in
``results_to_latex`` and are re-exported here rather than reimplemented. What
this module adds is everything those tables never needed: per-graph
predictions with tier and design recovered from the path, the W&B run and
history exports, and the Optuna trial log.

Expected layout under ``--results-dir`` (extract results/archives/*.tar.gz):

    results/inference_results/*.csv     test.py per-config metrics
    results/training_benchmark/*.csv    benchmark.py per-config aggregates
    results/benchmark_per_graph/*.csv   benchmark.py per-graph rows
    results/predictions/*.csv           test.py per-graph predictions
    results/measurements/*.csv          measure_sparsity.py / measure_partition.py
    results/wandb_export/*.csv          W&B run summary + training history
    results/hp_tuning/*.log|*.out       Optuna sweep logs
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.results_to_latex import (  # noqa: F401  (re-exported)
    _method_matches,
    load_benchmark_per_graph,
    load_inference_results,
    load_offline_stats,
    load_training_benchmark,
)


def load_all_inference(directory: Path, *, device: str = "cuda") -> pd.DataFrame:
    """Every inference row for one device, val split included.

    ``load_inference_results`` deliberately drops the val split because the
    thesis tables report test only. The val rows are still worth plotting as a
    generalisation-gap diagnostic, so this variant keeps them.

    The device filter is not optional. Downstream figures pivot these rows with
    a mean aggregation, so a CPU pass sitting next to its GPU counterpart would
    be silently averaged into it: every accuracy and throughput number in RQ3
    and RQ5 would be the mean of two different machines. Rows written before
    the column existed are treated as CUDA, which is what they were.
    """
    if not directory.is_dir():
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in sorted(directory.glob("*.csv"))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["split"] = df["split"].fillna("test")
    if "device" in df.columns:
        df = df[df["device"].fillna("cuda") == device].reset_index(drop=True)
    return df


# --- Predictions -------------------------------------------------------------
_PRED_STEM = re.compile(r"^(?P<algo>[^_]+)_(?P<rest>.*)$")


def parse_prediction_stem(stem: str) -> dict:
    """``Orchestrate_and_gate_only_matched_reduction_val`` ->
    method ``and_gate_only``, eval mode ``matched_reduction``, split ``val``."""
    rest = _PRED_STEM.match(stem).group("rest")
    split = "test"
    if rest.endswith("_val"):
        split, rest = "val", rest[: -len("_val")]
    for mode in ("matched_reduction", "full_graph"):
        if rest.endswith(mode):
            method = rest[: -len(mode)].rstrip("_") or "none"
            return {"reduction_method": method, "eval_mode": mode, "split": split}
    return {"reduction_method": rest or "none", "eval_mode": "unknown", "split": split}


def annotate_graph_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Recover tier, source design and source algorithm from the cached path.

    Paths are ``.../graphs/tier0/<design>/<file>.pt`` and
    ``.../graphs/tier1/<algorithm>/<design>/<file>.pt``, so the design is
    always the parent directory and the tier is the first component under
    ``graphs/``.

    ``source_algorithm`` is which synthesis script produced a tier-1 graph, and
    it is not cosmetic: it is the strongest single predictor of the label. A
    graph already optimized by C2RS or Deepsyn has almost nothing left for
    Orchestrate to remove, while a Syn4 graph has plenty.
    """
    parts = df["graph_id"].str.split("/")
    df = df.copy()
    df["design"] = parts.str[-2]
    df["tier"] = df["graph_id"].str.extract(r"/(tier\d)/", expand=False)
    df["source_algorithm"] = (
        df["graph_id"].str.extract(r"/tier1/([^/]+)/", expand=False).fillna("tier0 base")
    )
    return df


# Designs on which Orchestrate is at a fixed point: across all ~16k graphs each,
# the largest optimizability observed is 0.0003. They contribute no variance for
# R^2 to explain, only noise for it to divide by.
DEAD_DESIGNS = ("16384", "8192")


def label_strata(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks for the subsets every accuracy claim should be checked on.

    The pooled label is ~73% below 0.005 and ~49% exactly zero on the test
    split, so a pooled R^2 is dominated by a minority of graphs. Re-scoring the
    same persisted predictions on these strata costs nothing and shows which
    conclusions are properties of the model rather than of the label skew.
    """
    return {
        "all": np.ones(len(df), dtype=bool),
        "live_designs": ~df["design"].isin(DEAD_DESIGNS).to_numpy(),
        "optimizable": (df["target"] > 0).to_numpy(),
        "syn4_source": (df["source_algorithm"] == "Syn4").to_numpy(),
    }


STRATUM_LABELS = {
    "all": "All test graphs",
    "live_designs": "Designs Orchestrate can touch",
    "optimizable": "Graphs with $y > 0$",
    "syn4_source": "Syn4-derived graphs only",
}


def grouped_metrics(
    predictions_dir: Path, by: str, split: str = "test", min_n: int = 200
) -> pd.DataFrame:
    """Re-score every persisted prediction file within each level of ``by``.

    Unlike :func:`stratified_metrics`, whose subsets overlap and answer a
    robustness question, this partitions the test set: one row per
    (configuration, evaluation mode, group). Use it to ask whether a reduction
    costs more on one part of the corpus than another --- by tier, or by which
    synthesis script produced the graph.

    Groups smaller than ``min_n``, or whose labels are constant, yield NaN
    rather than a metric computed on too little variance to mean anything.
    """
    from scipy.stats import spearmanr

    rows = []
    for path in sorted(predictions_dir.glob("*.csv")):
        tags = parse_prediction_stem(path.stem)
        if tags["split"] != split:
            continue
        df = annotate_graph_ids(
            pd.read_csv(path, usecols=["graph_id", "num_nodes", "target", "prediction"])
        )
        for group, sub in df.groupby(by):
            y, p = sub["target"].to_numpy(), sub["prediction"].to_numpy()
            usable = len(y) >= min_n and y.std() > 0
            rows.append({
                "reduction_method": tags["reduction_method"],
                "eval_mode": tags["eval_mode"],
                by: group,
                "n": len(y),
                "rmse": float(np.sqrt(((p - y) ** 2).mean())) if usable else np.nan,
                "r2": float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
                if usable else np.nan,
                "spearman": float(spearmanr(y, p).statistic) if usable else np.nan,
                "mean_y": float(y.mean()),
            })
        del df
    return pd.DataFrame(rows)


def banded_metrics(predictions_dir: Path, split: str = "test") -> pd.DataFrame:
    """Per-configuration behaviour within bands of the true target.

    Inside a narrow band there is almost no label variance, so R^2 is not
    reported: the meaningful quantities are the mean prediction against the
    band's mean truth and the mean absolute error.
    """
    from analysis.fig_rq1 import TARGET_BINS, TARGET_LABELS

    rows = []
    for path in sorted(predictions_dir.glob("*.csv")):
        tags = parse_prediction_stem(path.stem)
        if tags["split"] != split:
            continue
        df = pd.read_csv(path, usecols=["target", "prediction", "abs_error"])
        df["bin"] = pd.cut(df["target"], TARGET_BINS, labels=TARGET_LABELS)
        for band, sub in df.groupby("bin", observed=True):
            rows.append({
                "reduction_method": tags["reduction_method"],
                "eval_mode": tags["eval_mode"],
                "bin": band,
                "n": len(sub),
                "true": float(sub["target"].mean()),
                "pred": float(sub["prediction"].mean()),
                "mae": float(sub["abs_error"].mean()),
                "bias": float((sub["prediction"] - sub["target"]).mean()),
            })
        del df
    out = pd.DataFrame(rows)
    if not out.empty:
        out["bin"] = pd.Categorical(out["bin"], categories=TARGET_LABELS, ordered=True)
    return out


def stratified_metrics(predictions_dir: Path, split: str = "test") -> pd.DataFrame:
    """Re-score every persisted prediction file on each label stratum.

    Files are read and released one at a time: the prediction set is ~340 MB
    and holding all 26 frames at once is what makes this unusable on a laptop.
    """
    from scipy.stats import spearmanr

    def metrics(y: np.ndarray, p: np.ndarray) -> dict:
        if len(y) < 3 or y.std() == 0:
            return {"n": len(y), "rmse": np.nan, "r2": np.nan, "spearman": np.nan}
        return {
            "n": len(y),
            "rmse": float(np.sqrt(((p - y) ** 2).mean())),
            "r2": float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "spearman": float(spearmanr(y, p).statistic),
        }

    rows = []
    for path in sorted(predictions_dir.glob("*.csv")):
        tags = parse_prediction_stem(path.stem)
        if tags["split"] != split:
            continue
        df = annotate_graph_ids(
            pd.read_csv(path, usecols=["graph_id", "num_nodes", "target", "prediction"])
        )
        y, p = df["target"].to_numpy(), df["prediction"].to_numpy()
        row = {"reduction_method": tags["reduction_method"], "eval_mode": tags["eval_mode"]}
        for name, mask in label_strata(df).items():
            for key, value in metrics(y[mask], p[mask]).items():
                row[f"{name}_{key}"] = value
        rows.append(row)
        del df
    return pd.DataFrame(rows)


def load_predictions(
    predictions_dir: Path,
    *,
    method: str | None = None,
    eval_mode: str | None = None,
    split: str = "test",
) -> pd.DataFrame:
    """Per-graph predictions, optionally filtered before anything is read.

    The prediction CSVs total ~340 MB, so the filter is applied to filenames
    rather than to a concatenated frame — loading all 26 at once is what makes
    this script unusable on a laptop.
    """
    if not predictions_dir.is_dir():
        print(f"[loaders] {predictions_dir} not found — no predictions.")
        return pd.DataFrame()
    frames = []
    for path in sorted(predictions_dir.glob("*.csv")):
        tags = parse_prediction_stem(path.stem)
        if method is not None and tags["reduction_method"] != method:
            continue
        if eval_mode is not None and tags["eval_mode"] != eval_mode:
            continue
        if split is not None and tags["split"] != split:
            continue
        df = pd.read_csv(path)
        for key, value in tags.items():
            df[key] = value
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return annotate_graph_ids(pd.concat(frames, ignore_index=True))


# --- W&B export --------------------------------------------------------------
def load_wandb_runs(wandb_dir: Path) -> pd.DataFrame:
    path = wandb_dir / "runs.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def load_wandb_history(wandb_dir: Path) -> pd.DataFrame:
    """Per-epoch training history, collapsed to one row per (run, epoch).

    W&B logs step-level and epoch-level metrics into the same table, so most
    rows carry only a subset of the columns. Grouping by epoch and taking the
    first non-null value of each column reassembles the epoch record.
    """
    path = wandb_dir / "train_history.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["epoch"].notna()]
    numeric = [c for c in df.columns if c not in {"run_id", "name"}]
    grouped = (
        df.groupby(["name", "epoch"], as_index=False)[numeric]
        .first()
        .sort_values(["name", "epoch"])
    )
    return grouped


def wandb_run_method(name: str) -> str:
    """Map a W&B run name onto a METHODS key."""
    from analysis.style import METHODS

    for key in sorted(METHODS, key=len, reverse=True):
        if key != "none" and name.endswith(key):
            return key
    return "none"


# --- Optuna sweep ------------------------------------------------------------
_TRIAL_HEADER = re.compile(r"^TRIAL (\d+) STARTED")
_TRIAL_PARAM = re.compile(r"^\s{2}(\w+): (.+)$")
_SELECTION = re.compile(
    r"^\[selection\] trial=(\d+) state=(\w+) outcome=(\w+) oom_like=(\w+)"
)
_VALUE = re.compile(r"value=([\d.]+|n/a)")


def load_hp_trials(hp_dir: Path) -> pd.DataFrame:
    """Parse the Optuna sweep logs into one row per trial.

    The SQLite studies were lost with the scratch workspaces, so the log text
    is the only surviving record. Each worker log restarts trial numbering at
    zero, hence the ``worker`` column — trial ids are unique only within a
    worker.
    """
    rows = []
    for path in sorted(hp_dir.glob("*")):
        if path.suffix not in {".log", ".out"}:
            continue
        worker = path.stem
        current: dict | None = None
        params: dict[int, dict] = {}
        for line in path.read_text(errors="replace").splitlines():
            header = _TRIAL_HEADER.match(line)
            if header:
                current = {"trial": int(header.group(1))}
                params[current["trial"]] = current
                continue
            if current is not None:
                param = _TRIAL_PARAM.match(line)
                if param:
                    key, raw = param.group(1), param.group(2).strip()
                    try:
                        current[key] = float(raw)
                    except ValueError:
                        current[key] = raw
                    continue
                if line.startswith("="):
                    current = None
            sel = _SELECTION.match(line)
            if sel:
                trial = int(sel.group(1))
                value = _VALUE.search(line)
                row = dict(params.get(trial, {"trial": trial}))
                row.update(
                    worker=worker,
                    state=sel.group(2),
                    outcome=sel.group(3),
                    oom_like=sel.group(4) == "True",
                    value=(
                        float(value.group(1))
                        if value and value.group(1) != "n/a"
                        else np.nan
                    ),
                )
                rows.append(row)
    return pd.DataFrame(rows)


# --- Derived views -----------------------------------------------------------
def offline_stats(measurements_dir: Path) -> pd.DataFrame:
    """One row per reduction method: mean node/edge retention and offline cost.

    Partitioning keeps every node by construction, so its node retention is 1.0
    and its edge retention is ``1 - edge_cut_ratio``; the two families' CSVs
    record different columns and are normalised onto the same schema here.
    """
    frames = []
    sparse = load_offline_stats(measurements_dir, "sparsification_stats")
    if not sparse.empty:
        agg = sparse.groupby("reduction_method").agg(
            node_retention=("node_retention", "mean"),
            node_retention_std=("node_retention", "std"),
            edge_retention=("edge_retention", "mean"),
            edge_retention_std=("edge_retention", "std"),
            offline_s=("time_s", "mean"),
            offline_s_std=("time_s", "std"),
        )
        agg["reduction_type"] = "sparsification"
        frames.append(agg.reset_index())

    part = load_offline_stats(measurements_dir, "partition_stats")
    if not part.empty:
        part = part.assign(edge_retention=1.0 - part["edge_cut_ratio"], node_retention=1.0)
        agg = part.groupby("reduction_method").agg(
            node_retention=("node_retention", "mean"),
            node_retention_std=("node_retention", "std"),
            edge_retention=("edge_retention", "mean"),
            edge_retention_std=("edge_retention", "std"),
            offline_s=("time_s", "mean"),
            offline_s_std=("time_s", "std"),
            num_partitions=("num_partitions", "mean"),
            balance_std=("std_nodes_per_partition", "mean"),
        )
        agg["reduction_type"] = "partition"
        frames.append(agg.reset_index())

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
