"""Non-learned reference points for RQ1: what R^2 = 0 is worth on this label.

Seven predictors are scored on TEST. Six are fair competitors, seeing only what
the encoder sees; the seventh is an oracle. All are fitted on the TRAIN split,
except ``zero``, which is an a-priori hypothesis and is fitted on nothing.

    mean        constant, the train target mean
    median      constant, the train target median
    zero        constant 0, the hypothesis that the script removes nothing
    size        ordinary least squares on log node count and log edge count
    counts      ordinary least squares on the sum-pooled node-type count vector
    size_level  train mean within a cell of size-by-level quantile bins

    source_script_mean   ORACLE: train mean of the script that produced the graph

``counts`` is the structure-agnostic reference of Errica et al. (ICLR 2020),
transposed from molecules to AIGs. Their Molecular Fingerprint sum-pools node
features over the graph and fits a small model on the result, so it reads
composition and no topology; a significant gain over it is what shows a GNN
exploited the graph. The node feature here is the four-way type one-hot over
(constant, PI, AND, PO), so summing it gives exactly the type counts, and this
is therefore the ceiling of what those features carry with zero message passing.
It also spans ``size``: the node and edge totals are both linear functions of the
count vector, which is why ``size`` is reported but no longer the load-bearing
non-constant reference.

``size_level`` is the one rung that is NOT structure agnostic, since the maximum
topological level is a topological quantity. It asks whether a learned structure
reader is needed at all, or whether one scalar structural summary already does
the work, and it asks it in the most generous form available without a graph
pass: a lookup table of train means over quantile cells, which is close to the
nonparametric optimum on two features at this sample size. Mean and standard
deviation of the level distribution would sharpen it but are not in the CSV, and
computing them means loading every graph, which this script deliberately does not
do.

``source_script_mean`` is an oracle and is written to a SEPARATE file, never the
one holding the fair competitors, so it cannot be ranked against the encoder by
accident. It reads which synthesis script produced each input graph, provenance
the encoder never sees and a practitioner holding one unoptimized AIG would not
have. Since the source script is the strongest single determinant of the target,
this bounds how much of an achievable R^2 is dataset construction rather than
circuit structure. The design-conditional mean is the other obvious oracle and is
not computed here: the split is by design, so every test design is unseen and its
train mean does not exist.

Fitting on train is the standard contract, not a preference. scikit-learn's
DummyRegressor -- the reference implementation every regression baseline is
measured against -- documents ``strategy="mean"`` as "always predicts the mean
of the training set" and stores it in ``constant_``, "Mean or median or quantile
of the training targets". A baseline exists to answer "what does the data alone
give you, under the same information the model had", and the model is fitted on
train. Fitting the baseline on validation instead hands it a different
information set than the model, which is what makes the comparison not
like-for-like; fitting it on test hands it the labels it is scored against,
which makes it an oracle rather than a baseline.

R^2 is reported against the TEST set's own mean, matching test.py and
sklearn.metrics.r2_score. That is what puts the zero point where it belongs: a
constant taken from any other split scores R^2 <= 0, with equality only where
the train and test means happen to coincide.

Nothing here loads a graph or runs a model. Every input is a column the
generation pipeline already wrote (data/creation/shell/10_algorithm_csv.sh):
``optimizability`` is the target, ``pre_depth`` the maximum topological level,
and both size totals come from the pre-optimization counts: the edge count is
``2 * pre_nodes + pre_num_PO`` and the node count is
``1 + pre_num_PI + pre_nodes + pre_num_PO`` as in data/dataset.py. Split
membership is READ from the splits JSON the training run wrote, never recomputed,
so the train set here is the train set the model saw.

Depends on numpy, pandas and scipy, plus config for the W&B and split-strategy
constants, which imports nothing but the standard library. It deliberately does
not import data.dataset to derive the splits filename: that module pulls in
torch and PyTorch Geometric, which costs about a minute of import time for a
script that never touches a tensor. Point ``--splits_path`` at the file instead,
which also removes the failure mode where a reconstructed filename silently
resolves to a different run's split.

Results go to W&B as ``baseline_trivial_<algorithm>``, one run holding every
predictor as a table, with per-predictor summary keys namespaced by role so an
oracle cannot be globbed into a panel of fair competitors. ``--wandb false``
skips it.

Usage:

    python src/trivial_baselines.py \\
        --csv_paths /path/to/Orchestrate.csv \\
        --splits_path /scratch-shared/$USER/aig_cache/Orchestrate_all_splits.json \\
        --out results/rq1_trivial_baselines.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config

# abc `print_stats` columns this script reads. The node-count triple is the same
# one data/dataset.py sizes graphs from; the rest are read here only.
_NODE_COUNT_COLS = ("pre_nodes", "pre_num_PI", "pre_num_PO")
_REQUIRED_COLS = (
    "unoptimized_graph_path",
    "optimizability",
    "pre_depth",
    *_NODE_COUNT_COLS,
)

# Quantile bins per axis for the size-by-level lookup. Ten squared is 100 cells
# against roughly 10^5 train graphs, so a typical cell still holds hundreds of
# graphs and its mean is not fitted to noise.
_N_BINS = 10


def _normalize_graph_path(graph_path: str) -> str:
    """Apply the rewrite AIGGraphRegressionDataset._normalize_graph_path applies.

    The splits JSON stores paths that already went through it, so CSV paths must
    too or nothing joins. data/dataset.py owns this rule; this is a second copy.
    """
    return str(graph_path).replace("/gpfs/scratch1/shared", "/scratch-shared")


def load_rows(csv_paths: list[str]) -> pd.DataFrame:
    """Per-graph target, type counts and level, straight from the generation CSVs."""
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        missing = [c for c in _REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing column(s) {missing}. Found: {sorted(df.columns)}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    graph_path = df["unoptimized_graph_path"].fillna("").map(_normalize_graph_path)
    out = pd.DataFrame(
        {
            "graph_path": graph_path,
            "target": pd.to_numeric(df["optimizability"], errors="coerce"),
            # The four-way node-type one-hot, sum-pooled. The constant node is
            # the fourth entry and is 1 for every AIG, so it carries no variance
            # and is absorbed into the intercept rather than given a column.
            "num_pi": pd.to_numeric(df["pre_num_PI"], errors="coerce"),
            "num_and": pd.to_numeric(df["pre_nodes"], errors="coerce"),
            "num_po": pd.to_numeric(df["pre_num_PO"], errors="coerce"),
            "num_nodes": 1
            + sum(pd.to_numeric(df[c], errors="coerce") for c in _NODE_COUNT_COLS),
            # Computed from the pre-optimization counts, NOT read from the CSV's
            # `edges` column. That column is a POST-optimization statistic: a
            # tier-1/tier-2 row is written with edges=0 and the node/PO counts of
            # the OPTIMIZED graph (data/creation/generate_csv.py), and
            # 8_full_csv.sh then imputes edges as 2*nodes + num_PO from that same
            # row. So the stored value is 2*post_nodes + post_num_PO, and since
            # optimizability = (pre_nodes - post_nodes)/pre_nodes, a model given
            # log10(edges) and log10(nodes) can reconstruct the target. Only the
            # tier-0 rows, which this CSV does not contain, carry a pre-optimization
            # edge count.
            "num_edges": 2 * pd.to_numeric(df["pre_nodes"], errors="coerce")
            + pd.to_numeric(df["pre_num_PO"], errors="coerce"),
            "level": pd.to_numeric(df["pre_depth"], errors="coerce"),
            # Which synthesis script produced the input graph. Tier-1 rows take a
            # tier-0 base as input and so have no upstream script; tier-2 rows
            # take `.../graphs/tier1/<script>/<design>/...`. Same rule as
            # analysis.loaders.annotate_graph_ids on the thesis-outline branch,
            # including its "tier0 base" fallback label, so the two agree if the
            # results tables are ever built from both.
            "source_script": graph_path.str.extract(r"/tier1/([^/]+)/", expand=False).fillna(
                "tier0 base"
            ),
        }
    )
    if "step_id" in df.columns:
        out["step_id"] = pd.to_numeric(df["step_id"], errors="coerce")

    # Level is deliberately NOT in this list. Only the size-by-level lookup uses
    # it, so a patchy `pre_depth` column must not shrink the corpus the other
    # six predictors are fitted and scored on.
    required = ["target", "num_pi", "num_and", "num_po", "num_nodes", "num_edges"]
    dropped = int(out[required].isna().any(axis=1).sum())
    if dropped:
        print(f"[trivial] dropping {dropped} row(s) with a missing target or size stat")
    out = out.dropna(subset=required)
    no_level = int(out["level"].isna().sum())
    if no_level:
        print(
            f"[trivial] {no_level} row(s) have no topological level; they are kept, and the "
            "size-by-level lookup scores them from its own group rather than a size cell"
        )
    return out[(out["num_nodes"] > 0) & (out["num_edges"] > 0)].reset_index(drop=True)


def load_splits(splits_path: Path) -> dict[str, set[str]]:
    """The train/val/test assignment the training run wrote. Never regenerated.

    Regenerating would silently produce a different split whenever a seed, a
    ratio or the sampling count differs from the run being compared against, and
    a baseline fitted on someone else's train set is not a baseline.
    """
    if not splits_path.is_file():
        raise SystemExit(
            f"[trivial] no splits file at {splits_path}.\n"
            "Run training (or the cache warmup) first so the split is fixed on disk. "
            "This script will not create one: a freshly generated split would not be "
            "the split the model was trained on."
        )
    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    splits = {name: set(payload.get(name, [])) for name in ("train", "val", "test")}
    for name, keys in splits.items():
        if not keys:
            raise SystemExit(f"[trivial] split {name!r} is empty in {splits_path}")
    # Checked here, on the sets, and not after assign_splits: that function
    # flattens the splits into a path -> name dict, where a duplicated path is
    # silently resolved last-wins rather than kept in both. A check downstream of
    # the flattening can never fire.
    overlap = splits["train"] & splits["test"]
    if overlap:
        raise SystemExit(
            f"[trivial] {len(overlap)} graph(s) are in both train and test in "
            f"{splits_path}; the split is broken and no baseline from it is meaningful"
        )
    return splits


def assign_splits(rows: pd.DataFrame, splits: dict[str, set[str]]) -> pd.DataFrame:
    lookup = {path: name for name, keys in splits.items() for path in keys}
    rows = rows.assign(split=rows["graph_path"].map(lookup))
    unassigned = int(rows["split"].isna().sum())
    if unassigned:
        # Expected: the held-out hyperparameter-tuning subset is removed before
        # splitting, so its rows are in the CSV but in no split.
        print(f"[trivial] {unassigned} CSV row(s) in no split (tuning holdout); ignored")
    return rows.dropna(subset=["split"]).reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class Predictor:
    """A fitted baseline, and whether it may be ranked against the model.

    ``role`` is load bearing, not a label: fair predictors and oracles are written
    to different files, so a table built from one cannot contain the other.
    """

    role: str  # "fair" or "oracle"
    fitted_on: str  # "train", or "none" for an a-priori constant
    predict: Callable[[pd.DataFrame], np.ndarray]


def _size_design(rows: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(rows)),
            np.log10(rows["num_nodes"].to_numpy(dtype=float)),
            np.log10(rows["num_edges"].to_numpy(dtype=float)),
        ]
    )


def _counts_design(rows: pd.DataFrame) -> np.ndarray:
    """The sum-pooled type counts, on a log scale.

    Counts span several orders of magnitude across this corpus, so a raw-count
    fit is decided by the largest graphs alone. The offset absorbs a count of
    zero, which a primary-input-free constant circuit can have.
    """
    return np.column_stack(
        [
            np.ones(len(rows)),
            *(
                np.log10(1.0 + rows[column].to_numpy(dtype=float))
                for column in ("num_pi", "num_and", "num_po")
            ),
        ]
    )


def _fit_ols(design: np.ndarray, y: np.ndarray, label: str) -> np.ndarray:
    """Least squares, reporting how far the columns are from independent.

    np.linalg.lstsq returns the minimum-norm solution when the design is rank
    deficient, so a collinear pair fits without raising. That is exactly the case
    for the size design and it must not pass silently: the edge count is the
    exact identity 2*AND + PO, so log10(edges) carries almost nothing
    log10(nodes) does not.
    """
    rank = int(np.linalg.matrix_rank(design))
    if rank < design.shape[1]:
        print(f"[trivial] {label} design matrix is rank {rank} of {design.shape[1]}: columns are collinear")
    print(f"[trivial] {label} design matrix condition number {np.linalg.cond(design):,.1f}")
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coefficients


def _linear_predictor(
    design_of: Callable[[pd.DataFrame], np.ndarray], coefficients: np.ndarray
) -> Callable[[pd.DataFrame], np.ndarray]:
    # The target is bounded by its definition, so a linear fit is clipped back
    # into range rather than allowed to predict an impossible optimizability.
    return lambda rows: np.clip(design_of(rows) @ coefficients, 0.0, 1.0)


def _bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior quantile edges, deduplicated and NaN free.

    Levels are small integers with heavy ties, so several quantiles can coincide.
    Deduplicating yields fewer bins rather than empty ones. A single NaN would
    otherwise poison every edge and put every graph in one cell.
    """
    interior = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.nanquantile(values, interior))
    return edges[~np.isnan(edges)]


def _cell_ids(rows: pd.DataFrame, size_edges: np.ndarray, level_edges: np.ndarray) -> np.ndarray:
    """Flatten the two bin indices into one cell identifier.

    Edges are fitted on train and applied unchanged to test, so a test graph
    larger than anything in train lands in the top cell rather than its own. A
    graph with no recorded level gets cell -1, which is its own group rather
    than a silent placement in the shallowest cell.
    """
    level = rows["level"].to_numpy(dtype=float)
    size_bin = np.searchsorted(size_edges, np.log10(rows["num_nodes"].to_numpy(dtype=float)), side="right")
    level_bin = np.searchsorted(level_edges, level, side="right")
    return np.where(np.isnan(level), -1, size_bin * (len(level_edges) + 1) + level_bin)


def _group_mean_predictor(
    train: pd.DataFrame, key_of: Callable[[pd.DataFrame], np.ndarray]
) -> Callable[[pd.DataFrame], np.ndarray]:
    """Predict the train mean of a graph's group, falling back to the global mean.

    Serves both the size-by-level lookup and the source-script oracle, and is the
    group-mean predictor R^2_grp scores in the metrics section. A group present in
    test but not in train has no train mean, so it falls back rather than failing.
    """
    keys = key_of(train)
    means = pd.Series(train["target"].to_numpy(dtype=float)).groupby(keys).mean()
    fallback = float(train["target"].mean())
    return lambda rows: (
        pd.Series(key_of(rows)).map(means).fillna(fallback).to_numpy(dtype=float)
    )


def fit_predictors(train: pd.DataFrame) -> dict[str, Predictor]:
    """Fit every predictor on train. Fair competitors first, oracles last."""
    y = train["target"].to_numpy(dtype=float)
    train_mean = float(np.mean(y))
    train_median = float(np.median(y))

    size_edges = _bin_edges(np.log10(train["num_nodes"].to_numpy(dtype=float)), _N_BINS)
    level_edges = _bin_edges(train["level"].to_numpy(dtype=float), _N_BINS)

    def constant(value: float, fitted_on: str = "train") -> Predictor:
        return Predictor("fair", fitted_on, lambda rows: np.full(len(rows), value, dtype=float))

    return {
        "mean": constant(train_mean),
        "median": constant(train_median),
        # Not fitted at all: the hypothesis that the script removes nothing is
        # stated a priori, so it stays the zero-inflation reference wherever the
        # train median happens to land.
        "zero": constant(0.0, fitted_on="none"),
        "size": Predictor(
            "fair", "train", _linear_predictor(_size_design, _fit_ols(_size_design(train), y, "size"))
        ),
        "counts": Predictor(
            "fair",
            "train",
            _linear_predictor(_counts_design, _fit_ols(_counts_design(train), y, "counts")),
        ),
        "size_level": Predictor(
            "fair",
            "train",
            _group_mean_predictor(train, lambda rows: _cell_ids(rows, size_edges, level_edges)),
        ),
        "source_script_mean": Predictor(
            "oracle",
            "train",
            _group_mean_predictor(train, lambda rows: rows["source_script"].to_numpy()),
        ),
    }


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, R^2 and Spearman, as defined in the metrics section."""
    residual = y_pred - y_true
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        # A constant predictor has zero variance, so Spearman is undefined for
        # the mean, median and zero arms. NaN says that; 0.0 would claim a
        # measurement.
        "spearman": float(spearmanr(y_pred, y_true).statistic)
        if np.ptp(y_pred) > 0
        else float("nan"),
    }


def _tie_groups(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Ascending tie-group index per element, and the group count."""
    groups = np.unique(values, return_inverse=True)[1]
    return groups, int(groups.max()) + 1


def _midranks(groups: np.ndarray, weights: np.ndarray, n_groups: int) -> np.ndarray:
    """Midrank of each element within the resample its weights describe.

    An element sits above every resampled element in a lower tie group and
    shares one rank block with its own group, so its midrank is the count below
    it plus half the block. This is what makes ties correct, and the target has a
    point mass at zero covering roughly half the corpus, so ties are the common
    case rather than an edge case.
    """
    size = np.bincount(groups, weights=weights, minlength=n_groups)
    below = np.concatenate(([0.0], np.cumsum(size)[:-1]))
    return below[groups] + (size[groups] + 1.0) / 2.0


def _weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray, n: int) -> float:
    mx, my = (w @ x) / n, (w @ y) / n
    cov = (w @ (x * y)) / n - mx * my
    var_x, var_y = (w @ (x * x)) / n - mx * mx, (w @ (y * y)) / n - my * my
    # A resample can draw a single distinct value on either side, leaving no
    # spread to correlate. scipy answers NaN there and so does this, quietly.
    if var_x <= 0.0 or var_y <= 0.0:
        return float("nan")
    return float(cov / np.sqrt(var_x * var_y))


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap over graphs, resampling the pair so it stays paired.

    A resample is carried as its multiplicity vector rather than as an index
    list, which turns every statistic into a weighted sum over the original
    arrays and never materializes a resampled copy. Spearman needs the ranks
    *within* each resample; those follow from the global tie groups by counting,
    so no resample is ever sorted. Sorting otherwise dominates this function, and
    this function otherwise dominates the whole script: at 96k test graphs the
    naive form spends about 40 seconds per non-constant predictor against about
    one second here. The result matches ranking each resample from scratch to
    floating-point noise, which test_bootstrap_matches_the_naive_resampling
    pins.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    residual = y_pred - y_true
    abs_residual, sq_residual, y_squared = np.abs(residual), residual**2, y_true**2

    # A constant predictor has no ranking, matching score()'s guard.
    ranked = bool(np.ptp(y_pred) > 0)
    if ranked:
        groups_pred, n_pred = _tie_groups(y_pred)
        groups_true, n_true = _tie_groups(y_true)

    draws: dict[str, list[float]] = {"mae": [], "rmse": [], "r2": [], "spearman": []}
    for _ in range(n_resamples):
        weights = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        ss_res = float(weights @ sq_residual)
        y_mean = float(weights @ y_true) / n
        ss_tot = float(weights @ y_squared) - n * y_mean**2
        draws["mae"].append(float(weights @ abs_residual) / n)
        draws["rmse"].append(np.sqrt(ss_res / n))
        draws["r2"].append(1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
        draws["spearman"].append(
            _weighted_pearson(
                _midranks(groups_pred, weights, n_pred),
                _midranks(groups_true, weights, n_true),
                weights,
                n,
            )
            if ranked
            else float("nan")
        )
    intervals = {}
    for metric, values in draws.items():
        # Spearman is NaN throughout for a constant predictor, and asking for a
        # percentile of an all-NaN sample warns rather than answering.
        if np.all(np.isnan(values)):
            intervals[metric] = (float("nan"), float("nan"))
            continue
        intervals[metric] = (
            float(np.nanpercentile(values, 100 * alpha / 2)),
            float(np.nanpercentile(values, 100 * (1 - alpha / 2))),
        )
    return intervals


def evaluate(
    predictors: dict[str, Predictor], test: pd.DataFrame, n_resamples: int, seed: int
) -> pd.DataFrame:
    """Score every predictor on test. Test targets are read here and nowhere else."""
    y_test = test["target"].to_numpy(dtype=float)
    records = []
    for name, predictor in predictors.items():
        y_pred = predictor.predict(test)
        metrics = score(y_test, y_pred)
        intervals = bootstrap_ci(y_test, y_pred, n_resamples, seed)
        record = {
            "predictor": name,
            "role": predictor.role,
            "n_test": len(test),
            "fitted_on": predictor.fitted_on,
        }
        for metric, value in metrics.items():
            record[metric] = value
            record[f"{metric}_lo"], record[f"{metric}_hi"] = intervals[metric]
        records.append(record)
        print(
            f"[trivial] {predictor.role:6} {name:18} rmse {metrics['rmse']:.4f}  "
            f"mae {metrics['mae']:.4f}  r2 {metrics['r2']:+.4f} "
            f"[{intervals['r2'][0]:+.4f}, {intervals['r2'][1]:+.4f}]"
        )
    return pd.DataFrame(records)


def wandb_run_name_for(algorithm: str, split_by: str) -> str:
    """Mirrors test.py's naming rule so this run sorts beside the model's runs.

    The default split strategy stays untagged and every other one gets a suffix,
    the same convention test.py and train.py use for their run labels.
    """
    name = f"baseline_trivial_{algorithm}"
    return name if split_by == config.SPLIT_BY else f"{name}_{split_by}"


def log_to_wandb(results: pd.DataFrame, args: argparse.Namespace, n_train: int) -> None:
    """One run holding every predictor, as a table plus per-predictor summaries.

    Fair predictors and oracles go to differently named summary keys, so a panel
    built by globbing the fair prefix cannot pick up an oracle.
    """
    import wandb

    run = wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        name=wandb_run_name_for(args.algorithm, args.split_by),
        dir=str(Path(args.out).parent),
        job_type="baseline",
        config={
            "algorithm": args.algorithm,
            "split_by": args.split_by,
            "fitted_on": "train",
            "n_train": n_train,
            "n_test": int(results["n_test"].iloc[0]),
            "n_resamples": args.n_resamples,
            "seed": args.seed,
            "splits_path": args.splits_path,
            "csv_paths": args.csv_paths,
        },
    )
    run.log({"baselines": wandb.Table(dataframe=results)})
    for record in results.to_dict(orient="records"):
        prefix = f"{record['role']}/{record['predictor']}"
        run.summary.update(
            {f"{prefix}/{m}": record[m] for m in ("mae", "rmse", "r2", "spearman")}
        )
    run.finish()
    print(f"[trivial] logged to W&B as {wandb_run_name_for(args.algorithm, args.split_by)}")


def main(args: argparse.Namespace) -> None:
    rows = load_rows(args.csv_paths)
    splits_path = Path(args.splits_path)
    print(f"[trivial] splits: {splits_path}")
    rows = assign_splits(rows, load_splits(splits_path))

    train = rows[rows["split"] == "train"]
    test = rows[rows["split"] == "test"]
    # The join, not the splits file, is what usually fails: the CSV's paths and
    # the splits file's paths are matched as strings, so a splits file written
    # under a different graph root matches nothing and every row lands in the
    # tuning-holdout branch above. Without this the run dies inside numpy on an
    # empty array, several screens later.
    for name, part in (("train", train), ("test", test)):
        if part.empty:
            raise SystemExit(
                f"[trivial] no CSV row joined to the {name} split. The splits file "
                "lists paths that none of the CSVs contain; check that both were "
                "produced by the same run."
            )

    for name, part in (("train", train), ("val", rows[rows["split"] == "val"]), ("test", test)):
        print(f"[trivial] {name:5} {len(part):>8,} graphs, mean y {part['target'].mean():.6f}")

    if "step_id" in rows.columns:
        affected = int((test["step_id"] == 21).sum())
        if affected:
            print(
                f"[trivial] {affected} test graph(s) at step 21, whose recorded node count "
                "understates the stored graph (choice networks). Both the target and the "
                "size features are wrong for those rows."
            )

    results = evaluate(fit_predictors(train), test, args.n_resamples, args.seed)

    # Separate files, not one file with a role column: an oracle that reaches a
    # results table as though it were a competitor is the failure this prevents,
    # and a column does not prevent it.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results[results["role"] == "fair"].to_csv(out, index=False)
    print(f"[trivial] wrote {out}")

    oracles = results[results["role"] == "oracle"]
    oracle_out = out.with_name(f"{out.stem}_oracles{out.suffix}")
    oracles.to_csv(oracle_out, index=False)
    print(
        f"[trivial] wrote {oracle_out}: {len(oracles)} predictor(s) using information "
        "unavailable at inference. Never rank these against the model."
    )

    # Logged after the CSVs are on disk, so an unreachable W&B backend cannot
    # cost the run its results.
    if args.wandb:
        log_to_wandb(results, args, n_train=len(train))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Non-learned RQ1 baselines: fitted on train, scored on test, from the CSVs alone."
    )
    parser.add_argument("--csv_paths", nargs="+", required=True)
    parser.add_argument(
        "--splits_path",
        type=str,
        required=True,
        help="The splits JSON the training run wrote (named by data.dataset.splits_cache_filename).",
    )
    parser.add_argument("--out", type=str, default="results/rq1_trivial_baselines.csv")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="Orchestrate",
        help="Names the W&B run; does not filter the CSV, which is already per-algorithm.",
    )
    parser.add_argument(
        "--split_by",
        type=str,
        default=config.SPLIT_BY,
        help="Names the W&B run. Must match the splits file given, which is not checked.",
    )
    parser.add_argument(
        "--wandb",
        type=lambda v: str(v).lower() not in ("false", "0", "no"),
        default=True,
        help="Log the results table and per-predictor summaries to W&B.",
    )
    parser.add_argument(
        "--n_resamples",
        type=int,
        default=2000,
        help="Bootstrap resamples behind each 95% interval.",
    )
    parser.add_argument("--seed", type=int, default=42)
    sys.exit(main(parser.parse_args()))
