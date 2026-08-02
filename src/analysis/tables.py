"""Generated booktabs tables for the Results chapter.

Every table is written to ``media/results/tables`` and pulled in with
``\\input``. Nothing here should ever be pasted into the chapter by hand: a
number typed into the .tex drifts from the run that produced it the first time
anything is re-run.

Tables carrying fabricated rows get a ``note`` saying so, the fabricated rows
are prefixed ``[TODO/FAKE]`` in their first column, and every number in them is
replaced by an absurd sentinel and typeset in red
(:func:`analysis.results_to_latex.write_booktabs_table`), so no single loss of
context can turn a placeholder into a result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.fake_data import (
    BASELINE_MODELS,
    CORPUS_STATS,
    SPLIT_PROTOCOL,
    SUMMARIZATION,
    TODO_BASELINE_MODELS,
    TODO_CORPUS_STATS,
    TODO_SPLIT_PROTOCOL,
    TODO_SUMMARIZATION,
)
from analysis.results_to_latex import BETTER, write_booktabs_table
from analysis.style import label_for, meta, sort_key

FAKE_TAG = "[TODO/FAKE] "


def _fake(methods) -> list[bool]:
    """Which rows are fabricated, read off the method registry rather than off a
    second list kept in step by hand."""
    return [not meta(m)["measured"] for m in methods]


def _fake_flags(measured: pd.Series) -> list[bool]:
    """Fabricated-row mask for the tables whose rows are models or protocols
    rather than reduction methods, so the registry does not know them. An
    unrecorded flag counts as fabricated."""
    return (~measured.fillna(False).astype(bool)).tolist()


def _best(df: pd.DataFrame, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Bold/underline spec: every column of ``df`` with a known direction, plus
    any generated column header named explicitly."""
    spec = {c: BETTER[c] for c in df.columns if c in BETTER}
    if extra:
        spec.update(extra)
    return spec


def _method_column(methods, *, tag_fake: bool = True) -> list[str]:
    return [
        (FAKE_TAG if tag_fake and fake else "") + label_for(m)
        for m, fake in zip(methods, _fake(methods))
    ]


def _family(methods) -> list[str]:
    return [meta(m)["family"].capitalize() for m in methods]


def _kind(methods) -> list[str]:
    return ["Domain" if meta(m)["domain"] else "Generic" for m in methods]


# --- Dataset -----------------------------------------------------------------
def dataset_stats(preds: pd.DataFrame, path: Path) -> None:
    rows = []
    for tier, sub in preds.groupby("tier"):
        rows.append(
            {
                "Tier": tier,
                "Graphs": len(sub),
                "Designs": sub["design"].nunique(),
                "Median nodes": int(sub["num_nodes"].median()),
                "Max nodes": int(sub["num_nodes"].max()),
                "Mean $y$": sub["target"].mean(),
                "SD $y$": sub["target"].std(),
                "$y < 0.005$": float((sub["target"] < 0.005).mean()),
            }
        )
    rows.append(
        {
            "Tier": "pooled",
            "Graphs": len(preds),
            "Designs": preds["design"].nunique(),
            "Median nodes": int(preds["num_nodes"].median()),
            "Max nodes": int(preds["num_nodes"].max()),
            "Mean $y$": preds["target"].mean(),
            "SD $y$": preds["target"].std(),
            "$y < 0.005$": float((preds["target"] < 0.005).mean()),
        }
    )
    write_booktabs_table(
        pd.DataFrame(rows),
        path / "dataset_stats.tex",
        caption=(
            "Held-out test corpus. The label is concentrated near zero, which is the "
            "denominator every $R^2$ in this chapter is measured against."
        ),
        label="tab:dataset_stats",
        wide=True,
    )


def corpus_stats(path: Path) -> None:  # noqa: ERA001  (kept for reference; not built)
    df = CORPUS_STATS.copy()
    df["tier"] = FAKE_TAG + df["tier"]
    df = df.drop(columns="measured").rename(
        columns={
            "tier": "Tier",
            "graphs": "Graphs",
            "median_nodes": "Median nodes",
            "mean_optimizability": "Mean $y$",
        }
    )
    write_booktabs_table(
        df,
        path / "corpus_stats.tex",
        caption=(
            "Whole-corpus statistics across the two stored tiers "
            "(\\ref{sec:method:data:tiers})."
        ),
        label="tab:corpus_stats",
        note="EVERY NUMBER IN THIS TABLE IS FABRICATED. " + TODO_CORPUS_STATS,
        fake=_fake_flags(CORPUS_STATS["measured"]),
    )


def dataset_composition(test: pd.DataFrame, val: pd.DataFrame, path: Path) -> None:
    """Split x tier x source-script composition of the evaluation corpus.

    The fact worth carrying into the methodology is that the two splits are
    materially different distributions rather than two samples of one pool,
    which is expected under a design-level split and is what produces the
    validation-to-test gap.
    """
    rows = []
    for name, d in [("val", val), ("test", test)]:
        if d.empty:
            continue
        rows.append({
            "Split": name,
            "Designs": d["design"].nunique(),
            "Graphs": len(d),
            "tier0": int((d["tier"] == "tier0").sum()),
            "tier1": int((d["tier"] == "tier1").sum()),
            "Median nodes": int(d["num_nodes"].median()),
            "Mean $y$": d["target"].mean(),
            "$y = 0$": float((d["target"] == 0).mean()),
            "Max $y$": d["target"].max(),
        })
    write_booktabs_table(
        pd.DataFrame(rows),
        path / "dataset_composition.tex",
        caption=(
            "Composition of the evaluation corpus over the two stored tiers. "
            "Validation and test differ substantially in label and size distribution, "
            "which is expected under a design-level split and is the direct explanation "
            "for the validation-to-test gap."
        ),
        label="tab:dataset_composition",
        wide=True,
    )


def dataset_structure(preds: pd.DataFrame, measurements_dir: Path, path: Path) -> None:
    """Structural statistics recoverable without a pass over the graph cache."""
    from analysis.results_to_latex import load_offline_stats

    sparse = load_offline_stats(measurements_dir, "sparsification_stats")
    ago = sparse[sparse["reduction_method"] == "and_gate_only"]
    if ago.empty:
        return
    keep = ago["node_retention"]
    density = preds["num_edges"] / preds["num_nodes"]
    rows = [
        ("AND gates and constants (share of nodes)", keep.mean(), keep.std(),
         keep.min(), keep.max()),
        ("Primary inputs and outputs (share of nodes)", 1 - keep.mean(), keep.std(),
         1 - keep.max(), 1 - keep.min()),
        ("Edges per node", density.mean(), density.std(), density.min(), density.max()),
    ]
    write_booktabs_table(
        pd.DataFrame(rows, columns=["Statistic", "Mean", "SD", "Min", "Max"]),
        path / "dataset_structure.tex",
        caption=(
            "Structural corpus statistics. The AND-gate share is recovered from the "
            "AND-gate-only node masks over 10,000 graphs: that method drops exactly the "
            "primary inputs and outputs, so its node retention is the AND-and-constant "
            "share. This is what fixes that method's compression, rather than a "
            "parameter that could be dialled."
        ),
        label="tab:dataset_structure",
    )


# --- RQ1 ---------------------------------------------------------------------
def rq1_baselines(tier1: pd.DataFrame, path: Path) -> None:
    # trivial_baselines already sets ``tier`` per row: the encoder is filed
    # under "This work", not as a trivial predictor. Overwriting the column
    # here would put the thesis's own model in tier 1.
    combined = pd.concat([tier1.assign(measured=True), BASELINE_MODELS], ignore_index=True)
    combined["model"] = [
        (FAKE_TAG if not m else "") + name
        for name, m in zip(combined["model"], combined["measured"])
    ]
    out = combined[["model", "tier", "rmse", "mae", "r2", "spearman"]].rename(
        columns={
            "model": "Model",
            "tier": "Tier",
            "rmse": "RMSE",
            "mae": "MAE",
            "r2": "$R^2$",
            "spearman": "Spearman",
        }
    )
    write_booktabs_table(
        out,
        path / "rq1_baselines.tex",
        caption=(
            "RQ1 baseline comparison on the design-disjoint test split. "
            "Trivial predictors are fitted on validation and scored on test. "
            "Upstream $R^2$ figures are NOT comparable: OpenABC-D z-scores its "
            "targets per design, which removes between-design variance from the "
            "denominator. Only the sign transfers."
        ),
        label="tab:rq1_baselines",
        wide=True,
        note="Rows marked [TODO/FAKE] are invented. " + TODO_BASELINE_MODELS,
        fake=_fake_flags(combined["measured"]),
        best=_best(out),
    )


def rq1_per_design(per_design: pd.DataFrame, path: Path) -> None:
    out = per_design[["design", "n", "mean_y", "rmse", "mae", "r2", "spearman"]].rename(
        columns={
            "design": "Design",
            "n": "Graphs",
            "mean_y": "Mean $y$",
            "rmse": "RMSE",
            "mae": "MAE",
            "r2": "$R^2$",
            "spearman": "Spearman",
        }
    )
    write_booktabs_table(
        out,
        path / "rq1_per_design.tex",
        caption=(
            "Per-design metrics on the design-disjoint test set. A pooled score over a "
            "design-level split is an average over this handful of whole circuits, not "
            "over a random sample."
        ),
        label="tab:rq1_per_design",
        wide=True,
    )


def rq1a_protocol(path: Path) -> None:
    df = SPLIT_PROTOCOL.copy()
    df["protocol"] = [
        (FAKE_TAG if not m else "") + p for p, m in zip(df["protocol"], df["measured"])
    ]
    base = df.loc[df["measured"], "r2"].iloc[0]
    df["Inflation vs. design-disjoint"] = df["r2"] - base
    out = df.drop(columns="measured").rename(
        columns={"protocol": "Protocol", "rmse": "RMSE", "r2": "$R^2$", "spearman": "Spearman"}
    )
    write_booktabs_table(
        out,
        path / "rq1a_protocol.tex",
        caption=(
            "RQ1a protocol sensitivity. Identical encoder, budget and seed; only the "
            "split changes. Random is the common default in circuit "
            "representation-learning evaluations, recipe-disjoint is OpenABC-D "
            "Variant~1, design-disjoint is Variant~2 and is what this thesis reports."
        ),
        label="tab:rq1a_protocol",
        note="Rows marked [TODO/FAKE] are invented. " + TODO_SPLIT_PROTOCOL,
        fake=_fake_flags(df["measured"]),
    )


def rq1_hyperparameters(runs: pd.DataFrame, path: Path) -> None:
    """The configuration the sweep selected, read off the W&B run config."""
    cfg = runs[runs["name"] == "train_Orchestrate"]
    if cfg.empty:
        return
    row = cfg.iloc[0]
    fields = [
        ("Encoder", "cfg/encoder_name"),
        ("Hidden dim", "cfg/hidden_dim"),
        ("Layers (message passing)", None),
        ("Positional encoding", "cfg/pe_type"),
        ("PE dim", "cfg/pos_enc_dim"),
        ("Pooling", "cfg/pooling_type"),
        ("Head dropout", "cfg/head_dropout"),
        ("Learning rate", "cfg/lr"),
        ("Min learning rate", "cfg/min_lr"),
        ("Weight decay", "cfg/weight_decay"),
        ("Warmup steps", "cfg/warmup_steps"),
        ("Scheduler patience", "cfg/scheduler_patience"),
        ("Node input dim", "cfg/node_input_dim"),
        ("Edge attr dim", "cfg/edge_attr_dim"),
        ("Seed", "cfg/seed"),
    ]
    rows = []
    for name, col in fields:
        if col is None:
            # Not in the W&B config: NUM_LAYERS lives in config.py and the run
            # never logged it. Marked so the caption's claim stays true.
            rows.append({"Hyperparameter": name, "Value": "4 (not logged)"})
            continue
        if col in cfg.columns and pd.notna(row[col]):
            rows.append({"Hyperparameter": name, "Value": f"{row[col]}"})
        else:
            # Dropping the row silently would leave no trace that the run did
            # not record it, which reads as though the field does not exist.
            rows.append({"Hyperparameter": name, "Value": "not recorded"})
    write_booktabs_table(
        pd.DataFrame(rows),
        path / "rq1_hyperparameters.tex",
        caption=(
            "Final configuration, as recorded in the run log of the reported run. "
            "Fields the run did not log are marked rather than omitted."
        ),
        label="tab:rq1_hyperparameters",
    )


# --- RQ2 ---------------------------------------------------------------------
def rq2_efficiency(offline: pd.DataFrame, bench: pd.DataFrame, savings: pd.DataFrame, path: Path) -> None:
    df = offline.merge(savings, on="reduction_method", how="left")
    bench_slim = bench.assign(
        reduction_method=bench["reduction_method"].fillna("none")
    )[["reduction_method", "avg_step_time_s", "peak_vram_allocated_mean_mb",
       "peak_vram_allocated_max_mb", "throughput_graphs_per_s"]]
    df = df.merge(bench_slim, on="reduction_method", how="left")

    fake = SUMMARIZATION.rename(
        columns={"peak_vram_mb": "peak_vram_allocated_mean_mb", "step_time_s": "avg_step_time_s"}
    )
    df = pd.concat([df.assign(measured=True), fake], ignore_index=True)
    df = df.sort_values("reduction_method", key=lambda s: s.map(sort_key))

    out = pd.DataFrame(
        {
            "Method": _method_column(df["reduction_method"]),
            "Family": _family(df["reduction_method"]),
            "Kind": _kind(df["reduction_method"]),
            "Node ret.": df["node_retention"],
            "Edge ret.": df["edge_retention"],
            "Offline (s/graph)": df["offline_s"],
            "Step time (s)": df["avg_step_time_s"],
            "VRAM mean (MB)": df["peak_vram_allocated_mean_mb"],
            "VRAM saving (%)": df["mean_vram_saving_pct"],
            "Time saving (%)": df["mean_time_saving_pct"],
        }
    )
    write_booktabs_table(
        out,
        path / "rq2_efficiency.tex",
        caption=(
            "RQ2 efficiency profile. Node and edge retention are reported separately "
            "because several methods preserve node counts exactly. Savings are per-graph "
            "means paired against the full-graph baseline."
        ),
        label="tab:rq2_efficiency",
        wide=True,
        note="Rows marked [TODO/FAKE] are invented. " + TODO_SUMMARIZATION,
        fake=_fake(df["reduction_method"]),
        best=_best(out),
    )


def rq2_paired_savings(savings: pd.DataFrame, path: Path) -> None:
    df = savings.sort_values("mean_vram_saving_pct", ascending=False)
    out = pd.DataFrame(
        {
            "Method": [label_for(m) for m in df["reduction_method"]],
            "Paired graphs": df["n_paired"],
            "VRAM saving (%)": df["mean_vram_saving_pct"],
            "95% CI low": df["vram_saving_ci_low_pct"],
            "95% CI high": df["vram_saving_ci_high_pct"],
            "Time saving (%)": df["mean_time_saving_pct"],
            "95% CI low ": df["time_saving_ci_low_pct"],
            "95% CI high ": df["time_saving_ci_high_pct"],
            "Wilcoxon $p$ (time)": df["time_saving_wilcoxon_p"],
        }
    )
    write_booktabs_table(
        out,
        path / "rq2_paired_savings.tex",
        caption=(
            "Per-graph savings against the full-graph baseline, paired by graph id, with "
            "percentile bootstrap intervals and a Wilcoxon signed-rank test on the paired "
            "deltas. Savings are not normally distributed, so bare means would be "
            "misleading."
        ),
        label="tab:rq2_paired_savings",
        wide=True,
        best=_best(out),
    )


def rq2_partition_balance(measurements_dir: Path, path: Path) -> None:
    from analysis.results_to_latex import load_offline_stats

    part = load_offline_stats(measurements_dir, "partition_stats")
    if part.empty:
        return
    agg = part.groupby("reduction_method").agg(
        k=("num_partitions", "mean"),
        cut=("edge_cut_ratio", "mean"),
        cut_sd=("edge_cut_ratio", "std"),
        balance=("std_nodes_per_partition", "mean"),
        max_nodes=("max_nodes_per_partition", "mean"),
        seconds=("time_s", "mean"),
    ).reset_index()
    agg = agg.sort_values("reduction_method", key=lambda s: s.map(sort_key))
    out = pd.DataFrame(
        {
            "Partitioner": _method_column(agg["reduction_method"]),
            "Kind": _kind(agg["reduction_method"]),
            "Mean $k$": agg["k"],
            "Edge cut": agg["cut"],
            "Edge cut sd": agg["cut_sd"],
            "Imbalance (sd nodes)": agg["balance"],
            "Max part. nodes": agg["max_nodes"].round().astype(int),
            "Offline (s)": agg["seconds"],
        }
    )
    write_booktabs_table(
        out,
        path / "rq2_partition_balance.tex",
        caption=(
            "The four partitioners keep every node and differ only in which edges they "
            "cut, at the same $k$. That makes them the cleanest domain-informed / generic "
            "pairs in the thesis."
        ),
        label="tab:rq2_partition_balance",
        wide=True,
        fake=_fake(agg["reduction_method"]),
        best=_best(out),
    )


# --- RQ3 ---------------------------------------------------------------------
def rq3_retention(matched: pd.DataFrame, path: Path) -> None:
    base = matched[matched["reduction_method"] == "none"].iloc[0]
    df = pd.concat(
        [matched.assign(measured=True), SUMMARIZATION], ignore_index=True
    ).sort_values("reduction_method", key=lambda s: s.map(sort_key))

    out = pd.DataFrame(
        {
            "Method": _method_column(df["reduction_method"]),
            "Family": _family(df["reduction_method"]),
            "Kind": _kind(df["reduction_method"]),
            "Smooth L1": df.get("smooth_l1", pd.Series(np.nan, index=df.index)),
            "RMSE": df["rmse"],
            "$R^2$": df["r2"],
            "Spearman": df["spearman"],
            "$R^2$ retained": df["r2"] / base["r2"],
        }
    )
    write_booktabs_table(
        out,
        path / "rq3_retention.tex",
        caption=(
            "RQ3 matched-state accuracy: models trained and tested under the same "
            "reduction. RMSE, $R^2$ and Spearman are reported together throughout, "
            "because a reduction can hold mean error steady while destroying "
            "explained variance."
        ),
        label="tab:rq3_retention",
        wide=True,
        note="Rows marked [TODO/FAKE] are invented. " + TODO_SUMMARIZATION,
        fake=_fake(df["reduction_method"]),
        best=_best(out),
    )


def rq3_pareto(matched: pd.DataFrame, savings: pd.DataFrame, path: Path) -> None:
    df = matched.merge(savings, on="reduction_method", how="left")
    df.loc[df["reduction_method"] == "none", ["mean_vram_saving_pct", "mean_time_saving_pct"]] = 0.0
    df = df.sort_values("r2", ascending=False)
    out = pd.DataFrame(
        {
            "Method": [label_for(m) for m in df["reduction_method"]],
            "$R^2$": df["r2"],
            "Spearman": df["spearman"],
            "VRAM saving (%)": df["mean_vram_saving_pct"],
            "Time saving (%)": df["mean_time_saving_pct"],
        }
    )
    write_booktabs_table(
        out,
        path / "rq3_pareto.tex",
        caption="Accuracy against efficiency, ranked by matched-state $R^2$.",
        label="tab:rq3_pareto",
        best=_best(out),
    )


def rq3_stratified(strata: pd.DataFrame, path: Path) -> None:
    """The matched-state ranking re-scored on each label stratum."""
    from analysis.loaders import STRATUM_LABELS

    df = strata[strata["eval_mode"] == "matched_reduction"].sort_values(
        "all_r2", ascending=False
    )
    out = pd.DataFrame({"Method": [label_for(m) for m in df["reduction_method"]]})
    for stratum, name in STRATUM_LABELS.items():
        out[f"$R^2$ {name}"] = df[f"{stratum}_r2"].to_numpy()
    for stratum, name in STRATUM_LABELS.items():
        out[f"$\\rho$ {name}"] = df[f"{stratum}_spearman"].to_numpy()
    write_booktabs_table(
        out,
        path / "rq3_stratified.tex",
        caption=(
            "Matched-state accuracy re-scored on four subsets of the same test split, "
            "from the persisted per-graph predictions. The pooled label is 49\\% exactly "
            "zero, so a pooled $R^2$ is carried by a minority of graphs; a conclusion "
            "that survives all four columns is a property of the reduction rather than "
            "of the label distribution."
        ),
        label="tab:rq3_stratified",
        wide=True,
        best=_best(out, {c: "max" for c in out.columns if c != "Method"}),
        note=(
            "Strata: all test graphs; excluding the two designs on which Orchestrate is "
            "at a fixed point (16384, 8192, max y = 0.0003 over ~32,000 graphs); graphs "
            "with a non-zero target; and graphs derived from Syn4, the one source script "
            "that leaves substantial optimizability behind."
        ),
    )


def rq3_by_subgroup(by_tier: pd.DataFrame, by_source: pd.DataFrame, path: Path) -> None:
    """Matched-state accuracy partitioned by tier and by source script."""
    frames = []
    for df, col, order in [
        (by_tier, "tier", ["tier0", "tier1"]),
        (by_source, "source_algorithm", ["tier0 base", "Syn4", "Deepsyn", "C2RS"]),
    ]:
        sub = df[(df["eval_mode"] == "matched_reduction")
                 | (df["reduction_method"] == "none")]
        for level in order:
            rows = sub[sub[col] == level]
            if rows.empty:
                continue
            frames.append(rows.assign(group=level, cut=col))
    if not frames:
        return
    joined = pd.concat(frames)
    pivot = joined.pivot_table(
        index="reduction_method", columns="group", values="r2", observed=True
    )
    spear = joined.pivot_table(
        index="reduction_method", columns="group", values="spearman", observed=True
    )
    pivot = pivot.reindex(sorted(pivot.index, key=sort_key))
    spear = spear.reindex(pivot.index)

    out = pd.DataFrame({"Method": [label_for(m) for m in pivot.index]})
    for c in pivot.columns:
        out[f"$R^2$ {c}"] = pivot[c].to_numpy()
    for c in spear.columns:
        out[f"$\\rho$ {c}"] = spear[c].to_numpy()
    write_booktabs_table(
        out,
        path / "rq3_by_subgroup.tex",
        caption=(
            "Matched-state accuracy partitioned two ways: by dataset tier, and by the "
            "synthesis script that produced the graph. Both partition the same test "
            "split, so the columns within each cut are disjoint. The source script "
            "separates the corpus far more sharply than the tier does."
        ),
        label="tab:rq3_by_subgroup",
        wide=True,
        best=_best(out, {c: "max" for c in out.columns if c != "Method"}),
    )


def rq3_by_target_bin(by_band: pd.DataFrame, path: Path) -> None:
    """Mean prediction and mean absolute error within bands of the true target."""
    df = by_band[
        (by_band["eval_mode"] == "matched_reduction")
        | (by_band["reduction_method"] == "none")
    ]
    if df.empty:
        return
    pred = df.pivot_table(index="reduction_method", columns="bin", values="pred",
                          observed=True)
    mae = df.pivot_table(index="reduction_method", columns="bin", values="mae",
                         observed=True)
    pred = pred.reindex(sorted(pred.index, key=sort_key))
    mae = mae.reindex(pred.index)
    truth = df.groupby("bin", observed=True)["true"].mean()

    out = pd.DataFrame({"Method": [label_for(m) for m in pred.index]})
    for c in pred.columns:
        out[f"pred {c}"] = pred[c].to_numpy()
    for c in mae.columns:
        out[f"MAE {c}"] = mae[c].to_numpy()
    truth_row = {"Method": "(true mean in band)"}
    truth_row.update({f"pred {c}": truth[c] for c in pred.columns})
    truth_row.update({f"MAE {c}": float("nan") for c in mae.columns})
    out = pd.concat([pd.DataFrame([truth_row]), out], ignore_index=True)

    write_booktabs_table(
        out,
        path / "rq3_by_target_bin.tex",
        caption=(
            "Mean prediction and mean absolute error within bands of the true "
            "optimizability. $R^2$ is not reported: inside a narrow band there is "
            "almost no label variance to divide by. Compare each method's predicted "
            "mean against the true mean in the first row. A row that stays flat "
            "across the bands is a model that has collapsed to a constant."
        ),
        label="tab:rq3_by_target_bin",
        wide=True,
    )


# --- RQ4 ---------------------------------------------------------------------
def rq4_pairings(pairings: pd.DataFrame, path: Path) -> None:
    out = pd.DataFrame(
        {
            "Domain-informed": [label_for(m) for m in pairings["domain"]],
            "Generic": [label_for(m) for m in pairings["generic"]],
            "Matched": pairings["kind"],
            "Edge ret. (dom.)": pairings["domain_edge_retention"],
            "Edge ret. (gen.)": pairings["generic_edge_retention"],
            "$\\Delta$ RMSE": pairings["delta_rmse"],
            "$\\Delta R^2$": pairings["delta_r2"],
            "$\\Delta$ Spearman": pairings["delta_spearman"],
        }
    )
    write_booktabs_table(
        out,
        path / "rq4_pairings.tex",
        caption=(
            "RQ4 pairings. $\\Delta$ is domain-informed minus generic, so a positive "
            "$\\Delta R^2$ favours the domain heuristic. The fourth pairing is NOT "
            "matched: random edge dropout at its configured rate keeps 69.7\\% of edges "
            "against spanning forest's 58.1\\%, so its gap conflates the heuristic with "
            "how much each arm removed."
        ),
        label="tab:rq4_pairings",
        wide=True,
        note=(
            "Each configuration is trained ONCE. These are point estimates with no "
            "run-to-run variance behind them, and RQ4's expected effect is small: a gap "
            "of a few percent cannot be distinguished from noise."
        ),
    )


# --- RQ5 ---------------------------------------------------------------------
def rq5_cross_state(cross: pd.DataFrame, path: Path) -> None:
    df = cross.copy()
    out = pd.DataFrame(
        {
            "Method": [label_for(m) for m in df["reduction_method"]],
            "Family": _family(df["reduction_method"]),
            "$R^2$ matched": df["matched_reduction_r2"],
            "$R^2$ full": df["full_graph_r2"],
            "$\\Delta R^2$": df["full_graph_r2"] - df["matched_reduction_r2"],
            "$\\rho$ matched": df["matched_reduction_spearman"],
            "$\\rho$ full": df["full_graph_spearman"],
            "Throughput full (g/s)": df["full_graph_throughput_graphs_per_s"],
        }
    )
    write_booktabs_table(
        out,
        path / "rq5_cross_state.tex",
        caption=(
            "RQ5: the same weights evaluated under the reduction they were trained with "
            "and on unreduced graphs. For every method here a full graph needs no "
            "conversion, it already is a valid input. The exact colour-refinement "
            "track is the exception and has not been run."
        ),
        label="tab:rq5_cross_state",
        wide=True,
        best=_best(out, {
            "$R^2$ matched": "max",
            "$R^2$ full": "max",
            "$\\rho$ matched": "max",
            "$\\rho$ full": "max",
        }),
    )


# --- Summary -----------------------------------------------------------------
def summary(offline, matched, cross, savings, bench, path: Path) -> None:
    """The one consolidated table the Discussion and Conclusion refer back to."""
    df = offline.merge(savings, on="reduction_method", how="left")
    df = df.merge(matched[["reduction_method", "rmse", "r2", "spearman"]], on="reduction_method",
                  how="right")
    df = df.merge(
        cross[["reduction_method", "full_graph_r2", "matched_reduction_r2"]],
        on="reduction_method", how="left",
    )
    fake = SUMMARIZATION.rename(columns={"cross_state_r2": "full_graph_r2"})
    df = pd.concat([df.assign(measured=True), fake], ignore_index=True)
    df = df.sort_values("reduction_method", key=lambda s: s.map(sort_key))

    out = pd.DataFrame(
        {
            "Method": _method_column(df["reduction_method"]),
            "Family": _family(df["reduction_method"]),
            "Kind": _kind(df["reduction_method"]),
            "Node ret.": df["node_retention"],
            "Edge ret.": df["edge_retention"],
            "Offline (s)": df["offline_s"],
            "VRAM sav. (%)": df["mean_vram_saving_pct"],
            "Time sav. (%)": df["mean_time_saving_pct"],
            "RMSE": df["rmse"],
            "$R^2$": df["r2"],
            "$\\rho$": df["spearman"],
            "$R^2$ cross-state": df["full_graph_r2"],
        }
    )
    write_booktabs_table(
        out,
        path / "summary_all.tex",
        caption=(
            "Every reduction method in one row: what it removes, what it costs offline, "
            "what it saves in training, what accuracy survives, and whether the model "
            "transfers to full graphs. Generic vs. domain-informed is a column, not a "
            "split, because it is one attribute of a method rather than the organising "
            "axis."
        ),
        label="tab:summary_all",
        wide=True,
        note="Rows marked [TODO/FAKE] are invented. " + TODO_SUMMARIZATION,
        fake=_fake(df["reduction_method"]),
        best=_best(out),
    )


def build(ctx, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    dataset_stats(ctx["preds_test"], path)
    dataset_composition(ctx["preds_test"], ctx["preds_val"], path)
    dataset_structure(ctx["preds_test"], ctx["measurements_dir"], path)
    # corpus_stats() is deliberately not built: it emitted fabricated rows. The real
    # whole-corpus table comes from data/creation/corpus_tier_stats.py on the cluster.
    rq1_baselines(ctx["tier1"], path)
    rq1_per_design(ctx["per_design"], path)
    rq1a_protocol(path)
    rq1_hyperparameters(ctx["runs"], path)
    rq2_efficiency(ctx["offline"], ctx["bench"], ctx["savings"], path)
    rq2_paired_savings(ctx["savings"], path)
    rq2_partition_balance(ctx["measurements_dir"], path)
    rq3_retention(ctx["matched"], path)
    rq3_pareto(ctx["matched"], ctx["savings"], path)
    rq3_stratified(ctx["strata"], path)
    rq3_by_subgroup(ctx["by_tier"], ctx["by_source"], path)
    rq3_by_target_bin(ctx["by_band"], path)
    rq4_pairings(ctx["pairings"], path)
    rq5_cross_state(ctx["cross"], path)
    summary(ctx["offline"], ctx["matched"], ctx["cross"], ctx["savings"], ctx["bench"], path)
