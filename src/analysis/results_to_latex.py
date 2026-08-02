"""Turns the results/ and logs/ CSVs produced by test.py, benchmark.py, and
measure_partition.py/measure_sparsity.py into booktabs-style .tex tables,
ready to \\input{} into sections/4-results.tex, plus a pareto_front.csv for
plot_results.py.

Grouping is always by (reduction_type, reduction_method) rather than a
hardcoded sparsification/partition split, so a future summarization row
(reduction_type="summarization") joins every table automatically without
code changes here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _escape_latex(value) -> str:
    s = str(value)
    return s.replace("_", r"\_").replace("%", r"\%")


def _format_cell(value) -> str:
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return "--"
        return f"{value:.3f}"
    if isinstance(value, str):
        escaped = _escape_latex(value)
        # A cell opening with "[" directly after a row's "\\" is parsed as the
        # optional vertical-space argument of \\ and kills the build. Bracing it
        # is the standard fix.
        return "{" + escaped + "}" if escaped.startswith("[") else escaped
    return str(value)


def write_booktabs_table(
    df: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    *,
    wide: bool = False,
    small: bool = True,
    note: str | None = None,
) -> None:
    """Write ``df`` as a booktabs table.

    ``wide`` selects the starred ``table*`` float, which is what a table with
    more than about four columns needs in the two-column thesis layout; ``note``
    adds an italic line under the rule, used to carry the TODO on any table
    holding placeholder rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print(f"[results_to_latex] Skipping {path.name} — no rows to write.")
        return

    col_spec = "".join(
        "l" if not pd.api.types.is_numeric_dtype(df[c]) else "r" for c in df.columns
    )
    header = " & ".join(f"\\textbf{{{_escape_latex(c)}}}" for c in df.columns) + " \\\\"
    env = "table*" if wide else "table"
    # A wide table still overflows \textwidth once it carries ten or more
    # columns, so the tabular is scaled to fit rather than left to run into the
    # margin.
    resize = len(df.columns) >= 8

    lines = [
        f"\\begin{{{env}}}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
    ]
    if small:
        lines.append("\\small")
    if resize:
        lines.append("\\resizebox{\\textwidth}{!}{%")
    lines += [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(_format_cell(v) for v in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if resize:
        lines.append("}")
    if note:
        # Notes are plain prose, so they are escaped: an unescaped underscore
        # from a filename or a flag name is a fatal error outside math mode.
        lines.append(
            f"\\\\[4pt]\\begin{{minipage}}{{\\linewidth}}\\footnotesize\\emph{{"
            f"{_escape_latex(note)}}}\\end{{minipage}}"
        )
    lines += [f"\\end{{{env}}}", ""]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[results_to_latex] Wrote {path}")


def _method_matches(series: pd.Series, method) -> pd.Series:
    """NaN-safe equality: pandas reads an empty reduction_method (the "none"
    config's CSV cell) back as NaN, and NaN == NaN is always False — so a
    plain `==` filter silently drops the baseline config from every join."""
    target = method if pd.notna(method) else ""
    return series.fillna("") == target


def _load_csv_dir(directory: Path, kind: str) -> pd.DataFrame:
    """Concatenate every per-config CSV that test.py / benchmark.py wrote into
    ``directory`` (one file per config, to avoid concurrent-append races in the
    SLURM array). Returns empty if the directory has no CSVs."""
    if not directory.is_dir():
        print(f"[results_to_latex] {directory} not found — skipping {kind} tables.")
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in sorted(directory.glob("*.csv"))]
    if not frames:
        print(f"[results_to_latex] no CSVs in {directory} — skipping {kind} tables.")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_inference_results(directory: Path) -> pd.DataFrame:
    df = _load_csv_dir(directory, "inference-based")
    if df.empty or "split" not in df.columns:
        # Rows written before --split existed are test-split results.
        return df
    # Drop val-split diagnostic rows (test.py --split val); only the test split
    # belongs in the thesis tables.
    return df[df["split"].fillna("test") == "test"].reset_index(drop=True)


def load_training_benchmark(directory: Path) -> pd.DataFrame:
    return _load_csv_dir(directory, "benchmark-based")


def load_offline_stats(logs_dir: Path, prefix: str) -> pd.DataFrame:
    """Loads logs/{prefix}_{method}.csv (measure_sparsity.py / measure_partition.py
    output) into one DataFrame tagged with the method parsed from the filename."""
    frames = []
    for csv_path in sorted(logs_dir.glob(f"{prefix}_*.csv")):
        method = csv_path.stem[len(prefix) + 1 :]
        df = pd.read_csv(csv_path)
        df["reduction_method"] = method
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_benchmark_per_graph(per_graph_dir: Path) -> pd.DataFrame:
    """Loads results/benchmark_per_graph/{run_label}_{run_id}.csv (benchmark.py
    per-graph output) into one DataFrame.

    Prefers the CSV's own ``run_label`` column (benchmark.py writes it into
    every row) over deriving it from the filename stem: filenames now carry a
    ``run_id`` suffix so repeated benchmark submissions don't overwrite each
    other, which would otherwise pollute a filename-derived label. Falls back
    to the stem for older per-graph files written before that column existed.
    """
    frames = []
    for csv_path in sorted(per_graph_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if "run_label" not in df.columns:
            df["run_label"] = csv_path.stem
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap CI on the mean of ``values``."""
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _wilcoxon_p(values: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank test that the paired per-graph savings
    have non-zero median. NaN when scipy can't run it (too few pairs, or a
    degenerate all-identical sample)."""
    if len(values) < 10 or np.allclose(values, values[0]):
        return float("nan")
    try:
        return float(wilcoxon(values).pvalue)
    except ValueError:
        return float("nan")


def build_paired_savings(training_df: pd.DataFrame, per_graph_df: pd.DataFrame) -> pd.DataFrame:
    """Per-config mean VRAM/time savings vs. the baseline, paired by graph_id,
    with a bootstrap CI and a Wilcoxon signed-rank test on the paired deltas.

    Because the benchmark measures one graph per batch on the same seeded
    sample for every config, a reduced graph can be matched to its own
    full-size version (same graph_id) and the per-graph delta averaged — a
    stronger comparison than differencing aggregate means over different
    batch compositions.
    """
    if training_df.empty or per_graph_df.empty:
        return pd.DataFrame()

    baseline_labels = training_df[training_df["reduction_type"] == "none"]["run_label"].unique()
    if len(baseline_labels) == 0:
        return pd.DataFrame()
    baseline_label = baseline_labels[0]

    # groupby-mean on graph_id (not drop_duplicates): a graph_id can
    # legitimately repeat within one run's per-graph CSV, and a config
    # benchmarked more than once (repeated array submissions, each its own
    # run_id) now also contributes one row per repeat for the same graph_id —
    # averaging is correct for both; drop_duplicates' arbitrary
    # first-occurrence pick silently discarded real measurements.
    base = (
        per_graph_df[per_graph_df["run_label"] == baseline_label]
        .groupby("graph_id", as_index=False)[["step_time_s", "peak_vram_allocated_mb"]]
        .mean()
        .rename(columns={"step_time_s": "base_time", "peak_vram_allocated_mb": "base_vram"})
    )
    if base.empty:
        return pd.DataFrame()

    label_map = (
        training_df.drop_duplicates("run_label")
        .set_index("run_label")[["reduction_type", "reduction_method"]]
        .to_dict("index")
    )

    rows = []
    for label in per_graph_df["run_label"].unique():
        if label == baseline_label:
            continue
        red = (
            per_graph_df[per_graph_df["run_label"] == label]
            .groupby("graph_id", as_index=False)[["step_time_s", "peak_vram_allocated_mb"]]
            .mean()
        )
        merged = red.merge(base, on="graph_id")
        if merged.empty:
            continue
        vram_saving = ((1 - merged["peak_vram_allocated_mb"] / merged["base_vram"]) * 100).to_numpy()
        time_saving = ((1 - merged["step_time_s"] / merged["base_time"]) * 100).to_numpy()
        vram_ci = _bootstrap_ci(vram_saving)
        time_ci = _bootstrap_ci(time_saving)
        meta = label_map.get(label, {"reduction_type": "", "reduction_method": ""})
        rows.append(
            {
                "reduction_type": meta["reduction_type"],
                "reduction_method": meta["reduction_method"],
                "n_paired": len(merged),
                "mean_vram_saving_pct": float(vram_saving.mean()),
                "vram_saving_ci_low_pct": vram_ci[0],
                "vram_saving_ci_high_pct": vram_ci[1],
                "vram_saving_wilcoxon_p": _wilcoxon_p(vram_saving),
                "mean_time_saving_pct": float(time_saving.mean()),
                "time_saving_ci_low_pct": time_ci[0],
                "time_saving_ci_high_pct": time_ci[1],
                "time_saving_wilcoxon_p": _wilcoxon_p(time_saving),
            }
        )
    return pd.DataFrame(rows)


def build_baseline_accuracy_table(inference_df: pd.DataFrame) -> pd.DataFrame:
    """RQ1: baseline predictive accuracy on full, unreduced AIGs.

    Filtered to device == "cuda", matching every other table here — test.sh
    and test_cpu.sh both append to the same inference_results.csv, so without
    this filter the baseline config would show up twice (once per device).
    """
    sub = inference_df[
        (inference_df["reduction_type"] == "none") & (inference_df["device"] == "cuda")
    ]
    cols = ["device", "num_graphs", "smooth_l1", "rmse", "r2", "spearman", "throughput_graphs_per_s"]
    cols = [c for c in cols if c in sub.columns]
    return sub[cols].reset_index(drop=True)


def build_reduction_efficiency_table(
    inference_df: pd.DataFrame,
    training_df: pd.DataFrame,
    sparsification_stats: pd.DataFrame,
    partition_stats: pd.DataFrame,
    paired_savings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """RQ2: joins offline reduction cost with runtime (train + inference) cost."""
    if training_df.empty:
        return pd.DataFrame()

    rows = []
    configs = training_df[["reduction_type", "reduction_method"]].drop_duplicates()
    for _, cfg in configs.iterrows():
        rtype, rmethod = cfg["reduction_type"], cfg["reduction_method"]

        train_row = training_df[
            (training_df["reduction_type"] == rtype)
            & _method_matches(training_df["reduction_method"], rmethod)
        ]

        vram_saving = float("nan")
        time_saving = float("nan")
        vram_ci_low = vram_ci_high = float("nan")
        time_ci_low = time_ci_high = float("nan")
        vram_p = time_p = float("nan")
        if paired_savings is not None and not paired_savings.empty:
            s = paired_savings[
                (paired_savings["reduction_type"] == rtype)
                & _method_matches(paired_savings["reduction_method"], rmethod)
            ]
            if not s.empty:
                vram_saving = float(s["mean_vram_saving_pct"].mean())
                time_saving = float(s["mean_time_saving_pct"].mean())
                vram_ci_low = float(s["vram_saving_ci_low_pct"].mean())
                vram_ci_high = float(s["vram_saving_ci_high_pct"].mean())
                time_ci_low = float(s["time_saving_ci_low_pct"].mean())
                time_ci_high = float(s["time_saving_ci_high_pct"].mean())
                vram_p = float(s["vram_saving_wilcoxon_p"].mean())
                time_p = float(s["time_saving_wilcoxon_p"].mean())
        infer_mode = "full_graph" if rtype == "none" else "matched_reduction"
        infer_row = pd.DataFrame()
        if not inference_df.empty:
            infer_row = inference_df[
                (inference_df["reduction_type"] == rtype)
                & _method_matches(inference_df["reduction_method"], rmethod)
                & (inference_df["eval_mode"] == infer_mode)
                & (inference_df["device"] == "cuda")
            ]

        offline_metric = ""
        offline_value = float("nan")
        offline_time_ms = float("nan")
        if rtype == "sparsification" and not sparsification_stats.empty:
            m = sparsification_stats[_method_matches(sparsification_stats["reduction_method"], rmethod)]
            if not m.empty:
                offline_metric = "edge_retention"
                offline_value = float(m["edge_retention"].mean())
                offline_time_ms = float(m["time_s"].mean()) * 1000
        elif rtype == "partition" and not partition_stats.empty:
            m = partition_stats[_method_matches(partition_stats["reduction_method"], rmethod)]
            if not m.empty:
                offline_metric = "edge_cut_ratio"
                offline_value = float(m["edge_cut_ratio"].mean())
                offline_time_ms = float(m["time_s"].mean()) * 1000

        rows.append(
            {
                "reduction_type": rtype,
                "reduction_method": rmethod,
                "offline_metric": offline_metric,
                "offline_value": offline_value,
                "offline_time_ms": offline_time_ms,
                "train_step_time_s": (
                    float(train_row["avg_step_time_s"].mean()) if not train_row.empty else float("nan")
                ),
                "train_peak_vram_allocated_mean_mb": (
                    float(train_row["peak_vram_allocated_mean_mb"].mean())
                    if not train_row.empty
                    else float("nan")
                ),
                "train_peak_vram_allocated_max_mb": (
                    float(train_row["peak_vram_allocated_max_mb"].mean())
                    if not train_row.empty
                    else float("nan")
                ),
                "train_peak_vram_reserved_mean_mb": (
                    float(train_row["peak_vram_reserved_mean_mb"].mean())
                    if not train_row.empty
                    else float("nan")
                ),
                "infer_throughput_graphs_per_s": (
                    float(infer_row["throughput_graphs_per_s"].mean()) if not infer_row.empty else float("nan")
                ),
                "infer_peak_vram_mb": (
                    float(infer_row["peak_vram_mb"].mean()) if not infer_row.empty else float("nan")
                ),
                "vram_saving_pct": vram_saving,
                "vram_saving_ci_low_pct": vram_ci_low,
                "vram_saving_ci_high_pct": vram_ci_high,
                "vram_saving_wilcoxon_p": vram_p,
                "time_saving_pct": time_saving,
                "time_saving_ci_low_pct": time_ci_low,
                "time_saving_ci_high_pct": time_ci_high,
                "time_saving_wilcoxon_p": time_p,
            }
        )
    return pd.DataFrame(rows)


def build_predictive_retention_table(inference_df: pd.DataFrame) -> pd.DataFrame:
    """RQ3: matched-reduction accuracy vs. the full-graph baseline."""
    baseline = inference_df[
        (inference_df["reduction_type"] == "none")
        & (inference_df["eval_mode"] == "full_graph")
        & (inference_df["device"] == "cuda")
    ]
    baseline_rmse = float(baseline["rmse"].mean()) if not baseline.empty else float("nan")
    baseline_r2 = float(baseline["r2"].mean()) if not baseline.empty else float("nan")

    matched = inference_df[
        (inference_df["eval_mode"] == "matched_reduction") & (inference_df["device"] == "cuda")
    ].copy()
    if matched.empty:
        return matched
    matched["rmse_delta_vs_baseline"] = matched["rmse"] - baseline_rmse
    matched["r2_delta_vs_baseline"] = matched["r2"] - baseline_r2
    cols = [
        "reduction_type",
        "reduction_method",
        "rmse",
        "r2",
        "spearman",
        "rmse_delta_vs_baseline",
        "r2_delta_vs_baseline",
    ]
    return matched[cols].reset_index(drop=True)


def build_cross_state_table(inference_df: pd.DataFrame) -> pd.DataFrame:
    """RQ4: matched-state vs. cross-state (full-graph) accuracy for reduced-trained models."""
    reduced = inference_df[
        (inference_df["reduction_type"] != "none") & (inference_df["device"] == "cuda")
    ]
    matched = reduced[reduced["eval_mode"] == "matched_reduction"].set_index(
        ["reduction_type", "reduction_method"]
    )
    cross = reduced[reduced["eval_mode"] == "full_graph"].set_index(
        ["reduction_type", "reduction_method"]
    )

    rows = []
    for key in matched.index:
        if key not in cross.index:
            continue
        m, c = matched.loc[key], cross.loc[key]
        rows.append(
            {
                "reduction_type": key[0],
                "reduction_method": key[1],
                "matched_rmse": m["rmse"],
                "matched_r2": m["r2"],
                "cross_state_rmse": c["rmse"],
                "cross_state_r2": c["r2"],
                "rmse_drop": c["rmse"] - m["rmse"],
                "r2_drop": m["r2"] - c["r2"],
            }
        )
    return pd.DataFrame(rows)


def build_pareto_front(inference_df: pd.DataFrame, training_df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy (RMSE/R2) vs. cost (VRAM/step-time) per config, for the RQ3 Pareto scatter."""
    if training_df.empty:
        return pd.DataFrame()

    rows = []
    configs = training_df[["reduction_type", "reduction_method"]].drop_duplicates()
    for _, cfg in configs.iterrows():
        rtype, rmethod = cfg["reduction_type"], cfg["reduction_method"]
        infer_mode = "full_graph" if rtype == "none" else "matched_reduction"
        infer_row = pd.DataFrame()
        if not inference_df.empty:
            infer_row = inference_df[
                (inference_df["reduction_type"] == rtype)
                & _method_matches(inference_df["reduction_method"], rmethod)
                & (inference_df["eval_mode"] == infer_mode)
                & (inference_df["device"] == "cuda")
            ]
        train_row = training_df[
            (training_df["reduction_type"] == rtype)
            & _method_matches(training_df["reduction_method"], rmethod)
        ]
        rows.append(
            {
                "reduction_type": rtype,
                "reduction_method": rmethod,
                "rmse": float(infer_row["rmse"].mean()) if not infer_row.empty else float("nan"),
                "r2": float(infer_row["r2"].mean()) if not infer_row.empty else float("nan"),
                "train_step_time_s": (
                    float(train_row["avg_step_time_s"].mean()) if not train_row.empty else float("nan")
                ),
                "train_peak_vram_allocated_mean_mb": (
                    float(train_row["peak_vram_allocated_mean_mb"].mean())
                    if not train_row.empty
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def build_vram_scaling_table(per_graph_df: pd.DataFrame) -> pd.DataFrame:
    """Peak allocated VRAM vs. graph size, fit per config (OLS slope/intercept
    over the measured sample's (num_nodes, peak_vram_allocated_mb) pairs).

    A single mean peak VRAM over ~100 sampled graphs doesn't generalize to
    graphs outside the sample; the slope (MB per 1,000 nodes) does, and is
    the more defensible number for arguing a method's memory benefit holds at
    sizes larger than anything actually measured.
    """
    if per_graph_df.empty or "peak_vram_allocated_mb" not in per_graph_df.columns:
        return pd.DataFrame()

    rows = []
    for label, group in per_graph_df.groupby("run_label"):
        valid = group.dropna(subset=["num_nodes", "peak_vram_allocated_mb"])
        if len(valid) < 2:
            continue
        slope, intercept = np.polyfit(valid["num_nodes"], valid["peak_vram_allocated_mb"], deg=1)
        rows.append(
            {
                "run_label": label,
                "n_graphs": len(valid),
                "vram_mb_per_1k_nodes": float(slope * 1000),
                "vram_intercept_mb": float(intercept),
            }
        )
    return pd.DataFrame(rows)


def main(args: argparse.Namespace) -> None:
    inference_df = load_inference_results(Path(args.inference_dir))
    training_df = load_training_benchmark(Path(args.training_dir))
    sparsification_stats = load_offline_stats(Path(args.logs_dir), "sparsification_stats")
    partition_stats = load_offline_stats(Path(args.logs_dir), "partition_stats")
    per_graph_df = load_benchmark_per_graph(Path(args.per_graph_dir))
    paired_savings = build_paired_savings(training_df, per_graph_df)

    tables_dir = Path(args.tables_dir)

    if not inference_df.empty:
        write_booktabs_table(
            build_baseline_accuracy_table(inference_df),
            tables_dir / "baseline_accuracy.tex",
            "Baseline predictive accuracy on full, unreduced AIGs (RQ1).",
            "tab:baseline_accuracy",
        )
        write_booktabs_table(
            build_predictive_retention_table(inference_df),
            tables_dir / "predictive_retention.tex",
            "Predictive retention of reduced-graph training vs. the full-graph baseline (RQ3).",
            "tab:predictive_retention",
        )
        write_booktabs_table(
            build_cross_state_table(inference_df),
            tables_dir / "cross_state_generalization.tex",
            "Matched-state vs. cross-state (full-graph) accuracy for models trained on reduced AIGs (RQ4).",
            "tab:cross_state_generalization",
        )

    if not training_df.empty:
        write_booktabs_table(
            build_reduction_efficiency_table(
                inference_df, training_df, sparsification_stats, partition_stats,
                paired_savings=paired_savings,
            ),
            tables_dir / "reduction_efficiency.tex",
            "Offline reduction cost and runtime (training + inference) efficiency across reduction methods (RQ2).",
            "tab:reduction_efficiency",
        )

        pareto_df = build_pareto_front(inference_df, training_df)
        if not pareto_df.empty:
            pareto_path = tables_dir / "pareto_front.csv"
            pareto_path.parent.mkdir(parents=True, exist_ok=True)
            pareto_df.to_csv(pareto_path, index=False)
            print(f"[results_to_latex] Wrote {pareto_path}")

    write_booktabs_table(
        build_vram_scaling_table(per_graph_df),
        tables_dir / "vram_scaling.tex",
        "Peak allocated training-step VRAM vs. graph size (MB per 1{,}000 nodes) per reduction method.",
        "tab:vram_scaling",
    )

    print(f"[results_to_latex] Done. Tables in {tables_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Turn results/ + logs/ CSVs into booktabs .tex tables for the thesis Results chapter."
    )
    parser.add_argument("--inference_dir", type=str, default="results/inference_results")
    parser.add_argument("--training_dir", type=str, default="results/training_benchmark")
    parser.add_argument("--logs_dir", type=str, default="logs")
    parser.add_argument("--per_graph_dir", type=str, default="results/benchmark_per_graph")
    parser.add_argument("--tables_dir", type=str, default="results/tables")

    main(parser.parse_args())
