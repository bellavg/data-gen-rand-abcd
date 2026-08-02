"""Build every Results-chapter figure and table in one pass.

    PYTHONPATH=src python -m analysis.make_all

Reads the exported result CSVs under ``--results-dir`` (extract
``results/archives/*.tar.gz`` first) and writes:

    <thesis>/media/results/figures/*.pdf
    <thesis>/media/results/tables/*.tex

Figures whose data does not exist are still produced, from
``analysis.fake_data``, and are red-framed, cross-hatched and watermarked. See
that module for what each one is waiting on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis import (
    fig_dataset,
    fig_placeholders,
    fig_rq1,
    fig_rq2,
    fig_rq3,
    fig_rq4,
    fig_rq5,
    loaders,
    tables,
)
from analysis.fake_data import BASELINE_MODELS
from analysis.results_to_latex import build_paired_savings
from analysis.style import apply_style

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "results"
DEFAULT_THESIS = REPO_ROOT / "IV_Gardner___Master_AI_Thesis_Outline"


def load_context(results_dir: Path) -> dict:
    ctx = {
        "measurements_dir": results_dir / "measurements",
        "offline": loaders.offline_stats(results_dir / "measurements"),
        "bench": loaders.load_training_benchmark(results_dir / "training_benchmark"),
        "per_graph": loaders.load_benchmark_per_graph(results_dir / "benchmark_per_graph"),
        "inference": loaders.load_all_inference(results_dir / "inference_results"),
        "runs": loaders.load_wandb_runs(results_dir / "wandb_export"),
        "history": loaders.load_wandb_history(results_dir / "wandb_export"),
        "trials": loaders.load_hp_trials(results_dir / "hp_tuning"),
        "preds_test": loaders.load_predictions(
            results_dir / "predictions", method="none", eval_mode="full_graph", split="test"
        ),
        "preds_val": loaders.load_predictions(
            results_dir / "predictions", method="none", eval_mode="full_graph", split="val"
        ),
    }
    ctx["savings"] = build_paired_savings(ctx["bench"], ctx["per_graph"])
    if not ctx["bench"].empty:
        baseline = ctx["bench"][ctx["bench"]["reduction_type"] == "none"]
        if not baseline.empty:
            ctx["savings"].attrs["baseline_step_s"] = float(
                baseline["avg_step_time_s"].iloc[0]
            )
    ctx["matched"] = fig_rq3.matched_state(ctx["inference"])
    ctx["strata"] = loaders.stratified_metrics(results_dir / "predictions")
    ctx["by_tier"] = loaders.grouped_metrics(results_dir / "predictions", "tier")
    ctx["by_source"] = loaders.grouped_metrics(
        results_dir / "predictions", "source_algorithm"
    )
    ctx["by_band"] = loaders.banded_metrics(results_dir / "predictions")
    return ctx


def main(args: argparse.Namespace) -> None:
    apply_style()
    figures = args.media_dir / "figures"
    tables_dir = args.media_dir / "tables"
    ctx = load_context(args.results_dir)

    missing = [k for k in ("inference", "bench", "preds_test") if _empty(ctx[k])]
    if missing:
        raise SystemExit(
            f"Missing input data for: {', '.join(missing)}. "
            f"Extract results/archives/*.tar.gz into {args.results_dir} first."
        )

    fig_dataset.build(
        ctx["preds_test"], figures,
        measurements_dir=ctx["measurements_dir"], preds_val=ctx["preds_val"],
    )
    ctx["tier1"], ctx["per_design"] = fig_rq1.build(
        ctx["preds_test"], ctx["preds_val"], ctx["history"], ctx["trials"],
        BASELINE_MODELS, figures,
    )
    fig_rq2.build(
        ctx["offline"], ctx["bench"], ctx["per_graph"], ctx["savings"],
        ctx["measurements_dir"], figures,
    )
    fig_rq3.build(
        ctx["matched"], ctx["inference"], ctx["offline"], ctx["savings"],
        ctx["strata"], ctx["by_tier"], ctx["by_source"], ctx["by_band"], figures,
    )
    ctx["pairings"] = fig_rq4.build(ctx["matched"], ctx["offline"], figures)
    ctx["cross"] = fig_rq5.build(ctx["inference"], figures)
    fig_placeholders.build(figures)

    tables.build(ctx, tables_dir)

    n_figs = len(list(figures.glob("*.pdf")))
    n_tabs = len(list(tables_dir.glob("*.tex")))
    print(f"\n[make_all] {n_figs} figures in {figures}")
    print(f"[make_all] {n_tabs} tables in {tables_dir}")


def _empty(value) -> bool:
    return isinstance(value, pd.DataFrame) and value.empty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--media-dir",
        type=Path,
        default=DEFAULT_THESIS / "media" / "results",
        help="Where figures/ and tables/ are written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
