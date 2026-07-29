"""Unit tests for src/results_to_latex.py — the per-graph/aggregate CSV
loaders and the paired-savings statistics, which is where a real bug
(drop_duplicates silently discarding repeat measurements instead of
averaging them) was fixed."""

import math

import numpy as np
import pandas as pd
import pytest

from results_to_latex import (
    _bootstrap_ci,
    _wilcoxon_p,
    build_paired_savings,
    build_vram_scaling_table,
    load_benchmark_per_graph,
)


class TestLoadBenchmarkPerGraph:
    def test_prefers_csv_run_label_column_over_filename(self, tmp_path):
        # Filename carries a run_id suffix (as benchmark.py now writes), so a
        # filename-derived label would be wrong unless the CSV's own column
        # is used instead.
        df = pd.DataFrame({"graph_id": ["g0"], "run_label": ["Orchestrate_metis"]})
        df.to_csv(tmp_path / "Orchestrate_metis_12345.csv", index=False)

        loaded = load_benchmark_per_graph(tmp_path)
        assert loaded["run_label"].tolist() == ["Orchestrate_metis"]

    def test_falls_back_to_filename_stem_without_run_label_column(self, tmp_path):
        df = pd.DataFrame({"graph_id": ["g0"]})
        df.to_csv(tmp_path / "Orchestrate.csv", index=False)

        loaded = load_benchmark_per_graph(tmp_path)
        assert loaded["run_label"].tolist() == ["Orchestrate"]

    def test_empty_dir_returns_empty_frame(self, tmp_path):
        assert load_benchmark_per_graph(tmp_path).empty


class TestBootstrapCi:
    def test_too_few_values_returns_nan(self):
        low, high = _bootstrap_ci(np.array([1.0]))
        assert math.isnan(low) and math.isnan(high)

    def test_brackets_the_true_mean_for_a_tight_sample(self):
        values = np.array([10.0] * 50)  # zero-variance sample
        low, high = _bootstrap_ci(values)
        assert low == pytest.approx(10.0)
        assert high == pytest.approx(10.0)


class TestWilcoxonP:
    def test_too_few_pairs_returns_nan(self):
        assert math.isnan(_wilcoxon_p(np.array([1.0, 2.0, 3.0])))

    def test_all_identical_returns_nan(self):
        assert math.isnan(_wilcoxon_p(np.array([5.0] * 20)))

    def test_clear_nonzero_shift_is_significant(self):
        rng = np.random.default_rng(0)
        values = rng.normal(loc=20.0, scale=1.0, size=30)  # consistently >> 0
        p = _wilcoxon_p(values)
        assert p < 0.01


class TestBuildPairedSavings:
    def _training_df(self):
        return pd.DataFrame(
            [
                {"reduction_type": "none", "reduction_method": "", "run_label": "Orchestrate"},
                {
                    "reduction_type": "sparsification",
                    "reduction_method": "pagerank",
                    "run_label": "Orchestrate_pagerank",
                },
            ]
        )

    def test_averages_repeat_measurements_instead_of_picking_one(self):
        # Same graph_id measured twice per config (two "repeat" benchmark
        # submissions) — the OLD drop_duplicates behavior would have kept an
        # arbitrary one of the two rows per config instead of averaging.
        per_graph_df = pd.DataFrame(
            [
                {"run_label": "Orchestrate", "graph_id": "g0", "step_time_s": 1.0, "peak_vram_allocated_mb": 100.0},
                {"run_label": "Orchestrate", "graph_id": "g0", "step_time_s": 2.0, "peak_vram_allocated_mb": 200.0},
                {"run_label": "Orchestrate_pagerank", "graph_id": "g0", "step_time_s": 0.5, "peak_vram_allocated_mb": 50.0},
                {"run_label": "Orchestrate_pagerank", "graph_id": "g0", "step_time_s": 1.5, "peak_vram_allocated_mb": 150.0},
            ]
        )
        result = build_paired_savings(self._training_df(), per_graph_df)
        assert len(result) == 1
        row = result.iloc[0]
        # baseline mean: time=1.5, vram=150; reduced mean: time=1.0, vram=100
        assert row["mean_time_saving_pct"] == pytest.approx((1 - 1.0 / 1.5) * 100)
        assert row["mean_vram_saving_pct"] == pytest.approx((1 - 100.0 / 150.0) * 100)

    def test_empty_inputs_return_empty_frame(self):
        assert build_paired_savings(pd.DataFrame(), pd.DataFrame()).empty
        assert build_paired_savings(self._training_df(), pd.DataFrame()).empty

    def test_no_baseline_config_returns_empty_frame(self):
        training_df = pd.DataFrame(
            [{"reduction_type": "sparsification", "reduction_method": "pagerank", "run_label": "Orchestrate_pagerank"}]
        )
        per_graph_df = pd.DataFrame(
            [{"run_label": "Orchestrate_pagerank", "graph_id": "g0", "step_time_s": 1.0, "peak_vram_allocated_mb": 100.0}]
        )
        assert build_paired_savings(training_df, per_graph_df).empty


class TestBuildVramScalingTable:
    def test_fits_slope_per_run_label(self):
        # Perfectly linear: vram = 2 * num_nodes + 10
        per_graph_df = pd.DataFrame(
            [
                {"run_label": "Orchestrate", "num_nodes": n, "peak_vram_allocated_mb": 2 * n + 10}
                for n in [10, 20, 30, 40]
            ]
        )
        result = build_vram_scaling_table(per_graph_df)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["vram_mb_per_1k_nodes"] == pytest.approx(2000.0)
        assert row["vram_intercept_mb"] == pytest.approx(10.0)

    def test_empty_input_returns_empty_frame(self):
        assert build_vram_scaling_table(pd.DataFrame()).empty

    def test_single_graph_config_is_skipped(self):
        per_graph_df = pd.DataFrame(
            [{"run_label": "Orchestrate", "num_nodes": 10, "peak_vram_allocated_mb": 30.0}]
        )
        assert build_vram_scaling_table(per_graph_df).empty
