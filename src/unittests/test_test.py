"""Unit tests for src/test.py's pure logic — metric math, reduction-config
dispatch, and CSV I/O helpers. Deliberately excludes anything that needs a
real checkpoint/dataloader/GPU (that's exercised on Snellius via test.sh)."""

import csv
import math

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch as PyGBatch

from test import (
    _batch_per_graph_counts,
    append_csv_row,
    compute_accuracy_metrics,
    resolve_reduction_kwargs,
    run_label_for,
    write_predictions_csv,
)


class TestRunLabelFor:
    def test_baseline_uses_bare_algorithm(self):
        assert run_label_for("Orchestrate", "none", None) == "Orchestrate"

    def test_reduction_appends_method_suffix(self):
        assert run_label_for("Orchestrate", "sparsification", "pagerank") == "Orchestrate_pagerank"
        assert run_label_for("Orchestrate", "partition", "metis") == "Orchestrate_metis"


class TestResolveReductionKwargs:
    def test_none(self):
        assert resolve_reduction_kwargs("none", None) == {"sparsification": None, "partition": None}

    def test_sparsification(self):
        assert resolve_reduction_kwargs("sparsification", "pagerank") == {
            "sparsification": "pagerank",
            "partition": None,
        }

    def test_partition(self):
        assert resolve_reduction_kwargs("partition", "metis") == {
            "sparsification": None,
            "partition": "metis",
        }

    def test_summarization_raises_not_implemented(self):
        """Summarization isn't wired into AIGDataModule yet — the dispatch
        must fail loudly, not silently no-op, so a future summarization run
        can't accidentally fall through to the baseline path."""
        with pytest.raises(NotImplementedError):
            resolve_reduction_kwargs("summarization", "snap")

    def test_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError):
            resolve_reduction_kwargs("bogus", "x")


class TestComputeAccuracyMetrics:
    def test_perfect_predictions(self):
        targets = torch.tensor([0.1, 0.5, 0.9, 0.3])
        preds = targets.clone()
        metrics = compute_accuracy_metrics(preds, targets)
        assert metrics["smooth_l1"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["spearman"] == pytest.approx(1.0, abs=1e-6)

    def test_matches_manual_rmse_and_r2(self):
        targets = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8])
        preds = torch.tensor([0.1, 0.1, 0.5, 0.5, 0.9])
        metrics = compute_accuracy_metrics(preds, targets)

        errors = (preds - targets).tolist()
        manual_rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
        assert metrics["rmse"] == pytest.approx(manual_rmse, rel=1e-5)

        mean_target = sum(targets.tolist()) / len(targets)
        ss_res = sum((p - t) ** 2 for p, t in zip(preds.tolist(), targets.tolist()))
        ss_tot = sum((t - mean_target) ** 2 for t in targets.tolist())
        manual_r2 = 1.0 - ss_res / ss_tot
        assert metrics["r2"] == pytest.approx(manual_r2, rel=1e-5)

    def test_inverse_ranking_gives_negative_spearman(self):
        targets = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        preds = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1])
        metrics = compute_accuracy_metrics(preds, targets)
        assert metrics["spearman"] == pytest.approx(-1.0, abs=1e-6)

    def test_empty_inputs_return_nan_not_crash(self):
        metrics = compute_accuracy_metrics(torch.empty(0), torch.empty(0))
        assert math.isnan(metrics["smooth_l1"])
        assert math.isnan(metrics["rmse"])
        assert math.isnan(metrics["r2"])
        assert math.isnan(metrics["spearman"])

    def test_single_sample_spearman_is_nan_not_crash(self):
        """scipy's spearmanr is undefined for n=1; must not raise."""
        metrics = compute_accuracy_metrics(torch.tensor([0.5]), torch.tensor([0.4]))
        assert math.isnan(metrics["spearman"])
        assert not math.isnan(metrics["rmse"])


class TestBatchPerGraphCounts:
    def test_counts_match_two_graph_batch(self):
        g1 = Data(
            x=torch.zeros(3, 4),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
        )
        g2 = Data(
            x=torch.zeros(2, 4),
            edge_index=torch.tensor([[0], [1]]),
        )
        batch = PyGBatch.from_data_list([g1, g2])

        node_counts, edge_counts = _batch_per_graph_counts(batch)
        assert node_counts == [3, 2]
        assert edge_counts == [2, 1]


class TestCsvHelpers:
    def test_append_csv_row_writes_header_once(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        append_csv_row(csv_path, {"a": 1, "b": "x"})
        append_csv_row(csv_path, {"a": 2, "b": "y"})

        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["a", "b"]
        assert rows[1] == ["1", "x"]
        assert rows[2] == ["2", "y"]

    def test_write_predictions_csv_writes_all_rows_when_under_cap(self, tmp_path):
        path = tmp_path / "preds.csv"
        per_graph = {
            "graph_id": ["g0", "g1", "g2"],
            "num_nodes": [10, 20, 30],
            "num_edges": [15, 25, 35],
            "target": [0.1, 0.2, 0.3],
            "prediction": [0.15, 0.18, 0.35],
        }
        write_predictions_csv(path, per_graph, max_rows=10)

        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert float(rows[0]["abs_error"]) == pytest.approx(abs(0.15 - 0.1))

    def test_write_predictions_csv_caps_and_is_seeded_reproducible(self, tmp_path):
        n = 1000
        per_graph = {
            "graph_id": [f"g{i}" for i in range(n)],
            "num_nodes": list(range(n)),
            "num_edges": list(range(n)),
            "target": [0.5] * n,
            "prediction": [0.5] * n,
        }
        path_a = tmp_path / "preds_a.csv"
        path_b = tmp_path / "preds_b.csv"
        write_predictions_csv(path_a, per_graph, max_rows=50)
        write_predictions_csv(path_b, per_graph, max_rows=50)

        with open(path_a, newline="") as f:
            rows_a = list(csv.DictReader(f))
        with open(path_b, newline="") as f:
            rows_b = list(csv.DictReader(f))

        assert len(rows_a) == 50
        assert [r["graph_id"] for r in rows_a] == [r["graph_id"] for r in rows_b]
