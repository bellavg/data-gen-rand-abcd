"""Unit tests for src/benchmark.py — reduction-config dispatch, CSV helpers,
and the one-graph-per-batch measurement loop (run_benchmark) exercised on CPU
against the adder.aig mock fixture (same pattern as test_final_config.py's
_mock_dataset). CUDA-only assertions are skipped when unavailable."""

import csv
import math

import pytest
import torch

import config
from benchmark import (
    _safe_nanmean,
    build_model,
    resolve_reduction_kwargs,
    run_benchmark,
    run_label_for,
    write_per_graph_csv,
    write_single_row_csv,
)
from data.data_utils import aig_to_pytorch_geometric
from data.datamodule import AIGDataModule


class TestRunLabelFor:
    def test_baseline_uses_bare_algorithm(self):
        assert run_label_for("Orchestrate", "none", None) == "Orchestrate"

    def test_reduction_appends_method_suffix(self):
        assert run_label_for("Orchestrate", "partition", "metis") == "Orchestrate_metis"


class TestResolveReductionKwargs:
    def test_none(self):
        assert resolve_reduction_kwargs("none", None) == {"sparsification": None, "partition": None}

    def test_sparsification(self):
        assert resolve_reduction_kwargs("sparsification", "and_gate_only") == {
            "sparsification": "and_gate_only",
            "partition": None,
        }

    def test_partition(self):
        assert resolve_reduction_kwargs("partition", "random") == {
            "sparsification": None,
            "partition": "random",
        }

    def test_summarization_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            resolve_reduction_kwargs("summarization", "snap")

    def test_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError):
            resolve_reduction_kwargs("bogus", "x")


class TestSafeNanmean:
    def test_all_nan_returns_nan_without_warning(self):
        assert math.isnan(_safe_nanmean([float("nan"), float("nan")]))

    def test_mixed_ignores_nan(self):
        assert _safe_nanmean([2.0, float("nan"), 4.0]) == pytest.approx(3.0)


class TestWriteSingleRowCsv:
    def test_header_and_single_row(self, tmp_path):
        csv_path = tmp_path / "sub" / "bench.csv"  # parent auto-created
        write_single_row_csv(csv_path, {"reduction_type": "none", "avg_step_time_s": 0.1})
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        assert rows == [["reduction_type", "avg_step_time_s"], ["none", "0.1"]]

    def test_overwrites_on_rerun(self, tmp_path):
        csv_path = tmp_path / "bench.csv"
        write_single_row_csv(csv_path, {"reduction_type": "none", "avg_step_time_s": 0.1})
        write_single_row_csv(csv_path, {"reduction_type": "partition", "avg_step_time_s": 0.2})
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        assert rows == [["reduction_type", "avg_step_time_s"], ["partition", "0.2"]]


class TestWritePerGraphCsv:
    def _rows(self, n):
        return [
            {
                "graph_id": f"g{i}",
                "num_nodes": i,
                "num_edges": 2 * i,
                "step_time_s": 0.1,
                "peak_vram_mb": float("nan"),
                "activation_vram_mb": float("nan"),
            }
            for i in range(n)
        ]

    def test_writes_all_under_cap(self, tmp_path):
        path = tmp_path / "pg.csv"
        write_per_graph_csv(path, self._rows(3), max_rows=10)
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_caps_and_is_seeded_reproducible(self, tmp_path):
        rows = self._rows(1000)
        a, b = tmp_path / "a.csv", tmp_path / "b.csv"
        write_per_graph_csv(a, rows, max_rows=50)
        write_per_graph_csv(b, rows, max_rows=50)
        with open(a, newline="") as f:
            rows_a = list(csv.DictReader(f))
        with open(b, newline="") as f:
            rows_b = list(csv.DictReader(f))
        assert len(rows_a) == 50
        assert [r["graph_id"] for r in rows_a] == [r["graph_id"] for r in rows_b]


class TestBuildModel:
    def test_uses_config_defaults_and_respects_compile_flag(self):
        model = build_model(compile_model=False)
        assert model.hparams.encoder_name == config.ENCODER_NAME
        assert model.hparams.hidden_dim == config.HIDDEN_DIM
        assert model.hparams.pe_type == config.PE_TYPE
        assert bool(model.hparams.compile_model) is False


def _mock_dataset(tmp_path, algorithm: str = "Orchestrate", n: int = 20):
    """Clones adder.aig several times with injected feature noise — same
    pattern used in test_final_config.py's fast_dev_run tests."""
    from pathlib import Path

    aig_path = Path("src/unittests/data/adder.aig")
    assert aig_path.exists(), f"Dummy AIG missing at {aig_path}!"

    base_data = aig_to_pytorch_geometric(aig_path)
    pt_paths = []
    for i in range(n):
        data = base_data.clone()
        data.x = data.x.float() + torch.randn_like(data.x.float()) * 0.05
        pt = tmp_path / f"graph_{i}.pt"
        torch.save(data, pt)
        pt_paths.append(pt)

    csv_path = tmp_path / f"dummy_{algorithm}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "unoptimized_graph_path",
                "design",
                "algorithm",
                "tier_id",
                "optimizability",
            ],
        )
        writer.writeheader()
        for i, pt in enumerate(pt_paths):
            writer.writerow(
                {
                    "unoptimized_graph_path": str(pt),
                    "design": "adder",
                    "algorithm": algorithm,
                    "tier_id": "1",
                    "optimizability": 0.5 + (i * 0.01),
                }
            )
    return csv_path


class TestRunBenchmarkIntegration:
    def test_per_graph_loop_excludes_warmup_and_reports_sane_aggregates(self, tmp_path):
        csv_path = _mock_dataset(tmp_path)

        datamodule = AIGDataModule(
            csv_paths=[str(csv_path)],
            positional_encoding=config.PE_TYPE if config.PE_TYPE != "none" else None,
            batch_size=1,
            num_workers=0,
        )
        datamodule.setup("fit")
        n_train = len(datamodule.train_ds)

        model = build_model(compile_model=False)
        num_warmup = 2
        aggregate, per_graph = run_benchmark(
            model,
            datamodule.train_ds,
            torch.device("cpu"),
            num_warmup=num_warmup,
            precision="32-true",
        )

        # Warmup graphs are excluded from every aggregate.
        assert aggregate["num_measured_graphs"] == n_train - num_warmup
        assert len(per_graph) == n_train - num_warmup

        assert aggregate["avg_step_time_s"] > 0
        assert aggregate["throughput_graphs_per_s"] > 0
        assert aggregate["avg_nodes_per_graph"] > 0
        assert aggregate["avg_edges_per_graph"] > 0

        # Each per-graph row carries the fields the pairing/plots depend on.
        row = per_graph[0]
        assert set(row) == {
            "graph_id",
            "num_nodes",
            "num_edges",
            "step_time_s",
            "peak_vram_mb",
            "activation_vram_mb",
        }
        assert row["num_nodes"] > 0

        # No CUDA: VRAM must be NaN (unmeasured), not a stale/garbage value.
        if not torch.cuda.is_available():
            assert math.isnan(aggregate["peak_vram_mb"])
            assert math.isnan(aggregate["activation_vram_mb"])
            assert math.isnan(row["peak_vram_mb"])
            assert math.isnan(row["activation_vram_mb"])
