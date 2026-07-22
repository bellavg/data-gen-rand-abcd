"""Unit tests for src/benchmark.py — reduction-config dispatch, batch-stats
bookkeeping, and BenchmarkCallback's step-timing/VRAM aggregation via a tiny
CPU fast_dev_run-style Trainer run (same pattern as test_final_config.py's
_mock_dataset). CUDA-only assertions are skipped when unavailable."""

import csv

import pytest
import pytorch_lightning as pl
import torch

import config
from benchmark import (
    BenchmarkCallback,
    _batch_stats,
    append_csv_row,
    build_model,
    resolve_reduction_kwargs,
)
from data.data_utils import aig_to_pytorch_geometric
from data.datamodule import AIGDataModule


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


class TestBatchStats:
    def test_counts_from_synthetic_batch(self):
        class _FakeBatch:
            num_graphs = 4
            x = torch.zeros(9, 4)
            edge_index = torch.zeros(2, 14, dtype=torch.long)

        num_graphs, num_nodes, num_edges = _batch_stats(_FakeBatch())
        assert (num_graphs, num_nodes, num_edges) == (4, 9, 14)


class TestAppendCsvRow:
    def test_writes_header_once(self, tmp_path):
        csv_path = tmp_path / "bench.csv"
        append_csv_row(csv_path, {"reduction_type": "none", "avg_step_time_s": 0.1})
        append_csv_row(csv_path, {"reduction_type": "partition", "avg_step_time_s": 0.2})

        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["reduction_type", "avg_step_time_s"]
        assert rows[1][0] == "none"
        assert rows[2][0] == "partition"


class TestBuildModel:
    def test_uses_config_defaults_and_respects_compile_flag(self):
        model = build_model(compile_model=False)
        assert model.hparams.encoder_name == config.ENCODER_NAME
        assert model.hparams.hidden_dim == config.HIDDEN_DIM
        assert model.hparams.pe_type == config.PE_TYPE
        assert bool(model.hparams.compile_model) is False


def _mock_dataset(tmp_path, algorithm: str = "Orchestrate"):
    """Clones adder.aig several times with injected feature noise — same
    pattern used in test_final_config.py's fast_dev_run tests."""
    from pathlib import Path

    aig_path = Path("src/unittests/data/adder.aig")
    assert aig_path.exists(), f"Dummy AIG missing at {aig_path}!"

    base_data = aig_to_pytorch_geometric(aig_path)
    pt_paths = []
    # 20 clones for a non-empty val split, matching test_final_config.py's
    # _mock_dataset (its comment: "healthy 8/1/1 data split").
    for i in range(20):
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


class TestBenchmarkCallbackIntegration:
    def test_step_timing_excludes_warmup_and_summary_is_sane(self, tmp_path):
        csv_path = _mock_dataset(tmp_path)

        model = build_model(compile_model=False)
        # No train_num_samples here: it subsets both train AND val candidates
        # (see AIGDataModule.setup) before the design-level split runs, which
        # can starve val to zero with this tiny single-design mock fixture.
        # limit_train_batches below already bounds how many steps actually run.
        datamodule = AIGDataModule(
            csv_paths=[str(csv_path)],
            positional_encoding=config.PE_TYPE if config.PE_TYPE != "none" else None,
            batch_size=2,
            num_workers=0,
        )
        datamodule.setup("fit")

        num_warmup_steps = 1
        num_measure_steps = 2
        callback = BenchmarkCallback(
            num_warmup_steps=num_warmup_steps,
            num_measure_steps=num_measure_steps,
            gpu_util_sample_every=1,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            limit_train_batches=num_warmup_steps + num_measure_steps,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            callbacks=[callback],
        )
        trainer.fit(model, datamodule=datamodule)

        # Warmup steps must be excluded from every aggregate.
        assert len(callback._step_times) == num_measure_steps
        assert len(callback._graph_counts) == num_measure_steps

        summary = callback.summary(model.device)
        assert summary["avg_step_time_s"] > 0
        assert summary["throughput_graphs_per_s"] > 0
        assert summary["avg_nodes_per_batch"] > 0
        assert summary["avg_edges_per_batch"] > 0
        # No CUDA device — VRAM must be reported as NaN, not a stale/garbage value.
        if not torch.cuda.is_available():
            import math

            assert math.isnan(summary["peak_vram_mb"])
