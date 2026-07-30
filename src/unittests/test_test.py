"""Unit tests for src/test.py — metric math, reduction-config dispatch, CSV
I/O helpers, and the forward sweep itself (run_eval_pass) driven on CPU
against a stub datamodule and model. Still excludes anything needing a real
checkpoint, graph cache, or GPU (that's exercised on Snellius via test.sh)."""

import csv
import math

import pytest
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Batch as PyGBatch

import config
import test as test_module
from data.sampler import BalancedDynamicBatchSampler
from test import (
    _batch_per_graph_counts,
    batching_label,
    build_eval_passes,
    compute_accuracy_metrics,
    emitted_sample_order,
    resolve_checkpoint_path,
    resolve_reduction_kwargs,
    run_eval_pass,
    run_label_for,
    wandb_run_name_for,
    write_predictions_csv,
    write_single_row_csv,
)


class TestRunLabelFor:
    def test_baseline_uses_bare_algorithm(self):
        assert run_label_for("Orchestrate", "none", None) == "Orchestrate"

    def test_reduction_appends_method_suffix(self):
        assert run_label_for("Orchestrate", "sparsification", "pagerank") == "Orchestrate_pagerank"
        assert run_label_for("Orchestrate", "partition", "metis") == "Orchestrate_metis"

    def test_default_split_by_adds_no_suffix(self):
        """train.py leaves design-split checkpoint dirs unsuffixed, so eval
        must not go looking for an "Orchestrate_design" directory."""
        assert run_label_for("Orchestrate", "none", None, config.SPLIT_BY) == "Orchestrate"

    def test_non_default_split_by_matches_train_pys_directory(self):
        assert run_label_for("Orchestrate", "none", None, "recipe") == "Orchestrate_recipe"
        assert (
            run_label_for("Orchestrate", "sparsification", "pagerank", "random")
            == "Orchestrate_pagerank_random"
        )


class TestWandbRunNameFor:
    """Mirrors train.py's scheme (train_<algo>[_<type>_<method>]) with a
    test_ prefix, so each config's eval run sits beside its training run."""

    def test_baseline(self):
        assert (
            wandb_run_name_for("Orchestrate", "none", None, "cuda")
            == "test_Orchestrate_cuda"
        )

    def test_sparsification_and_partition(self):
        assert (
            wandb_run_name_for("Orchestrate", "sparsification", "pagerank", "cuda")
            == "test_Orchestrate_sparsification_pagerank_cuda"
        )
        assert (
            wandb_run_name_for("Orchestrate", "partition", "metis", "cuda")
            == "test_Orchestrate_partition_metis_cuda"
        )

    def test_device_disambiguates_the_gpu_and_cpu_passes(self):
        """test.sh and test_cpu.sh evaluate the same 9 configs; without device
        in the name both would collide on one WandB run."""
        assert wandb_run_name_for(
            "Orchestrate", "none", None, "cuda"
        ) != wandb_run_name_for("Orchestrate", "none", None, "cpu")

    def test_split_by_suffix_mirrors_train_py(self):
        assert (
            wandb_run_name_for("Orchestrate", "none", None, "cuda", config.SPLIT_BY)
            == "test_Orchestrate_cuda"
        )
        assert (
            wandb_run_name_for("Orchestrate", "none", None, "cuda", "recipe")
            == "test_Orchestrate_cuda_recipe"
        )


class TestBatchingLabel:
    def test_dynamic_reports_the_node_budget(self):
        assert (
            batching_label(
                dynamic_batching=True, batch_size=32, max_total_nodes=5_000_000
            )
            == "dynamic_nodes=5000000"
        )

    def test_fixed_reports_the_graph_count(self):
        assert (
            batching_label(
                dynamic_batching=False, batch_size=32, max_total_nodes=5_000_000
            )
            == "fixed_graphs=32"
        )

    def test_label_ignores_the_setting_that_is_not_in_force(self):
        """Under dynamic batching the node budget alone packs batches, so
        batch_size must not leak into the label (and vice versa) — otherwise
        two runs that batched identically would compare as different."""
        assert batching_label(
            dynamic_batching=True, batch_size=8, max_total_nodes=5_000_000
        ) == batching_label(
            dynamic_batching=True, batch_size=512, max_total_nodes=5_000_000
        )
        assert batching_label(
            dynamic_batching=False, batch_size=32, max_total_nodes=1
        ) == batching_label(
            dynamic_batching=False, batch_size=32, max_total_nodes=9_000_000
        )


class TestResolveCheckpointPath:
    """Checkpoint filenames really look like 'epoch=03-val_loss=val_loss=0.0024.ckpt'
    — the ModelCheckpoint template double-prints the 'val_loss=' prefix."""

    @staticmethod
    def _make_run_dir(tmp_path, filenames):
        run_dir = tmp_path / "Orchestrate_pagerank"
        run_dir.mkdir()
        for name in filenames:
            (run_dir / name).write_text("ckpt")
        return str(tmp_path), "Orchestrate_pagerank"

    def test_picks_lowest_val_loss_across_reruns(self, tmp_path):
        # A dir holding two runs' top-3 checkpoints plus both last files.
        root, label = self._make_run_dir(
            tmp_path,
            [
                "epoch=01-val_loss=val_loss=0.0023.ckpt",
                "epoch=02-val_loss=val_loss=0.0044.ckpt",
                "last.ckpt",
                "epoch=06-val_loss=val_loss=0.0016.ckpt",
                "epoch=07-val_loss=val_loss=0.0017.ckpt",
                "last-v1.ckpt",
            ],
        )
        assert (
            resolve_checkpoint_path(root, label, "best").name
            == "epoch=06-val_loss=val_loss=0.0016.ckpt"
        )

    def test_lowest_need_not_be_the_newest_run(self, tmp_path):
        # The completed run's best (0.0034) predates later, worse checkpoints.
        root, label = self._make_run_dir(
            tmp_path,
            [
                "epoch=02-val_loss=val_loss=0.0034.ckpt",
                "epoch=05-val_loss=val_loss=0.0041.ckpt",
                "epoch=06-val_loss=val_loss=0.0041.ckpt",
                "last.ckpt",
            ],
        )
        assert (
            resolve_checkpoint_path(root, label, "best").name
            == "epoch=02-val_loss=val_loss=0.0034.ckpt"
        )

    def test_includes_lightning_dedup_suffix(self, tmp_path):
        root, label = self._make_run_dir(
            tmp_path,
            [
                "epoch=03-val_loss=val_loss=0.0024.ckpt",
                "epoch=03-val_loss=val_loss=0.0011-v1.ckpt",
            ],
        )
        assert (
            resolve_checkpoint_path(root, label, "best").name
            == "epoch=03-val_loss=val_loss=0.0011-v1.ckpt"
        )

    def test_tie_at_minimum_is_deterministic_and_warns(self, tmp_path, capsys):
        root, label = self._make_run_dir(
            tmp_path,
            [
                "epoch=08-val_loss=val_loss=0.0024.ckpt",
                "epoch=05-val_loss=val_loss=0.0024.ckpt",
            ],
        )
        chosen = resolve_checkpoint_path(root, label, "best")
        assert chosen.name == "epoch=05-val_loss=val_loss=0.0024.ckpt"
        assert "tie at val_loss=0.0024" in capsys.readouterr().out

    def test_literal_filename_bypasses_selection(self, tmp_path):
        root, label = self._make_run_dir(
            tmp_path, ["last.ckpt", "epoch=01-val_loss=val_loss=0.0023.ckpt"]
        )
        assert resolve_checkpoint_path(root, label, "last.ckpt").name == "last.ckpt"

    def test_missing_literal_filename_raises(self, tmp_path):
        root, label = self._make_run_dir(tmp_path, ["last.ckpt"])
        with pytest.raises(FileNotFoundError):
            resolve_checkpoint_path(root, label, "absent.ckpt")

    def test_only_last_ckpt_raises(self, tmp_path):
        root, label = self._make_run_dir(tmp_path, ["last.ckpt", "last-v1.ckpt"])
        with pytest.raises(FileNotFoundError):
            resolve_checkpoint_path(root, label, "best")

    def test_missing_run_dir_names_the_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="directory not found"):
            resolve_checkpoint_path(str(tmp_path), "Orchestrate_absent", "best")


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


class TestBuildEvalPasses:
    """The val sanity pass is appended automatically (test.sh's default), but
    must never appear alongside --skip_val or the manual --split val
    diagnostic (which already covers val via the primary pair)."""

    def test_baseline_gets_full_graph_and_val_passes(self):
        red_kwargs = resolve_reduction_kwargs("none", None)
        passes = build_eval_passes("test", "none", red_kwargs, skip_val=False)
        assert passes == [
            ("test", "full_graph", {"sparsification": None, "partition": None}),
            ("val", "full_graph", {"sparsification": None, "partition": None}),
        ]

    def test_reduction_gets_three_passes_val_matches_the_checkpoints_own_reduction(self):
        red_kwargs = resolve_reduction_kwargs("partition", "metis")
        passes = build_eval_passes("test", "partition", red_kwargs, skip_val=False)
        assert passes == [
            ("test", "full_graph", {"sparsification": None, "partition": None}),
            ("test", "matched_reduction", {"sparsification": None, "partition": "metis"}),
            ("val", "matched_reduction", {"sparsification": None, "partition": "metis"}),
        ]

    def test_skip_val_drops_the_validation_pass(self):
        red_kwargs = resolve_reduction_kwargs("sparsification", "pagerank")
        passes = build_eval_passes("test", "sparsification", red_kwargs, skip_val=True)
        assert [p[0] for p in passes] == ["test", "test"]
        assert all(split != "val" for split, _, _ in passes)

    def test_manual_split_val_does_not_duplicate_the_automatic_pass(self):
        """The pre-existing manual diagnostic (EXTRA_ARGS='--split val') must
        keep running exactly its own full_graph/matched_reduction pair against
        val, not that pair *plus* the automatic val pass on top."""
        red_kwargs = resolve_reduction_kwargs("partition", "metis")
        passes = build_eval_passes("val", "partition", red_kwargs, skip_val=False)
        assert passes == [
            ("val", "full_graph", {"sparsification": None, "partition": None}),
            ("val", "matched_reduction", {"sparsification": None, "partition": "metis"}),
        ]

    def test_manual_split_val_baseline_is_a_single_pass(self):
        red_kwargs = resolve_reduction_kwargs("none", None)
        passes = build_eval_passes("val", "none", red_kwargs, skip_val=False)
        assert passes == [
            ("val", "full_graph", {"sparsification": None, "partition": None}),
        ]

    def test_skip_full_graph_drops_only_the_test_side_full_graph_pass(self):
        """Targeted resubmission after fixing an unrelated failure (e.g. a
        missing mask) whose full_graph pass already succeeded: rerun only
        matched_reduction plus the still-pending val pass, not full_graph
        again."""
        red_kwargs = resolve_reduction_kwargs("sparsification", "random_edge_dropout")
        passes = build_eval_passes(
            "test", "sparsification", red_kwargs, skip_val=False, skip_full_graph=True
        )
        assert passes == [
            (
                "test",
                "matched_reduction",
                {"sparsification": "random_edge_dropout", "partition": None},
            ),
            (
                "val",
                "matched_reduction",
                {"sparsification": "random_edge_dropout", "partition": None},
            ),
        ]

    def test_skip_full_graph_and_skip_val_leaves_only_matched_reduction(self):
        red_kwargs = resolve_reduction_kwargs("partition", "metis")
        passes = build_eval_passes(
            "test", "partition", red_kwargs, skip_val=True, skip_full_graph=True
        )
        assert passes == [
            ("test", "matched_reduction", {"sparsification": None, "partition": "metis"}),
        ]

    def test_skip_full_graph_on_baseline_leaves_only_the_val_pass(self):
        """Baseline has no matched_reduction pass, so skipping full_graph
        drops its only test-side pass entirely — an intentional, if unusual,
        consequence of asking for "reduced only" on a config with nothing
        reduced."""
        red_kwargs = resolve_reduction_kwargs("none", None)
        passes = build_eval_passes(
            "test", "none", red_kwargs, skip_val=False, skip_full_graph=True
        )
        assert passes == [
            ("val", "full_graph", {"sparsification": None, "partition": None}),
        ]


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
    def test_write_single_row_csv_header_and_row(self, tmp_path):
        csv_path = tmp_path / "sub" / "results.csv"  # parent auto-created
        write_single_row_csv(csv_path, {"a": 1, "b": "x"})

        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        assert rows == [["a", "b"], ["1", "x"]]

    def test_write_single_row_csv_overwrites(self, tmp_path):
        """Re-running a config must cleanly replace its own file, not append a
        duplicate row (the whole point of per-config output files)."""
        csv_path = tmp_path / "results.csv"
        write_single_row_csv(csv_path, {"a": 1, "b": "x"})
        write_single_row_csv(csv_path, {"a": 2, "b": "y"})

        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        assert rows == [["a", "b"], ["2", "y"]]

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


# ---------------------------------------------------------------------------
# Forward-sweep (run_eval_pass) harness
#
# run_eval_pass builds its own AIGDataModule from dm_kwargs, so the stub below
# is monkeypatched over test.AIGDataModule. It mirrors the two batching modes
# AIGDataModule.test_dataloader actually supports (fixed graph count vs
# node-budget dynamic batching) so the ordering/timing logic is exercised for
# real rather than mocked away.
# ---------------------------------------------------------------------------


def _target_for(num_nodes: int) -> float:
    """Target that is a deterministic but *non-monotonic* function of node
    count, so it differs from _CountingModel's prediction. Equal preds and
    targets would drive every metric to its perfect value and make the
    batching-invariance test below pass vacuously."""
    return ((num_nodes * 7) % 100) / 100.0


def _make_graph(num_nodes: int, num_edges: int) -> Data:
    """Graph whose node count uniquely identifies it, so predictions derived
    from node count can be traced back to the right graph_id."""
    src = torch.arange(num_edges, dtype=torch.long) % num_nodes
    dst = (src + 1) % num_nodes
    return Data(
        x=torch.zeros(num_nodes, 4),
        edge_index=torch.stack([src, dst]),
        y=torch.tensor([[_target_for(num_nodes)]], dtype=torch.float32),
    )


class _FakeSample:
    def __init__(self, graph_path: str) -> None:
        self.graph_path = graph_path


class _CountingModel(torch.nn.Module):
    """Predicts num_nodes/1000 per graph — a deterministic, order-revealing
    stand-in for the real regressor."""

    def forward(self, batch):
        counts = torch.bincount(batch.batch, minlength=batch.num_graphs).float()
        return (counts / 1000.0).unsqueeze(-1)


def _install_fake_datamodule(monkeypatch, graphs, captured: dict):
    class _FakeDataModule:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.batch_size = kwargs.get("batch_size", 32)
            self.dynamic_batching = kwargs.get("dynamic_batching", False)
            self.max_total_nodes = kwargs.get("max_total_nodes", 3_000_000)
            self.test_ds = type(
                "_FakeTestDS",
                (),
                {"samples": [_FakeSample(f"/graphs/g{i}.pt") for i in range(len(graphs))]},
            )()

        def setup(self, stage):
            assert stage == "test"

        def test_dataloader(self):
            if self.dynamic_batching:
                plan = BalancedDynamicBatchSampler.build_batch_plan(
                    [g.num_nodes for g in graphs],
                    max_total_nodes=self.max_total_nodes,
                )
                sampler = BalancedDynamicBatchSampler(
                    batch_size=self.batch_size,
                    shuffle=False,
                    seed=42,
                    max_total_nodes=self.max_total_nodes,
                    precomputed_batches=plan,
                )
                return DataLoader(
                    graphs,
                    batch_sampler=sampler,
                    collate_fn=PyGBatch.from_data_list,
                )
            return DataLoader(
                graphs,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=PyGBatch.from_data_list,
            )

    monkeypatch.setattr(test_module, "AIGDataModule", _FakeDataModule)


# Deliberately NOT in ascending node order: dataset order and size-sorted
# order must differ, or the reordering bug this guards against is invisible.
_GRAPH_SIZES = [50, 10, 90, 30, 70, 20]


def _sweep(monkeypatch, *, dynamic: bool, max_total_nodes: int = 100, sizes=None):
    sizes = list(_GRAPH_SIZES if sizes is None else sizes)
    graphs = [_make_graph(n, n) for n in sizes]
    captured: dict = {}
    _install_fake_datamodule(monkeypatch, graphs, captured)
    dm_kwargs = {
        "batch_size": 2,
        "dynamic_batching": dynamic,
        "max_total_nodes": max_total_nodes,
    }
    metrics, per_graph = run_eval_pass(
        _CountingModel(),
        dm_kwargs,
        device=torch.device("cpu"),
    )
    return metrics, per_graph, captured


class TestEmittedSampleOrder:
    def test_none_for_plain_sequential_loader(self):
        """A stock DataLoader still exposes a .batch_sampler, so the check has
        to key on the sampler *type*, not on the attribute existing."""
        loader = DataLoader([_make_graph(4, 4)], batch_size=1)
        assert emitted_sample_order(loader) is None

    def test_dynamic_plan_covers_every_index_exactly_once(self):
        sizes = [50, 10, 90, 30]
        plan = BalancedDynamicBatchSampler.build_batch_plan(sizes, max_total_nodes=100)
        sampler = BalancedDynamicBatchSampler(
            batch_size=2, shuffle=False, seed=42, precomputed_batches=plan
        )
        loader = DataLoader([_make_graph(n, n) for n in sizes], batch_sampler=sampler)

        order = emitted_sample_order(loader)
        assert sorted(order) == list(range(len(sizes)))

    def test_dynamic_order_differs_from_dataset_order(self):
        """Guards the premise of the alignment fix: if the plan happened to be
        emitted in dataset order the reordering test below would pass for the
        wrong reason."""
        plan = BalancedDynamicBatchSampler.build_batch_plan(
            _GRAPH_SIZES, max_total_nodes=100
        )
        sampler = BalancedDynamicBatchSampler(
            batch_size=2, shuffle=False, seed=42, precomputed_batches=plan
        )
        loader = DataLoader(
            [_make_graph(n, n) for n in _GRAPH_SIZES], batch_sampler=sampler
        )
        assert emitted_sample_order(loader) != list(range(len(_GRAPH_SIZES)))


class TestRunEvalPassAlignment:
    def test_sequential_rows_stay_in_dataset_order(self, monkeypatch):
        _, per_graph, _ = _sweep(monkeypatch, dynamic=False)

        assert per_graph["graph_id"] == [f"/graphs/g{i}.pt" for i in range(len(_GRAPH_SIZES))]
        assert per_graph["num_nodes"] == _GRAPH_SIZES

    def test_dynamic_batching_keeps_graph_id_aligned_with_its_prediction(
        self, monkeypatch
    ):
        """Regression test: the batch plan sorts by node count, so the loader
        emits samples out of dataset order. graph_id is read from
        dataset.samples while predictions come back in emission order — zipping
        them naively mislabels every row of the predictions CSV."""
        _, per_graph, _ = _sweep(monkeypatch, dynamic=True)

        size_of = {f"/graphs/g{i}.pt": n for i, n in enumerate(_GRAPH_SIZES)}
        for gid, n_nodes, pred, target in zip(
            per_graph["graph_id"],
            per_graph["num_nodes"],
            per_graph["prediction"],
            per_graph["target"],
        ):
            assert size_of[gid] == n_nodes
            assert pred == pytest.approx(n_nodes / 1000.0)
            assert target == pytest.approx(_target_for(n_nodes))

    def test_dynamic_and_sequential_cover_the_same_graphs(self, monkeypatch):
        _, seq, _ = _sweep(monkeypatch, dynamic=False)
        _, dyn, _ = _sweep(monkeypatch, dynamic=True)

        assert sorted(dyn["graph_id"]) == sorted(seq["graph_id"])
        assert sorted(dyn["num_nodes"]) == sorted(seq["num_nodes"])


class TestRunEvalPassBatchingInvariance:
    def test_accuracy_metrics_are_invariant_to_batching(self, monkeypatch):
        """The property that makes dynamic batching safe for the thesis
        numbers: metrics are computed over the full concatenated prediction
        tensor, the model's norm layers are per-node, and pooling is per-graph
        — so batch composition cannot move RMSE/R2/Spearman."""
        seq_metrics, _, _ = _sweep(monkeypatch, dynamic=False)
        dyn_metrics, _, _ = _sweep(monkeypatch, dynamic=True)

        assert dyn_metrics["num_graphs"] == seq_metrics["num_graphs"]
        for key in ("smooth_l1", "rmse", "r2", "spearman"):
            assert dyn_metrics[key] == pytest.approx(seq_metrics[key], abs=1e-6)

    def test_node_budget_controls_packing(self, monkeypatch):
        """A budget below the largest graph still yields batches (one graph
        each); a budget above the whole split collapses to a single batch."""
        _, tight, _ = _sweep(monkeypatch, dynamic=True, max_total_nodes=1)
        _, loose, _ = _sweep(monkeypatch, dynamic=True, max_total_nodes=10_000)

        assert sorted(tight["num_nodes"]) == sorted(_GRAPH_SIZES)
        assert sorted(loose["num_nodes"]) == sorted(_GRAPH_SIZES)

    def test_batching_kwargs_reach_the_datamodule(self, monkeypatch):
        _, _, captured = _sweep(monkeypatch, dynamic=True, max_total_nodes=12345)

        assert captured["dynamic_batching"] is True
        assert captured["max_total_nodes"] == 12345


class TestRunEvalPassTiming:
    def test_first_batch_is_excluded_from_timing(self, monkeypatch):
        """Batch 0 absorbs CUDA context init, cuDNN autotune and worker spawn,
        so it is deliberately not charged to steady-state throughput."""
        metrics, _, _ = _sweep(monkeypatch, dynamic=False)

        # batch_size=2 over 6 graphs -> 3 batches; the first 2 graphs are warmup.
        assert metrics["num_graphs"] == 6
        assert metrics["num_timed_graphs"] == 4

    def test_single_batch_reports_nan_throughput_instead_of_dividing_by_zero(
        self, monkeypatch
    ):
        """With one batch there is no steady-state region at all. Accuracy is
        still reported; only the timing columns go NaN."""
        metrics, _, _ = _sweep(monkeypatch, dynamic=False, sizes=[40, 60])

        assert metrics["num_graphs"] == 2
        assert metrics["num_timed_graphs"] == 0
        assert math.isnan(metrics["throughput_graphs_per_s"])
        # total_time_s must be NaN too, not a microsecond-scale number that
        # measures nothing but reads like a real duration in the CSV.
        assert math.isnan(metrics["total_time_s"])
        assert not math.isnan(metrics["rmse"])

    def test_hardware_columns_are_populated_on_cpu(self, monkeypatch):
        """peak VRAM / GPU util are NaN off-GPU, but the host-memory columns
        must still be real numbers on the CPU pass (test_cpu.sh depends on
        them)."""
        metrics, _, _ = _sweep(monkeypatch, dynamic=False)

        assert metrics["peak_process_rss_mb"] > 0
        assert not math.isnan(metrics["avg_system_memory_pct"])
        assert math.isnan(metrics["peak_vram_mb"])
        assert metrics["avg_nodes_per_graph"] == pytest.approx(
            sum(_GRAPH_SIZES) / len(_GRAPH_SIZES)
        )


class TestWritePredictionsNoCap:
    """max_rows=0 (the default) must write every graph. A capped file samples
    by row position, and a reduced config's batch plan orders graphs
    differently, so capped files from two configs cover different graphs and
    cannot be paired design-by-design."""

    @staticmethod
    def _per_graph(n):
        return {
            "graph_id": [f"g{i}" for i in range(n)],
            "num_nodes": list(range(n)),
            "num_edges": list(range(n)),
            "target": [0.5] * n,
            "prediction": [0.5] * n,
        }

    def test_zero_writes_every_row(self, tmp_path):
        path = tmp_path / "preds.csv"
        write_predictions_csv(path, self._per_graph(500), max_rows=0)

        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert [r["graph_id"] for r in rows] == [f"g{i}" for i in range(500)]

    def test_negative_also_means_no_cap(self, tmp_path):
        path = tmp_path / "preds.csv"
        write_predictions_csv(path, self._per_graph(50), max_rows=-1)

        with open(path, newline="") as f:
            assert len(list(csv.DictReader(f))) == 50

    def test_cap_equal_to_size_keeps_dataset_order(self, tmp_path):
        """max_rows == n must not trigger sampling — sampling would reorder
        nothing but still exercise the random path for no reason."""
        path = tmp_path / "preds.csv"
        write_predictions_csv(path, self._per_graph(30), max_rows=30)

        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert [r["graph_id"] for r in rows] == [f"g{i}" for i in range(30)]
