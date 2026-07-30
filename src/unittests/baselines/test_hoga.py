from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch_geometric.data import Batch, Data

from baselines.common.lightning_wrapper import BaselineRegressionLightningModule
from baselines.hoga.hop_features import (
    HopFeatureCache,
    collate_hoga_batch,
    compute_hop_features,
    num_hop_slots,
)
from baselines.hoga.regressor import HOGAGraphRegressor


class TestNumHopSlots(unittest.TestCase):
    def test_directed_and_undirected_widths(self):
        self.assertEqual(num_hop_slots(4, directed=True), 9)
        self.assertEqual(num_hop_slots(4, directed=False), 5)
        self.assertEqual(num_hop_slots(1, directed=True), 3)


class TestComputeHopFeatures(unittest.TestCase):
    def test_directed_propagation_matches_hand_computed_path_graph(self):
        # Path graph 0 -> 1 -> 2, feat_dim=1, num_hops=1.
        # Forward (fan-out) propagation and reverse (fan-in) propagation must
        # differ -- this guards against the upstream copy-paste bug (see
        # baselines/hoga/hop_features.py module docstring) where both
        # directions silently reused the forward adjacency.
        x = torch.tensor([[1.0], [2.0], [3.0]])
        edge_index = torch.tensor([[0, 1], [1, 2]])

        out = compute_hop_features(x, edge_index, num_nodes=3, num_hops=1, directed=True)

        self.assertEqual(out.shape, (3, 3, 1))
        expected = torch.tensor(
            [
                [[1.0], [2.0], [0.0]],  # node 0: self, fwd-hop, rev-hop
                [[2.0], [3.0], [1.0]],  # node 1
                [[3.0], [0.0], [2.0]],  # node 2
            ]
        )
        self.assertTrue(torch.allclose(out, expected))

    def test_undirected_width_and_no_crash(self):
        x = torch.rand(5, 3)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        out = compute_hop_features(x, edge_index, num_nodes=5, num_hops=2, directed=False)
        self.assertEqual(out.shape, (5, 3, 3))

    def test_single_node_graph_does_not_crash(self):
        # Regression test: a single-node graph's degree-sum has shape (1, 1);
        # _graph2adj used to flatten it with .squeeze() (collapsing to a 0-d
        # array) instead of .reshape(-1), which crashed the boolean-mask
        # assignment inside _graph2adj with a TypeError.
        x = torch.tensor([[1.0, 2.0]])
        edge_index = torch.empty((2, 0), dtype=torch.long)
        out = compute_hop_features(x, edge_index, num_nodes=1, num_hops=2, directed=True)
        self.assertEqual(out.shape, (1, 5, 2))
        self.assertFalse(torch.isnan(out).any())

    def test_isolated_node_gets_zero_propagated_features(self):
        # Node 2 has no edges at all; its hop features should be all zero
        # (not NaN/inf) thanks to the degree-normalization's isinf/isnan guard.
        x = torch.tensor([[1.0], [2.0], [5.0]])
        edge_index = torch.tensor([[0], [1]])
        out = compute_hop_features(x, edge_index, num_nodes=3, num_hops=1, directed=True)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())
        self.assertTrue(torch.allclose(out[2, 1:, :], torch.zeros(2, 1)))


class _FakeSample:
    def __init__(self, graph_path: str) -> None:
        self.graph_path = graph_path


class _FakeBaseDataset:
    """Minimal stand-in for AIGGraphRegressionDataset's public surface that
    HopFeatureCache relies on: `.samples[idx].graph_path` and `[idx]`, plus
    the `get_num_nodes_list()`/`release_runtime_caches()` pair that
    train_baseline._hoga_loader uses to build its node-budget batch plan."""

    def __init__(self, data_list: list[Data]) -> None:
        self._data_list = data_list
        self.samples = [_FakeSample(f"design_{i}.pt") for i in range(len(data_list))]

    def __len__(self) -> int:
        return len(self._data_list)

    def __getitem__(self, idx: int) -> Data:
        d = self._data_list[idx]
        return Data(x=d.x.clone(), edge_index=d.edge_index.clone(), y=d.y.clone())

    def get_num_nodes_list(self) -> list[int]:
        return [int(d.x.shape[0]) for d in self._data_list]

    def release_runtime_caches(self) -> None:
        return None


def _make_graph_with_nodes(num_nodes: int, seed: int) -> Data:
    """Path graph on `num_nodes` nodes, so node counts differ per sample."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(num_nodes, 4, generator=g)
    src = torch.arange(num_nodes - 1)
    edge_index = torch.stack([src, src + 1])
    return Data(x=x, edge_index=edge_index, y=torch.rand(1, 1, generator=g))


def _make_small_graph(seed: int) -> Data:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(4, 4, generator=g)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    return Data(x=x, edge_index=edge_index, y=torch.rand(1, 1, generator=g))


class TestHopFeatureCache(unittest.TestCase):
    def test_cache_file_written_and_reloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = _FakeBaseDataset([_make_small_graph(0), _make_small_graph(1)])
            cache = HopFeatureCache(base, num_hops=1, cache_dir=tmp, directed=True)

            cache_path = cache._cache_path(0)
            self.assertFalse(cache_path.exists())

            data0 = cache[0]
            self.assertTrue(cache_path.exists())
            self.assertTrue(hasattr(data0, "hoga_x"))
            self.assertEqual(data0.hoga_x.shape, (4, 3, 4))

            # Second access must hit the cache and return identical values.
            data0_again = cache[0]
            self.assertTrue(torch.allclose(data0.hoga_x, data0_again.hoga_x))

    def test_cache_key_changes_when_underlying_file_is_regenerated(self):
        # Regression test: the cache used to key purely on the graph_path
        # string, so a graph regenerated at the same path (same filename,
        # different content/mtime -- e.g. a re-run of the data-creation
        # pipeline) would silently reuse stale cached hop features forever.
        with tempfile.TemporaryDirectory() as tmp:
            real_graph_path = Path(tmp) / "design_0.pt"
            real_graph_path.write_bytes(b"x" * 10)

            base = _FakeBaseDataset([_make_small_graph(0)])
            base.samples[0].graph_path = str(real_graph_path)
            cache = HopFeatureCache(base, num_hops=1, cache_dir=tmp, directed=True)
            path_before = cache._cache_path(0)

            # Simulate regeneration: same path, different size and a later mtime.
            real_graph_path.write_bytes(b"y" * 99)
            os_stat = real_graph_path.stat()
            os.utime(real_graph_path, (os_stat.st_atime, os_stat.st_mtime + 10))

            path_after = cache._cache_path(0)
            self.assertNotEqual(path_before, path_after)

    def test_different_num_hops_do_not_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = _FakeBaseDataset([_make_small_graph(0)])
            cache_h1 = HopFeatureCache(base, num_hops=1, cache_dir=tmp, directed=True)
            cache_h2 = HopFeatureCache(base, num_hops=2, cache_dir=tmp, directed=True)

            self.assertNotEqual(cache_h1._cache_path(0), cache_h2._cache_path(0))
            self.assertEqual(cache_h1[0].hoga_x.shape[1], 3)
            self.assertEqual(cache_h2[0].hoga_x.shape[1], 5)

    def test_precompute_all_populates_every_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = _FakeBaseDataset([_make_small_graph(i) for i in range(3)])
            cache = HopFeatureCache(base, num_hops=1, cache_dir=tmp, directed=True)
            cache.precompute_all(log_every=0)
            for idx in range(3):
                self.assertTrue(cache._cache_path(idx).exists())

    def test_cache_dir_none_computes_without_touching_disk(self):
        # The on-disk hop cache is not size-viable at full dataset scale
        # (~5.7 TB / ~788k files for train+val), so cache_dir=None is the
        # production path: compute in-process, write nothing.
        base = _FakeBaseDataset([_make_small_graph(0), _make_small_graph(1)])
        cache = HopFeatureCache(base, num_hops=1, cache_dir=None, directed=True)

        self.assertIsNone(cache.cache_dir)
        data0 = cache[0]
        self.assertTrue(hasattr(data0, "hoga_x"))
        self.assertEqual(data0.hoga_x.shape, (4, 3, 4))

        # No cache path is even derivable, so nothing can be written: this is
        # what stops the disk path being reached, so assert it directly rather
        # than checking that some unrelated temp dir stayed empty.
        with self.assertRaises(TypeError):
            cache._cache_path(0)

        # Recomputation must be deterministic, since every access recomputes.
        self.assertTrue(torch.allclose(data0.hoga_x, cache[0].hoga_x))

    def test_cache_dir_none_matches_cached_values(self):
        # The two paths must be numerically identical, so a run with the disk
        # cache disabled trains on exactly the same features as one with it.
        with tempfile.TemporaryDirectory() as tmp:
            graphs = [_make_small_graph(i) for i in range(3)]
            cached = HopFeatureCache(
                _FakeBaseDataset(list(graphs)), num_hops=2, cache_dir=tmp, directed=True
            )
            uncached = HopFeatureCache(
                _FakeBaseDataset(list(graphs)), num_hops=2, cache_dir=None, directed=True
            )
            for idx in range(3):
                self.assertTrue(
                    torch.allclose(cached[idx].hoga_x, uncached[idx].hoga_x),
                    msg=f"hop features diverged at idx={idx}",
                )

    def test_precompute_all_raises_without_cache_dir(self):
        base = _FakeBaseDataset([_make_small_graph(0)])
        cache = HopFeatureCache(base, num_hops=1, cache_dir=None, directed=True)
        with self.assertRaises(ValueError):
            cache.precompute_all(log_every=0)

    def test_index_alignment_with_base_dataset(self):
        # train_baseline._hoga_loader builds a node-budget batch plan from the
        # BASE dataset's get_num_nodes_list() and then feeds those indices to
        # the HopFeatureCache wrapper. That is only sound if entry i of the
        # node-count list describes the same sample the wrapper returns at
        # index i -- otherwise the budget is enforced against the wrong sizes.
        # Deliberately varied node counts, so a mis-ordering actually fails.
        sizes = [3, 9, 5, 12, 7]
        base = _FakeBaseDataset([_make_graph_with_nodes(n, seed=n) for n in sizes])
        cache = HopFeatureCache(base, num_hops=1, cache_dir=None, directed=True)

        self.assertEqual(len(cache), len(base))
        self.assertEqual(base.get_num_nodes_list(), sizes)
        for idx, expected_nodes in enumerate(sizes):
            wrapped = cache[idx]
            self.assertEqual(int(wrapped.x.shape[0]), expected_nodes)
            self.assertEqual(int(wrapped.hoga_x.shape[0]), expected_nodes)
            self.assertTrue(torch.allclose(wrapped.x, base[idx].x))

    def test_collate_hoga_batch_builds_pyg_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = _FakeBaseDataset([_make_small_graph(0), _make_small_graph(1)])
            cache = HopFeatureCache(base, num_hops=1, cache_dir=tmp, directed=True)
            batch = collate_hoga_batch([cache[0], cache[1]])
            self.assertEqual(int(batch.num_graphs), 2)
            self.assertTrue(hasattr(batch, "hoga_x"))
            self.assertEqual(batch.hoga_x.shape[0], batch.x.shape[0])


def _make_batch_with_hoga_x(num_hops: int, directed: bool, seeds: list[int]) -> Batch:
    data_list = []
    for seed in seeds:
        data = _make_small_graph(seed)
        data.hoga_x = compute_hop_features(
            data.x, data.edge_index, data.num_nodes, num_hops, directed
        )
        data_list.append(data)
    return Batch.from_data_list(data_list)


class TestHOGAGraphRegressor(unittest.TestCase):
    def test_forward_pass_shape_and_range(self):
        model = HOGAGraphRegressor(
            in_channels=4, hidden_channels=8, num_layers=1, dropout=0.0,
            num_hops=num_hop_slots(1, directed=True), heads=2,
        )
        model.eval()
        batch = _make_batch_with_hoga_x(num_hops=1, directed=True, seeds=[0])
        out = model(batch)
        self.assertEqual(out.shape, (1, 1))
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0))

    def test_batch_independence(self):
        model = HOGAGraphRegressor(
            in_channels=4, hidden_channels=8, num_layers=1, dropout=0.0,
            num_hops=num_hop_slots(1, directed=True), heads=2,
        )
        model.eval()
        batch1 = _make_batch_with_hoga_x(num_hops=1, directed=True, seeds=[1])
        batch2 = _make_batch_with_hoga_x(num_hops=1, directed=True, seeds=[2])
        combined = _make_batch_with_hoga_x(num_hops=1, directed=True, seeds=[1, 2])

        out1_alone = model(batch1)
        out2_alone = model(batch2)
        out_combined = model(combined)

        self.assertTrue(torch.allclose(out1_alone[0], out_combined[0], atol=1e-4))
        self.assertTrue(torch.allclose(out2_alone[0], out_combined[1], atol=1e-4))

    def test_gradient_flows(self):
        model = HOGAGraphRegressor(
            in_channels=4, hidden_channels=8, num_layers=1, dropout=0.0,
            num_hops=num_hop_slots(1, directed=True), heads=2,
        )
        batch = _make_batch_with_hoga_x(num_hops=1, directed=True, seeds=[0, 1])
        out = model(batch)
        out.mean().backward()
        # lins[1] and lins[2] are allocated unconditionally in vendored
        # HOGA.__init__ (model.py) but never referenced in forward() -- only
        # lins[0] is ever called, regardless of num_layers. This is a
        # pre-existing quirk of the unmodified upstream model, not something
        # introduced by this project's adaptation, so those two are excluded
        # here rather than "fixed" in the vendored file.
        always_unused = {"lins.1.weight", "lins.2.weight"}
        for name, p in model.named_parameters():
            if p.requires_grad and name not in always_unused:
                self.assertIsNotNone(p.grad, f"Broken graph at {name}")


class TestHOGALightningTraining(unittest.TestCase):
    def setUp(self):
        self.num_hops = 1
        self.directed = True
        slots = num_hop_slots(self.num_hops, directed=self.directed)
        self.dataset = [
            _make_batch_with_hoga_x(self.num_hops, self.directed, [i]).to_data_list()[0]
            for i in range(10)
        ]
        self.slots = slots

    def test_training_and_testing_loop(self):
        base_model = HOGAGraphRegressor(
            in_channels=4,
            hidden_channels=8,
            num_layers=1,
            dropout=0.0,
            num_hops=self.slots,
            heads=2,
        )
        model = BaselineRegressionLightningModule(
            base_model, lr=1e-3, loss_fn=torch.nn.MSELoss()
        )

        from torch.utils.data import DataLoader as TorchDataLoader

        train_loader = TorchDataLoader(
            self.dataset[:6], batch_size=2, collate_fn=collate_hoga_batch
        )
        val_loader = TorchDataLoader(
            self.dataset[6:8], batch_size=2, collate_fn=collate_hoga_batch
        )
        test_loader = TorchDataLoader(
            self.dataset[8:], batch_size=2, collate_fn=collate_hoga_batch
        )

        trainer = pl.Trainer(
            fast_dev_run=True,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    unittest.main()
