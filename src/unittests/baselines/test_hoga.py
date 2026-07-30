from __future__ import annotations

import math
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
from baselines.hoga.model import MultiheadAttention, MultiheadAttentionMix
from baselines.hoga.regressor import HOGAGraphRegressor


class TestMultiheadAttention(unittest.TestCase):
    """Covers the one deliberate change to the vendored model.py.

    Upstream's `MultiheadAttention` softmaxed over the head axis rather than
    the key axis (see model.py's module docstring). These tests pin the fixed
    behaviour to the paper's Equation (5) and to `MultiheadAttentionMix`, the
    sibling class upstream's own run.sh actually instantiates.
    """

    def test_output_is_convex_combination_of_values(self):
        """The property that distinguishes key-softmax from head-softmax.

        Normalizing over keys makes every output row a convex combination of
        the value rows, so it must lie within their elementwise min/max. The
        upstream head-axis softmax gives no such guarantee. Checked with V and
        the output projection pinned to identity so the attention weights are
        the only thing acting on the values.
        """
        torch.manual_seed(0)
        dim, seq = 8, 5
        attn = MultiheadAttention(dim, num_heads=1)
        attn.eval()
        with torch.no_grad():
            attn.value_projection.weight.copy_(torch.eye(dim))
            attn.value_projection.bias.zero_()
            attn.output_projection.weight.copy_(torch.eye(dim))
            attn.output_projection.bias.zero_()

        x = torch.randn(3, seq, dim)
        with torch.no_grad():
            out, _ = attn(x, x, x)

        lo = x.min(dim=1, keepdim=True).values
        hi = x.max(dim=1, keepdim=True).values
        self.assertTrue(torch.all(out >= lo - 1e-5))
        self.assertTrue(torch.all(out <= hi + 1e-5))

    def test_matches_explicit_key_softmax_reference(self):
        torch.manual_seed(1)
        dim, heads, seq, batch = 16, 4, 6, 3
        attn = MultiheadAttention(dim, num_heads=heads)
        attn.eval()
        x = torch.randn(batch, seq, dim)

        def _split(t):
            return t.view(batch, seq, heads, dim // heads).transpose(1, 2)

        with torch.no_grad():
            q = _split(attn.query_projection(x))
            k = _split(attn.key_projection(x))
            v = _split(attn.value_projection(x))
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(dim // heads)
            probs = scores.softmax(dim=-1)  # over KEYS, per paper Eq. (5)
            ref = (probs @ v).transpose(1, 2).reshape(batch, seq, dim)
            ref = attn.output_projection(ref)
            out, _ = attn(x, x, x)

        self.assertTrue(torch.allclose(out, ref, atol=1e-5))

    def test_matches_torch_native_multihead_attention(self):
        """Independent oracle for the head split, at the production head count.

        `test_matches_explicit_key_softmax_reference` re-derives the same
        reshape it is checking, so it validates the softmax axis and the
        1/sqrt(head_dim) scale but is a tautology with respect to head layout.
        torch's own `nn.MultiheadAttention` is written by someone else, so
        agreeing with it (weights mapped across) pins the layout too. Run at
        heads=32, the value train_baseline_hoga.sh actually uses.
        """
        torch.manual_seed(4)
        dim, heads, seq, batch = 256, 32, 6, 3
        attn = MultiheadAttention(dim, num_heads=heads)
        attn.eval()

        ref = torch.nn.MultiheadAttention(
            dim, heads, dropout=0.0, bias=True, batch_first=True
        )
        ref.eval()
        with torch.no_grad():
            ref.in_proj_weight.copy_(
                torch.cat(
                    [
                        attn.query_projection.weight,
                        attn.key_projection.weight,
                        attn.value_projection.weight,
                    ]
                )
            )
            ref.in_proj_bias.copy_(
                torch.cat(
                    [
                        attn.query_projection.bias,
                        attn.key_projection.bias,
                        attn.value_projection.bias,
                    ]
                )
            )
            ref.out_proj.weight.copy_(attn.output_projection.weight)
            ref.out_proj.bias.copy_(attn.output_projection.bias)

            x = torch.randn(batch, seq, dim)
            ours, _ = attn(x, x, x)
            theirs, _ = ref(x, x, x, need_weights=False)

        self.assertTrue(torch.allclose(ours, theirs, atol=1e-4))

    def test_single_head_matches_upstream_mix_implementation(self):
        """Ties the fixed class to the one upstream actually ran -- at heads=1.

        `MultiheadAttentionMix` is what `main_gamora.py` builds
        (`attn_type="mix"`), and it softmaxes over keys. Scope is deliberately
        num_heads=1: Mix's own reshape,
        `view(batch_size * num_heads, -1, head_dim)`, splits the flattened
        (seq x feature) axis rather than the feature axis, so above one head
        its "heads" are chopped-up sequence rows and it is not a valid
        multi-head reference. At heads=1 that reshape is the identity, which
        makes this a clean check of the softmax axis and nothing more. Head
        layout is covered by test_matches_torch_native_multihead_attention.
        """
        torch.manual_seed(2)
        dim, seq, batch = 12, 4, 2
        fixed = MultiheadAttention(dim, num_heads=1)
        mix = MultiheadAttentionMix(dim, num_heads=1)
        mix.load_state_dict(fixed.state_dict())
        fixed.eval()
        mix.eval()

        x = torch.randn(batch, seq, dim)
        with torch.no_grad():
            out_fixed, _ = fixed(x, x, x)
            out_mix, _ = mix(x, x, x)

        self.assertTrue(torch.allclose(out_fixed, out_mix, atol=1e-5))

    def test_returns_none_for_probs(self):
        """Both call sites discard the second return value via `[0]`.

        Deliberately does NOT assert that no score tensor is materialized:
        whether SDPA builds one depends on the backend it dispatches to for a
        given shape and device, which a unit test should not pin.
        """
        attn = MultiheadAttention(8, num_heads=2)
        attn.eval()
        x = torch.randn(2, 3, 8)
        with torch.no_grad():
            out, probs = attn(x, x, x)
        self.assertIsNone(probs)
        self.assertEqual(out.shape, (2, 3, 8))

    def test_mask_zeros_are_excluded_from_attention(self):
        torch.manual_seed(3)
        dim, seq = 8, 4
        attn = MultiheadAttention(dim, num_heads=1)
        attn.eval()
        x = torch.randn(1, seq, dim)

        mask = torch.ones(1, seq, seq)
        mask[:, :, -1] = 0  # forbid attending to the last key
        with torch.no_grad():
            masked, _ = attn(x, x, x, mask=mask)
            # Changing the masked-out key must not change the output.
            x_perturbed = x.clone()
            x_perturbed[:, -1, :] += 10.0
            masked_perturbed, _ = attn(x_perturbed, x_perturbed, x_perturbed, mask=mask)

        self.assertTrue(
            torch.allclose(masked[:, :-1], masked_perturbed[:, :-1], atol=1e-5)
        )


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

    def test_forward_pass_undirected_slots(self):
        # train_baseline_hoga.sh now defaults HOGA_DIRECTED=false (paper
        # Section 3.1 uses a single normalized adjacency; upstream's
        # --directed defaults off and its run.sh never sets it), so the
        # undirected slot width needs the same end-to-end coverage as the
        # directed one.
        model = HOGAGraphRegressor(
            in_channels=4, hidden_channels=8, num_layers=1, dropout=0.0,
            num_hops=num_hop_slots(2, directed=False), heads=2,
        )
        model.eval()
        batch = _make_batch_with_hoga_x(num_hops=2, directed=False, seeds=[0, 1])
        out = model(batch)
        self.assertEqual(out.shape, (2, 1))
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


class _FakeSizedDataset:
    """Minimal stand-in for AIGGraphRegressionDataset for _hoga_loader."""

    def __init__(self, sizes: list[int]) -> None:
        self._sizes = sizes

    def __len__(self) -> int:
        return len(self._sizes)

    def __getitem__(self, idx: int) -> Data:
        n = self._sizes[idx]
        return Data(
            x=torch.ones(n, 4),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            y=torch.tensor([0.5]),
            num_nodes=n,
        )

    def get_num_nodes_list(self) -> list[int]:
        return list(self._sizes)

    def release_runtime_caches(self) -> None:
        pass


class TestHOGABatchPlanOrdering(unittest.TestCase):
    """The val split is capped by --limit_val_batches, which takes a PREFIX.

    build_batch_plan sorts by node count and anchors each batch on the largest
    remaining graph, backfilling with the smallest that fit, so the raw plan is
    ordered by descending anchor size. An unshuffled prefix is therefore "the
    biggest graphs plus the smallest" with nothing from the middle -- which
    would feed a bimodal, unrepresentative val_loss to ModelCheckpoint,
    PreciseEarlyStopping and ReduceLROnPlateau. _hoga_loader shuffles the plan
    once off a fixed seed to fix this.
    """

    def _args(self, seed: int = 42):
        from types import SimpleNamespace

        return SimpleNamespace(
            hoga_num_hops=1,
            hoga_hop_cache_dir=None,
            hoga_directed=False,
            hoga_max_nodes_per_batch=4000,
            batch_size=8,
            seed=seed,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2,
        )

    def test_raw_plan_prefix_is_size_biased(self):
        # Documents the hazard the shuffle exists to defeat. If this ever
        # stops holding, build_batch_plan changed and the fix may be moot.
        from data.sampler import BalancedDynamicBatchSampler

        # Fixture sized like production: every graph fits the budget, which
        # packs ~3-4 per batch (the real run averages 150k/40k ~= 4). A budget
        # small enough to make most graphs singletons would show a bias in the
        # opposite direction and would not represent the actual failure.
        sizes = list(range(1, 2001))
        plan = BalancedDynamicBatchSampler.build_batch_plan(
            sizes, max_total_nodes=4000
        )
        self.assertGreater(sum(len(b) for b in plan) / len(plan), 3.0)

        prefix_sizes = [sizes[i] for b in plan[: len(plan) // 4] for i in b]
        population_median = sorted(sizes)[len(sizes) // 2]
        prefix_median = sorted(prefix_sizes)[len(prefix_sizes) // 2]
        # Each batch anchors on the largest remaining graph and backfills with
        # the smallest that fit, so an early batch is one huge graph plus
        # several tiny ones -- median ~442 against a population median of
        # ~1001. test_val_loader_plan_is_shuffled_and_deterministic asserts the
        # mirror image once the plan is shuffled.
        self.assertGreater(abs(prefix_median - population_median), 300)

    def test_val_loader_plan_is_shuffled_and_deterministic(self):
        from train_baseline import _hoga_loader

        sizes = list(range(1, 2001))
        args = self._args()

        loader_a = _hoga_loader(_FakeSizedDataset(sizes), args, shuffle=False)
        loader_b = _hoga_loader(_FakeSizedDataset(sizes), args, shuffle=False)
        plan_a = list(loader_a.batch_sampler)
        plan_b = list(loader_b.batch_sampler)

        # Deterministic: same seed, same order, so val_loss is stable epoch to
        # epoch even when truncated.
        self.assertEqual(plan_a, plan_b)

        # Representative: a prefix now spans the size distribution rather than
        # its two extremes.
        from data.sampler import BalancedDynamicBatchSampler

        raw = BalancedDynamicBatchSampler.build_batch_plan(
            sizes, max_total_nodes=4000
        )
        self.assertNotEqual(plan_a, raw)

        quarter = plan_a[: len(plan_a) // 4]
        prefix_sizes = [sizes[i] for b in quarter for i in b]
        population_median = sorted(sizes)[len(sizes) // 2]
        prefix_median = sorted(prefix_sizes)[len(prefix_sizes) // 2]
        self.assertLess(abs(prefix_median - population_median), 200)

    def test_val_plan_covers_every_sample_exactly_once(self):
        from train_baseline import _hoga_loader

        sizes = [7, 3, 50, 11, 90, 2, 45, 30]
        loader = _hoga_loader(_FakeSizedDataset(sizes), self._args(), shuffle=False)
        seen = [i for batch in loader.batch_sampler for i in batch]
        self.assertEqual(sorted(seen), list(range(len(sizes))))


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
