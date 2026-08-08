from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

import config
from baselines.common.lightning_wrapper import BaselineRegressionLightningModule
from baselines.polargate import layers as polargate_layers
from baselines.polargate.layers import PolarGateConv, restPolarGateConv
from baselines.polargate.regressor import (
    PolarGateGraphRegressor,
    split_signed_edge_index,
)
from data.data_utils import aig_to_pytorch_geometric


def _graph(num_nodes: int, num_edges: int, seed: int, *, inv_rate: float = 0.4) -> Data:
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=g)
    inv = (torch.rand(num_edges, generator=g) < inv_rate).to(torch.float32)
    node_type = torch.randint(0, config.NODE_INPUT_DIM, (num_nodes,), generator=g)
    return Data(
        x=torch.eye(config.NODE_INPUT_DIM)[node_type],
        edge_index=torch.stack([src, dst]),
        edge_attr=torch.stack([1.0 - inv, inv], dim=1),
        y=torch.rand(1, generator=g),
    )


class TestSplitSignedEdgeIndex(unittest.TestCase):
    """The pos/neg split is the single point where this project's edge encoding
    meets upstream's. Upstream reads a literal +1/-1 sign column written by its
    `.bench` preprocessor; here inversion lives in `edge_attr` as
    `[1 - inv, inv]` (data/data_utils.py:148). Get the column wrong and the
    model still runs -- it just swaps every inverted edge for a plain one and
    quietly stops being PolarGate.
    """

    def test_partitions_edges_by_the_inverted_column(self):
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        # rows 1 and 3 inverted
        edge_attr = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        )

        pos, neg = split_signed_edge_index(edge_index, edge_attr)

        self.assertTrue(torch.equal(pos, torch.tensor([[0, 2], [1, 3]])))
        self.assertTrue(torch.equal(neg, torch.tensor([[1, 3], [2, 4]])))

    def test_split_is_a_partition_with_no_loss_or_overlap(self):
        data = _graph(40, 120, seed=3)
        pos, neg = split_signed_edge_index(data.edge_index, data.edge_attr)

        self.assertEqual(pos.size(1) + neg.size(1), data.edge_index.size(1))
        recombined = torch.cat([pos, neg], dim=1)
        # Order is not preserved by a boolean mask split, so compare as sets of
        # (src, dst) pairs rather than elementwise.
        self.assertEqual(
            sorted(map(tuple, recombined.t().tolist())),
            sorted(map(tuple, data.edge_index.t().tolist())),
        )

    def test_all_non_inverted_and_all_inverted_extremes(self):
        edge_index = torch.tensor([[0, 1], [1, 2]])

        pos, neg = split_signed_edge_index(
            edge_index, torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        )
        self.assertEqual(pos.size(1), 2)
        self.assertEqual(neg.size(1), 0)
        # An empty side must still be [2, 0], not [0] -- propagate indexes row 0
        # and row 1 unconditionally.
        self.assertEqual(tuple(neg.shape), (2, 0))

        pos, neg = split_signed_edge_index(
            edge_index, torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        )
        self.assertEqual(pos.size(1), 0)
        self.assertEqual(neg.size(1), 2)
        self.assertEqual(tuple(pos.shape), (2, 0))

    def test_matches_the_real_pipeline_encoding_on_a_real_aig(self):
        """Pins the split against data/data_utils.py rather than a hand-built
        tensor, because config.py's EDGE_ATTR_DIM comment claims the two columns
        are `[normal edge, primary output edge]`. That comment is stale. If it
        ever became true, this test fails and the model stops seeing
        inversions -- which is the whole method.
        """
        aig_path = Path(__file__).resolve().parents[1] / "data" / "adder.aig"
        self.assertTrue(aig_path.exists(), f"Missing fixture AIG: {aig_path}")
        data = aig_to_pytorch_geometric(aig_path)

        # The real encoding is one-hot over two columns summing to 1.
        self.assertEqual(data.edge_attr.size(1), config.EDGE_ATTR_DIM)
        self.assertTrue(
            torch.allclose(
                data.edge_attr.sum(dim=1), torch.ones(data.edge_attr.size(0))
            )
        )
        self.assertTrue(
            torch.all((data.edge_attr == 0.0) | (data.edge_attr == 1.0)),
            "edge_attr is expected to be a hard one-hot, not a soft weight",
        )

        pos, neg = split_signed_edge_index(data.edge_index, data.edge_attr)
        self.assertEqual(pos.size(1) + neg.size(1), data.edge_index.size(1))
        # An adder AIG contains inverted edges; a split that produced none
        # would mean column 1 is not the inversion flag.
        self.assertGreater(neg.size(1), 0)
        self.assertGreater(pos.size(1), 0)
        self.assertEqual(int(data.edge_attr[:, 1].sum()), neg.size(1))

    def test_rejects_missing_or_malformed_edge_attr(self):
        edge_index = torch.tensor([[0, 1], [1, 2]])
        with self.assertRaises(ValueError):
            split_signed_edge_index(edge_index, None)
        with self.assertRaises(ValueError):
            split_signed_edge_index(edge_index, torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            split_signed_edge_index(edge_index, torch.zeros(5, 2))


class TestVendoredLayers(unittest.TestCase):
    def test_spectral_feature_fallback_is_not_importable(self):
        """Deletion 1 in layers.py. Upstream reaches
        `create_spectral_features` whenever `init_emb is None`; it runs
        TruncatedSVD(n_components=64, n_iter=128) over the adjacency, which is
        not viable at 366,040 nodes x ~707k graphs. Its absence is what makes
        that path unreachable rather than merely unused.
        """
        self.assertFalse(hasattr(polargate_layers, "create_spectral_features"))

    def test_conv_aggregators_match_upstream(self):
        """`mean` for the first conv, `min` for the rest (OPAND, the paper's
        differentiable Boolean intersection). Upstream sets these via
        `kwargs.setdefault`, so a caller passing `aggr=` would silently change
        the operator.
        """
        self.assertEqual(PolarGateConv(4, 8, first_aggr=True).aggr, "mean")
        self.assertEqual(restPolarGateConv(8, 8).aggr, "min")

    def test_nodes_with_no_incoming_edges_aggregate_to_zero(self):
        """Paper Equation (6): a primary input contributes `[0, 0, h_own]`.

        Upstream gets that for free from the scatter reduction's empty-row fill
        rather than by branching on node type, so this port inherits the
        behaviour only as long as PyG keeps filling empty rows with 0 -- for
        `min` as well as `mean`. The identity `min` fill is +inf, and a PyG
        upgrade that switched to it would not announce itself: every primary
        input's aggregate would silently become the largest float instead of
        zero, the `[0, 0, h]` form of Equation (6) would be gone, and the
        `inf - inf` inside the next Linear would turn the trunk to NaN. (tanh
        alone would not: tanh(+inf) is 1.0, finite.) The explicit
        zero-comparison below is what catches it.
        """
        conv = restPolarGateConv(2, 2)
        x = torch.randn(3, 4)
        # Only 0 -> 1. Nodes 0 and 2 receive nothing.
        edge_index = torch.tensor([[0], [1]])
        empty = torch.zeros(2, 0, dtype=torch.long)
        out = conv(x, edge_index, empty)
        self.assertTrue(torch.isfinite(out).all())

        with torch.no_grad():
            aggregated = conv.propagate(edge_index, x=(x[:, :2], x[:, :2]))
        self.assertTrue(torch.equal(aggregated[0], torch.zeros(2)))
        self.assertTrue(torch.equal(aggregated[2], torch.zeros(2)))


class TestPolarGateGraphRegressor(unittest.TestCase):
    def _model(self, **kwargs) -> PolarGateGraphRegressor:
        torch.manual_seed(0)
        defaults = dict(in_dim=config.NODE_INPUT_DIM, out_dim=32, layer_num=3)
        defaults.update(kwargs)
        return PolarGateGraphRegressor(**defaults).eval()

    def test_output_shape_and_range(self):
        model = self._model()
        batch = Batch.from_data_list([_graph(20, 50, 1), _graph(30, 70, 2)])
        with torch.no_grad():
            out = model(batch)
        self.assertEqual(out.shape, (2, config.TASK_OUT_DIM))
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0))

    def test_two_graph_batch_matches_two_single_graph_forwards(self):
        """PyG batches block-diagonally, so nothing may leak between graphs.

        Not automatic here: the readout is graph-level, so a batch-coupled
        head or a mis-scoped pooling would mix graphs. What this covers is
        pooling scope and the size covariates. It says nothing about the
        head_norm_layer default -- the model under test has no norm layer at
        all, and even with one it would run in eval mode here, where BatchNorm
        uses running statistics and is per-row.
        `test_batchnorm_head_raises_on_a_singleton_batch` is what pins that.

        Compared with a tolerance rather than `torch.equal`: the batched and
        single-graph runs reduce over different tensor shapes, so BLAS and the
        scatter kernels are free to accumulate in a different order. The
        observed gap is ~6e-8 in float32, i.e. rounding. Anything larger than
        `atol` here means real cross-graph leakage, not arithmetic.
        """
        model = self._model()
        a, b = _graph(23, 61, 11), _graph(37, 90, 12)
        with torch.no_grad():
            single = torch.cat(
                [model(Batch.from_data_list([a])), model(Batch.from_data_list([b]))]
            )
            batched = model(Batch.from_data_list([a, b]))
        self.assertTrue(
            torch.allclose(single, batched, rtol=0.0, atol=1e-6),
            f"single={single.tolist()} batched={batched.tolist()}",
        )

    def test_batch_equivalence_holds_for_sum_pooling_too(self):
        model = self._model(pooling="sum")
        a, b = _graph(23, 61, 11), _graph(37, 90, 12)
        with torch.no_grad():
            single = torch.cat(
                [model(Batch.from_data_list([a])), model(Batch.from_data_list([b]))]
            )
            batched = model(Batch.from_data_list([a, b]))
        self.assertTrue(
            torch.allclose(single, batched, rtol=0.0, atol=1e-6),
            f"single={single.tolist()} batched={batched.tolist()}",
        )

    def test_inversion_reaches_the_model(self):
        """The failure this exists to catch: a port that never wires edge_attr
        through still trains and still produces numbers, it just is not
        PolarGate. Flipping every edge from non-inverted to inverted must move
        the prediction.
        """
        model = self._model()
        base = _graph(30, 80, 7)

        all_pos = base.clone()
        all_pos.edge_attr = torch.stack(
            [torch.ones(base.edge_index.size(1)), torch.zeros(base.edge_index.size(1))],
            dim=1,
        )
        all_neg = base.clone()
        all_neg.edge_attr = all_pos.edge_attr.flip(1)

        with torch.no_grad():
            out_pos = model(Batch.from_data_list([all_pos]))
            out_neg = model(Batch.from_data_list([all_neg]))
        self.assertFalse(torch.allclose(out_pos, out_neg))

    def test_requires_edge_attr(self):
        model = self._model()
        data = _graph(10, 20, 5)
        data.edge_attr = None
        with self.assertRaises(ValueError):
            model(Batch.from_data_list([data]))

    def test_requires_node_features(self):
        """Upstream would fall back to create_spectral_features here."""
        model = self._model()
        data = _graph(10, 20, 5)
        batch = Batch.from_data_list([data])
        batch.x = None
        with self.assertRaises(ValueError):
            model(batch)

    def test_size_covariates_make_the_readout_size_aware(self):
        """Defect this addresses: mean pooling cannot represent |V| or |E|, and
        on this dataset a two-parameter OLS on log node and edge count already
        outranks the primary encoder on Spearman.

        Two graphs built so that mean pooling sees identical node embeddings --
        the same 6-node motif, once and twice over as disconnected copies -- so
        the ONLY difference the head can see is the size covariate.
        """
        motif = Data(
            x=torch.eye(config.NODE_INPUT_DIM)[torch.tensor([1, 2, 2, 2, 2, 3])],
            edge_index=torch.tensor([[0, 1, 0, 2, 3, 4], [1, 3, 2, 4, 5, 5]]),
            edge_attr=torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
            ),
            y=torch.zeros(1),
        )
        # Two disjoint copies of the same motif inside ONE graph: every node has
        # an identical neighbourhood to its counterpart, so the per-node
        # embeddings are the same multiset and the mean is unchanged.
        doubled = Batch.from_data_list([motif, motif])
        doubled = Data(
            x=doubled.x,
            edge_index=doubled.edge_index,
            edge_attr=doubled.edge_attr,
            y=torch.zeros(1),
        )

        blind = self._model(size_covariates=False)
        with torch.no_grad():
            b_small = blind(Batch.from_data_list([motif]))
            b_large = blind(Batch.from_data_list([doubled]))
        self.assertTrue(
            torch.allclose(b_small, b_large, atol=1e-6),
            "mean pooling was expected to be size-blind here; if this fails the "
            "fixture is no longer size-degenerate and the test below proves nothing",
        )

        aware = self._model(size_covariates=True)
        with torch.no_grad():
            a_small = aware(Batch.from_data_list([motif]))
            a_large = aware(Batch.from_data_list([doubled]))
        self.assertFalse(torch.allclose(a_small, a_large, atol=1e-6))

    def test_batchnorm_head_raises_on_a_singleton_batch(self):
        """Documents the real failure mode of --polargate_head_norm_layer.

        Upstream's readout MLP is built with `norm_layer='batchnorm'`, where it
        sees one row per NODE. Here it sees one row per GRAPH, and a graph
        larger than the node budget forms a singleton micro-batch that cannot
        be split -- guaranteed at config.MAX_NUM_GATES. nn.BatchNorm1d rejects a
        1-row input in training mode outright rather than degrading, so the job
        dies mid-epoch. Note DeepGate4's port behaves differently here (its
        vendored MLP pads a 1-row input by repeating it), so the two flags share
        a default but not a failure mode.
        """
        torch.manual_seed(0)
        model = PolarGateGraphRegressor(
            in_dim=config.NODE_INPUT_DIM,
            out_dim=32,
            layer_num=2,
            head_norm_layer="batchnorm",
        ).train()
        one_graph = Batch.from_data_list([_graph(12, 25, 4)])
        with self.assertRaises(ValueError):
            model(one_graph)

        # Two graphs is enough for BatchNorm to run, which is why this cannot
        # be caught by the ordinary multi-graph tests.
        two_graphs = Batch.from_data_list([_graph(12, 25, 4), _graph(9, 18, 5)])
        self.assertEqual(model(two_graphs).shape, (2, config.TASK_OUT_DIM))

    def test_rejects_odd_out_dim(self):
        with self.assertRaises(ValueError):
            PolarGateGraphRegressor(in_dim=4, out_dim=33, layer_num=2)

    def test_layer_num_one_uses_only_the_first_conv(self):
        model = self._model(layer_num=1)
        self.assertEqual(len(model.convs), 0)
        with torch.no_grad():
            out = model(Batch.from_data_list([_graph(12, 25, 4)]))
        self.assertEqual(out.shape, (1, config.TASK_OUT_DIM))

    def test_backward_reaches_every_parameter(self):
        torch.manual_seed(0)
        model = PolarGateGraphRegressor(
            in_dim=config.NODE_INPUT_DIM, out_dim=32, layer_num=3
        ).train()
        out = model(Batch.from_data_list([_graph(25, 60, 9), _graph(18, 40, 10)]))
        out.sum().backward()
        for name, param in model.named_parameters():
            with self.subTest(param=name):
                self.assertIsNotNone(param.grad, f"{name} received no gradient")
                self.assertTrue(torch.isfinite(param.grad).all())


class TestPolarGateRunLabel(unittest.TestCase):
    def test_size_covariate_arms_get_separate_checkpoint_dirs(self):
        """The size-covariate ablation is the whole basis for attributing any
        PolarGate win, since it is the only model in the suite that sees |V|
        and |E|. The two arms differ in nothing else, so without a suffix the
        second overwrites the first's last.ckpt and neither can be attributed.
        """
        import train_baseline

        def label(**overrides):
            args = SimpleNamespace(
                baseline="polargate",
                algorithm="Orchestrate",
                split_by=config.SPLIT_BY,
                loss=train_baseline._BASELINE_DEFAULTS["polargate"]["loss"],
                polargate_size_covariates=False,
                polargate_pooling="mean",
            )
            for key, value in overrides.items():
                setattr(args, key, value)
            return train_baseline._run_label(args)

        self.assertEqual(label(), "polargate_Orchestrate")
        self.assertEqual(
            label(polargate_size_covariates=True),
            "polargate_Orchestrate_sizecov",
        )
        self.assertEqual(
            label(polargate_pooling="sum"), "polargate_Orchestrate_sumpool"
        )
        # Suffixes stack, and neither is emitted for the other baselines.
        self.assertEqual(
            label(polargate_size_covariates=True, polargate_pooling="sum"),
            "polargate_Orchestrate_sizecov_sumpool",
        )
        self.assertEqual(
            label(baseline="hoga", loss="mse", polargate_size_covariates=True),
            "hoga_Orchestrate",
        )


class TestPolarGateLightningIntegration(unittest.TestCase):
    def test_one_training_step_runs_under_the_baseline_wrapper(self):
        torch.manual_seed(0)
        model = PolarGateGraphRegressor(
            in_dim=config.NODE_INPUT_DIM, out_dim=16, layer_num=2
        )
        module = BaselineRegressionLightningModule(
            model, lr=1e-3, loss_fn=torch.nn.SmoothL1Loss(beta=0.01)
        )
        graphs = [_graph(15 + i, 30 + i, seed=100 + i) for i in range(4)]
        loader = DataLoader(graphs, batch_size=2, collate_fn=Batch.from_data_list)
        # A val loader is required, not optional: the wrapper's
        # ReduceLROnPlateau monitors val_loss with strict=True, so a fit with
        # no validation stage raises MisconfigurationException at the first
        # epoch end.
        val_loader = DataLoader(
            graphs[:2], batch_size=2, collate_fn=Batch.from_data_list
        )

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=val_loader)
        self.assertGreater(trainer.global_step, 0)


if __name__ == "__main__":
    unittest.main()
