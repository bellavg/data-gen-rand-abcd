"""Tests for the Gamora baseline port.

Three groups, matching the three things that could silently go wrong:

1. `gamora_node_features` reproduces the released ABC exporter's encoding
   (acecXor.c:392-415). This is how inverted edges reach a model that consumes
   no `edge_attr`, so a bug here makes the baseline quietly blind to inversions
   -- the exact defect the HOGA port has.
2. The readout IS size-blind, deliberately. Mean pooling cannot distinguish a
   graph from two disjoint copies of it, and that limitation is pinned rather
   than fixed -- the primary model pools identically, so both lose the same
   information and the comparison stays about the encoder.
3. Nothing in the port samples. Upstream's released trainer builds a
   `NeighborSampler`; this project's baselines must not, so it is pinned at the
   source level rather than trusted.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

import config
from baselines.common.lightning_wrapper import BaselineRegressionLightningModule
from baselines.gamora.model import SAGE_MULT
from baselines.gamora.regressor import (
    GAMORA_NODE_FEATURE_DIM,
    GamoraGraphRegressor,
    gamora_node_features,
)

NODE_INPUT_DIM = 4  # [constant, pi, and_gate, po]
# gamora_node_features indexes x[:, 2] and x[:, 3] by position, so a change to
# the project's node-feature layout would break it while every test below --
# which builds its own one-hots against this local constant -- still passed.
assert NODE_INPUT_DIM == config.NODE_INPUT_DIM

_GAMORA_PKG_DIR = Path(__file__).resolve().parents[2] / "baselines" / "gamora"


def _make_aig_data(seed: int = 42, num_nodes: int = 12, num_edges: int = 16) -> Data:
    g = torch.Generator().manual_seed(seed)

    type_idx = torch.randint(0, NODE_INPUT_DIM, (num_nodes,), generator=g)
    x = torch.zeros(num_nodes, NODE_INPUT_DIM)
    x[torch.arange(num_nodes), type_idx] = 1.0

    src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=g)
    edge_index = torch.stack([src, dst], dim=0)

    inv = (torch.rand(num_edges, generator=g) > 0.5).float()
    edge_attr = torch.stack([1.0 - inv, inv], dim=1)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.rand(1, 1, generator=g),
    )


def _one_hot(type_index: int) -> list[float]:
    row = [0.0] * NODE_INPUT_DIM
    row[type_index] = 1.0
    return row


class TestGamoraNodeFeatures(unittest.TestCase):
    """Pins the encoding written by `Gia_edgelist` (acecXor.c:392, :399, :415).

        CI  (primary input) : 0,0,0,0
        AND                 : 1,1,<FaninC0>,<FaninC1>
        CO  (primary output): 0,0,1,1
    """

    def _features_for_and_gate(self, inverted_flags: list[float]) -> torch.Tensor:
        """Build a 1-AND-gate graph fed by `len(inverted_flags)` PIs."""
        num_fanins = len(inverted_flags)
        x = torch.tensor(
            [_one_hot(1) for _ in range(num_fanins)] + [_one_hot(2)],
            dtype=torch.float32,
        )
        and_idx = num_fanins
        edge_index = torch.tensor(
            [list(range(num_fanins)), [and_idx] * num_fanins], dtype=torch.long
        )
        inv = torch.tensor(inverted_flags, dtype=torch.float32)
        edge_attr = torch.stack([1.0 - inv, inv], dim=1)
        return gamora_node_features(x, edge_index, edge_attr)[and_idx]

    def test_and_gate_with_no_inverted_fanins(self):
        torch.testing.assert_close(
            self._features_for_and_gate([0.0, 0.0]),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
        )

    def test_and_gate_with_both_fanins_inverted(self):
        """The paper's own worked example: "node 17 has two inputs inverted,
        with the feature vector [1, 1, 1]" (Section III.B.1), which is
        `1,1,1,1` in the released 4-column form."""
        torch.testing.assert_close(
            self._features_for_and_gate([1.0, 1.0]),
            torch.tensor([1.0, 1.0, 1.0, 1.0]),
        )

    def test_one_inverted_fanin_follows_upstreams_literal_ordering(self):
        """`(FaninC0, FaninC1)` is ordered by literal, not by inversion count.

        `Gia_ManAppendAnd` (gia.h:670-682) branches on `if (iLit0 < iLit1)` and
        stores the smaller literal (`2 * node_id + complement`) as fanin 0. So
        when exactly one fanin is inverted, WHICH column carries the 1 depends
        on which fanin has the lower node id -- the pair is not symmetric.

        A count-based `(count >= 1, count >= 2)` reconstruction would emit
        `[1, 1, 1, 0]` for both cases below and pass a weaker test; these
        assertions are what distinguishes it from upstream's encoding.
        """
        # Fanins are PI 0 and PI 1, in that index order.
        torch.testing.assert_close(
            self._features_for_and_gate([1.0, 0.0]),  # lower-id fanin inverted
            torch.tensor([1.0, 1.0, 1.0, 0.0]),
        )
        torch.testing.assert_close(
            self._features_for_and_gate([0.0, 1.0]),  # higher-id fanin inverted
            torch.tensor([1.0, 1.0, 0.0, 1.0]),
        )

    def test_features_are_invariant_to_edge_column_order(self):
        """Nothing in the pipeline promises to preserve `edge_index` order.

        The ranking is a min/max over each node's incoming edges, so permuting
        the columns must not move a bit. Without this, the features would
        depend on graph-build order and two cache generations could disagree.
        """
        data = _make_aig_data(seed=17)
        reference = gamora_node_features(data.x, data.edge_index, data.edge_attr)

        g = torch.Generator().manual_seed(5)
        perm = torch.randperm(data.edge_index.size(1), generator=g)
        shuffled = gamora_node_features(
            data.x, data.edge_index[:, perm], data.edge_attr[perm]
        )
        torch.testing.assert_close(shuffled, reference)

    def test_primary_input_and_constant_are_all_zero(self):
        """Both map to `0,0,0,0`.

        The PI case is upstream's `Gia_ManForEachCi` row. The constant has no
        upstream analogue at all -- Gia's const0 is object 0 and never gets a
        feature row -- and is mapped to the same vector, the natural reading
        for a source node with no fanins.
        """
        x = torch.tensor([_one_hot(0), _one_hot(1)], dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.float32)
        feats = gamora_node_features(x, edge_index, edge_attr)
        torch.testing.assert_close(feats, torch.zeros(2, GAMORA_NODE_FEATURE_DIM))

    def test_primary_output_is_constant_regardless_of_its_edge(self):
        """Upstream writes `0,0,1,1` for every CO, discarding the inversion.

        `Gia_edgelist` has `Gia_ObjFaninC0` in scope at that point and does not
        use it. In this project's AIGs a PO edge does carry a real inversion
        bit (data/data_utils.py:166), so being faithful here means that bit is
        not visible to this baseline. Pinned so the omission stays deliberate.
        """
        x = torch.tensor([_one_hot(2), _one_hot(3)], dtype=torch.float32)
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        for inv in (0.0, 1.0):
            edge_attr = torch.tensor([[1.0 - inv, inv]], dtype=torch.float32)
            feats = gamora_node_features(x, edge_index, edge_attr)
            torch.testing.assert_close(feats[1], torch.tensor([0.0, 0.0, 1.0, 1.0]))

    def test_inversion_actually_changes_the_features(self):
        """The whole point of the featurisation: flipping an inverted fanin
        must change the model's input, or the baseline is inversion-blind."""
        data = _make_aig_data(seed=7)
        flipped = data.edge_attr.flip(1)
        self.assertFalse(
            torch.equal(
                gamora_node_features(data.x, data.edge_index, data.edge_attr),
                gamora_node_features(data.x, data.edge_index, flipped),
            )
        )


class TestGamoraHeadSurgery(unittest.TestCase):
    def test_classification_heads_are_removed(self):
        """Upstream's `self.linear` holds 4 entries: the shared neck plus one
        head each for xor / maj / adder-root. Only the neck may survive."""
        upstream = SAGE_MULT(
            in_channels=4, hidden_channels=8, out_channels=3, num_layers=4, dropout=0.5
        )
        self.assertEqual(len(upstream.linear), 4)

        model = GamoraGraphRegressor(hidden_channels=8, num_layers=4)
        self.assertEqual(len(model.linear), 1)
        self.assertEqual(model.linear[0].out_features, 8)

    def test_conv_depth_matches_num_layers(self):
        for num_layers in (4, 8):
            model = GamoraGraphRegressor(hidden_channels=8, num_layers=num_layers)
            self.assertEqual(len(model.convs), num_layers)

    def test_vendored_forward_nosampler_reads_self_dropout(self):
        """The one deliberate change to the vendored trunk (model.py).

        Upstream's `forward_nosampler` hardcodes `F.dropout(x, p=0.5)` while
        `__init__` stores `self.dropout` and never reads it, so upstream's
        `--dropout` flag is dead. This copy reads it. Exercised on `SAGE_MULT`
        directly, because `GamoraGraphRegressor` deletes the classification
        heads this method writes to and so can never call it.
        """
        torch.manual_seed(0)
        x = torch.randn(20, 4)
        edge_index = torch.randint(0, 20, (2, 40))

        model = SAGE_MULT(
            in_channels=4, hidden_channels=8, out_channels=3, num_layers=4, dropout=0.0
        )
        model.train()
        torch.manual_seed(1)
        quiet = model.forward_nosampler(x, edge_index, "cpu")
        torch.manual_seed(1)
        torch.testing.assert_close(
            quiet, model.forward_nosampler(x, edge_index, "cpu")
        )

        model.dropout = 0.9
        torch.manual_seed(1)
        noisy = model.forward_nosampler(x, edge_index, "cpu")
        self.assertFalse(torch.allclose(quiet[0], noisy[0]))

    def test_dropout_argument_is_wired_through(self):
        """Same knob, through the regressor's own re-inlined trunk.

        Uses a MULTI-graph batch deliberately: with one graph in train mode the
        prediction is constant regardless of the trunk, for the reason
        `test_single_graph_train_batch_collapses_to_the_bn_bias` below explains,
        and this assertion would then hold for the wrong reason.
        """
        batch = Batch.from_data_list([_make_aig_data(seed=i) for i in range(4)])

        model = GamoraGraphRegressor(hidden_channels=8, num_layers=4, dropout=0.0)
        model.train()
        torch.manual_seed(1)
        no_dropout = model(batch)
        torch.manual_seed(1)
        torch.testing.assert_close(no_dropout, model(batch))

        model.dropout = 0.9
        torch.manual_seed(1)
        heavy_dropout = model(batch)
        self.assertFalse(torch.allclose(no_dropout, heavy_dropout))

    def test_single_graph_train_batch_collapses_to_the_bn_bias(self):
        """An interaction between upstream's `bn0` and the pooling this port adds.

        `bn0` (gnn_multitask.py:58) normalises over every node in the batch, so
        the batch-wide mean of its output is exactly its bias. Mean pooling then
        averages over that same node set -- so when the batch holds ONE graph,
        the two operations are over identical sets and the graph embedding is
        the BN bias, whatever the circuit was. The encoder contributes nothing
        to that step.

        Harmless in the configured regime: at the 3M-node budget a batch holds
        ~75 graphs and no graph reaches the budget, so singleton batches do not
        arise. It becomes real if the budget is ever lowered below
        config.MAX_NUM_GATES, because an oversized graph forms a singleton batch
        that graph-level pooling cannot split. Pinned so that consequence is
        discovered here and not in a flat training curve.

        Train mode only -- eval uses running statistics, so nothing collapses.
        """
        torch.manual_seed(0)
        # Both dropouts off so the comparison is about the embedding, not about
        # the head drawing a different mask on each of the three calls.
        model = GamoraGraphRegressor(
            hidden_channels=8, num_layers=4, dropout=0.0, head_dropout=0.0
        )
        model.train()

        predictions = [
            model(Batch.from_data_list([_make_aig_data(seed=s)])) for s in (1, 2, 3)
        ]
        for other in predictions[1:]:
            torch.testing.assert_close(predictions[0], other)

        model.eval()
        with torch.no_grad():
            eval_predictions = [
                model(Batch.from_data_list([_make_aig_data(seed=s)])) for s in (1, 2, 3)
            ]
        self.assertFalse(torch.allclose(eval_predictions[0], eval_predictions[1]))


class TestGamoraGraphRegressor(unittest.TestCase):
    def test_forward_pass_shape_and_range(self):
        model = GamoraGraphRegressor(hidden_channels=8, num_layers=4)
        model.eval()
        batch = Batch.from_data_list([_make_aig_data(seed=i) for i in range(3)])
        with torch.no_grad():
            out = model(batch)
        self.assertEqual(out.shape, (3, 1))
        self.assertTrue(torch.all(out >= 0.0))
        self.assertTrue(torch.all(out <= 1.0))

    def test_batch_independence(self):
        """A two-graph batch must equal two one-graph forwards.

        `bn0` is a BatchNorm over the node axis, so this only holds in eval
        mode (running statistics); in train mode the batch composition legitimately
        changes the result. Eval is the mode the reported metrics come from.
        """
        torch.manual_seed(0)
        model = GamoraGraphRegressor(hidden_channels=8, num_layers=4)
        model.eval()

        graphs = [_make_aig_data(seed=11), _make_aig_data(seed=12)]
        with torch.no_grad():
            together = model(Batch.from_data_list(graphs))
            apart = torch.cat(
                [model(Batch.from_data_list([g])) for g in graphs], dim=0
            )
        torch.testing.assert_close(together, apart, rtol=1e-5, atol=1e-6)

    def test_forward_accepts_an_uncollated_data(self):
        """A bare `Data` has `.batch is None` and no `.num_graphs`.

        The SynthNet and HOGA ports both work on one, because they read
        `.batch` and nothing else. This one computes per-graph sizes, so it has
        to handle the single-graph case explicitly -- and must agree with the
        collated answer, or an eval path that forwards one graph at a time
        would silently score differently from training.
        """
        model = GamoraGraphRegressor(hidden_channels=8, num_layers=4)
        model.eval()
        data = _make_aig_data(seed=31)
        with torch.no_grad():
            bare = model(data)
            collated = model(Batch.from_data_list([data]))
        self.assertEqual(bare.shape, (1, 1))
        torch.testing.assert_close(bare, collated)

    def test_gradient_flows(self):
        model = GamoraGraphRegressor(hidden_channels=8, num_layers=4)
        batch = Batch.from_data_list([_make_aig_data(seed=i) for i in range(4)])
        model(batch).sum().backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, msg=f"no gradient for {name}")
                self.assertTrue(torch.isfinite(param.grad).all(), msg=name)


class TestMeanPoolingIsSizeBlind(unittest.TestCase):
    """Pins a known, deliberate limitation rather than a feature.

    `G` and `G disjoint-union G` have byte-identical mean-pooled embeddings:
    message passing never crosses components, so every node's embedding is
    unchanged, and the mean over two identical copies is the mean over one. So
    this model cannot tell a graph from twice itself, and carries no |V| or |E|
    information.

    That is not a bug to fix here. The primary model pools the same way
    (`config.POOLING_TYPE = "mean"`), so the baseline and the model it is
    compared against lose exactly the same information and the comparison stays
    about the encoder. Feeding graph size to the head was tried and removed:
    nothing else in the project receives it, and Gamora has no readout at all
    to take it from. This test exists so that if anyone adds size information
    later, they are forced to notice it must be added everywhere or nowhere.
    """

    @staticmethod
    def _duplicated(data: Data) -> Data:
        n = data.x.size(0)
        return Data(
            x=torch.cat([data.x, data.x], dim=0),
            edge_index=torch.cat([data.edge_index, data.edge_index + n], dim=1),
            edge_attr=torch.cat([data.edge_attr, data.edge_attr], dim=0),
            y=data.y,
        )

    def test_a_graph_and_two_copies_of_it_predict_identically(self):
        torch.manual_seed(0)
        model = GamoraGraphRegressor(hidden_channels=8, num_layers=4)
        model.eval()
        single = _make_aig_data(seed=5)
        with torch.no_grad():
            one = model(Batch.from_data_list([single]))
            two = model(Batch.from_data_list([self._duplicated(single)]))
        torch.testing.assert_close(one, two, rtol=1e-5, atol=1e-6)

    def test_head_takes_the_embedding_width_with_nothing_appended(self):
        model = GamoraGraphRegressor(hidden_channels=32)
        self.assertEqual(model.regression_head[0].in_features, 32)


class TestNoSamplingInPort(unittest.TestCase):
    """This project's baselines must not sample, partition, or decompose.

    Upstream's released trainer builds
    `NeighborSampler(data.adj_t, sizes=[8, 5, 5, 5], batch_size=20)`
    (gnn_multitask.py:570-572) and `train()` iterates it. This port trains
    full-graph instead -- a documented deviation from the published procedure.
    A source-level assertion, because the failure mode is someone vendoring one
    more upstream method later and nobody noticing.
    """

    FORBIDDEN = frozenset(
        {
            "NeighborSampler",
            "NeighborLoader",
            "ClusterLoader",
            "ClusterData",
            "subgraph",
            "k_hop_subgraph",
        }
    )

    @staticmethod
    def _identifiers(source: str) -> set[str]:
        """Every name and attribute the module's CODE references.

        Walks the AST rather than grepping the text, because these docstrings
        quote upstream's `NeighborSampler(...)` call verbatim on purpose --
        naming it is the documentation, calling it is the violation. String
        constants are not `Name`/`Attribute` nodes, so they are excluded for
        free.
        """
        names: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.alias):
                names.add(node.name.split(".")[-1])
        return names

    # Every module the Gamora training path actually executes, not just the
    # baseline package. PROVENANCE.md claims nothing samples anywhere on that
    # path; checking baselines/gamora/*.py alone would leave the claim about
    # the entrypoint, the batch sampler and the Lightning wrapper unpinned --
    # and those are where a sampler would realistically be introduced.
    TRAINING_PATH_MODULES = (
        "train_baseline.py",
        "train_utils.py",
        "data/sampler.py",
        "baselines/common/lightning_wrapper.py",
        # Imported by regressor.py for derive_num_inverted_predecessors.
        "baselines/openabc_synthnet/regressor.py",
    )

    def _modules_under_test(self) -> list[Path]:
        src = _GAMORA_PKG_DIR.parents[1]
        paths = sorted(_GAMORA_PKG_DIR.glob("*.py"))
        self.assertTrue(paths, "no gamora modules found -- did the package move?")
        for relative in self.TRAINING_PATH_MODULES:
            path = src / relative
            self.assertTrue(path.is_file(), f"{relative} moved -- update this list")
            paths.append(path)
        return paths

    def test_no_sampler_or_subgraph_call_on_the_gamora_training_path(self):
        for path in self._modules_under_test():
            with self.subTest(module=path.name):
                found = self._identifiers(path.read_text()) & self.FORBIDDEN
                self.assertEqual(
                    found, set(), msg=f"{path.name} references {sorted(found)}"
                )

    def test_the_assertion_would_actually_catch_a_sampler(self):
        """Guards the guard: an AST walk that silently matched nothing would
        make the test above pass for the wrong reason."""
        source = "from torch_geometric.loader import NeighborSampler\nx = NeighborSampler(a, sizes=[8, 5, 5, 5])\n"
        self.assertIn("NeighborSampler", self._identifiers(source) & self.FORBIDDEN)


class TestGamoraLightningTraining(unittest.TestCase):
    def setUp(self):
        self.dataset = [_make_aig_data(seed=i) for i in range(10)]
        self.train_loader = DataLoader(self.dataset[:6], batch_size=2)
        self.val_loader = DataLoader(self.dataset[6:8], batch_size=2)
        self.test_loader = DataLoader(self.dataset[8:], batch_size=2)

    def test_training_and_testing_loop(self):
        base_model = GamoraGraphRegressor(hidden_channels=8, num_layers=4)
        model = BaselineRegressionLightningModule(
            base_model, lr=1e-3, loss_fn=torch.nn.SmoothL1Loss(beta=0.01)
        )

        trainer = pl.Trainer(
            fast_dev_run=True,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(
            model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
        )
        trainer.test(model, dataloaders=self.test_loader)


if __name__ == "__main__":
    unittest.main()
