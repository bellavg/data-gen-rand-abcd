from __future__ import annotations

import copy
import unittest

import numpy as np
import pytorch_lightning as pl
import torch
from torch_geometric.data import Data

from baselines.common.lightning_wrapper import BaselineRegressionLightningModule
from baselines.deepgate4.aig_features import (
    GATE_AND,
    GATE_NOT,
    GATE_PI,
    OUT_DEGREE_TABLE_SIZE,
    DeepGateGraphAdapter,
    collate_deepgate_batch,
    expand_not_nodes,
    node_out_degrees,
    check_topological,
    to_deepgate_graph,
    virtual_edges,
)
from baselines.deepgate4.regressor import DeepGate4GraphRegressor
from models.layers.positional_encodings import get_pe_transform


def _tiny_aig(inverted_mask=(False, True, False, True, False), y=0.5):
    """const, PI, PI, AND(<-1,2), AND(<-1,3), PO(<-4). Levels are pre-expansion.

    `pos_enc` is produced by the REAL `get_pe_transform('level')`, not
    hand-written as integers. That matters: the transform stores log1p of the
    level, and a fixture that hard-codes integers would silently assert a
    pipeline contract the pipeline does not honour.
    """
    x = torch.tensor(
        [
            [1.0, 0, 0, 0],  # 0 constant
            [0, 1.0, 0, 0],  # 1 PI
            [0, 1.0, 0, 0],  # 2 PI
            [0, 0, 1.0, 0],  # 3 AND
            [0, 0, 1.0, 0],  # 4 AND
            [0, 0, 0, 1.0],  # 5 PO
        ]
    )
    edge_index = torch.tensor([[1, 2, 1, 3, 4], [3, 3, 4, 4, 5]])
    edge_attr = torch.tensor(
        [[0.0, 1.0] if inv else [1.0, 0.0] for inv in inverted_mask]
    )
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        level=torch.tensor([[0], [0], [0], [1], [2], [3]], dtype=torch.float32),
        y=torch.tensor([[y]]),
    )
    data.num_nodes = 6
    return get_pe_transform("level", attr_name="pos_enc")(data)


def _brute_force_reachable(edge_index, num_nodes, k):
    """`{(u, v) : there is a path u -> v of length <= k}`, by explicit BFS."""
    succ = [[] for _ in range(num_nodes)]
    for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        succ[s].append(d)
    pairs = set()
    for start in range(num_nodes):
        frontier = {start}
        for _ in range(k):
            frontier = {n for u in frontier for n in succ[u]}
            if not frontier:
                break
            for n in frontier:
                pairs.add((start, n))
    return pairs


class TestNotNodeExpansion(unittest.TestCase):
    """Covers the one structural change this port makes to the input graphs.

    This project stores inversion on the edge; DeepGate stores it as a NOT
    node and keeps separate aggregators/GRUs for AND and NOT. See
    aig_features.expand_not_nodes.
    """

    def test_one_shared_not_node_per_inverted_source(self):
        """Upstream memoises via `has_not[fanin]`, so a source that drives two
        inverted edges gets ONE NOT node, not two."""
        x = torch.tensor([[0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 1.0, 0]])
        # node 0 drives both node 1 and node 2, inverted in both cases.
        edge_index = torch.tensor([[0, 0], [1, 2]])
        edge_attr = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

        gate, expanded, num_nodes, not_of = expand_not_nodes(x, edge_index, edge_attr)

        self.assertEqual(num_nodes, 4)  # 3 original + exactly 1 NOT
        self.assertEqual(gate[3], GATE_NOT)
        self.assertEqual(not_of[0], 3)
        edges = sorted(map(tuple, expanded.T.tolist()))
        self.assertEqual(edges, [(0, 3), (3, 1), (3, 2)])

    def test_no_inversions_leaves_graph_untouched(self):
        data = _tiny_aig(inverted_mask=(False,) * 5)
        gate, expanded, num_nodes, not_of = expand_not_nodes(
            data.x, data.edge_index, data.edge_attr
        )
        self.assertEqual(num_nodes, 6)
        self.assertTrue((not_of == -1).all())
        self.assertEqual(expanded.shape[1], 5)

    def test_node_type_mapping(self):
        data = _tiny_aig(inverted_mask=(False,) * 5)
        gate, _, _, _ = expand_not_nodes(data.x, data.edge_index, data.edge_attr)
        # constant and PI both map to the in-degree-0 type; PO is in-degree 1.
        self.assertEqual(list(gate[:6]), [GATE_PI, GATE_PI, GATE_PI, GATE_AND, GATE_AND, GATE_NOT])

    def test_inverted_primary_output_becomes_a_two_hop_not_chain(self):
        """Pins a known, deliberate distortion rather than leaving it to chance.

        This project adds a synthetic PO node (in-degree 1, mapped to
        GATE_NOT). If the driver->PO edge is itself inverted, the generic
        expansion inserts a real NOT ahead of it, so the chain is
        `driver -> NOT -> PO`, both typed GATE_NOT. Upstream emits a single
        terminal NOT for an inverted PO and no PO node at all, so its NOT
        update runs once where ours runs twice. Documented in
        aig_features.expand_not_nodes.
        """
        x = torch.tensor(
            [[0, 1.0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
        )
        edge_index = torch.tensor([[0, 1, 2], [2, 2, 3]])
        edge_attr = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # PO edge inverted

        gate, expanded, num_nodes, _ = expand_not_nodes(x, edge_index, edge_attr)

        self.assertEqual(num_nodes, 5)  # 4 original + 1 inserted NOT
        self.assertEqual(gate.tolist(), [GATE_PI, GATE_PI, GATE_AND, GATE_NOT, GATE_NOT])
        edges = sorted(map(tuple, expanded.T.tolist()))
        self.assertEqual(edges, [(0, 2), (1, 2), (2, 4), (4, 3)])

        # Both NOT hops keep in-degree 1, so the paper's type-by-in-degree rule
        # still holds and the level walk stays well-defined.
        in_deg = torch.zeros(num_nodes, dtype=torch.long)
        e = torch.from_numpy(expanded)
        in_deg.scatter_add_(0, e[1], torch.ones_like(e[1]))
        self.assertEqual(int(in_deg[3]), 1)
        self.assertEqual(int(in_deg[4]), 1)

    def test_missing_edge_attr_means_no_inversions(self):
        data = _tiny_aig()
        _, _, num_nodes, not_of = expand_not_nodes(data.x, data.edge_index, None)
        self.assertEqual(num_nodes, 6)
        self.assertTrue((not_of == -1).all())


class TestForwardLevels(unittest.TestCase):
    """`DeepGate2.forward` walks levels in order and assumes every fanin was
    finalised at a strictly lower level, so the only property that actually
    matters is that levels are a topological stratification of the EXPANDED
    graph -- which the cached pre-expansion `pos_enc` is not."""

    def test_levels_stratify_the_expanded_graph(self):
        for mask in [
            (False,) * 5,
            (True,) * 5,
            (False, True, False, True, False),
            (True, False, True, False, True),
        ]:
            with self.subTest(mask=mask):
                g = to_deepgate_graph(_tiny_aig(inverted_mask=mask), num_hops=3)
                lvl = g.forward_level
                src, dst = g.edge_index
                self.assertTrue(
                    torch.all(lvl[src] < lvl[dst]),
                    "forward_level is not a topological stratification",
                )

    def test_inversion_lengthens_the_path(self):
        """An inverted fanin inserts a NOT node, so the consumer sits one level
        deeper than it would with a direct edge."""
        plain = to_deepgate_graph(_tiny_aig(inverted_mask=(False,) * 5), num_hops=3)
        inv = to_deepgate_graph(_tiny_aig(inverted_mask=(True, False, False, False, False)), num_hops=3)
        # node 3 is fed by edge 0 (1 -> 3); inverting it costs one extra level.
        self.assertEqual(int(plain.forward_level[3]) + 1, int(inv.forward_level[3]))

    def test_a_cycle_raises_rather_than_looping_or_returning_garbage(self):
        """Kahn's terminates early on a cycle instead of settling every node.

        An AIG is acyclic by construction, so this should never fire in
        practice -- but the failure mode without the check is a silently
        under-relaxed level array, i.e. exactly the class of silent corruption
        the rest of this file exists to prevent.
        """
        from baselines.deepgate4.aig_features import forward_levels

        # 0 -> 1 -> 2 -> 1 : nodes 1 and 2 never reach in-degree 0.
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 1]])
        with self.assertRaises(ValueError) as ctx:
            forward_levels(
                edge_index,
                np.zeros(3, dtype=bool),
                np.full(3, -1, dtype=np.int64),
                3,
            )
        self.assertIn("cycle", str(ctx.exception))

    def test_self_loop_is_detected_as_a_cycle(self):
        from baselines.deepgate4.aig_features import forward_levels

        edge_index = torch.tensor([[0, 1], [1, 1]])
        with self.assertRaises(ValueError):
            forward_levels(
                edge_index,
                np.zeros(2, dtype=bool),
                np.full(2, -1, dtype=np.int64),
                2,
            )

    def test_disconnected_and_dangling_nodes_do_not_trip_cycle_detection(self):
        """The complement of the two tests above: a valid DAG with isolated
        nodes and a dangling branch must NOT raise."""
        from baselines.deepgate4.aig_features import forward_levels

        # 0->1->2 live; 3 isolated; 4->5 dangling branch off 0.
        edge_index = torch.tensor([[0, 1, 0, 4], [1, 2, 4, 5]])
        level = forward_levels(
            edge_index,
            np.zeros(4, dtype=bool),
            np.full(6, -1, dtype=np.int64),
            6,
        )
        src, dst = edge_index
        self.assertTrue(bool((level[src.numpy()] < level[dst.numpy()]).all()))
        self.assertEqual(int(level[3]), 0)  # isolated node

    def test_primary_inputs_are_level_zero(self):
        g = to_deepgate_graph(_tiny_aig(), num_hops=3)
        for pi in (0, 1, 2):
            self.assertEqual(int(g.forward_level[pi]), 0)


class TestOutDegrees(unittest.TestCase):
    """Paper Eq. 2's OutAND / OutNOT, the structural encoding's inputs."""

    def test_counts_match_successor_types(self):
        data = _tiny_aig(inverted_mask=(False, True, False, False, False))
        gate, expanded, num_nodes, _ = expand_not_nodes(
            data.x, data.edge_index, data.edge_attr
        )
        out_and, out_not = node_out_degrees(gate, expanded, num_nodes)
        # node 1 drives AND 3 and AND 4 directly.
        self.assertEqual(int(out_and[1]), 2)
        self.assertEqual(int(out_not[1]), 0)
        # node 2's only edge was inverted, so it now drives its NOT node.
        self.assertEqual(int(out_and[2]), 0)
        self.assertEqual(int(out_not[2]), 1)
        # node 4 drives the PO, which maps to the NOT type.
        self.assertEqual(int(out_not[4]), 1)

    def test_counts_are_clamped_to_the_embedding_table(self):
        """Upstream's nn.Embedding(5000, ...) would index out of bounds on a
        high-fanout net; this dataset reaches 366k-gate AIGs."""
        fanout = OUT_DEGREE_TABLE_SIZE + 50
        x = torch.cat(
            [torch.tensor([[0, 1.0, 0, 0]]), torch.tensor([[0, 0, 1.0, 0]]).repeat(fanout, 1)]
        )
        edge_index = torch.stack(
            [torch.zeros(fanout, dtype=torch.long), torch.arange(1, fanout + 1)]
        )
        edge_attr = torch.tensor([[1.0, 0.0]]).repeat(fanout, 1)
        gate, expanded, num_nodes, _ = expand_not_nodes(x, edge_index, edge_attr)
        out_and, _ = node_out_degrees(gate, expanded, num_nodes)
        self.assertEqual(int(out_and[0]), OUT_DEGREE_TABLE_SIZE - 1)
        self.assertLess(int(out_and.max()), OUT_DEGREE_TABLE_SIZE)


class TestVirtualEdges(unittest.TestCase):
    """Paper Section 3.5's `E_bar = {(u, v) : u <=_k v}`."""

    def test_matches_brute_force_reachability(self):
        torch.manual_seed(0)
        g = to_deepgate_graph(_tiny_aig(), num_hops=3, symmetric=False)
        expected = _brute_force_reachable(g.edge_index, g.num_nodes, 3)
        got = set(map(tuple, g.global_virtual_edge.T.tolist()))
        self.assertEqual(got, expected)

    def test_radius_is_monotonic_and_saturates(self):
        """More hops can only add pairs, and once k exceeds the expanded
        graph's longest path the set stops growing."""
        expanded = to_deepgate_graph(_tiny_aig(), num_hops=1)
        edge_index = expanded.edge_index.numpy()
        num_nodes = expanded.num_nodes

        counts = [
            virtual_edges(edge_index, num_nodes, k, symmetric=False).shape[1]
            for k in (1, 2, 3, 4, 8, 16)
        ]
        self.assertEqual(counts, sorted(counts))
        self.assertLess(counts[0], counts[-1])
        # k=8 and k=16 both exceed this graph's longest path, so both saturate.
        self.assertEqual(counts[-1], counts[-2])

    def test_symmetric_doubles_and_reverses(self):
        data = _tiny_aig()
        one = to_deepgate_graph(data, num_hops=3, symmetric=False).global_virtual_edge
        two = to_deepgate_graph(data, num_hops=3, symmetric=True).global_virtual_edge
        self.assertEqual(two.shape[1], 2 * one.shape[1])
        fwd = set(map(tuple, one.T.tolist()))
        both = set(map(tuple, two.T.tolist()))
        self.assertEqual(both, fwd | {(v, u) for u, v in fwd})

    def test_default_is_one_way_matching_paper_and_upstream(self):
        """Both paper Section 3.5 (`E_bar = {(u,v) : u <=_k v}`) and the
        released code emit ancestor -> descendant only.

        The code is easy to misread as symmetric: `get_fanin_fanout_cone`
        computes BOTH cones, but marks the fanin cone 1 and the fanout cone 2
        (data_preparation.py:222, :233), and the consumer keeps
        `argwhere(ff_cone.T == 1)` (line 523) -- so fanout pairs never become
        edges. This pins the default, because getting it wrong silently
        doubles the baseline's dominant memory term AND departs from both
        sources.
        """
        import inspect

        from baselines.deepgate4 import aig_features

        for fn in (aig_features.virtual_edges, aig_features.to_deepgate_graph):
            self.assertIs(
                inspect.signature(fn).parameters["symmetric"].default, False,
                f"{fn.__name__} must default to one-way virtual edges",
            )
        self.assertIs(
            inspect.signature(aig_features.DeepGateGraphAdapter.__init__)
            .parameters["symmetric"].default,
            False,
        )

        # Every emitted edge must run strictly downstream: ancestor first.
        g = to_deepgate_graph(_tiny_aig(), num_hops=4)
        src, dst = g.global_virtual_edge
        self.assertGreater(src.numel(), 0)
        self.assertTrue(
            torch.all(g.forward_level[src] < g.forward_level[dst]),
            "virtual edges must point from ancestor to descendant",
        )

    def test_no_self_loops_emitted(self):
        """GATConv adds its own after calling remove_self_loops, so emitting
        them here (as upstream's `ff_cone + eye` does) would be redundant."""
        g = to_deepgate_graph(_tiny_aig(), num_hops=4)
        src, dst = g.global_virtual_edge
        self.assertFalse(bool((src == dst).any()))

    def test_empty_edge_index(self):
        got = virtual_edges(torch.zeros((2, 0), dtype=torch.long).numpy(), 3, 8)
        self.assertEqual(tuple(got.shape), (2, 0))


class TestDegenerateInputs(unittest.TestCase):
    """Shapes the node-budget sampler and the dataset can genuinely produce."""

    def test_single_node_no_edges(self):
        data = Data(
            x=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 2)),
            level=torch.tensor([[0.0]]),
            y=torch.tensor([[0.5]]),
        )
        data.num_nodes = 1
        data = get_pe_transform("level", attr_name="pos_enc")(data)
        g = to_deepgate_graph(data, num_hops=8)

        self.assertEqual(g.num_nodes, 1)
        self.assertEqual(tuple(g.global_virtual_edge.shape), (2, 0))
        self.assertEqual(g.forward_level.tolist(), [0])
        model = DeepGate4GraphRegressor(hidden=16, num_tf_layers=2, heads=2, max_level=64)
        self.assertEqual(tuple(model(collate_deepgate_batch([g])).shape), (1, 1))

    def test_every_edge_inverted(self):
        data = Data(
            x=torch.tensor([[0, 1.0, 0, 0], [0, 0, 1.0, 0]]),
            edge_index=torch.tensor([[0], [1]]),
            edge_attr=torch.tensor([[0.0, 1.0]]),
            level=torch.tensor([[0.0], [1.0]]),
            y=torch.tensor([[0.3]]),
        )
        data.num_nodes = 2
        g = to_deepgate_graph(get_pe_transform("level", "pos_enc")(data), num_hops=8)

        self.assertEqual(g.gate.view(-1).tolist(), [GATE_PI, GATE_AND, GATE_NOT])
        src, dst = g.edge_index
        self.assertTrue(torch.all(g.forward_level[src] < g.forward_level[dst]))

    def test_no_level_attribute_needed(self):
        """Levels come from the edge list, so a graph carrying neither `level`
        nor `pos_enc` converts fine. This used to raise; see
        TestLevelIndependence for why the dependency was removed."""
        data = Data(
            x=torch.tensor([[0, 1.0, 0, 0], [0, 0, 1.0, 0]]),
            edge_index=torch.tensor([[0], [1]]),
            edge_attr=torch.tensor([[1.0, 0.0]]),
            y=torch.tensor([[0.3]]),
        )
        data.num_nodes = 2
        g = to_deepgate_graph(data, num_hops=3)
        self.assertEqual(g.forward_level.tolist(), [0, 1])


class TestLevelIndependence(unittest.TestCase):
    """`to_deepgate_graph` must derive levels from the edge list alone.

    It deliberately ignores the cached `pos_enc`, for two independent reasons
    that each caused a real bug during development:
      1. `pos_enc` holds log1p(level), not level -- `get_pe_transform('level')`
         builds `ExtractPrecomputedPE(discrete=False)`, whose branch applies
         `log1p_()`. Truncating that to int collapsed a depth-25k circuit into
         ~10 levels.
      2. Even unscaled, the cached level is not topological on circuits with
         dangling logic (see the aigverse test below).
    """

    @staticmethod
    def _chain(depth):
        """PI -> AND -> AND -> ... , so the true level of node i is exactly i."""
        n = depth + 1
        x = torch.zeros(n, 4)
        x[0, 1] = 1.0
        x[1:, 2] = 1.0
        data = Data(
            x=x,
            edge_index=torch.stack([torch.arange(n - 1), torch.arange(1, n)]),
            edge_attr=torch.tensor([[1.0, 0.0]]).repeat(n - 1, 1),
            level=torch.arange(n, dtype=torch.float32).view(-1, 1),
            y=torch.tensor([[0.5]]),
        )
        data.num_nodes = n
        return data

    def test_pos_enc_is_log1p_not_level(self):
        """Pins the trap that motivates all of this. If the transform ever
        stops rescaling, this fails and the docstrings need revisiting."""
        data = get_pe_transform("level", "pos_enc")(self._chain(8))
        self.assertTrue(
            torch.allclose(
                data.pos_enc.view(-1),
                torch.log1p(torch.arange(9, dtype=torch.float32)),
                atol=1e-6,
            )
        )

    def test_works_with_no_level_or_pos_enc_at_all(self):
        """The strongest form of the guarantee: strip both attributes and the
        conversion must still produce exact longest-path levels."""
        for depth in (1, 8, 100, 500):
            with self.subTest(depth=depth):
                data = self._chain(depth)
                del data.level
                self.assertIsNone(getattr(data, "pos_enc", None))
                g = to_deepgate_graph(data, num_hops=2)
                self.assertEqual(
                    g.forward_level.tolist(), list(range(depth + 1))
                )

    def test_a_wrong_cached_level_cannot_affect_the_result(self):
        """Feed a deliberately corrupted `level` and require identical output."""
        good = self._chain(40)
        bad = self._chain(40)
        bad.level = torch.zeros_like(bad.level)  # every node claims level 0
        self.assertTrue(
            torch.equal(
                to_deepgate_graph(good, num_hops=2).forward_level,
                to_deepgate_graph(bad, num_hops=2).forward_level,
            )
        )

    def test_check_topological_rejects_a_rescaled_level(self):
        """The postcondition guard that would catch a regression here."""
        chain = self._chain(30)
        rescaled = torch.log1p(chain.level).view(-1).numpy().astype("int64")
        with self.assertRaises(ValueError) as ctx:
            check_topological(chain.edge_index, rescaled, chain.num_nodes)
        self.assertIn("topological", str(ctx.exception))

    def test_dangling_logic_still_yields_valid_levels(self):
        """The cached level is NOT reliably topological, so `forward_levels`
        must not depend on it being so.

        `aigverse`'s `DepthAig.level()` -- which data/data_utils.py uses to
        populate `level` -- returns 0 for every node outside some primary
        output's fanin cone. A dangling AND gate and its dangling fanins
        therefore all report level 0, and `level[src] < level[dst]` fails on
        those edges. Reproduced here against the real library rather than a
        hand-made approximation, because the whole point is that the upstream
        data source behaves this way.
        """
        aigverse_networks = __import__(
            "aigverse.networks", fromlist=["Aig", "DepthAig"]
        )
        aig = aigverse_networks.Aig()
        a, b = aig.create_pi(), aig.create_pi()
        live = aig.create_and(a, b)
        aig.create_po(live)
        d1 = aig.create_and(a, ~b)      # dangling
        d2 = aig.create_and(d1, live)   # dangling, deeper
        aig.create_and(d2, d1)          # dangling, deeper still

        depth = aigverse_networks.DepthAig(aig)
        cached = np.array([depth.level(n) for n in aig.nodes()], dtype=np.int64)

        src, dst, inv = [], [], []
        for n in aig.nodes():
            for f in aig.fanins(n):
                src.append(f.index)
                dst.append(n)
                inv.append(bool(f.complement))
        num_base = len(list(aig.nodes()))

        # Precondition of this test: the cached level really is broken here.
        s, d = np.array(src), np.array(dst)
        self.assertTrue(
            (cached[s] >= cached[d]).any(),
            "expected DepthAig to report level 0 for dangling nodes",
        )

        x = torch.zeros(num_base, 4)
        for n in aig.nodes():
            col = 0 if aig.is_constant(n) else (1 if aig.is_pi(n) else 2)
            x[n, col] = 1.0
        edge_attr = torch.tensor(
            [[0.0, 1.0] if i else [1.0, 0.0] for i in inv]
        )
        data = Data(
            x=x,
            edge_index=torch.tensor([src, dst]),
            edge_attr=edge_attr,
            level=torch.from_numpy(cached).float().view(-1, 1),
            y=torch.tensor([[0.5]]),
        )
        data.num_nodes = num_base

        g = to_deepgate_graph(data, num_hops=3)
        gs, gd = g.edge_index
        self.assertTrue(
            torch.all(g.forward_level[gs] < g.forward_level[gd]),
            "forward_levels produced a non-topological result on dangling logic",
        )


class TestBatching(unittest.TestCase):
    """`DeepGateData.__inc__`/`__cat_dim__` -- without them `Batch` would
    concatenate `global_virtual_edge` on the wrong axis and leave `nodes`
    un-offset, silently mixing graphs together."""

    def test_indices_are_offset_into_the_batch(self):
        graphs = [to_deepgate_graph(_tiny_aig(y=v), num_hops=3) for v in (0.1, 0.2, 0.9)]
        batch = collate_deepgate_batch(graphs)

        self.assertEqual(batch.num_nodes, sum(g.num_nodes for g in graphs))
        self.assertEqual(batch.nodes.tolist(), list(range(batch.num_nodes)))
        self.assertEqual(batch.forward_index.tolist(), list(range(batch.num_nodes)))
        self.assertEqual(batch.global_virtual_edge.shape[0], 2)
        self.assertEqual(
            batch.global_virtual_edge.shape[1],
            sum(g.global_virtual_edge.shape[1] for g in graphs),
        )
        self.assertLess(int(batch.global_virtual_edge.max()), batch.num_nodes)
        self.assertLess(int(batch.edge_index.max()), batch.num_nodes)
        self.assertTrue(
            torch.allclose(batch.y.view(-1), torch.tensor([0.1, 0.2, 0.9]), atol=1e-6)
        )

    def test_virtual_edges_never_cross_graphs(self):
        graphs = [to_deepgate_graph(_tiny_aig(), num_hops=4) for _ in range(3)]
        batch = collate_deepgate_batch(graphs)
        src, dst = batch.global_virtual_edge
        self.assertTrue(torch.all(batch.batch[src] == batch.batch[dst]))

    def test_adapter_is_index_aligned_with_its_base_dataset(self):
        """train_baseline.py builds the node-budget plan from the WRAPPED
        dataset's node counts and indexes the adapter with it."""
        base = [_tiny_aig(y=0.1), _tiny_aig(y=0.7)]
        adapter = DeepGateGraphAdapter(base, num_hops=3)
        self.assertEqual(len(adapter), len(base))
        self.assertAlmostEqual(float(adapter[1].y), 0.7)


class TestDeepGate4GraphRegressor(unittest.TestCase):
    def _model(self, **kw):
        kw.setdefault("hidden", 16)
        kw.setdefault("num_tf_layers", 2)
        kw.setdefault("heads", 2)
        kw.setdefault("max_level", 64)
        return DeepGate4GraphRegressor(**kw)

    def _batch(self, n=3, **kw):
        return collate_deepgate_batch(
            [to_deepgate_graph(_tiny_aig(y=0.1 * i, **kw), num_hops=3) for i in range(n)]
        )

    def test_output_shape_and_range(self):
        model = self._model()
        out = model(self._batch(n=3))
        self.assertEqual(tuple(out.shape), (3, 1))
        self.assertTrue(bool(((out >= 0) & (out <= 1)).all()))

    def test_single_graph_batch(self):
        """Node-budget batching produces singleton batches for large graphs."""
        out = self._model()(self._batch(n=1))
        self.assertEqual(tuple(out.shape), (1, 1))

    def test_not_pathway_receives_gradient(self):
        """The whole point of NOT-node expansion. Without it, `gate == 2` never
        occurs, half the tokenizer never fires, and inversion -- which nothing
        else in DeepGate4 reads -- becomes invisible to the model.

        Checks the NOT GRUs specifically. The NOT *attention* projections
        (`aggr_not_*.msg_q/msg_k`) legitimately get zero gradient: a NOT gate
        has in-degree 1, so their softmax is identically 1. Upstream behaves
        the same way, and the paper says so in Section 3.7.
        """
        model = self._model()
        batch = self._batch(n=2, inverted_mask=(True, True, False, True, False))
        loss = torch.nn.functional.mse_loss(model(batch).squeeze(-1), batch.y.view(-1))
        model.zero_grad()
        loss.backward()

        named = dict(model.named_parameters())
        for name in (
            "tokenizer.update_not_strc.weight_ih_l0",
            "tokenizer.update_not_func.weight_ih_l0",
        ):
            grad = named[name].grad
            self.assertIsNotNone(grad, f"{name} got no gradient")
            self.assertGreater(float(grad.abs().sum()), 0.0, f"{name} gradient is zero")

    def test_inversion_changes_the_prediction(self):
        """End-to-end consequence of the above: two graphs identical except for
        edge inversion must not produce the same output."""
        torch.manual_seed(0)
        model = self._model().eval()
        with torch.no_grad():
            plain = model(
                collate_deepgate_batch(
                    [to_deepgate_graph(_tiny_aig(inverted_mask=(False,) * 5), num_hops=3)]
                )
            )
            inverted = model(
                collate_deepgate_batch(
                    [to_deepgate_graph(_tiny_aig(inverted_mask=(True,) * 5), num_hops=3)]
                )
            )
        self.assertGreater(float((plain - inverted).abs().max()), 1e-6)

    def test_structural_encoding_is_wired_in(self):
        """Paper Eq. 2: SE(v) = Emb_l(level) + Emb_and(OutAND) + Emb_not(OutNOT)."""
        model = self._model()
        batch = self._batch(n=2)
        loss = model(batch).sum()
        model.zero_grad()
        loss.backward()
        for name in ("out_and.weight", "out_not.weight", "abs_pe_embedding.weight"):
            grad = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(grad, f"{name} got no gradient")
            self.assertGreater(float(grad.abs().sum()), 0.0, f"{name} gradient is zero")

    def test_gradient_checkpointing_is_numerically_transparent(self):
        """It is on by default purely for memory (12 GATConv layers over ~7.36M
        virtual edges retain ~45 GB for one average graph). It must not change
        the forward value or a single gradient.

        Run at the SHIPPED dropout values, not with dropout disabled. Dropout
        is the one mechanism that could make checkpointing non-transparent --
        the backward pass has to replay the same mask, which relies on
        `use_reentrant=False` preserving RNG state. Testing at 0.0 would skip
        exactly the case worth testing.
        """
        for tf_dropout, head_dropout in ((0.0, 0.0), (0.1, 0.3)):
            with self.subTest(tf_dropout=tf_dropout, head_dropout=head_dropout):
                torch.manual_seed(0)
                plain = self._model(
                    num_tf_layers=4,
                    tf_dropout=tf_dropout,
                    head_dropout=head_dropout,
                    gradient_checkpointing=False,
                )
                ckpt = self._model(
                    num_tf_layers=4,
                    tf_dropout=tf_dropout,
                    head_dropout=head_dropout,
                    gradient_checkpointing=True,
                )
                ckpt.load_state_dict(copy.deepcopy(plain.state_dict()))
                batch = self._batch(n=3)

                outs, grads = [], []
                for model in (plain, ckpt):
                    model.train()
                    torch.manual_seed(1234)
                    out = model(batch)
                    model.zero_grad()
                    torch.nn.functional.mse_loss(
                        out.squeeze(-1), batch.y.view(-1)
                    ).backward()
                    outs.append(out.detach().clone())
                    grads.append(
                        {n: (p.grad.clone() if p.grad is not None else None)
                         for n, p in model.named_parameters()}
                    )

                self.assertTrue(torch.allclose(outs[0], outs[1], atol=1e-6))
                for name, ref in grads[0].items():
                    got = grads[1][name]
                    self.assertEqual(
                        ref is None, got is None, f"grad presence differs for {name}"
                    )
                    if ref is not None:
                        self.assertTrue(
                            torch.allclose(ref, got, atol=1e-5),
                            f"gradient differs for {name}",
                        )

    def test_pretrained_tokenizer_rejects_mismatched_hidden_dim(self):
        """`DeepGate2.load` skips shape-mismatched tensors with a printed
        warning, so a silent no-op is the failure mode this guards against."""
        model = self._model(hidden=32)
        with self.assertRaises(ValueError):
            model.load_pretrained_tokenizer("/nonexistent/model_last.pth")


class TestPublishedHyperparameters(unittest.TestCase):
    """Pins every value Zheng et al. ICLR'25 publishes, so a future edit that
    silently drifts off the paper fails here rather than in a results table.

    Sources: Section 4.1 ("we set k to 8 and delta to 6. The dimensions of both
    the structural and functional embedding are set to 128. The depth of Sparse
    Transformer is 12 ... All training task heads are 3-layer multilayer
    perceptrons (MLPs). We train all models for 200 epochs ... We utilize the
    Adam optimizer with a learning rate of 10^-4"), plus upstream's argparse
    defaults (`--mlp_hidden 128 --mlp_layer 3`) and `run/train_large.sh`
    (`--hidden 128 --lr 1e-4 --epoch 200`).
    """

    def test_paper_constants(self):
        from baselines.deepgate4 import aig_features, regressor

        self.assertEqual(regressor.DEFAULT_HIDDEN_DIM, 128)
        self.assertEqual(regressor.DEFAULT_NUM_TF_LAYERS, 12)
        self.assertEqual(regressor.DEFAULT_LR, 1e-4)
        self.assertEqual(regressor.DEFAULT_NUM_EPOCHS, 200)
        self.assertEqual(regressor.DEFAULT_MLP_HIDDEN, 128)
        self.assertEqual(regressor.DEFAULT_MLP_LAYER, 3)
        self.assertEqual(aig_features.DEFAULT_NUM_HOPS, 8)

    def test_upstream_transformer_defaults(self):
        """heads/concat/dropout are NOT in the paper -- they are
        `Sparse_Transformer.__init__`'s defaults, which upstream never
        overrides (dg4.py passes only `args` and `hidden`)."""
        import inspect

        from baselines.deepgate4.plain_tf_linear import Sparse_Transformer
        from baselines.deepgate4.regressor import DEFAULT_HEADS, DEFAULT_TF_DROPOUT

        sig = inspect.signature(Sparse_Transformer.__init__).parameters
        self.assertEqual(sig["num_layers"].default, 12)
        self.assertEqual(sig["heads"].default, DEFAULT_HEADS)
        self.assertEqual(sig["dropout"].default, DEFAULT_TF_DROPOUT)
        self.assertIs(sig["concat"].default, True)

    def test_readout_head_matches_published_shape(self):
        """3 layers of width 128, ReLU, upstream's MLP class -- not a
        hand-rolled head."""
        from baselines.deepgate4.mlp import MLP

        model = DeepGate4GraphRegressor(hidden=128)
        head = model.regression_head
        self.assertIsInstance(head, MLP)

        linears = [m for m in head.fc if isinstance(m, torch.nn.Linear)]
        self.assertEqual(len(linears), 3)
        self.assertEqual(linears[0].in_features, 256)  # cat([hf, hs])
        self.assertEqual(linears[0].out_features, 128)
        self.assertEqual(linears[-1].out_features, 1)
        self.assertTrue(any(isinstance(m, torch.nn.ReLU) for m in head.fc))
        self.assertTrue(any(isinstance(m, torch.nn.Dropout) for m in head.fc))

    def test_default_head_has_no_batchnorm(self):
        """Documented deviation from upstream's `--norm_layer batchnorm`.
        Justified by the next test."""
        model = DeepGate4GraphRegressor(hidden=32)
        self.assertFalse(
            any(isinstance(m, torch.nn.BatchNorm1d) for m in model.regression_head.fc)
        )

    def test_batchnorm_head_would_be_constant_at_graph_batch_size_one(self):
        """Why `head_norm_layer` defaults to None rather than upstream's value.

        `MLP.forward` pads a 1-row input with `x.repeat(2, 1)`. BatchNorm1d then
        sees two identical rows -> variance 0 -> every input normalises to 0, so
        the head emits its bias no matter which circuit went in. The node budget
        puts ~1 graph in a micro-batch, making that the normal case, not an edge
        case.
        """
        from baselines.deepgate4.mlp import MLP

        torch.manual_seed(0)
        head = MLP(
            dim_in=8, dim_hidden=8, dim_pred=1, num_layer=3,
            norm_layer="batchnorm", act_layer="relu", p_drop=0.0,
        ).train()

        a = head(torch.randn(1, 8))
        b = head(torch.randn(1, 8))
        self.assertAlmostEqual(float(a), float(b), places=5,
                               msg="expected BatchNorm to collapse a 1-row batch")

        # The shipped head (no norm) must NOT have that property.
        plain = MLP(
            dim_in=8, dim_hidden=8, dim_pred=1, num_layer=3,
            norm_layer=None, act_layer="relu", p_drop=0.0,
        ).train()
        self.assertNotAlmostEqual(
            float(plain(torch.randn(1, 8))), float(plain(torch.randn(1, 8))), places=5
        )


class TestDataAdaptation(unittest.TestCase):
    """The converted graph must satisfy the invariants DeepGate4's own code
    assumes about an AIG, since this project's AIGs are stored differently
    (4 node types, inversion on the edge) from the 3-type explicit-NOT form
    the model was written against."""

    def test_gate_type_agrees_with_in_degree(self):
        """Paper Section 3.2: "The gate type can be easily identified by its
        in-degree: the in-degree of a PI is 0, the in-degree of an AND gate is
        2, and the in-degree of a NOT gate is 1."

        This is the single sharpest check that the data was adapted correctly:
        if the node-type mapping or the NOT expansion were wrong, gate labels
        and in-degrees would disagree.
        """
        for mask in [(False,) * 5, (True,) * 5, (False, True, False, True, False)]:
            with self.subTest(mask=mask):
                g = to_deepgate_graph(_tiny_aig(inverted_mask=mask), num_hops=3)
                in_degree = torch.zeros(g.num_nodes, dtype=torch.long)
                in_degree.scatter_add_(
                    0, g.edge_index[1], torch.ones_like(g.edge_index[1])
                )
                gate = g.gate.view(-1)
                self.assertTrue(torch.all(in_degree[gate == GATE_PI] == 0))
                self.assertTrue(torch.all(in_degree[gate == GATE_AND] == 2))
                self.assertTrue(torch.all(in_degree[gate == GATE_NOT] == 1))

    def test_edge_direction_is_fanin_to_node(self):
        """Both this project (data/data_utils.py appends `(fanin, node)`) and
        upstream (data_preparation.py appends `[fanin_idx, idx]`) put the fanin
        in row 0. `get_slices` slices on `forward_level[edge_index[1]]`, so a
        flipped convention would silently mis-level every gate."""
        g = to_deepgate_graph(_tiny_aig(inverted_mask=(False,) * 5), num_hops=3)
        src, dst = g.edge_index
        self.assertTrue(torch.all(g.forward_level[src] < g.forward_level[dst]))

    def test_every_model_input_attribute_is_present_and_typed(self):
        """dg2.get_slices / DeepGate2.forward / Sparse_Transformer.forward read
        exactly these. A missing or wrongly-typed one is an AttributeError or a
        silent dtype bug deep inside vendored code."""
        g = to_deepgate_graph(_tiny_aig(), num_hops=3)
        for attr in (
            "gate", "edge_index", "forward_level", "forward_index",
            "nodes", "out_and", "out_not", "global_virtual_edge",
        ):
            self.assertTrue(hasattr(g, attr), f"missing {attr}")
            self.assertEqual(getattr(g, attr).dtype, torch.long, f"{attr} dtype")
        # gate must be [N, 1]: get_slices does `(G.gate == 1).squeeze(1)`.
        self.assertEqual(g.gate.dim(), 2)
        self.assertEqual(g.gate.shape[1], 1)
        # out_and/out_not index nn.Embedding tables, so must be in range.
        self.assertLess(int(g.out_and.max()), OUT_DEGREE_TABLE_SIZE)
        self.assertLess(int(g.out_not.max()), OUT_DEGREE_TABLE_SIZE)

    def test_sinusoidal_level_table_covers_this_dataset(self):
        """Upstream sizes it at 10,000 rows; config.MAX_DEPTH is 24,972 and
        NOT expansion can nearly double a path, so an out-of-bounds index would
        be a hard crash on deep circuits."""
        import config as project_config
        from baselines.deepgate4.regressor import DEFAULT_MAX_LEVEL

        self.assertGreaterEqual(DEFAULT_MAX_LEVEL, 2 * project_config.MAX_DEPTH)
        model = DeepGate4GraphRegressor(hidden=8, num_tf_layers=1, heads=2)
        self.assertGreater(model.sinu_pe.shape[0], 2 * project_config.MAX_DEPTH)


class TestEncoderComposition(unittest.TestCase):
    """Covers how the encoder's pieces are combined, which the shape/gradient
    tests above cannot see.

    Both cases below were found by mutation testing: dropping the residual, and
    pooling only `hf`, each left the entire suite green. They are exactly the
    kind of fidelity error that produces plausible numbers from the wrong
    model, so they get direct tests rather than indirect ones.

    Gradient-based tests specifically CANNOT catch the `hs` case: the tokenizer
    feeds `node_state = cat([hs, hf])` into `aggr_and_func`, so `hs` influences
    `hf` regardless, and every `hs`-side parameter keeps receiving gradient even
    when the readout ignores `hs` entirely. Both tests therefore stub the
    transformer's output and check what actually reaches the prediction.
    """

    class _StubTransformer(torch.nn.Module):
        """Returns constants in place of the sparse transformer's output.

        Must be an `nn.Module`: `transformer` is a registered child, and
        `nn.Module.__setattr__` rejects a plain function for such a name.
        """

        def __init__(self, hf_const=0.0, hs_const=0.0):
            super().__init__()
            self.hf_const = hf_const
            self.hs_const = hs_const

        def forward(self, g, hf, hs, mk):
            return (
                torch.full_like(hf, self.hf_const),
                torch.full_like(hs, self.hs_const),
            )

    @staticmethod
    def _model():
        return DeepGate4GraphRegressor(
            hidden=16, num_tf_layers=2, heads=2, max_level=64,
            tf_dropout=0.0, head_dropout=0.0,
        ).eval()

    def test_transformer_output_is_added_not_replaced(self):
        """`hf = hf + hf_tf`, not `hf = hf_tf` (upstream dg4.py:391-392).

        With the transformer stubbed to return zeros, a residual keeps the
        tokenizer's embeddings alive, so two structurally different circuits
        still get different predictions. Without it every graph collapses to
        the same vector.
        """
        model = self._model()
        model.transformer = self._StubTransformer(0.0, 0.0)

        a = collate_deepgate_batch(
            [to_deepgate_graph(_tiny_aig(inverted_mask=(False,) * 5), num_hops=3)]
        )
        b = collate_deepgate_batch(
            [to_deepgate_graph(_tiny_aig(inverted_mask=(True,) * 5), num_hops=3)]
        )
        with torch.no_grad():
            self.assertGreater(
                float((model(a) - model(b)).abs().max()), 1e-6,
                "transformer output replaces the tokenizer's instead of adding "
                "to it -- the residual is missing",
            )

    def test_structural_embedding_reaches_the_readout(self):
        """The readout pools `cat([hf, hs])`; `hs` must not be dropped.

        Stubs the transformer to emit a constant into the `hs` channel only.
        If the readout consumed `hf` alone, varying that constant could not
        change the prediction.
        """
        model = self._model()
        batch = collate_deepgate_batch(
            [to_deepgate_graph(_tiny_aig(), num_hops=3)]
        )

        with torch.no_grad():
            model.transformer = self._StubTransformer(hs_const=0.0)
            zero = model(batch)
            model.transformer = self._StubTransformer(hs_const=5.0)
            five = model(batch)
        self.assertGreater(
            float((zero - five).abs().max()), 1e-6,
            "changing the structural embedding did not change the prediction "
            "-- the readout is ignoring hs",
        )

    def test_transformer_receives_hf_first_then_hs(self):
        """Argument ORDER into the sparse transformer, which the stubbing tests
        above cannot see.

        `Sparse_Transformer.forward(g, hf, hs, mk)` concatenates its two inputs
        and chunks the result back apart in the same order, so swapping the
        call's arguments silently routes the structural embedding through the
        functional channel and vice versa. Everything still runs and still
        trains -- it is just no longer DeepGate4. Caught by recomputing the
        tokenizer's output independently and comparing.

        NOT tested, deliberately: the ORDER of `cat([hf, hs])` in the readout.
        That one feeds a learned `Linear(2*hidden, hidden)`, which can
        represent either arrangement by permuting its own columns, so the two
        are the same model -- a mutation there is benign rather than a defect.
        """
        model = self._model()
        batch = collate_deepgate_batch(
            [to_deepgate_graph(_tiny_aig(), num_hops=3)]
        )

        captured = {}

        class _Recorder(torch.nn.Module):
            def forward(self, g, hf, hs, mk):
                captured["hf"] = hf.detach().clone()
                captured["hs"] = hs.detach().clone()
                return torch.zeros_like(hf), torch.zeros_like(hs)

        model.transformer = _Recorder()
        with torch.no_grad():
            model(batch)

        # Independently reproduce what the tokenizer should have handed over.
        with torch.no_grad():
            mk = torch.zeros(int(batch.num_nodes))
            level = batch.forward_level.clamp(max=model.max_level)
            abs_pe = model.abs_pe_embedding(model.sinu_pe[level])
            init_lhs = (
                abs_pe + model.out_not(batch.out_not) + model.out_and(batch.out_and)
            )
            hs_ref, hf_ref = model.tokenizer(batch, mk=mk, lhs=init_lhs)

        # Guard against a vacuous assertion: hf and hs must actually differ.
        self.assertFalse(torch.allclose(hf_ref, hs_ref, atol=1e-4))
        self.assertTrue(
            torch.allclose(captured["hf"], hf_ref, atol=1e-6),
            "transformer's 1st tensor argument is not the functional embedding",
        )
        self.assertTrue(
            torch.allclose(captured["hs"], hs_ref, atol=1e-6),
            "transformer's 2nd tensor argument is not the structural embedding",
        )

    def test_checkpointing_is_skipped_outside_training(self):
        """`CheckpointedSparseTransformer` guards on `self.training and
        torch.is_grad_enabled()`. Under inference the recompute buys nothing,
        so dropping the guard would silently pay for it on every eval batch."""
        import torch.utils.checkpoint as ckpt

        calls = []
        original = ckpt.checkpoint

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        model = DeepGate4GraphRegressor(
            hidden=16, num_tf_layers=2, heads=2, max_level=64,
            tf_dropout=0.0, head_dropout=0.0, gradient_checkpointing=True,
        )
        batch = collate_deepgate_batch(
            [to_deepgate_graph(_tiny_aig(), num_hops=3)]
        )

        ckpt.checkpoint = counting
        try:
            model.eval()
            with torch.no_grad():
                model(batch)
            self.assertEqual(len(calls), 0, "checkpointing ran during eval")

            model.train()
            model(batch)
            self.assertEqual(
                len(calls), 2, "checkpointing did not run for all training layers"
            )
        finally:
            ckpt.checkpoint = original

    def test_tokenizer_runs_outside_autocast_but_transformer_inside(self):
        """Regression: bf16-mixed AMP crashed on the first training batch.

        The vendored tokenizer writes its GRU output back into the fp32 `hs`/`hf`
        buffers in place (dg2.py:220, :229, :246, :252, :263, :269). Under CUDA
        autocast the GRU returns reduced precision, and index-put demands
        matching dtypes, so training died with

            RuntimeError: Index put requires the source and destination dtypes
            match, got Float for the destination and Half for the source

        The fix draws the autocast boundary in `regressor.forward` rather than
        editing a file PROVENANCE.md records as vendored unmodified.

        This asserts the boundary's PLACEMENT, not the dtype, deliberately: CPU
        autocast does not cover GRU, so a plain autocast forward passes on this
        machine with or without the fix and would be a test with no teeth. The
        transformer half of the assertion matters just as much -- disabling
        autocast for the whole forward would "fix" the crash while silently
        giving up the AMP the memory budget in train_baseline_deepgate4.sh is
        sized around.
        """
        model = self._model()
        batch = collate_deepgate_batch(
            [to_deepgate_graph(_tiny_aig(y=0.1 * i), num_hops=3) for i in range(2)]
        )
        seen = {}

        def spy(name, fn):
            def wrapper(*args, **kwargs):
                seen[name] = torch.is_autocast_enabled("cpu")
                return fn(*args, **kwargs)
            return wrapper

        model.tokenizer.forward = spy("tokenizer", model.tokenizer.forward)
        model.transformer.forward = spy("transformer", model.transformer.forward)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            model(batch)

        self.assertFalse(
            seen["tokenizer"],
            "tokenizer ran under autocast; its in-place GRU writeback will "
            "raise an index-put dtype mismatch on CUDA",
        )
        self.assertTrue(
            seen["transformer"],
            "sparse transformer lost AMP; the node budget assumes bf16 activations",
        )

    def test_functional_embedding_reaches_the_readout(self):
        """Mirror of the above for `hf`, so neither half can be dropped."""
        model = self._model()
        batch = collate_deepgate_batch(
            [to_deepgate_graph(_tiny_aig(), num_hops=3)]
        )

        with torch.no_grad():
            model.transformer = self._StubTransformer(hf_const=0.0)
            zero = model(batch)
            model.transformer = self._StubTransformer(hf_const=5.0)
            five = model(batch)
        self.assertGreater(float((zero - five).abs().max()), 1e-6)


class TestLearning(unittest.TestCase):
    """End-to-end evidence that the whole stack is wired, not merely runnable.

    Most tests here check a component in isolation; this one checks the
    composition. If the structural encoding were dropped, the readout pooled a
    constant, the head degenerated (see the BatchNorm case above), or NOT
    expansion failed to make inversion visible, the model could still produce
    finite losses and correct shapes -- but it could not fit four graphs whose
    ONLY difference is their inversion pattern.
    """

    def test_overfits_four_graphs_differing_only_in_inversion(self):
        specs = [
            ((False,) * 5, 0.05),
            ((True,) * 5, 0.95),
            ((True, False, True, False, True), 0.35),
            ((False, True, False, True, False), 0.75),
        ]
        batch = collate_deepgate_batch(
            [to_deepgate_graph(_tiny_aig(inverted_mask=m, y=y), num_hops=4)
             for m, y in specs]
        )

        torch.manual_seed(0)
        model = DeepGate4GraphRegressor(
            hidden=32, num_tf_layers=2, heads=2, max_level=64,
            tf_dropout=0.0, head_dropout=0.0,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        first = None
        for step in range(120):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(
                model(batch).squeeze(-1), batch.y.view(-1)
            )
            loss.backward()
            opt.step()
            if step == 0:
                first = loss.item()
        last = loss.item()

        self.assertLess(
            last, first * 0.2,
            f"failed to overfit 4 graphs (loss {first:.5f} -> {last:.5f}); "
            "the encoder or readout is probably not wired end to end",
        )

        model.eval()
        with torch.no_grad():
            preds = model(batch).view(-1)
        for pred, (_, target) in zip(preds.tolist(), specs):
            self.assertAlmostEqual(pred, target, places=2)


class TestLightningIntegration(unittest.TestCase):
    def test_one_training_and_validation_step(self):
        pl.seed_everything(0, workers=True)
        model = BaselineRegressionLightningModule(
            DeepGate4GraphRegressor(hidden=16, num_tf_layers=2, heads=2, max_level=64),
            lr=1e-4,
        )
        graphs = [to_deepgate_graph(_tiny_aig(y=0.1 * i), num_hops=3) for i in range(4)]
        loader = torch.utils.data.DataLoader(
            graphs, batch_size=2, collate_fn=collate_deepgate_batch
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
        trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
        self.assertIn("val_loss", trainer.callback_metrics)
        self.assertTrue(torch.isfinite(trainer.callback_metrics["val_loss"]))


if __name__ == "__main__":
    unittest.main()
