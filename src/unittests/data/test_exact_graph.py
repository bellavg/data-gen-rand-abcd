"""Tests for the exact-compression track's data transform.

Tests cover:
- fold_inversions_into_x: inversion-count computation, edge_attr dropped,
  edge_weight defaulted, purity, empty-edge graphs
- apply_exact_merge_map: identity losslessness, representative (not summed)
  x, per-target-member multiplicity (not raw coalesced total), node_size,
  internal-edge counting, validation reuse, purity, heterogeneous clusters
"""
from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from data.exact_graph import apply_exact_merge_map, fold_inversions_into_x


@pytest.fixture
def aig_graph() -> Data:
    """A 6-node AIG: const, 2 PIs, 2 ANDs, 1 PO — one inverted edge (2->3)."""
    x = torch.tensor(
        [
            [1, 0, 0, 0],  # 0: const
            [0, 1, 0, 0],  # 1: PI
            [0, 1, 0, 0],  # 2: PI
            [0, 0, 1, 0],  # 3: AND
            [0, 0, 1, 0],  # 4: AND
            [0, 0, 0, 1],  # 5: PO
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor([[1, 2, 3, 0, 4], [3, 3, 4, 4, 5]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]], dtype=torch.float32
    )
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = 6
    data.num_edges = 5
    return data


@pytest.fixture
def symmetric_graph() -> Data:
    """Two structurally identical PI-pair->AND cones feeding one PO, no
    inversions — the shape used to verify multiplicity arithmetic by hand.

    Class C = {1,2,4,5} (4 PIs), class D = {3,6} (2 ANDs); each D-member has
    exactly 2 class-C fanins, so total coalesced C->D edges = 4 but the
    correct per-target-member multiplicity is 4/2 = 2.
    """
    x = torch.zeros(8, 4, dtype=torch.float32)
    x[0, 0] = 1.0
    x[[1, 2, 4, 5], 1] = 1.0
    x[[3, 6], 2] = 1.0
    x[7, 3] = 1.0
    edge_index = torch.tensor(
        [[1, 2, 4, 5, 3, 6], [3, 3, 6, 6, 7, 7]], dtype=torch.long
    )
    edge_attr = torch.tensor([[1, 0]] * 6, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = 8
    return data


# =====================================================================
# fold_inversions_into_x
# =====================================================================


class TestFoldInversions:
    def test_inversion_counts(self, aig_graph: Data) -> None:
        out = fold_inversions_into_x(aig_graph)

        assert out.x.shape == (6, 5)
        # node3 <- {1(normal), 2(inverted)} -> 1 inverted incoming
        # node4 <- {3(normal), 0(inverted)} -> 1 inverted incoming
        # node5 <- {4(normal)}              -> 0 inverted incoming
        expected_inverted = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        assert torch.equal(out.x[:, 4], expected_inverted)
        assert torch.equal(out.x[:, :4], aig_graph.x)

    def test_edge_attr_dropped_edge_weight_added(self, aig_graph: Data) -> None:
        out = fold_inversions_into_x(aig_graph)
        assert getattr(out, "edge_attr", None) is None
        assert torch.equal(out.edge_weight, torch.ones(5))
        assert torch.equal(out.edge_index, aig_graph.edge_index)

    def test_empty_edge_index(self) -> None:
        data = Data(
            x=torch.eye(4),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 2), dtype=torch.float32),
        )
        out = fold_inversions_into_x(data)
        assert torch.equal(out.x[:, 4], torch.zeros(4))
        assert out.edge_weight.shape == (0,)

    def test_source_graph_not_mutated(self, aig_graph: Data) -> None:
        before_x = aig_graph.x.clone()
        before_attr = aig_graph.edge_attr.clone()
        fold_inversions_into_x(aig_graph)
        assert torch.equal(aig_graph.x, before_x)
        assert torch.equal(aig_graph.edge_attr, before_attr)


# =====================================================================
# apply_exact_merge_map
# =====================================================================


class TestApplyExactMergeMap:
    def test_identity_merge_is_lossless(self, aig_graph: Data) -> None:
        folded = fold_inversions_into_x(aig_graph)
        out = apply_exact_merge_map(folded, torch.arange(6), 6)

        assert torch.equal(out.x, folded.x)  # mean of 1 member == itself
        assert torch.equal(out.node_size, torch.ones(6, 1, dtype=torch.long))
        assert int(out.internal_edges) == 0
        # Every super-edge came from exactly 1 original edge into a
        # size-1 target class, so per-target-member multiplicity is 1.
        assert torch.allclose(out.edge_weight, torch.ones(5))
        assert out.num_nodes == 6

    def test_representative_not_sum(self, symmetric_graph: Data) -> None:
        folded = fold_inversions_into_x(symmetric_graph)
        cluster = torch.tensor([0, 1, 1, 2, 1, 1, 2, 3])  # {1,2,4,5}->1, {3,6}->2
        out = apply_exact_merge_map(folded, cluster, 4)

        # Representative, not sum: a PI-class super-node still reads as one
        # PI (mean of 4 identical one-hots), not "[0,4,0,0,0]".
        assert torch.equal(out.x[1], folded.x[1])
        assert torch.equal(out.x[2], folded.x[3])
        assert torch.equal(out.node_size.flatten(), torch.tensor([1, 4, 2, 1]))

    def test_per_target_member_multiplicity(self, symmetric_graph: Data) -> None:
        # The case verified by hand: 4 original C->D edges, target class D
        # has 2 members, so the correct multiplicity is 4/2 = 2 — not 4.
        folded = fold_inversions_into_x(symmetric_graph)
        cluster = torch.tensor([0, 1, 1, 2, 1, 1, 2, 3])
        out = apply_exact_merge_map(folded, cluster, 4)

        edges = {
            (int(u), int(v)): float(w)
            for u, v, w in zip(*out.edge_index, out.edge_weight, strict=True)
        }
        assert edges[(1, 2)] == pytest.approx(2.0)  # C -> D
        assert edges[(2, 3)] == pytest.approx(2.0)  # D -> PO: 2 edges, target size 1
        assert int(out.internal_edges) == 0

    def test_internal_edges_counted(self, symmetric_graph: Data) -> None:
        folded = fold_inversions_into_x(symmetric_graph)
        # Merge one AND (3) with one of its own fanins (1): creates a
        # self-loop that must be dropped and counted.
        cluster = torch.tensor([0, 1, 2, 1, 3, 4, 5, 6])
        out = apply_exact_merge_map(folded, cluster, 7)
        assert int(out.internal_edges) == 1

    def test_heterogeneous_cluster_averages_gracefully(self, aig_graph: Data) -> None:
        # Not a real WL class (mixes a PI and an AND) -- representative
        # convention degrades to an average rather than crashing.
        folded = fold_inversions_into_x(aig_graph)
        cluster = torch.tensor([0, 1, 2, 1, 3, 4])  # merge node1 (PI) with node3 (AND)
        out = apply_exact_merge_map(folded, cluster, 5)
        assert torch.allclose(out.x[1], (folded.x[1] + folded.x[3]) / 2)

    def test_validation_reused(self, aig_graph: Data) -> None:
        folded = fold_inversions_into_x(aig_graph)
        with pytest.raises(ValueError, match="but the graph has 6 nodes"):
            apply_exact_merge_map(folded, torch.arange(5), 5)
        with pytest.raises(ValueError, match="unused"):
            apply_exact_merge_map(folded, torch.arange(6), 7)

    def test_source_graph_not_mutated(self, symmetric_graph: Data) -> None:
        folded = fold_inversions_into_x(symmetric_graph)
        before_x = folded.x.clone()
        apply_exact_merge_map(folded, torch.tensor([0, 1, 1, 2, 1, 1, 2, 3]), 4)
        assert torch.equal(folded.x, before_x)

    def test_empty_edge_index(self) -> None:
        data = Data(
            x=torch.eye(4),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_weight=torch.empty((0,), dtype=torch.float32),
        )
        out = apply_exact_merge_map(data, torch.arange(4), 4)
        assert out.edge_index.shape == (2, 0)
        assert out.edge_weight.shape == (0,)
