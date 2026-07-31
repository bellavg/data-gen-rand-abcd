"""Tests for the summarization module.

Tests cover:
- apply_merge_map identity losslessness (requirement R1) on both the raw
  schema (level) and the cached schema (pos_enc)
- Every registered method: contiguous cluster ids, acceptance by
  apply_merge_map, determinism, and rejection of invalid parameters
- Per-method behaviour: cone chain/width merging and DAG preservation, LSH
  bucket monotonicity and type purity, spectral ratio targeting and
  eigensolver fallback, ConvMatch twin merging and polarity blindness
- Feature/edge merging: type counts, summed polarity counts on coalesced
  super-edges, level minimum, internal (intra-cluster) edge counting
- Cluster-vector validation: wrong length, out-of-range ids, unused ids
- Purity: the source graph is never mutated
- Dropped attributes (pi_paths, local_sp_sum, edge_weight)
- Serialization round-trip under weights_only=True and PyG batching
- SUMMARIZATION_REGISTRY dispatch via summarize_graph
- The precompute driver: manifest task building, output layout, per-shard
  index emission and merging, resume-skipping, and failure isolation
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

from data.sparsification import _register_pyg_safe_globals
from data.summarization import (
    SUMMARIZATION_REGISTRY,
    apply_merge_map,
    color_refinement,
    cone_coarsening,
    convmatch_coarsening,
    identity_clustering,
    lsh_coarsening,
    spectral_coarsening,
    summarize_graph,
)
from data.summarize_graphs import (
    _NUM_NODES_GLOBAL,
    _shard_index_name,
    build_tasks,
    merge_shard_indexes,
    summarize_from_manifests,
)

# Node/edge counts of the fixture, named so the assertions below read as
# claims about the merge rather than as magic numbers.
N_NODES = 10
N_EDGES = 10
N_PIS = 3
N_POS = 2


# =====================================================================
# FIXTURES
# =====================================================================


@pytest.fixture
def aig_graph() -> Data:
    """A 10-node AIG matching the real schema built by data_utils.

    Node types (one-hot ``x`` [N,4] = [const, PI, AND, PO]):
        0: const   1-3: PI   4-7: AND   8-9: PO

    The type census is 1/3/4/2 — deliberately all different, so a mix-up
    between the columns read for num_pis and num_pos cannot pass unnoticed.

    Edges are fanin → node, with no parallel edges:
        1→4, 2→4, 2→5, 3→5, 4→6, 5→6, 0→7, 6→7, 6→8, 7→9

    ``edge_attr`` [E,2] is the one-hot inverter polarity [normal, inverted]
    and ``level`` is float32 [N,1], both as ``aig_to_pytorch_geometric``
    produces them.
    """
    x = torch.zeros(N_NODES, 4, dtype=torch.float32)
    x[0, 0] = 1.0            # const
    x[1:4, 1] = 1.0          # PI
    x[4:8, 2] = 1.0          # AND
    x[8:10, 3] = 1.0         # PO
    edge_index = torch.tensor(
        [[1, 2, 2, 3, 4, 5, 0, 6, 6, 7], [4, 4, 5, 5, 6, 6, 7, 7, 8, 9]],
        dtype=torch.long,
    )
    edge_attr = torch.tensor(
        [[1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]],
        dtype=torch.float32,
    )
    level = torch.tensor(
        [[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [2.0], [3.0], [3.0], [4.0]]
    )
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, level=level)
    data.num_nodes = N_NODES
    data.num_edges = N_EDGES
    data.num_pis = N_PIS
    data.num_pos = N_POS
    return data


def _edges(data: Data) -> list[tuple[int, int, tuple[float, ...]]]:
    """Edges as a sorted (u, v, attr) list.

    coalesce() sorts row-major while data_utils emits edges grouped by
    target, so the tensors of an identity merge are a permutation of the
    originals rather than equal to them.  Comparing the multiset is the
    meaningful check.
    """
    return sorted(
        (int(u), int(v), tuple(a.tolist()))
        for u, v, a in zip(
            data.edge_index[0], data.edge_index[1], data.edge_attr, strict=True
        )
    )


# =====================================================================
# IDENTITY (REQUIREMENT R1)
# =====================================================================


class TestIdentityMerge:
    def test_identity_merge_is_lossless(self, aig_graph: Data) -> None:
        out = apply_merge_map(aig_graph, torch.arange(N_NODES), N_NODES)

        assert torch.equal(out.x, aig_graph.x)
        assert torch.equal(out.level, aig_graph.level)
        assert _edges(out) == _edges(aig_graph)
        assert int(out.internal_edges) == 0
        assert out.num_nodes == N_NODES
        assert out.num_edges == N_EDGES
        assert out.num_pis == N_PIS
        assert out.num_pos == N_POS

    def test_identity_merge_on_cached_shape(self, aig_graph: Data) -> None:
        # Cache files carry pos_enc = log1p(level) and no level at all.
        pos_enc = torch.log1p(aig_graph.level)
        aig_graph.pos_enc = pos_enc.clone()
        del aig_graph.level

        out = apply_merge_map(aig_graph, torch.arange(N_NODES), N_NODES)

        assert torch.equal(out.pos_enc, pos_enc)
        assert getattr(out, "level", None) is None

    def test_missing_level_and_pos_enc_raises(self, aig_graph: Data) -> None:
        del aig_graph.level
        with pytest.raises(ValueError, match="neither 'level' nor 'pos_enc'"):
            apply_merge_map(aig_graph, torch.arange(N_NODES), N_NODES)

    def test_num_nodes_is_stored(self, aig_graph: Data) -> None:
        # PyG infers num_nodes from x, so asserting the value is vacuous;
        # the dataset's edge-normalization path needs it stored explicitly.
        out = apply_merge_map(aig_graph, torch.arange(N_NODES), N_NODES)
        assert "num_nodes" in out.keys()


# =====================================================================
# MERGING
# =====================================================================


class TestMerge:
    def test_two_nodes_merge(self, aig_graph: Data) -> None:
        # Merge PIs 1 and 2 into super-node 1.  Both drive node 4 with the
        # same polarity, so the resulting super-edge must carry a count of
        # 2 — this is what distinguishes summing from any other reduction.
        cluster = torch.tensor([0, 1, 1, 2, 3, 4, 5, 6, 7, 8])
        out = apply_merge_map(aig_graph, cluster, 9)

        assert out.num_nodes == 9
        assert torch.equal(out.x[1], torch.tensor([0.0, 2.0, 0.0, 0.0]))
        assert torch.equal(out.level[1], torch.tensor([0.0]))
        assert _edges(out) == [
            (0, 6, (1.0, 0.0)),
            (1, 3, (2.0, 0.0)),   # 1→4 and 2→4, both [1,0], summed
            (1, 4, (0.0, 1.0)),
            (2, 4, (1.0, 0.0)),
            (3, 5, (1.0, 0.0)),
            (4, 5, (0.0, 1.0)),
            (5, 6, (0.0, 1.0)),
            (5, 7, (1.0, 0.0)),
            (6, 8, (0.0, 1.0)),
        ]
        assert int(out.internal_edges) == 0
        assert out.num_edges == 9
        assert out.num_pis == N_PIS
        assert out.num_pos == N_POS

    def test_opposite_polarity_super_edge_sums_both_columns(
        self, aig_graph: Data
    ) -> None:
        # Merge PIs 2 and 3: 2→5 is inverted and 3→5 is not, so the merged
        # super-edge carries one of each.
        cluster = torch.tensor([0, 1, 2, 2, 3, 4, 5, 6, 7, 8])
        out = apply_merge_map(aig_graph, cluster, 9)

        merged = [e for e in _edges(out) if e[0] == 2 and e[1] == 4]
        assert merged == [(2, 4, (1.0, 1.0))]
        assert out.num_edges == 9

    def test_internal_edges_counted(self, aig_graph: Data) -> None:
        # Merge ANDs 4 and 6, which are directly connected 4→6.
        cluster = torch.tensor([0, 1, 2, 3, 4, 5, 4, 6, 7, 8])
        out = apply_merge_map(aig_graph, cluster, 9)

        assert int(out.internal_edges) == 1
        assert out.num_edges == 9
        assert (out.edge_index[0] == out.edge_index[1]).sum() == 0
        assert torch.equal(out.x[4], torch.tensor([0.0, 0.0, 2.0, 0.0]))
        assert torch.equal(out.level[4], torch.tensor([1.0]))

    def test_all_nodes_into_one_cluster(self, aig_graph: Data) -> None:
        out = apply_merge_map(aig_graph, torch.zeros(N_NODES, dtype=torch.long), 1)

        assert out.num_nodes == 1
        assert torch.equal(out.x, torch.tensor([[1.0, 3.0, 4.0, 2.0]]))
        assert int(out.internal_edges) == N_EDGES
        assert out.num_edges == 0
        # Every edge became internal, so the edge tensors are emptied by the
        # merge rather than being empty on input.
        assert out.edge_index.shape == (2, 0)
        assert out.edge_index.dtype == torch.long
        assert out.edge_attr.shape == (0, 2)
        assert out.edge_attr.dtype == torch.float32

    def test_empty_edge_index(self, aig_graph: Data) -> None:
        aig_graph.edge_index = torch.empty((2, 0), dtype=torch.long)
        aig_graph.edge_attr = torch.empty((0, 2), dtype=torch.float32)

        out = apply_merge_map(aig_graph, torch.arange(N_NODES), N_NODES)

        assert out.edge_index.shape == (2, 0)
        assert out.edge_index.dtype == torch.long
        assert out.edge_attr.shape == (0, 2)
        assert out.edge_attr.dtype == torch.float32
        assert int(out.internal_edges) == 0


# =====================================================================
# VALIDATION
# =====================================================================


class TestValidation:
    def test_wrong_cluster_length_raises(self, aig_graph: Data) -> None:
        with pytest.raises(ValueError, match="but the graph has 10 nodes"):
            apply_merge_map(aig_graph, torch.arange(5), 5)

    def test_unused_cluster_id_raises(self, aig_graph: Data) -> None:
        with pytest.raises(ValueError, match="unused"):
            apply_merge_map(aig_graph, torch.arange(N_NODES), N_NODES + 1)

    @pytest.mark.parametrize("bad_id", [-1, N_NODES])
    def test_out_of_range_cluster_raises(self, aig_graph: Data, bad_id: int) -> None:
        # Must be a ValueError, not a RuntimeError from bincount/index_add_.
        cluster = torch.arange(N_NODES)
        cluster[0] = bad_id
        with pytest.raises(ValueError, match=r"cluster ids must lie in"):
            apply_merge_map(aig_graph, cluster, N_NODES)

    def test_invalid_num_clusters_raises(self, aig_graph: Data) -> None:
        with pytest.raises(ValueError, match="num_clusters must be >= 1"):
            apply_merge_map(aig_graph, torch.zeros(N_NODES, dtype=torch.long), 0)

    @pytest.mark.parametrize(
        "cluster",
        [
            torch.arange(N_NODES, dtype=torch.int32),
            torch.arange(N_NODES, dtype=torch.float32),
            torch.arange(N_NODES).reshape(-1, 1),
        ],
        ids=["int32", "float32", "column-vector"],
    )
    def test_cluster_shape_and_dtype_normalized(
        self, aig_graph: Data, cluster: torch.Tensor
    ) -> None:
        # Methods are free to return any integer-valued dtype or a column
        # vector; bincount rejects both float and non-1-D input, so the
        # reshape/long normalization has to happen before validation.
        out = apply_merge_map(aig_graph, cluster, N_NODES)
        assert torch.equal(out.x, aig_graph.x)
        assert torch.equal(out.level, aig_graph.level)


# =====================================================================
# PURITY AND ATTRIBUTE HANDLING
# =====================================================================


class TestPurityAndAttrs:
    def test_source_graph_not_mutated(self, aig_graph: Data) -> None:
        before = {
            key: getattr(aig_graph, key).clone()
            for key in ("x", "edge_index", "edge_attr", "level")
        }

        apply_merge_map(aig_graph, torch.tensor([0, 1, 1, 2, 3, 4, 5, 6, 7, 8]), 9)

        for key, original in before.items():
            assert torch.equal(getattr(aig_graph, key), original), key

    def test_output_tensors_are_not_aliases(self, aig_graph: Data) -> None:
        # ExtractPrecomputedPE mutates its input in place, so returning a
        # source tensor by reference would corrupt the graph the offline
        # job still holds.
        out = apply_merge_map(aig_graph, torch.arange(N_NODES), N_NODES)
        for key in ("x", "edge_index", "edge_attr", "level"):
            assert (
                getattr(out, key).data_ptr() != getattr(aig_graph, key).data_ptr()
            ), key

    def test_dropped_attrs_absent(self, aig_graph: Data) -> None:
        aig_graph.pi_paths = torch.zeros(N_NODES, 1)
        aig_graph.local_sp_sum = torch.zeros(N_NODES, 1)
        aig_graph.edge_weight = torch.ones(N_EDGES)

        out = apply_merge_map(aig_graph, torch.arange(N_NODES), N_NODES)

        for key in ("pi_paths", "local_sp_sum", "edge_weight"):
            assert getattr(out, key, None) is None, key

    def test_collates_and_roundtrips(self, aig_graph: Data, tmp_path: Path) -> None:
        out = apply_merge_map(aig_graph, torch.tensor([0, 1, 1, 2, 3, 4, 5, 6, 7, 8]), 9)

        _register_pyg_safe_globals()
        path = tmp_path / "summarized.pt"
        torch.save(out, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert torch.equal(loaded.x, out.x)
        assert loaded.num_nodes == 9

        batch = Batch.from_data_list([out, out])
        assert batch.x.shape == (18, 4)
        assert batch.internal_edges.tolist() == [0, 0]


# =====================================================================
# REGISTRY
# =====================================================================


class TestRegistry:
    def test_identity_clustering(self, aig_graph: Data) -> None:
        cluster = identity_clustering(aig_graph)
        assert torch.equal(cluster, torch.arange(N_NODES))
        assert cluster.dtype == torch.long

    def test_registry_identity_roundtrip(self, aig_graph: Data) -> None:
        out = summarize_graph(aig_graph, "identity")
        assert torch.equal(out.x, aig_graph.x)
        assert _edges(out) == _edges(aig_graph)
        assert int(out.internal_edges) == 0

    def test_unknown_method_raises(self, aig_graph: Data) -> None:
        with pytest.raises(ValueError, match="Unknown summarization method"):
            summarize_graph(aig_graph, "not_a_method")

    def test_registry_passes_params_through(self, aig_graph: Data) -> None:
        out = summarize_graph(aig_graph, "wl", depth=0, pe_aware=False)
        # depth=0, ignoring the encoding, groups purely by node type.
        assert out.num_nodes == 4


# =====================================================================
# COLOUR REFINEMENT (WL / BISIMULATION)
# =====================================================================


@pytest.fixture
def symmetric_graph() -> Data:
    """Two structurally identical PI→AND cones feeding one PO.

    Nodes 1,2 and 4,5 are PIs; 3 and 6 are ANDs with the same fanin
    signature; 7 is the shared PO.  Node 0 is an isolated const.
    """
    x = torch.zeros(8, 4, dtype=torch.float32)
    x[0, 0] = 1.0
    x[[1, 2, 4, 5], 1] = 1.0
    x[[3, 6], 2] = 1.0
    x[7, 3] = 1.0
    edge_index = torch.tensor(
        [[1, 2, 4, 5, 3, 6], [3, 3, 6, 6, 7, 7]], dtype=torch.long
    )
    edge_attr = torch.tensor(
        [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]], dtype=torch.float32
    )
    level = torch.tensor([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [1.0], [2.0]])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, level=level)
    data.num_nodes = 8
    return data


def _classes(cluster: torch.Tensor) -> set[frozenset[int]]:
    return {
        frozenset((cluster == c).nonzero().flatten().tolist())
        for c in cluster.unique()
    }


class TestColorRefinement:
    def test_depth_zero_groups_by_features(self, aig_graph: Data) -> None:
        cluster = color_refinement(aig_graph, depth=0, pe_aware=False)
        # One class per node type present.
        assert len(cluster.unique()) == 4
        assert _classes(cluster) == {
            frozenset({0}),
            frozenset({1, 2, 3}),
            frozenset({4, 5, 6, 7}),
            frozenset({8, 9}),
        }

    def test_symmetric_cones_merge(self, symmetric_graph: Data) -> None:
        cluster = color_refinement(symmetric_graph, depth=4)
        # The two cones are indistinguishable, so their counterparts merge;
        # the const, the ANDs and the PO stay in their own classes.
        assert _classes(cluster) == {
            frozenset({0}),
            frozenset({1, 2, 4, 5}),
            frozenset({3, 6}),
            frozenset({7}),
        }

    def test_asymmetry_prevents_merging(self, symmetric_graph: Data) -> None:
        # Invert one fanin of the second cone: it is no longer the same shape.
        symmetric_graph.edge_attr[2] = torch.tensor([0.0, 1.0])
        cluster = color_refinement(symmetric_graph, depth=4)
        assert cluster[3] != cluster[6]

    def test_bisimulation_is_coarser_than_exact(self) -> None:
        # Node 3 has two fanins of the same type, node 6 has one.  Exact
        # refinement counts them and keeps them apart; count_cap=1 compares
        # only the set of neighbour kinds and merges them.
        x = torch.zeros(5, 4, dtype=torch.float32)
        x[[0, 1, 2], 1] = 1.0
        x[[3, 4], 2] = 1.0
        data = Data(
            x=x,
            edge_index=torch.tensor([[0, 1, 2], [3, 3, 4]], dtype=torch.long),
            edge_attr=torch.ones(3, 2, dtype=torch.float32) * torch.tensor([1.0, 0.0]),
            level=torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0]]),
        )

        exact = color_refinement(data, depth=2, count_cap=None)
        bisim = color_refinement(data, depth=2, count_cap=1)

        assert exact[3] != exact[4]
        assert bisim[3] == bisim[4]
        assert len(bisim.unique()) < len(exact.unique())

    def test_pe_aware_keeps_levels_apart(self, symmetric_graph: Data) -> None:
        # Give one PI of each cone a different level; pe_aware must not merge
        # across levels, because apply_merge_map pools the encoding by min.
        symmetric_graph.level[2] = 5.0
        aware = color_refinement(symmetric_graph, depth=4, pe_aware=True)
        blind = color_refinement(symmetric_graph, depth=4, pe_aware=False)

        assert aware[1] != aware[2]
        assert len(aware.unique()) > len(blind.unique())

    def test_direction_changes_partition(self, symmetric_graph: Data) -> None:
        backward = color_refinement(symmetric_graph, depth=4, direction="backward")
        forward = color_refinement(symmetric_graph, depth=4, direction="forward")
        both = color_refinement(symmetric_graph, depth=4, direction="both")

        # Backward sees only fanins, so the four PIs (all sourceless) collapse;
        # forward distinguishes nothing among them either, but "both" is at
        # least as fine as either one alone.
        assert len(both.unique()) >= max(
            len(backward.unique()), len(forward.unique())
        )

    def test_deterministic(self, symmetric_graph: Data) -> None:
        first = color_refinement(symmetric_graph, depth=4)
        second = color_refinement(symmetric_graph, depth=4)
        assert torch.equal(first, second)

    def test_returns_contiguous_ids_usable_as_cluster(
        self, symmetric_graph: Data
    ) -> None:
        cluster = color_refinement(symmetric_graph, depth=4)
        assert cluster.dtype == torch.long
        assert set(cluster.tolist()) == set(range(len(cluster.unique())))
        # The whole point: it must be accepted by apply_merge_map unchanged.
        out = apply_merge_map(cluster=cluster, data=symmetric_graph,
                              num_clusters=int(cluster.max()) + 1)
        assert out.num_nodes == 4

    def test_sourceless_nodes_merge_under_backward_refinement(
        self, aig_graph: Data
    ) -> None:
        # Backward refinement only ever looks at fanins, so nodes with none
        # — the primary inputs — are indistinguishable from each other and
        # collapse into a single class.  Everything downstream stays split.
        cluster = color_refinement(aig_graph, depth=4, direction="backward")
        assert cluster[1] == cluster[2] == cluster[3]
        assert len(cluster.unique()) == N_NODES - 2

    def test_forward_refinement_separates_primary_inputs(
        self, aig_graph: Data
    ) -> None:
        # PI 1 drives one gate and PI 2 drives two, which only following
        # fanouts can see.
        cluster = color_refinement(aig_graph, depth=4, direction="both")
        assert cluster[1] != cluster[2]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"direction": "sideways"}, "direction must be"),
            ({"depth": -1}, "depth must be"),
            ({"count_cap": 0}, "count_cap must be"),
        ],
    )
    def test_invalid_params_raise(
        self, aig_graph: Data, kwargs: dict, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            color_refinement(aig_graph, **kwargs)


class TestEncoderInvariance:
    """Does coarsening actually preserve what the model computes?

    The design notes treat exact colour refinement as provably lossless for
    this encoder, following Bollen et al.  That result assumes a plain
    sum-aggregating message-passing network; GCN+ is not one.  Measured here
    rather than assumed, and currently expected to fail — see the xfail
    reason on the test below.
    """

    @staticmethod
    def _encode(data: Data) -> torch.Tensor:
        from models.base_model import UnifiedGraphBaseModel

        torch.manual_seed(0)
        model = UnifiedGraphBaseModel(
            encoder_name="gcn",
            hidden_dim=16,
            encoder_kwargs={"num_layers": 2, "hid_dim": 16},
            pe_type="none",
            task_out_dim=1,
            pooling_type="mean",
        )
        model.eval()
        batch = Batch.from_data_list([data])
        with torch.no_grad():
            return model.forward_batch(batch)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Exact colour refinement is not lossless for GCN+ as built. "
            "base_model applies GraphNorm to the projected node features, so "
            "every node's representation depends on the graph's node "
            "multiset; and mean pooling counts a super-node once regardless "
            "of how many nodes it stands for. Restoring losslessness needs "
            "architecture changes (size-weighted pooling, and normalization "
            "that is invariant to quotienting), which is a thesis-level "
            "decision. Remove this marker if that lands."
        ),
    )
    def test_exact_refinement_preserves_graph_output(
        self, symmetric_graph: Data
    ) -> None:
        cluster = color_refinement(symmetric_graph, depth=2, count_cap=None)
        assert len(cluster.unique()) < symmetric_graph.x.size(0), "no compression"

        coarse = apply_merge_map(
            symmetric_graph, cluster, int(cluster.max()) + 1
        )
        original = self._encode(symmetric_graph)
        summarized = self._encode(coarse)

        assert torch.allclose(original, summarized, atol=1e-5), (
            f"graph-level output changed: {original.tolist()} vs "
            f"{summarized.tolist()}"
        )


# =====================================================================
# EVERY REGISTERED METHOD
# =====================================================================


def _build(types: list[int], edges: list[tuple[int, int]], inverted: set[int] = frozenset()) -> Data:
    """Assemble an AIG in the schema data_utils produces.

    *types* indexes the one-hot ``x`` columns [const, PI, AND, PO], *edges*
    are fanin → node pairs, and *inverted* holds the positions in *edges*
    that carry an inverter.  Levels are derived, so a fixture only has to
    state its topology.
    """
    num_nodes = len(types)
    x = torch.zeros(num_nodes, 4, dtype=torch.float32)
    x[torch.arange(num_nodes), torch.tensor(types)] = 1.0
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.zeros(len(edges), 2, dtype=torch.float32)
    for position in range(len(edges)):
        edge_attr[position, int(position in inverted)] = 1.0

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = num_nodes
    from data.compute_levels import compute_node_levels

    data.level = compute_node_levels(data).to(torch.float32).reshape(-1, 1)
    return data


def _quotient_is_acyclic(data: Data, cluster: torch.Tensor) -> bool:
    import networkx as nx

    quotient = cluster[data.edge_index]
    external = quotient[:, quotient[0] != quotient[1]]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(int(cluster.max()) + 1))
    graph.add_edges_from(external.t().tolist())
    return nx.is_directed_acyclic_graph(graph)


class TestEveryMethod:
    """Contract every method has to satisfy to be usable at all."""

    @pytest.fixture(params=sorted(SUMMARIZATION_REGISTRY))
    def method(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_returns_contiguous_ids(self, method: str, aig_graph: Data) -> None:
        cluster = SUMMARIZATION_REGISTRY[method](aig_graph)
        assert cluster.dtype == torch.long
        assert cluster.shape == (N_NODES,)
        # apply_merge_map rejects gaps rather than relabelling, so producing a
        # contiguous vector is the method's job, not the rewrite's.
        assert set(cluster.tolist()) == set(range(len(cluster.unique())))

    def test_round_trips_through_apply_merge_map(
        self, method: str, aig_graph: Data
    ) -> None:
        out = summarize_graph(aig_graph, method)
        assert out.num_nodes <= N_NODES
        assert out.x.shape == (out.num_nodes, 4)
        assert out.edge_attr.shape == (out.edge_index.size(1), 2)
        # Members are conserved: a merge moves counts around, never adds them.
        assert out.x.sum() == N_NODES
        assert out.num_pis == N_PIS
        assert out.num_pos == N_POS

    def test_deterministic(self, method: str, aig_graph: Data) -> None:
        first = SUMMARIZATION_REGISTRY[method](aig_graph)
        second = SUMMARIZATION_REGISTRY[method](aig_graph)
        assert torch.equal(first, second)

    def test_runs_on_the_cached_schema(self, method: str, aig_graph: Data) -> None:
        # Cached graphs carry pos_enc = log1p(level) and no level at all, and
        # that is the form the precompute job actually reads.
        expected = SUMMARIZATION_REGISTRY[method](aig_graph)
        aig_graph.pos_enc = torch.log1p(aig_graph.level)
        del aig_graph.level

        assert torch.equal(SUMMARIZATION_REGISTRY[method](aig_graph), expected)

    def test_runs_on_an_edgeless_graph(self, method: str, aig_graph: Data) -> None:
        aig_graph.edge_index = torch.empty((2, 0), dtype=torch.long)
        aig_graph.edge_attr = torch.empty((0, 2), dtype=torch.float32)

        cluster = SUMMARIZATION_REGISTRY[method](aig_graph)
        assert set(cluster.tolist()) == set(range(len(cluster.unique())))

    @pytest.mark.parametrize(
        "degenerate",
        ["single_node", "single_type", "self_loop", "parallel_edges", "max_depth"],
        ids=lambda case: case,
    )
    def test_survives_degenerate_graphs(
        self, method: str, aig_graph: Data, degenerate: str
    ) -> None:
        # None of these appear in the corpus as it stands, but all of them are
        # one upstream change away, and the failure mode of a bad cluster
        # vector is an exception ~hours into a 700k-graph cluster job.
        if degenerate == "single_node":
            data = _build(types=[1], edges=[])
        elif degenerate == "single_type":
            data = _build(types=[2, 2, 2], edges=[(0, 1), (1, 2)])
        elif degenerate == "self_loop":
            data = _build(types=[1, 2, 3], edges=[(0, 1), (1, 1), (1, 2)])
        elif degenerate == "parallel_edges":
            data = _build(types=[1, 2, 3], edges=[(0, 1), (0, 1), (1, 2)])
        else:
            data = aig_graph
            data.level = data.level * 24972.0

        cluster = SUMMARIZATION_REGISTRY[method](data)
        assert cluster.shape == (data.x.size(0),)
        assert set(cluster.tolist()) == set(range(len(cluster.unique())))


class TestConfigParams:
    """config.SUMMARIZATION_PARAMS is what a cluster run actually passes."""

    def test_every_method_has_an_entry(self) -> None:
        import config

        assert set(config.SUMMARIZATION_PARAMS) == set(SUMMARIZATION_REGISTRY)

    def test_every_entry_is_accepted_by_its_method(self, aig_graph: Data) -> None:
        # A typo'd key raises TypeError inside a worker, which the driver
        # swallows per graph — so the whole corpus would come back "0 graphs,
        # 700k errors" rather than failing fast.  Catch it here instead.
        import config

        for method, params in config.SUMMARIZATION_PARAMS.items():
            cluster = SUMMARIZATION_REGISTRY[method](aig_graph, **params)
            assert cluster.shape == (N_NODES,), method


# =====================================================================
# S1 — LEVEL-BOUNDED CONE COARSENING
# =====================================================================


@pytest.fixture
def chain_aig() -> Data:
    """A fanout-free cascade: PI,PI → 4 chained ANDs → PO.

    Each AND drives exactly one successor, so the whole cascade is one
    contractible chain and nothing merges on the width axis (every gate sits
    on its own level).
    """
    return _build(
        types=[1, 1, 2, 2, 2, 2, 3],
        edges=[(0, 2), (1, 2), (2, 3), (0, 3), (3, 4), (1, 4), (4, 5), (0, 5), (5, 6)],
    )


@pytest.fixture
def reconvergent_aig() -> Data:
    """Two same-level cones reconverging on one gate: 4 PIs → 2 ANDs → AND → PO."""
    return _build(
        types=[1, 1, 1, 1, 2, 2, 2, 3],
        edges=[(0, 4), (1, 4), (2, 5), (3, 5), (4, 6), (5, 6), (6, 7)],
    )


@pytest.fixture
def banded_aig() -> Data:
    """Two branches reconverging at gate 9 from *different* levels.

    Gates 5 and 8 both have gate 9 as their immediate post-dominator but sit
    on levels 2 and 3, so only a widened band puts them in the same group.
    """
    return _build(
        types=[1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3],
        edges=[
            (0, 4), (1, 4), (4, 5), (2, 5),
            (2, 6), (3, 6), (6, 7), (3, 7), (7, 8), (0, 8),
            (5, 9), (8, 9), (9, 10),
        ],
    )


@pytest.fixture
def wide_aig() -> Data:
    """A ~440-node AIG with graded structure, for the hashing properties.

    The LSH guarantees are asymptotic in the number of distinct descriptors:
    on a ten-node fixture every node lands in its own bucket whatever the
    parameters, so a violation cannot show up.  Gates here vary in level, in
    fanout and in polarity mix, which is what makes the buckets contend.
    """
    types: list[int] = [1] * 20
    edges: list[tuple[int, int]] = []
    inverted: set[int] = set()
    frontier = list(range(20))
    for depth in range(1, 15):
        layer = []
        for offset in range(len(frontier) - 1):
            node = len(types)
            types.append(2)
            for source in (frontier[offset], frontier[(offset + depth) % len(frontier)]):
                if (source + node) % 3 == 0:
                    inverted.add(len(edges))
                edges.append((source, node))
            layer.append(node)
        frontier = layer[: max(2, len(layer) - 2)]
        if len(frontier) < 2:
            break
    for source in frontier:
        node = len(types)
        types.append(3)
        edges.append((source, node))
    return _build(types=types, edges=edges, inverted=inverted)


@pytest.fixture
def ladder_aig() -> Data:
    """A cascade of five ANDs, each one level deeper than the last.

    Levels and fanout counts vary from gate to gate, which is what gives the
    hashed descriptors of S5 something to spread out over.
    """
    return _build(
        types=[1, 1, 2, 2, 2, 2, 2, 3],
        edges=[
            (0, 2), (1, 2), (2, 3), (0, 3), (3, 4), (1, 4),
            (4, 5), (0, 5), (5, 6), (1, 6), (6, 7),
        ],
    )


class TestConeCoarsening:
    def test_fanout_free_chain_contracts(self, chain_aig: Data) -> None:
        cluster = cone_coarsening(chain_aig, max_chain_length=4)
        # The four ANDs collapse into one super-node; the PIs and the PO,
        # being the circuit's fixed interface, are untouched.
        assert cluster[2] == cluster[3] == cluster[4] == cluster[5]
        assert len(cluster.unique()) == 4

    def test_max_chain_length_caps_contraction(self, chain_aig: Data) -> None:
        cluster = cone_coarsening(chain_aig, max_chain_length=2)
        assert cluster[2] == cluster[3]
        assert cluster[4] == cluster[5]
        assert cluster[3] != cluster[4]

    def test_chain_length_one_disables_the_depth_axis(self, chain_aig: Data) -> None:
        # Nothing merges on the width axis here either, so this is identity.
        cluster = cone_coarsening(chain_aig, max_chain_length=1)
        assert len(cluster.unique()) == chain_aig.num_nodes

    def test_width_axis_merges_reconvergent_siblings(
        self, reconvergent_aig: Data
    ) -> None:
        cluster = cone_coarsening(reconvergent_aig, max_chain_length=1)
        # Gates 4 and 5 are on the same level and both cones reconverge at 6.
        assert cluster[4] == cluster[5]
        assert len(cluster.unique()) == reconvergent_aig.num_nodes - 1

    def test_width_axis_keeps_levels_exact(self, reconvergent_aig: Data) -> None:
        # Merging only within a level is what keeps the level PE exact: the
        # member minimum apply_merge_map stores is the members' actual level.
        cluster = cone_coarsening(reconvergent_aig, max_chain_length=1)
        levels = reconvergent_aig.level.reshape(-1)
        for group in cluster.unique():
            assert len(levels[cluster == group].unique()) == 1

    def test_primary_inputs_and_outputs_never_merge(
        self, reconvergent_aig: Data
    ) -> None:
        cluster = cone_coarsening(reconvergent_aig)
        boundary = [0, 1, 2, 3, 7]
        assert len(cluster[boundary].unique()) == len(boundary)

    def test_multi_fanout_gate_is_not_absorbed(self) -> None:
        # Gate 3 drives two successors, so contracting it into either would
        # not be a fanout-free chain and would not be safe.
        data = _build(
            types=[1, 1, 2, 2, 2, 3, 3],
            edges=[(0, 2), (1, 2), (2, 3), (0, 3), (3, 4), (1, 4), (4, 5), (3, 6)],
        )
        cluster = cone_coarsening(data)
        assert cluster[3] != cluster[4]

    def test_gates_without_a_reconvergence_point_are_left_alone(self) -> None:
        # Two independent circuits, each fanning out to two outputs without
        # reconverging.  Gates 2 and 6 are on the same level and neither has
        # any single gate its whole cone passes through, so the only thing
        # they have in common is the output boundary — which is not a cone.
        # They are not even weakly connected, and must not be merged.
        data = _build(
            types=[1, 1, 2, 2, 2, 3, 3, 1, 1, 2, 2, 2, 3, 3],
            edges=[
                (0, 2), (1, 2), (2, 3), (0, 3), (2, 4), (1, 4), (3, 5), (4, 6),
                (7, 9), (8, 9), (9, 10), (7, 10), (9, 11), (8, 11), (10, 12),
                (11, 13),
            ],
        )
        cluster = cone_coarsening(data, max_chain_length=1)

        assert cluster[2] != cluster[9]
        assert len(cluster.unique()) == data.num_nodes

    def test_preserves_acyclicity(
        self, aig_graph: Data, reconvergent_aig: Data, chain_aig: Data
    ) -> None:
        for graph in (aig_graph, reconvergent_aig, chain_aig):
            assert _quotient_is_acyclic(graph, cone_coarsening(graph))

    def test_level_band_widens_the_groups(self, banded_aig: Data) -> None:
        tight = cone_coarsening(banded_aig, max_chain_length=1, level_band=0)
        wide = cone_coarsening(banded_aig, max_chain_length=1, level_band=1)

        assert len(tight.unique()) == banded_aig.num_nodes
        # Gates 5 and 8 reconverge at gate 9 but sit one level apart.
        assert wide[5] == wide[8]
        assert len(wide.unique()) == banded_aig.num_nodes - 1

    def test_both_axes_disabled_is_identity(self, aig_graph: Data) -> None:
        cluster = cone_coarsening(aig_graph, max_chain_length=1, level_band=None)
        assert torch.equal(cluster, torch.arange(N_NODES))

    def test_negative_level_band_raises(self, aig_graph: Data) -> None:
        with pytest.raises(ValueError, match="level_band must be"):
            cone_coarsening(aig_graph, level_band=-1)


# =====================================================================
# S5 — LSH / UGC HASH COARSENING
# =====================================================================


class TestLshCoarsening:
    def test_doubling_bin_width_never_increases_clusters(
        self, wide_aig: Data
    ) -> None:
        # Offsets are drawn independently of bin_width, so each doubling
        # produces a coarser bucketing rather than a differently shaped one.
        # This is a guarantee, not a trend, and is what makes the parameter
        # usable as a compression knob.  It needs a graph big enough for
        # buckets to contend: on a ten-node fixture the property cannot be
        # violated even by an implementation that does not have it.
        widths = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
        partitions = [lsh_coarsening(wide_aig, bin_width=w) for w in widths]

        # Non-increasing counts alone would not show refinement — two
        # partitions can have the same size and cut the graph differently.
        # Assert the real relation: every group of the finer partition lies
        # wholly inside one group of the coarser.
        for finer, coarser in zip(partitions, partitions[1:], strict=False):
            for group in finer.unique():
                assert len(coarser[finer == group].unique()) == 1

        counts = [len(p.unique()) for p in partitions]
        assert counts[0] > counts[-1], "the knob did nothing over this range"

    @pytest.mark.parametrize("bin_width", [0.25, 2.0, 1e6])
    def test_node_types_never_merge(self, wide_aig: Data, bin_width: float) -> None:
        # Node type is an exact part of the bucket key, so a primary input can
        # never dissolve into an AND super-node however coarse the hashing.
        # Needs a graph where the hashed columns alone *would* collide across
        # types; on a small one they never do, and the test proves nothing.
        cluster = lsh_coarsening(wide_aig, bin_width=bin_width)
        types = wide_aig.x.argmax(dim=1)
        for group in cluster.unique():
            assert len(types[cluster == group].unique()) == 1

    def test_structurally_identical_nodes_collide(
        self, reconvergent_aig: Data
    ) -> None:
        # PIs 0-3 have identical descriptors: same level, no fanin, one
        # regular fanout into an AND.
        cluster = lsh_coarsening(reconvergent_aig)
        assert len(cluster[[0, 1, 2, 3]].unique()) == 1

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"bin_width": 0.0}, "bin_width must be"),
            ({"num_projections": 0}, "num_projections must be"),
        ],
    )
    def test_invalid_params_raise(
        self, aig_graph: Data, kwargs: dict, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            lsh_coarsening(aig_graph, **kwargs)


# =====================================================================
# S4 — SPECTRAL / LOCAL VARIATION
# =====================================================================


class TestSpectralCoarsening:
    @pytest.mark.parametrize("ratio", [0.2, 0.5, 0.8])
    def test_reaches_the_target_ratio(self, aig_graph: Data, ratio: float) -> None:
        cluster = spectral_coarsening(aig_graph, reduction_ratio=ratio)
        assert len(cluster.unique()) == round(N_NODES * (1.0 - ratio))

    def test_zero_ratio_is_identity(self, aig_graph: Data) -> None:
        cluster = spectral_coarsening(aig_graph, reduction_ratio=0.0)
        assert torch.equal(cluster, torch.arange(N_NODES))

    def test_node_cap_falls_back_to_heavy_edge(self, aig_graph: Data) -> None:
        # Above the cap the eigensolver is skipped entirely.  The fallback is
        # part of the method's definition, so it has to be the same partition
        # heavy_edge would have produced, not an approximation of the spectral
        # one.
        capped = spectral_coarsening(aig_graph, max_spectral_nodes=0)
        heavy = spectral_coarsening(aig_graph, variant="heavy_edge")
        assert torch.equal(capped, heavy)

    def test_the_two_variants_pick_different_edges(self, aig_graph: Data) -> None:
        # Guards against the eigensolver silently failing and every graph
        # quietly taking the fallback path: heavy-edge scores by degree alone,
        # local variation by how far contraction moves the low Laplacian
        # eigenvectors, and on this graph they disagree.
        spectral = spectral_coarsening(aig_graph, reduction_ratio=0.3)
        heavy = spectral_coarsening(
            aig_graph, reduction_ratio=0.3, variant="heavy_edge"
        )
        assert not torch.equal(spectral, heavy)

    def test_edgeless_graph_is_identity(self, aig_graph: Data) -> None:
        aig_graph.edge_index = torch.empty((2, 0), dtype=torch.long)
        aig_graph.edge_attr = torch.empty((0, 2), dtype=torch.float32)
        cluster = spectral_coarsening(aig_graph, reduction_ratio=0.9)
        assert torch.equal(cluster, torch.arange(N_NODES))

    def test_cost_ranks_edges_as_the_reference_implementation_does(
        self, aig_graph: Data
    ) -> None:
        """Differential test against Loukas' ``contract_variation_edges``.

        That reference scores a candidate edge as
        ``||B^T L_e B||_F`` with ``B = (I - 11^T/2) A[edge]`` and
        ``L_e = [[2d_i - w, -w], [-w, 2d_j - w]]``.  Written out, the edge
        weight cancels and what is left is the movement of the preserved
        subspace scaled by the volume of the pair — so the degrees multiply
        rather than divide, which is exactly the kind of inversion that still
        produces a plausible-looking coarsening.  Costs are compared as a
        ranking, since only the ordering drives the matching.
        """
        import numpy as np

        from data.summarization import (
            _laplacian_subspace,
            _spectral_edge_costs,
            _undirected_simple,
        )

        num_nodes = N_NODES
        pairs, weight = _undirected_simple(aig_graph.edge_index, num_nodes)
        subspace = _laplacian_subspace(pairs, weight, num_nodes, 4)
        assert subspace is not None, "eigensolver did not run on this graph"

        degree = torch.zeros(num_nodes).index_add_(
            0, pairs.reshape(-1), weight.repeat(2)
        )
        projector = np.eye(2) - np.ones((2, 2)) / 2
        basis = subspace.double().numpy()
        reference = []
        for edge in range(pairs.size(1)):
            i, j = int(pairs[0, edge]), int(pairs[1, edge])
            edge_weight = float(weight[edge])
            diagonal = 2 * np.array([float(degree[i]), float(degree[j])]) - edge_weight
            local = np.array(
                [[diagonal[0], -edge_weight], [-edge_weight, diagonal[1]]]
            )
            projected = projector @ basis[[i, j], :]
            reference.append(np.linalg.norm(projected.T @ local @ projected))

        actual = _spectral_edge_costs(pairs, weight, subspace, num_nodes)

        ratio = actual / torch.tensor(reference, dtype=actual.dtype)
        assert torch.allclose(ratio, ratio[0].expand_as(ratio), atol=1e-4), (
            "cost is not a constant multiple of the reference, so it ranks "
            f"edges differently: ratios {ratio.tolist()}"
        )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"variant": "kron"}, "variant must be"),
            ({"reduction_ratio": 1.0}, "reduction_ratio must be"),
            ({"reduction_ratio": -0.1}, "reduction_ratio must be"),
        ],
    )
    def test_invalid_params_raise(
        self, aig_graph: Data, kwargs: dict, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            spectral_coarsening(aig_graph, **kwargs)


# =====================================================================
# S3 — CONVOLUTION MATCHING
# =====================================================================


@pytest.fixture
def twin_aig() -> Data:
    """Three AND gates of equal degree, two of which are convolution twins.

    Gates 4 and 5 take the same fanins (PIs 0 and 1) and drive the same PO,
    so they have identical neighbourhoods and are indistinguishable to an
    undirected convolution.  Gate 6 has the same degree and the same features
    but a different neighbourhood (PIs 2 and 3), which makes ``(4, 6)`` the
    controlled comparison for ``(4, 5)``.

    One of gate 5's fanins is inverted, so the twins compute *different*
    functions: the pair separates a polarity-blind method from a
    polarity-aware one.  (A real AIG is structurally hashed, so an
    identical-fanin pair only survives because of that inversion.)
    """
    return _build(
        types=[1, 1, 1, 1, 2, 2, 2, 3],
        edges=[
            (0, 4), (1, 4), (0, 5), (1, 5), (2, 6), (3, 6), (4, 7), (5, 7), (6, 7),
        ],
        inverted={2},
    )


class TestConvMatchCoarsening:
    @pytest.mark.parametrize("ratio", [0.2, 0.5, 0.8])
    def test_reaches_the_target_ratio(self, aig_graph: Data, ratio: float) -> None:
        cluster = convmatch_coarsening(aig_graph, reduction_ratio=ratio)
        assert len(cluster.unique()) == round(N_NODES * (1.0 - ratio))

    def test_zero_ratio_is_identity(self, aig_graph: Data) -> None:
        cluster = convmatch_coarsening(aig_graph, reduction_ratio=0.0)
        assert torch.equal(cluster, torch.arange(N_NODES))

    def test_identical_neighbourhoods_are_the_cheaper_merge(
        self, twin_aig: Data
    ) -> None:
        # The defining property, isolated: gates 4, 5 and 6 have the same
        # degree and the same features, so the only thing separating the
        # pairs is that 4 and 5 aggregate over the same neighbours and 4 and
        # 6 do not.  Compared at matched degree because the objective is an
        # unweighted L1 sum, which independently favours low-degree nodes.
        from data.summarization import (
            _convmatch_costs,
            _node_levels,
            _undirected_simple,
        )

        num_nodes = twin_aig.num_nodes
        pairs, weight = _undirected_simple(twin_aig.edge_index, num_nodes)
        features = torch.cat(
            [twin_aig.x, torch.log1p(_node_levels(twin_aig).float().reshape(-1, 1))],
            dim=1,
        )
        twins, mismatched = _convmatch_costs(
            torch.tensor([[4, 4], [5, 6]]),
            pairs,
            weight,
            features,
            torch.ones(num_nodes),
            num_nodes,
        ).tolist()

        assert twins < mismatched

    def test_representation_matches_the_gcn_operator(self, aig_graph: Data) -> None:
        # The cost is only ConvMatch's if the representation it perturbs is
        # really D^-1/2 (A+I) D^-1/2 X.  Checked against a dense reference,
        # because an extra or missing degree factor changes which pairs get
        # merged while leaving every behavioural assertion in this class
        # passing.
        from data.summarization import _convmatch_representation, _undirected_simple

        num_nodes = N_NODES
        pairs, weight = _undirected_simple(aig_graph.edge_index, num_nodes)
        features = aig_graph.x

        dense = torch.zeros(num_nodes, num_nodes)
        dense[pairs[0], pairs[1]] = weight
        dense[pairs[1], pairs[0]] = weight
        dense = dense + torch.eye(num_nodes)
        inverse_sqrt = dense.sum(1).rsqrt()
        expected = (
            inverse_sqrt.unsqueeze(1) * dense * inverse_sqrt.unsqueeze(0)
        ) @ features

        _, representation, _, _ = _convmatch_representation(
            pairs, weight, features, torch.ones(num_nodes), num_nodes
        )
        assert torch.allclose(representation, expected, atol=1e-5)

    def test_merged_degree_accounts_for_the_shared_edge(self) -> None:
        # Merging two adjacent nodes turns the edge between them into an
        # internal one, so the super-node's degree is d_u + d_v - 2w, not
        # d_u + d_v.  Nearly every candidate is a graph edge, so getting this
        # wrong skews the whole cost ranking rather than a few entries.
        from data.summarization import _candidate_edge_weight, _undirected_simple

        data = _build(
            types=[1, 1, 2, 2, 3],
            edges=[(0, 2), (1, 2), (2, 3), (0, 3), (3, 4)],
        )
        pairs, weight = _undirected_simple(data.edge_index, data.num_nodes)

        # (2,3) is an edge; (0,4) is not; (0,2) is an edge.
        shared = _candidate_edge_weight(
            torch.tensor([[2, 0, 0], [3, 4, 2]]), pairs, weight, data.num_nodes
        )
        assert shared.tolist() == [1.0, 0.0, 1.0]

    def test_cost_matches_the_reference_implementation(self, aig_graph: Data) -> None:
        """Differential test against amazon-science/convolution-matching.

        The body below is the cost of that repository's
        ``ApproximateConvolutionMatchingCoarsener._compute_edge_costs_internal``
        transcribed in its own terms — ``current_sum``, ``influence``,
        ``scaled_feat``, ``self_loop_weight``.  Each piece of the published
        bound is easy to get subtly wrong in a way no behavioural assertion
        notices (the influence sum ranges over neighbours only, and the edge
        between an adjacent candidate pair has to leave the degree, the
        neighbour sums *and* the influences), so the whole formula is pinned
        at once.

        One deliberate difference is parameterised here rather than hidden:
        the reference keeps an internal edge as a self-loop on the super-node,
        while ``apply_merge_map`` drops it, so this graph's degrees lose it.
        """
        from data.summarization import (
            _candidate_edge_weight,
            _convmatch_costs,
            _node_levels,
            _projection_neighbour_pairs,
            _sgc_embedding,
            _undirected_simple,
        )

        num_nodes = N_NODES
        pairs, weight = _undirected_simple(aig_graph.edge_index, num_nodes)
        features = torch.cat(
            [aig_graph.x, torch.log1p(_node_levels(aig_graph).float().reshape(-1, 1))],
            dim=1,
        )
        size = torch.ones(num_nodes)
        candidates = torch.cat(
            [
                pairs,
                _projection_neighbour_pairs(
                    _sgc_embedding(pairs, weight, features, num_nodes, 4), 2, 42
                ),
            ],
            dim=1,
        )

        degree = torch.zeros(num_nodes).index_add_(
            0, pairs.reshape(-1), weight.repeat(2)
        )
        inv_sqrt_degree = 1.0 / torch.sqrt(degree + size)
        scaled_feat = features * inv_sqrt_degree.unsqueeze(1)
        current_sum = torch.zeros_like(scaled_feat)
        current_sum.index_add_(0, pairs[0], weight.unsqueeze(1) * scaled_feat[pairs[1]])
        current_sum.index_add_(0, pairs[1], weight.unsqueeze(1) * scaled_feat[pairs[0]])
        influence = torch.zeros(num_nodes)
        influence.index_add_(0, pairs[0], weight * inv_sqrt_degree[pairs[1]])
        influence.index_add_(0, pairs[1], weight * inv_sqrt_degree[pairs[0]])
        h = (current_sum + size.unsqueeze(1) * scaled_feat) * inv_sqrt_degree.unsqueeze(1)

        src, dst = candidates[0], candidates[1]
        edge_weight = _candidate_edge_weight(candidates, pairs, weight, num_nodes)
        sn_size = size[src] + size[dst]
        sn_feat = (
            size[src].unsqueeze(1) * features[src]
            + size[dst].unsqueeze(1) * features[dst]
        ) / sn_size.unsqueeze(1)
        # self_loop_weight stays zero: apply_merge_map drops internal edges.
        sn_degree = degree[src] + degree[dst] - 2.0 * edge_weight
        sqrt_degrees = torch.sqrt(sn_degree + sn_size)
        sn_scaled = sn_feat / sqrt_degrees.unsqueeze(1)
        sn_sums = (
            current_sum[src]
            + current_sum[dst]
            - edge_weight.unsqueeze(1) * scaled_feat[src]
            - edge_weight.unsqueeze(1) * scaled_feat[dst]
        )
        sn_h = (
            sn_sums + sn_size.unsqueeze(1) * sn_scaled
        ) / sqrt_degrees.unsqueeze(1)
        expected = (
            (sn_h - h[src]).abs().sum(1)
            + (sn_h - h[dst]).abs().sum(1)
            + (scaled_feat[src] - sn_scaled).abs().sum(1)
            * (influence[src] - edge_weight * inv_sqrt_degree[dst])
            + (scaled_feat[dst] - sn_scaled).abs().sum(1)
            * (influence[dst] - edge_weight * inv_sqrt_degree[src])
        )

        actual = _convmatch_costs(
            candidates, pairs, weight, features, size, num_nodes
        )
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_merged_degree_matches_an_exact_recompute(self) -> None:
        # The O(1) shortcut has to agree with actually performing the merge
        # and rebuilding the coarse graph.  Checked on an adjacent pair,
        # where the two differ.
        from data.summarization import (
            _candidate_edge_weight,
            _coarse_degree,
            _relabel,
            _undirected_simple,
        )

        data = _build(
            types=[1, 1, 2, 2, 3],
            edges=[(0, 2), (1, 2), (2, 3), (0, 3), (3, 4)],
        )
        num_nodes = data.num_nodes
        pairs, weight = _undirected_simple(data.edge_index, num_nodes)
        size = torch.ones(num_nodes)
        degree = _coarse_degree(pairs, weight, size, num_nodes)

        left, right = 2, 3
        candidates = torch.tensor([[left], [right]])
        shared = _candidate_edge_weight(candidates, pairs, weight, num_nodes)
        shortcut = degree[left] + degree[right] - 2.0 * shared[0]

        # Exact: merge, rebuild, and read the super-node's degree back.
        cluster_map = _relabel(
            torch.arange(num_nodes).index_put_(
                (torch.tensor([left]),), torch.tensor([right])
            )
        )
        coarse_pairs, coarse_weight = _undirected_simple(
            cluster_map[pairs], int(cluster_map.max()) + 1, weight
        )
        coarse_size = torch.zeros(int(cluster_map.max()) + 1).index_add_(
            0, cluster_map, size
        )
        exact = _coarse_degree(
            coarse_pairs, coarse_weight, coarse_size, int(cluster_map.max()) + 1
        )[cluster_map[right]]

        assert torch.allclose(shortcut, exact)

    def test_is_polarity_blind_where_colour_refinement_is_not(
        self, twin_aig: Data
    ) -> None:
        # The documented cost of using the paper's operator unchanged: gates 4
        # and 5 compute different functions, ConvMatch merges them anyway, and
        # only the AIG-adapted method can tell them apart.  This is the
        # domain-blindness that makes ConvMatch the honest general-purpose bar
        # rather than a strawman.
        merged = convmatch_coarsening(twin_aig, reduction_ratio=0.5)
        refined = color_refinement(twin_aig, depth=4, direction="backward")

        assert merged[4] == merged[5]
        assert refined[4] != refined[5]

    def test_edgeless_graph_is_identity(self, aig_graph: Data) -> None:
        aig_graph.edge_index = torch.empty((2, 0), dtype=torch.long)
        aig_graph.edge_attr = torch.empty((0, 2), dtype=torch.float32)
        cluster = convmatch_coarsening(aig_graph, reduction_ratio=0.9)
        assert torch.equal(cluster, torch.arange(N_NODES))

    def test_sgc_depth_changes_the_candidates(self, aig_graph: Data) -> None:
        # sgc_depth only enters through candidate generation, so it must move
        # the result without moving the compression the caller asked for.
        shallow = convmatch_coarsening(aig_graph, reduction_ratio=0.3, sgc_depth=0)
        deep = convmatch_coarsening(aig_graph, reduction_ratio=0.3, sgc_depth=4)
        assert not torch.equal(shallow, deep)
        assert len(shallow.unique()) == len(deep.unique())

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"reduction_ratio": 1.0}, "reduction_ratio must be"),
            ({"sgc_depth": -1}, "sgc_depth must be"),
        ],
    )
    def test_invalid_params_raise(
        self, aig_graph: Data, kwargs: dict, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            convmatch_coarsening(aig_graph, **kwargs)


# =====================================================================
# PRECOMPUTE DRIVER
# =====================================================================


@pytest.fixture
def manifest_workspace(tmp_path: Path, aig_graph: Data) -> tuple[Path, Path]:
    """A cache directory of 3 graphs plus the manifest that lists them."""
    cache_dir = tmp_path / "aig_train_run" / "shared_tier0_cache"
    cache_dir.mkdir(parents=True)
    meta_dir = tmp_path / "aig_train_run" / "Orchestrate" / "cache" / "metadata"
    meta_dir.mkdir(parents=True)

    entries = []
    for i in range(3):
        cache_path = cache_dir / f"graph{i}.pt"
        torch.save(aig_graph, cache_path)
        entries.append(
            {
                "graph_path": f"/raw/tier0/graph{i}.pt",
                "cache_path": str(cache_path),
                "num_nodes": N_NODES,
            }
        )
    (meta_dir / "algo_manifest.json").write_text(
        json.dumps({"version": 2, "num_samples": 3, "entries": entries})
    )
    return meta_dir, tmp_path / "out"


class TestDriver:
    def test_build_tasks_mirrors_layout(
        self, manifest_workspace: tuple[Path, Path]
    ) -> None:
        meta_dir, out_root = manifest_workspace
        tasks = build_tasks([meta_dir], out_root)

        assert len(tasks) == 3
        for cache_path, out_path, graph_path in tasks:
            assert Path(out_path).parent == out_root / "shared_tier0_cache"
            assert Path(out_path).name == Path(cache_path).name
            assert graph_path.startswith("/raw/tier0/")

    def test_build_tasks_shards_disjointly(
        self, manifest_workspace: tuple[Path, Path]
    ) -> None:
        meta_dir, out_root = manifest_workspace
        shards = [build_tasks([meta_dir], out_root, i, 2) for i in range(2)]

        assert sorted(shards[0] + shards[1]) == build_tasks([meta_dir], out_root)
        assert not set(shards[0]) & set(shards[1])

    def test_driver_writes_expected_layout(
        self, manifest_workspace: tuple[Path, Path]
    ) -> None:
        meta_dir, out_root = manifest_workspace
        summarize_from_manifests([meta_dir], "identity", out_root)

        out_dir = out_root / "shared_tier0_cache"
        assert sorted(p.name for p in out_dir.glob("*.pt")) == [
            "graph0.pt",
            "graph1.pt",
            "graph2.pt",
        ]

        # Keyed by source graph path, holding post-merge counts.
        num_nodes = json.loads((out_dir / _shard_index_name(0)).read_text())
        assert num_nodes == {f"/raw/tier0/graph{i}.pt": N_NODES for i in range(3)}

        stats = json.loads(
            (out_root / "_summary_stats_identity_shard000.json").read_text()
        )
        assert stats["graphs"] == 3
        assert stats["errors"] == 0
        assert stats["node_retention"] == 1.0

    def test_driver_output_is_a_summarized_graph(
        self, manifest_workspace: tuple[Path, Path]
    ) -> None:
        meta_dir, out_root = manifest_workspace
        summarize_from_manifests([meta_dir], "identity", out_root)

        _register_pyg_safe_globals()
        written = torch.load(
            out_root / "shared_tier0_cache" / "graph0.pt",
            map_location="cpu",
            weights_only=True,
        )
        # Not merely a copy of the input: the rewrite adds internal_edges
        # and drops nothing the dataset needs.
        assert "num_nodes" in written.keys()
        assert int(written.internal_edges) == 0
        assert written.x.shape == (N_NODES, 4)

    def test_driver_resumes(
        self, manifest_workspace: tuple[Path, Path], capsys: pytest.CaptureFixture
    ) -> None:
        meta_dir, out_root = manifest_workspace
        summarize_from_manifests([meta_dir], "identity", out_root)
        capsys.readouterr()

        summarize_from_manifests([meta_dir], "identity", out_root)
        assert "skipping 3 already done" in capsys.readouterr().out

    def test_driver_redoes_graphs_missing_from_index(
        self, manifest_workspace: tuple[Path, Path], capsys: pytest.CaptureFixture
    ) -> None:
        # An output file whose node count was never recorded must be redone,
        # or the index ends up with holes after an interrupted run.
        meta_dir, out_root = manifest_workspace
        summarize_from_manifests([meta_dir], "identity", out_root)
        index_path = out_root / "shared_tier0_cache" / _shard_index_name(0)
        index = json.loads(index_path.read_text())
        index.pop("/raw/tier0/graph1.pt")
        index_path.write_text(json.dumps(index))
        capsys.readouterr()

        summarize_from_manifests([meta_dir], "identity", out_root)

        assert "skipping 2 already done" in capsys.readouterr().out
        assert json.loads(index_path.read_text()) == {
            f"/raw/tier0/graph{i}.pt": N_NODES for i in range(3)
        }

    def test_driver_isolates_a_bad_graph(
        self, manifest_workspace: tuple[Path, Path]
    ) -> None:
        # One unusable graph must not abort the shard and discard the index
        # for everything already processed.
        meta_dir, out_root = manifest_workspace
        bad = Data(
            x=torch.zeros(2, 4),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 2)),
        )  # no level and no pos_enc
        torch.save(bad, meta_dir.parents[2] / "shared_tier0_cache" / "graph1.pt")

        summarize_from_manifests([meta_dir], "identity", out_root)

        out_dir = out_root / "shared_tier0_cache"
        stats = json.loads(
            (out_root / "_summary_stats_identity_shard000.json").read_text()
        )
        assert stats["graphs"] == 2
        assert stats["errors"] == 1
        assert json.loads((out_dir / _shard_index_name(0)).read_text()) == {
            "/raw/tier0/graph0.pt": N_NODES,
            "/raw/tier0/graph2.pt": N_NODES,
        }

    def test_shards_write_separate_indexes_and_merge(
        self, manifest_workspace: tuple[Path, Path]
    ) -> None:
        # Shards run on separate nodes and are unpacked into one directory,
        # so each must keep its own index until they are merged.
        meta_dir, out_root = manifest_workspace
        for shard_id in range(2):
            summarize_from_manifests(
                [meta_dir], "identity", out_root, shard_id=shard_id, num_shards=2
            )

        out_dir = out_root / "shared_tier0_cache"
        per_shard = [
            json.loads((out_dir / _shard_index_name(i)).read_text()) for i in range(2)
        ]
        assert not set(per_shard[0]) & set(per_shard[1])

        assert merge_shard_indexes(out_dir) == 3
        assert json.loads((out_dir / _NUM_NODES_GLOBAL).read_text()) == {
            f"/raw/tier0/graph{i}.pt": N_NODES for i in range(3)
        }

    def test_merge_shard_indexes_no_input(self, tmp_path: Path) -> None:
        assert merge_shard_indexes(tmp_path) == 0
        assert not (tmp_path / _NUM_NODES_GLOBAL).exists()
