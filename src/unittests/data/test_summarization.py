"""Tests for the summarization module.

Tests cover:
- apply_merge_map identity losslessness (requirement R1) on both the raw
  schema (level) and the cached schema (pos_enc)
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
    apply_merge_map,
    color_refinement,
    identity_clustering,
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
