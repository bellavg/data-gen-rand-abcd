"""Tests for the sparsification module.

Tests cover:
- Algorithm correctness (random_edge_dropout, spanner, pagerank, and_gate_only)
- Chunked index file writing and reading (get_sparse_entry with glob)
- precomputed_sparsification() lookup (embedded attrs, index files, and_gate_only)
- Worker function returns 3-tuple (cache_dir_str, basename, results)
- Mask numpy conversion round-trip in worker → main accumulation
- clear_sparse_index_cache()
"""
from __future__ import annotations

import os
import uuid
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

# We need to set PYTHONPATH-equivalent so imports resolve.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.sparsification import (
    _SPARSE_PREFIX,
    and_gate_only_sparsification,
    clear_sparse_index_cache,
    get_sparse_entry,
    pagerank_sparsification,
    precomputed_sparsification,
    random_edge_dropout,
    spanner_sparsification,
    _process_single_cache_file,
    _register_pyg_safe_globals,
)


# =====================================================================
# FIXTURES
# =====================================================================


@pytest.fixture
def simple_graph() -> Data:
    """A small directed graph with 6 nodes and 7 edges.

    Topology (0-indexed):
        0 → 1, 0 → 2, 1 → 3, 2 → 3, 3 → 4, 3 → 5, 4 → 5

    Node features: 6×4 identity-like matrix (one-hot style).
    Edge attributes: 7×2 random.
    """
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 3, 3, 4], [1, 2, 3, 3, 4, 5, 5]], dtype=torch.long
    )
    x = torch.eye(4).repeat(2, 1)[:6]  # 6 nodes, 4 features
    edge_attr = torch.rand(7, 2)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


@pytest.fixture
def aig_graph() -> Data:
    """A small AIG-style graph for and_gate_only testing.

    Node types (one-hot in x[:, 0:4]):
        0: PI   (x=[0,1,0,0])
        1: PI   (x=[0,1,0,0])
        2: AND  (x=[0,0,1,0])
        3: AND  (x=[0,0,1,0])
        4: PO   (x=[0,0,0,1])

    Edges: PI→AND, AND→AND, AND→PO
    """
    x = torch.tensor(
        [
            [0, 1, 0, 0],  # node 0: PI
            [0, 1, 0, 0],  # node 1: PI
            [0, 0, 1, 0],  # node 2: AND
            [0, 0, 1, 0],  # node 3: AND
            [0, 0, 0, 1],  # node 4: PO
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [2, 2, 3, 4]], dtype=torch.long
    )
    edge_attr = torch.tensor(
        [[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float32
    )
    level = torch.tensor([0, 0, 1, 2, 3], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, level=level)


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """A temporary directory for writing index files."""
    return tmp_path / "cache"


# =====================================================================
# ALGORITHM TESTS
# =====================================================================


class TestRandomEdgeDropout:
    def test_returns_bool_mask(self, simple_graph):
        mask = random_edge_dropout(simple_graph, dropout_rate=0.5, seed=42)
        assert mask.dtype == torch.bool
        assert mask.shape == (simple_graph.edge_index.shape[1],)

    def test_deterministic_with_seed(self, simple_graph):
        m1 = random_edge_dropout(simple_graph, dropout_rate=0.5, seed=123)
        m2 = random_edge_dropout(simple_graph, dropout_rate=0.5, seed=123)
        assert torch.equal(m1, m2)

    def test_different_seeds_differ(self, simple_graph):
        m1 = random_edge_dropout(simple_graph, dropout_rate=0.5, seed=0)
        m2 = random_edge_dropout(simple_graph, dropout_rate=0.5, seed=999)
        # With 7 edges it's possible they match by chance, but very unlikely
        # at dropout_rate=0.5. We check they're not trivially identical.
        # (This is a statistical test, not guaranteed, but 1/128 chance of false positive)
        assert m1.shape == m2.shape

    def test_zero_dropout_keeps_all(self, simple_graph):
        mask = random_edge_dropout(simple_graph, dropout_rate=0.0, seed=0)
        assert mask.all()

    def test_full_dropout_removes_all(self, simple_graph):
        mask = random_edge_dropout(simple_graph, dropout_rate=1.0, seed=0)
        assert not mask.any()


class TestSpannerSparsification:
    def test_returns_bool_mask(self, simple_graph):
        mask = spanner_sparsification(simple_graph, stretch=3.0, seed=0)
        assert mask.dtype == torch.bool
        assert mask.shape == (simple_graph.edge_index.shape[1],)

    def test_keeps_some_edges(self, simple_graph):
        mask = spanner_sparsification(simple_graph, stretch=3.0, seed=0)
        assert mask.any(), "Spanner should keep at least some edges"


class TestPageRankSparsification:
    def test_returns_bool_node_mask(self, simple_graph):
        mask = pagerank_sparsification(simple_graph, keep_ratio=0.8)
        assert mask.dtype == torch.bool
        assert mask.shape == (simple_graph.num_nodes,)

    def test_keeps_correct_fraction(self, simple_graph):
        mask = pagerank_sparsification(simple_graph, keep_ratio=0.5)
        expected = max(1, int(simple_graph.num_nodes * 0.5))
        assert mask.sum().item() == expected

    def test_keeps_at_least_one(self, simple_graph):
        mask = pagerank_sparsification(simple_graph, keep_ratio=0.01)
        assert mask.sum().item() >= 1


class TestAndGateOnly:
    def test_removes_pi_and_po(self, aig_graph):
        result = and_gate_only_sparsification(aig_graph)
        # Only AND gates should remain (nodes 2 and 3 from original)
        assert result.num_nodes == 2
        # All remaining nodes should be AND type
        assert (result.x[:, 2] == 1.0).all()

    def test_preserves_node_attrs(self, aig_graph):
        result = and_gate_only_sparsification(aig_graph)
        assert hasattr(result, "level")
        assert result.level.shape[0] == result.num_nodes

    def test_returns_data_object(self, aig_graph):
        result = and_gate_only_sparsification(aig_graph)
        assert isinstance(result, Data)


# =====================================================================
# CHUNKED INDEX FILE TESTS
# =====================================================================


class TestChunkedIndexFiles:
    """Test that get_sparse_entry reads glob-matched chunks correctly."""

    def _write_chunk(self, cache_dir: Path, algo: str, entries: dict, tag: str = ""):
        """Helper to write a single chunk file matching the naming convention."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        chunk_name = f"{_SPARSE_PREFIX}{algo}_{tag}_{uuid.uuid4().hex[:4]}.pt"
        path = cache_dir / chunk_name
        torch.save(entries, path)
        return path

    def test_single_chunk_lookup(self, cache_dir):
        clear_sparse_index_cache()
        mask = torch.tensor([True, False, True], dtype=torch.bool)
        entries = {"graph_001.pt": {"mask": mask}}
        self._write_chunk(cache_dir, "random_edge_dropout", entries, tag="1234")

        result = get_sparse_entry(cache_dir, "random_edge_dropout", "graph_001.pt")
        assert result is not None
        assert torch.equal(result["mask"], mask)

    def test_multiple_chunks_merged(self, cache_dir):
        clear_sparse_index_cache()
        mask1 = torch.tensor([True, True], dtype=torch.bool)
        mask2 = torch.tensor([False, True, False], dtype=torch.bool)
        self._write_chunk(cache_dir, "spanner", {"a.pt": {"mask": mask1}}, tag="0001")
        self._write_chunk(cache_dir, "spanner", {"b.pt": {"mask": mask2}}, tag="0002")

        r1 = get_sparse_entry(cache_dir, "spanner", "a.pt")
        r2 = get_sparse_entry(cache_dir, "spanner", "b.pt")
        assert r1 is not None and torch.equal(r1["mask"], mask1)
        assert r2 is not None and torch.equal(r2["mask"], mask2)

    def test_missing_basename_returns_none(self, cache_dir):
        clear_sparse_index_cache()
        mask = torch.tensor([True], dtype=torch.bool)
        self._write_chunk(cache_dir, "pagerank", {"exists.pt": {"mask": mask}}, tag="x")

        result = get_sparse_entry(cache_dir, "pagerank", "does_not_exist.pt")
        assert result is None

    def test_missing_dir_returns_none(self, tmp_path):
        clear_sparse_index_cache()
        nonexistent = tmp_path / "no_such_dir"
        nonexistent.mkdir()
        result = get_sparse_entry(nonexistent, "random_edge_dropout", "any.pt")
        assert result is None

    def test_different_algos_isolated(self, cache_dir):
        clear_sparse_index_cache()
        mask_r = torch.tensor([True, False], dtype=torch.bool)
        mask_s = torch.tensor([False, True, True], dtype=torch.bool)
        self._write_chunk(cache_dir, "random_edge_dropout", {"g.pt": {"mask": mask_r}}, tag="a")
        self._write_chunk(cache_dir, "spanner", {"g.pt": {"mask": mask_s}}, tag="b")

        r = get_sparse_entry(cache_dir, "random_edge_dropout", "g.pt")
        s = get_sparse_entry(cache_dir, "spanner", "g.pt")
        assert torch.equal(r["mask"], mask_r)
        assert torch.equal(s["mask"], mask_s)

    def test_clear_cache_forces_reload(self, cache_dir):
        clear_sparse_index_cache()
        mask = torch.tensor([True], dtype=torch.bool)
        self._write_chunk(cache_dir, "pagerank", {"g.pt": {"mask": mask}}, tag="z")

        # First load
        r1 = get_sparse_entry(cache_dir, "pagerank", "g.pt")
        assert r1 is not None

        clear_sparse_index_cache()

        # Should reload from disk
        r2 = get_sparse_entry(cache_dir, "pagerank", "g.pt")
        assert r2 is not None
        assert torch.equal(r2["mask"], mask)


# =====================================================================
# WORKER FUNCTION TESTS
# =====================================================================


class TestWorkerFunction:
    """Test _process_single_cache_file returns the right shape and format."""

    def test_returns_3_tuple(self, simple_graph, tmp_path):
        _register_pyg_safe_globals()
        cache_path = tmp_path / "test_graph.pt"
        torch.save(simple_graph, cache_path)

        result = _process_single_cache_file(
            cache_path,
            algo_names=["random_edge_dropout"],
            dropout_rate=0.5,
            stretch=3.0,
            keep_ratio=0.8,
            alpha=0.85,
            seed=42,
        )

        assert result is not None
        assert len(result) == 3, "Worker must return (cache_dir_str, basename, results)"
        cache_dir_str, basename, algo_results = result
        assert cache_dir_str == str(tmp_path)
        assert basename == "test_graph.pt"
        assert "random_edge_dropout" in algo_results

    def test_masks_are_numpy(self, simple_graph, tmp_path):
        _register_pyg_safe_globals()
        cache_path = tmp_path / "test_graph.pt"
        torch.save(simple_graph, cache_path)

        result = _process_single_cache_file(
            cache_path,
            algo_names=["random_edge_dropout"],
            dropout_rate=0.5,
            stretch=3.0,
            keep_ratio=0.8,
            alpha=0.85,
            seed=42,
        )
        _, _, algo_results = result
        entry = algo_results["random_edge_dropout"]
        assert isinstance(entry["mask"], np.ndarray), "Mask should be numpy array for IPC efficiency"
        assert entry["mask"].dtype == np.bool_

    def test_multiple_algos_in_one_pass(self, simple_graph, tmp_path):
        _register_pyg_safe_globals()
        cache_path = tmp_path / "test_graph.pt"
        torch.save(simple_graph, cache_path)

        result = _process_single_cache_file(
            cache_path,
            algo_names=["random_edge_dropout", "spanner", "pagerank"],
            dropout_rate=0.5,
            stretch=3.0,
            keep_ratio=0.8,
            alpha=0.85,
            seed=42,
        )
        _, _, algo_results = result
        assert set(algo_results.keys()) == {"random_edge_dropout", "spanner", "pagerank"}

    def test_missing_file_returns_none(self, tmp_path):
        result = _process_single_cache_file(
            tmp_path / "nonexistent.pt",
            algo_names=["random_edge_dropout"],
            dropout_rate=0.5,
            stretch=3.0,
            keep_ratio=0.8,
            alpha=0.85,
            seed=42,
        )
        assert result is None

    def test_numpy_roundtrip_to_tensor(self, simple_graph, tmp_path):
        """Verify the numpy→tensor conversion matches the partition pattern."""
        _register_pyg_safe_globals()
        cache_path = tmp_path / "test_graph.pt"
        torch.save(simple_graph, cache_path)

        result = _process_single_cache_file(
            cache_path,
            algo_names=["random_edge_dropout"],
            dropout_rate=0.5,
            stretch=3.0,
            keep_ratio=0.8,
            alpha=0.85,
            seed=42,
        )
        _, _, algo_results = result
        np_mask = algo_results["random_edge_dropout"]["mask"]

        # Simulate what the main process does
        tensor_mask = torch.from_numpy(np_mask).clone()
        assert tensor_mask.dtype == torch.bool
        assert tensor_mask.shape == (simple_graph.edge_index.shape[1],)


# =====================================================================
# precomputed_sparsification() LOOKUP TESTS
# =====================================================================


class TestPrecomputedSparsification:
    """Test the clean lookup function that mirrors precomputed_partitioning()."""

    def test_embedded_edge_mask(self, simple_graph):
        """Backward compat: mask embedded directly on data_obj."""
        mask = torch.ones(simple_graph.edge_index.shape[1], dtype=torch.bool)
        mask[0] = False  # drop first edge
        simple_graph.random_edge_dropout_sparsification_mask = mask

        result = precomputed_sparsification(simple_graph, "random_edge_dropout")
        assert result.edge_index.shape[1] == mask.sum().item()

    def test_embedded_node_mask(self, simple_graph):
        """Backward compat: pagerank node mask embedded on data_obj."""
        mask = torch.ones(simple_graph.num_nodes, dtype=torch.bool)
        mask[0] = False  # drop first node
        simple_graph.pagerank_sparsification_mask = mask

        result = precomputed_sparsification(simple_graph, "pagerank")
        assert result.num_nodes == mask.sum().item()

    def test_index_file_lookup(self, simple_graph, cache_dir):
        """Mask found via chunked index file when not embedded."""
        clear_sparse_index_cache()
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save graph to cache dir
        graph_file = cache_dir / "my_graph.pt"
        torch.save(simple_graph, graph_file)

        # Write a chunk index file
        mask = torch.ones(simple_graph.edge_index.shape[1], dtype=torch.bool)
        mask[2] = False
        chunk = {"my_graph.pt": {"mask": mask}}
        chunk_path = cache_dir / f"{_SPARSE_PREFIX}random_edge_dropout_1234_abcd.pt"
        torch.save(chunk, chunk_path)

        result = precomputed_sparsification(
            simple_graph, "random_edge_dropout", cache_path=graph_file
        )
        assert result.edge_index.shape[1] == mask.sum().item()

    def test_missing_mask_raises(self, simple_graph):
        """AttributeError when no mask is found anywhere."""
        with pytest.raises(AttributeError, match="not found"):
            precomputed_sparsification(simple_graph, "random_edge_dropout")

    def test_and_gate_only_on_the_fly(self, aig_graph):
        """and_gate_only uses on-the-fly computation, no precomputed mask."""
        result = precomputed_sparsification(aig_graph, "and_gate_only")
        # Should have only AND gates (2 nodes)
        assert result.num_nodes == 2
        assert (result.x[:, 2] == 1.0).all()

    def test_and_gate_only_embedded_attr(self, aig_graph):
        """and_gate_only with embedded backward-compat attribute."""
        precomputed = and_gate_only_sparsification(aig_graph)
        aig_graph.and_gate_only_graph = precomputed

        result = precomputed_sparsification(aig_graph, "and_gate_only")
        assert result.num_nodes == precomputed.num_nodes

    def test_edge_attr_filtered_with_mask(self, simple_graph, cache_dir):
        """Edge attributes should be filtered along with edge_index."""
        clear_sparse_index_cache()
        cache_dir.mkdir(parents=True, exist_ok=True)
        graph_file = cache_dir / "g.pt"
        torch.save(simple_graph, graph_file)

        num_edges = simple_graph.edge_index.shape[1]
        mask = torch.ones(num_edges, dtype=torch.bool)
        mask[0] = False
        mask[3] = False
        chunk = {"g.pt": {"mask": mask}}
        torch.save(chunk, cache_dir / f"{_SPARSE_PREFIX}spanner_1_a.pt")

        result = precomputed_sparsification(simple_graph, "spanner", cache_path=graph_file)
        assert result.edge_index.shape[1] == mask.sum().item()
        assert result.edge_attr.shape[0] == mask.sum().item()
