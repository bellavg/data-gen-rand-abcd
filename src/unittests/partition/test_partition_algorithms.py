from __future__ import annotations

import unittest
import torch
from torch_geometric.data import Data
from data.partition import run_metis, run_span_weighted_metis, run_level_slicing, run_random, compute_dynamic_k


class TestRunMetis(unittest.TestCase):
    def test_basic(self):
        # Symmetrized graph with 6 nodes; run_metis handles symmetrization internally.
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
             [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]], dtype=torch.long
        )
        data = Data(edge_index=edge_index, num_nodes=6)

        try:
            mask = run_metis(data, num_partitions=2)
        except ImportError as e:
            if "requires either" in str(e):
                self.skipTest(f"Skipping METIS test: {e}")
            raise

        self.assertEqual(mask.shape, (6,))
        self.assertEqual(mask.dtype, torch.long)
        self.assertEqual(mask.device.type, "cpu")
        self.assertTrue((mask >= 0).all())
        self.assertTrue((mask < 2).all())


class TestRunSpanWeightedMetis(unittest.TestCase):
    def test_basic(self):
        # Symmetrized graph with 6 nodes and levels attribute.
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
             [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]], dtype=torch.long
        )
        level = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
        data = Data(edge_index=edge_index, level=level, num_nodes=6)

        try:
            mask = run_span_weighted_metis(data, num_partitions=2)
        except ImportError as e:
            self.skipTest(f"Skipping Span-Weighted METIS test: {e}")
            return

        self.assertEqual(mask.shape, (6,))
        self.assertEqual(mask.dtype, torch.long)
        self.assertEqual(mask.device.type, "cpu")
        self.assertTrue((mask >= 0).all())
        self.assertTrue((mask < 2).all())

    def test_span_weighted_differs_from_unweighted(self):
        # Create a 4-node cycle: 0-1, 1-3, 3-2, 2-0
        edge_index = torch.tensor(
            [[0, 1, 1, 3, 3, 2, 2, 0],
             [1, 0, 3, 1, 2, 3, 0, 2]], dtype=torch.long
        )
        # Assign levels: 0 and 2 are 0, 1 and 3 are 100.
        # This makes edges (0,1) and (2,3) have huge span (100),
        # whereas edges (0,2) and (1,3) have span 0.
        level = torch.tensor([0, 100, 0, 100], dtype=torch.long)
        data = Data(edge_index=edge_index, level=level, num_nodes=4)

        try:
            mask_unweighted = run_metis(data, num_partitions=2)
            mask_weighted = run_span_weighted_metis(data, num_partitions=2, alpha=100.0)
        except ImportError as e:
            self.skipTest(f"Skipping METIS comparison test: {e}")
            return

        # 1. Verify shapes
        self.assertEqual(mask_unweighted.shape, (4,))
        self.assertEqual(mask_weighted.shape, (4,))

        # 2. For the weighted version, it should strictly avoid cutting the high-span edges:
        # (0,1) and (2,3). Therefore, node 0 and 1 must have the same partition,
        # and node 2 and 3 must have the same partition.
        self.assertEqual(mask_weighted[0], mask_weighted[1])
        self.assertEqual(mask_weighted[2], mask_weighted[3])

        # 3. Standard unweighted METIS has no level awareness, so it partitions
        # the graph such that it cuts (0,1) and (2,3) to get {0, 2} and {1, 3}.
        self.assertNotEqual(mask_unweighted[0], mask_unweighted[1])
        self.assertNotEqual(mask_unweighted[2], mask_unweighted[3])


class TestRunLevelSlicing(unittest.TestCase):
    def test_missing_level_raises(self):
        data = Data(num_nodes=5)
        with self.assertRaises(AttributeError):
            run_level_slicing(data, num_partitions=2)

    def test_invalid_num_partitions_raises(self):
        level = torch.tensor([1, 2, 3], dtype=torch.long)
        data = Data(level=level, num_nodes=3)
        with self.assertRaises(ValueError):
            run_level_slicing(data, num_partitions=0)

    def test_even_nodes_two_partitions(self):
        # 10 nodes, 2 equal buckets → exactly 5 each.
        level = torch.tensor([1, 4, 2, 5, 1, 2, 3, 3, 4, 5], dtype=torch.long)
        data = Data(level=level, num_nodes=10)

        mask = run_level_slicing(data, num_partitions=2)
        self.assertEqual(mask.shape, (10,))
        self.assertEqual(mask.dtype, torch.long)
        self.assertEqual(mask.device.type, "cpu")

        self.assertEqual((mask == 0).sum().item(), 5)
        self.assertEqual((mask == 1).sum().item(), 5)

        # Lower-level nodes belong to bucket 0.
        max_level_part0 = level[mask == 0].max().item()
        min_level_part1 = level[mask == 1].min().item()
        self.assertLessEqual(max_level_part0, min_level_part1)

    def test_odd_nodes_two_partitions(self):
        # 9 nodes: floor(i*2/9) for i=0..8 → [0,0,0,0,1,1,1,1,1]
        # → bucket 0 gets positions 0-4 (5 nodes), bucket 1 gets positions 5-8 (4 nodes).
        level = torch.tensor([5, 4, 3, 2, 1, 2, 3, 4, 5], dtype=torch.long)
        data = Data(level=level, num_nodes=9)

        mask = run_level_slicing(data, num_partitions=2)
        self.assertEqual(mask.shape, (9,))
        self.assertEqual((mask == 0).sum().item(), 5)
        self.assertEqual((mask == 1).sum().item(), 4)

        max_level_part0 = level[mask == 0].max().item()
        min_level_part1 = level[mask == 1].min().item()
        self.assertLessEqual(max_level_part0, min_level_part1)


    def test_four_partitions(self):
        # 12 nodes, 4 equal buckets → exactly 3 each.
        level = torch.arange(12, dtype=torch.long)
        data = Data(level=level, num_nodes=12)

        mask = run_level_slicing(data, num_partitions=4)
        self.assertEqual(mask.shape, (12,))
        self.assertTrue((mask >= 0).all())
        self.assertTrue((mask < 4).all())
        for p in range(4):
            self.assertEqual((mask == p).sum().item(), 3)

    def test_single_node(self):
        # 1 node → floor(0 * K / 1) = 0 for any K, clamped to K-1 at most.
        level = torch.tensor([3], dtype=torch.long)
        data = Data(level=level, num_nodes=1)

        mask = run_level_slicing(data, num_partitions=2)
        self.assertEqual(mask.shape, (1,))
        # New equal-bucket formula: floor(0 * 2 / 1) = 0
        self.assertEqual(mask[0].item(), 0)

    def test_empty(self):
        level = torch.tensor([], dtype=torch.long)
        data = Data(level=level, num_nodes=0)

        mask = run_level_slicing(data, num_partitions=2)
        self.assertEqual(mask.shape, (0,))


    def test_dynamic_k_heuristic(self):
        """compute_dynamic_k clamps correctly at min/max boundaries."""
        # 5000 nodes → k = 5000//10000 = 0, clamped to min_k=2
        self.assertEqual(compute_dynamic_k(5_000, 10_000, 2, 32), 2)
        # 50000 nodes → k = 50000//10000 = 5
        self.assertEqual(compute_dynamic_k(50_000, 10_000, 2, 32), 5)
        # 1_000_000 nodes → k = 100, clamped to max_k=32
        self.assertEqual(compute_dynamic_k(1_000_000, 10_000, 2, 32), 32)


class TestRunRandom(unittest.TestCase):
    def _make_data(self, n: int = 20) -> Data:
        return Data(
            x=torch.randn(n, 4),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            num_nodes=n,
        )

    def test_shape_and_dtype(self):
        data = self._make_data(20)
        mask = run_random(data, num_partitions=4, seed=0)
        self.assertEqual(mask.shape, (20,))
        self.assertEqual(mask.dtype, torch.long)

    def test_values_in_range(self):
        data = self._make_data(50)
        mask = run_random(data, num_partitions=3, seed=7)
        self.assertTrue((mask >= 0).all())
        self.assertTrue((mask < 3).all())

    def test_reproducible_with_same_seed(self):
        data = self._make_data(30)
        mask_a = run_random(data, num_partitions=2, seed=42)
        mask_b = run_random(data, num_partitions=2, seed=42)
        self.assertTrue(torch.equal(mask_a, mask_b))

    def test_different_seeds_differ(self):
        data = self._make_data(100)
        mask_a = run_random(data, num_partitions=4, seed=1)
        mask_b = run_random(data, num_partitions=4, seed=2)
        # With 100 nodes and 4 partitions, getting identical masks is astronomically unlikely.
        self.assertFalse(torch.equal(mask_a, mask_b))

    def test_two_partitions(self):
        data = self._make_data(10)
        mask = run_random(data, num_partitions=2, seed=0)
        self.assertTrue((mask >= 0).all())
        self.assertTrue((mask < 2).all())


class TestPrecomputedPartitioning(unittest.TestCase):
    def test_direct_lookup(self):
        from data.partition_utils import precomputed_partitioning
        data = Data(
            x=torch.randn(10, 4),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            num_nodes=10,
        )
        # Use the new dynamic-key format written by the precompute pipeline
        data.metis_dynamic_mask = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 0, 1], dtype=torch.long)
        data.metis_dynamic_num_partitions = torch.tensor([4], dtype=torch.long)

        part_data = precomputed_partitioning(data, "metis")
        self.assertEqual(part_data.num_partitions.item(), 4)
        self.assertTrue(torch.equal(part_data.partition_id, torch.sort(data.metis_dynamic_mask)[0]))

    def test_missing_mask_raises_attribute_error(self):
        from data.partition_utils import precomputed_partitioning
        data = Data(
            x=torch.randn(10, 4),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            num_nodes=10,
        )
        with self.assertRaises(AttributeError):
            precomputed_partitioning(data, "metis")

        with self.assertRaises(AttributeError):
            precomputed_partitioning(data, "non_existent")

    def test_index_file_roundtrip(self):
        """Full round-trip: precompute writes index → precomputed_partitioning reads it via cache_path."""
        import tempfile
        from pathlib import Path
        from data.partition import update_existing_cache_with_masks
        from data.partition_utils import precomputed_partitioning, clear_mask_index_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Graph with 8 nodes, no pre-existing mask.
            graph = Data(
                x=torch.randn(8, 4),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=8,
            )
            graph.level = torch.zeros(8, dtype=torch.long)
            cache_path = tmp_path / "graph.pt"
            torch.save(graph, cache_path)

            # Precompute: writes _masks_random.pt index file, does NOT touch graph.pt.
            update_existing_cache_with_masks(
                directories=[tmp_path],
                algo_names=["random"],
                seed=99,
            )

            # graph.pt must still have no embedded mask.
            reloaded = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.assertFalse(hasattr(reloaded, "random_dynamic_mask"),
                             "graph.pt should not be modified")

            # Clear the in-process cache so the lookup reads from disk.
            clear_mask_index_cache()

            # Read path: precomputed_partitioning finds the mask via cache_path.
            result = precomputed_partitioning(reloaded, "random", cache_path=cache_path)

            self.assertEqual(result.num_nodes, 8)
            self.assertEqual(result.num_partitions.item(), 2)  # 8 nodes / 10000 → k=2 (min_k)
            self.assertEqual(result.partition_id.shape, (8,))
            self.assertTrue((result.partition_id >= 0).all())
            self.assertTrue((result.partition_id < 2).all())

    def test_embedded_attrs_take_precedence_over_index(self):
        """Embedded attributes are used first; the index file is not consulted."""
        import tempfile
        from pathlib import Path
        from data.partition_utils import precomputed_partitioning, clear_mask_index_cache, _get_mask_entry

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cache_path = tmp_path / "graph.pt"

            # Write an index that says k=4, but embed k=2 directly on the object.
            index = {
                "graph.pt": {
                    "mask": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long),
                    "k":    torch.tensor([4], dtype=torch.long),
                }
            }
            torch.save(index, tmp_path / "_masks_random.pt")

            clear_mask_index_cache()

            graph = Data(
                x=torch.randn(8, 4),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=8,
            )
            # Embedded mask says k=2 — should win over the index's k=4.
            graph.random_dynamic_mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
            graph.random_dynamic_num_partitions = torch.tensor([2], dtype=torch.long)

            result = precomputed_partitioning(graph, "random", cache_path=cache_path)
            self.assertEqual(result.num_partitions.item(), 2,
                             "Embedded attr (k=2) should take precedence over index (k=4)")


class TestDatasetPartitionFallback(unittest.TestCase):
    """Verifies that dataset.get() raises RuntimeError when the precomputed
    mask is missing — dynamic on-the-fly computation is no longer supported."""

    def test_missing_mask_raises_runtime_error(self):
        from data.dataset import AIGGraphRegressionDataset
        from unittest.mock import MagicMock

        ds = MagicMock(spec=AIGGraphRegressionDataset)
        ds.partition = "level_slicing"
        ds.positional_encoding = None
        ds.normalize_edges = False
        ds.samples = [MagicMock(y_node_opt=0.5)]
        ds._y_tensors = [torch.tensor([[0.5]], dtype=torch.float32)]
        ds._load_graph_for_sample = MagicMock()
        ds._load_partition_cached_graph = MagicMock(return_value=(None, None))
        # _graph_cache_path_map is an instance attr set in __init__; mock needs it explicitly.
        ds._graph_cache_path_map = {}

        # Graph has NO precomputed mask and no index file → should raise AttributeError.
        mock_graph = Data(
            x=torch.randn(6, 4),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            level=torch.tensor([1, 2, 1, 2, 3, 3], dtype=torch.long),
            num_nodes=6,
        )
        ds._load_graph_for_sample.return_value = mock_graph

        with self.assertRaises(AttributeError):
            AIGGraphRegressionDataset.get(ds, 0)

    def test_precomputed_random_mask_is_applied(self):
        """random is now an offline algorithm; the precomputed dynamic mask should be used."""
        from data.dataset import AIGGraphRegressionDataset
        from unittest.mock import MagicMock

        ds = MagicMock(spec=AIGGraphRegressionDataset)
        ds.partition = "random"
        ds.positional_encoding = None
        ds.normalize_edges = False
        ds.samples = [MagicMock(y_node_opt=0.5)]
        ds._y_tensors = [torch.tensor([[0.5]], dtype=torch.float32)]
        ds._load_graph_for_sample = MagicMock()
        # _graph_cache_path_map is an instance attr set in __init__; mock needs it explicitly.
        ds._graph_cache_path_map = {}

        # Provide a precomputed random_dynamic_mask embedded in the graph object
        # (backward-compat path — precomputed_partitioning checks embedded attrs first).
        mock_graph = Data(
            x=torch.randn(6, 4),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            num_nodes=6,
        )
        mock_graph.random_dynamic_mask = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
        mock_graph.random_dynamic_num_partitions = torch.tensor([2], dtype=torch.long)
        ds._load_graph_for_sample.return_value = mock_graph

        result = AIGGraphRegressionDataset.get(ds, 0)
        self.assertEqual(result.num_partitions.item(), 2)


class TestUpdateExistingCacheWithMasks(unittest.TestCase):
    def test_update_existing_cache(self):
        import tempfile
        from pathlib import Path
        from data.partition import update_existing_cache_with_masks
        from torch_geometric.data import Data

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Flat cache dir — scandir is non-recursive (matches production layout).
            data1 = Data(x=torch.randn(10, 4), edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=10)
            data2 = Data(x=torch.randn(5, 4), edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=5)
            data1.level = torch.zeros(10, dtype=torch.long)
            data2.level = torch.zeros(5, dtype=torch.long)

            file1 = tmp_path / "graph1.pt"
            file2 = tmp_path / "graph2.pt"
            # Non-.pt file should be silently ignored by the scanner.
            file_other = tmp_path / "readme.txt"

            torch.save(data1, file1)
            torch.save(data2, file2)
            with open(file_other, "w") as f:
                f.write("This is a dummy text file")

            update_existing_cache_with_masks(
                directories=[tmp_path],
                algo_names=["random"],
                seed=42,
            )

            # Masks now live in the index file, NOT embedded in the graph .pt files.
            index_files = list(tmp_path.glob("_masks_random*.pt"))
            self.assertEqual(len(index_files), 1, "Exactly one index file should be created")
            index_path = index_files[0]
            self.assertTrue(index_path.is_file(), "Index file should have been created")

            index = torch.load(index_path, map_location="cpu", weights_only=True)
            self.assertIn("graph1.pt", index, "graph1.pt should be in the index")
            self.assertIn("graph2.pt", index, "graph2.pt should be in the index")
            self.assertIn("mask", index["graph1.pt"])
            self.assertIn("k", index["graph1.pt"])
            self.assertEqual(index["graph1.pt"]["mask"].shape[0], 10)
            self.assertEqual(index["graph2.pt"]["mask"].shape[0], 5)

            # Graph .pt files must NOT have been modified.
            updated1 = torch.load(file1, map_location="cpu", weights_only=False)
            updated2 = torch.load(file2, map_location="cpu", weights_only=False)
            self.assertFalse(hasattr(updated1, "random_dynamic_mask"),
                             "Graph .pt should not have embedded masks")
            self.assertFalse(hasattr(updated2, "random_dynamic_mask"),
                             "Graph .pt should not have embedded masks")

            # Non-.pt file must be untouched.
            with open(file_other) as f:
                self.assertEqual(f.read(), "This is a dummy text file")


if __name__ == "__main__":
    unittest.main()
