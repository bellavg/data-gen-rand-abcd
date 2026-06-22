from __future__ import annotations

import unittest
import torch
from torch_geometric.data import Data
from data.partition import run_metis, run_level_bisect

class TestPartitionAlgorithms(unittest.TestCase):
    def test_run_metis_basic(self):
        # Symmetrized edges are required for METIS, but run_metis does this internally.
        # Symmetrized graph with 6 nodes
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                                   [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]], dtype=torch.long)
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

    def test_run_level_bisect_missing_level(self):
        data = Data(num_nodes=5)
        with self.assertRaises(AttributeError):
            run_level_bisect(data, num_partitions=2)

    def test_run_level_bisect_even_nodes(self):
        # 10 nodes with various levels
        level = torch.tensor([1, 4, 2, 5, 1, 2, 3, 3, 4, 5], dtype=torch.long)
        data = Data(level=level, num_nodes=10)
        
        mask = run_level_bisect(data, num_partitions=2)
        self.assertEqual(mask.shape, (10,))
        self.assertEqual(mask.dtype, torch.long)
        self.assertEqual(mask.device.type, "cpu")
        
        # Verify 50/50 split
        num_zero = (mask == 0).sum().item()
        num_one = (mask == 1).sum().item()
        self.assertEqual(num_zero, 5)
        self.assertEqual(num_one, 5)

        # Verify partition ordering:
        # All nodes in partition 0 should have levels <= all nodes in partition 1
        max_level_part0 = level[mask == 0].max().item()
        min_level_part1 = level[mask == 1].min().item()
        self.assertLessEqual(max_level_part0, min_level_part1)

    def test_run_level_bisect_odd_nodes(self):
        # 9 nodes
        level = torch.tensor([5, 4, 3, 2, 1, 2, 3, 4, 5], dtype=torch.long)
        data = Data(level=level, num_nodes=9)
        
        mask = run_level_bisect(data, num_partitions=2)
        self.assertEqual(mask.shape, (9,))
        
        # Verify split: half_size = 9 // 2 = 4 assigned to 0, 5 assigned to 1
        num_zero = (mask == 0).sum().item()
        num_one = (mask == 1).sum().item()
        self.assertEqual(num_zero, 4)
        self.assertEqual(num_one, 5)

        max_level_part0 = level[mask == 0].max().item()
        min_level_part1 = level[mask == 1].min().item()
        self.assertLessEqual(max_level_part0, min_level_part1)

    def test_run_level_bisect_single_node(self):
        level = torch.tensor([3], dtype=torch.long)
        data = Data(level=level, num_nodes=1)
        
        mask = run_level_bisect(data, num_partitions=2)
        self.assertEqual(mask.shape, (1,))
        self.assertEqual(mask[0].item(), 1) # half_size = 0, so assigned to 1

    def test_run_level_bisect_empty(self):
        level = torch.tensor([], dtype=torch.long)
        data = Data(level=level, num_nodes=0)
        
        mask = run_level_bisect(data, num_partitions=2)
        self.assertEqual(mask.shape, (0,))

    def test_random_partitioning_positional_arguments(self):
        from data.partition_utils import random_partitioning
        # Test 2 nodes
        data = Data(x=torch.randn(10, 4), edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=10)
        
        # Test positional passing
        part_data = random_partitioning(data, 4)
        self.assertEqual(part_data.num_partitions.item(), 4)
        self.assertTrue((part_data.partition_id >= 0).all())
        self.assertTrue((part_data.partition_id < 4).all())

    def test_precomputed_partitioning_direct_lookup(self):
        from data.partition_utils import precomputed_partitioning
        data = Data(x=torch.randn(10, 4), edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=10)
        
        # Test direct lookup: f"{algo_name}_{num_partitions}_mask"
        data.metis_4_mask = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 0, 1], dtype=torch.long)
        part_data = precomputed_partitioning(data, "metis", 4)
        self.assertEqual(part_data.num_partitions.item(), 4)
        self.assertTrue(torch.equal(part_data.partition_id, torch.sort(data.metis_4_mask)[0]))

        # Test invalid name or mismatch raises AttributeError
        with self.assertRaises(AttributeError):
            precomputed_partitioning(data, "metis", 2)
            
        with self.assertRaises(AttributeError):
            precomputed_partitioning(data, "non_existent", 4)

    def test_dataset_dynamic_partitioning_fallback(self):
        from data.dataset import AIGGraphRegressionDataset
        from unittest.mock import MagicMock

        # Create a mock dataset instance
        ds = MagicMock(spec=AIGGraphRegressionDataset)
        ds.partition = "level_bisect"
        ds.positional_encoding = None
        ds.normalize_edges = False
        ds.samples = [MagicMock(y_node_opt=0.5)]
        ds._load_graph_for_sample = MagicMock()
        
        # A mock graph data object with no precomputed mask
        level = torch.tensor([1, 2, 1, 2, 3, 3], dtype=torch.long)
        mock_graph = Data(x=torch.randn(6, 4), edge_index=torch.zeros((2, 0), dtype=torch.long), level=level, num_nodes=6)
        ds._load_graph_for_sample.return_value = mock_graph
        
        # Let's call the actual get method logic (from the base class / class method)
        # using our mock dataset as 'self'
        result = AIGGraphRegressionDataset.get(ds, 0)
        
        # Verify it partitioned dynamically on-the-fly!
        self.assertEqual(result.num_partitions.item(), 2)
        # Partition 0 gets the 3 lowest level nodes (half_size = 3)
        # Partition 1 gets the rest
        self.assertEqual((result.partition_id == 0).sum().item(), 3)
        self.assertEqual((result.partition_id == 1).sum().item(), 3)

if __name__ == "__main__":
    unittest.main()
