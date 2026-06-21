"""Unit tests for data.partition_utils: PartitionedData and random_partitioning."""
from __future__ import annotations

import unittest

import torch
from torch_geometric.data import Batch, Data

from data.partition_utils import PartitionedData, random_partitioning


def _make_graph(num_nodes: int = 6, num_edges: int = 8, seed: int = 0) -> Data:
    """Return a small synthetic PyG Data object."""
    g = torch.Generator()
    g.manual_seed(seed)
    return Data(
        x=torch.randn(num_nodes, 4, generator=g),
        edge_index=torch.randint(0, num_nodes, (2, num_edges), generator=g),
        edge_attr=torch.randn(num_edges, 2, generator=g),
        pos_enc=torch.randn(num_nodes, 1, generator=g),
        num_nodes=num_nodes,
    )


class TestRandomPartitioning(unittest.TestCase):
    # ------------------------------------------------------------------
    # Return type & attribute presence
    # ------------------------------------------------------------------

    def test_returns_partitioned_data_instance(self):
        data = random_partitioning(_make_graph())
        self.assertIsInstance(data, PartitionedData)

    def test_partition_id_shape_and_dtype(self):
        g = _make_graph(num_nodes=10)
        data = random_partitioning(g)
        self.assertEqual(data.partition_id.shape, (10,))
        self.assertEqual(data.partition_id.dtype, torch.long)

    def test_num_partitions_shape_and_value(self):
        data = random_partitioning(_make_graph())
        self.assertEqual(data.num_partitions.shape, (1,))
        self.assertEqual(data.num_partitions.dtype, torch.long)
        self.assertEqual(data.num_partitions.item(), 2)

    def test_custom_num_partitions(self):
        data = random_partitioning(_make_graph(num_nodes=12), num_partitions=4)
        self.assertEqual(data.num_partitions.item(), 4)
        self.assertTrue((data.partition_id >= 0).all())
        self.assertTrue((data.partition_id < 4).all())

    # ------------------------------------------------------------------
    # All nodes and edges are retained
    # ------------------------------------------------------------------

    def test_all_nodes_retained(self):
        g = _make_graph(num_nodes=10)
        data = random_partitioning(g)
        self.assertEqual(data.x.shape[0], 10)
        self.assertEqual(data.num_nodes, 10)

    def test_cross_partition_edges_dropped(self):
        torch.manual_seed(42)
        g = _make_graph(num_nodes=24, num_edges=32)
        data = random_partitioning(g)
        src, dst = data.edge_index
        # Every surviving edge must connect nodes in the same partition
        self.assertTrue((data.partition_id[src] == data.partition_id[dst]).all())
        # Edge count is <= original (cross-partition edges removed)
        self.assertLessEqual(data.edge_index.shape[1], 32)

    def test_partition_id_is_sorted_and_contiguous(self):
        """Verify that partition_id values are grouped contiguously in non-decreasing order."""
        g = _make_graph(num_nodes=30)
        data = random_partitioning(g)
        # A sorted tensor will always have a non-negative diff between successive entries
        self.assertTrue(torch.all(data.partition_id[:-1] <= data.partition_id[1:]))

    def test_edge_attr_always_present_and_in_sync(self):
        """edge_attr is always present and must stay aligned with edge_index."""
        torch.manual_seed(42)
        g = _make_graph(num_nodes=24, num_edges=32)
        data = random_partitioning(g)
        self.assertIsNotNone(data.edge_attr)
        self.assertEqual(data.edge_attr.shape[0], data.edge_index.shape[1])

    def test_pos_enc_retained(self):
        g = _make_graph(num_nodes=6)
        data = random_partitioning(g)
        self.assertIsNotNone(getattr(data, "pos_enc", None))
        self.assertEqual(data.pos_enc.shape[0], 6)

    # ------------------------------------------------------------------
    # Partition label values
    # ------------------------------------------------------------------

    def test_partition_id_values_in_range(self):
        g = _make_graph(num_nodes=20)
        data = random_partitioning(g)
        self.assertTrue((data.partition_id >= 0).all())
        self.assertTrue((data.partition_id < 2).all())

    def test_both_partitions_represented(self):
        """With 100 nodes the probability of one partition being empty is ~2^-100."""
        torch.manual_seed(42)
        g = _make_graph(num_nodes=100)
        data = random_partitioning(g)
        unique = data.partition_id.unique()
        self.assertIn(0, unique)
        self.assertIn(1, unique)


class TestPartitionedDataBatching(unittest.TestCase):
    """Verify that Batch.from_data_list increments partition_id correctly."""

    def _make_partitioned(self, num_nodes: int, seed: int) -> PartitionedData:
        torch.manual_seed(seed)
        g = _make_graph(num_nodes=num_nodes, seed=seed)
        return random_partitioning(g)

    def test_batch_has_correct_total_nodes(self):
        g0 = self._make_partitioned(6, seed=0)
        g1 = self._make_partitioned(8, seed=1)
        batch = Batch.from_data_list([g0, g1])
        self.assertEqual(batch.x.shape[0], 14)

    def test_partition_id_globally_unique_across_batch(self):
        """Graph 0 gets [0,1], Graph 1 gets [2,3] (offset by num_partitions=2)."""
        g0 = self._make_partitioned(6, seed=0)
        g1 = self._make_partitioned(6, seed=1)
        batch = Batch.from_data_list([g0, g1])

        pid = batch.partition_id
        # First graph's nodes must have partition_id < 2
        g0_mask = batch.batch == 0
        self.assertTrue((pid[g0_mask] < 2).all(), pid[g0_mask])
        # Second graph's nodes must have partition_id in {2, 3}
        g1_mask = batch.batch == 1
        self.assertTrue((pid[g1_mask] >= 2).all(), pid[g1_mask])
        self.assertTrue((pid[g1_mask] < 4).all(), pid[g1_mask])

    def test_num_partitions_stacked_per_graph(self):
        g0 = self._make_partitioned(6, seed=0)
        g1 = self._make_partitioned(6, seed=1)
        g2 = self._make_partitioned(6, seed=2)
        batch = Batch.from_data_list([g0, g1, g2])
        # num_partitions should be [2, 2, 2] — one per graph
        self.assertEqual(batch.num_partitions.shape, (3,))
        torch.testing.assert_close(
            batch.num_partitions, torch.tensor([2, 2, 2], dtype=torch.long)
        )

    def test_three_graphs_partition_id_offsets(self):
        """With 3 graphs each having num_partitions=2, offsets are 0, 2, 4."""
        g0 = self._make_partitioned(4, seed=0)
        g1 = self._make_partitioned(4, seed=1)
        g2 = self._make_partitioned(4, seed=2)
        batch = Batch.from_data_list([g0, g1, g2])
        pid = batch.partition_id
        for graph_idx, expected_offset in enumerate([0, 2, 4]):
            mask = batch.batch == graph_idx
            self.assertTrue(
                (pid[mask] >= expected_offset).all(),
                f"graph {graph_idx}: partition_id not >= {expected_offset}",
            )
            self.assertTrue(
                (pid[mask] < expected_offset + 2).all(),
                f"graph {graph_idx}: partition_id not < {expected_offset + 2}",
            )

    def test_edge_index_incremented_correctly(self):
        """edge_index should still be offset by num_nodes (standard PyG behaviour)."""
        g0 = self._make_partitioned(4, seed=0)
        g1 = self._make_partitioned(4, seed=1)
        batch = Batch.from_data_list([g0, g1])
        # Nodes of g1 should start at index 4
        g1_mask = batch.batch == 1
        g1_node_indices = g1_mask.nonzero(as_tuple=True)[0]
        self.assertEqual(g1_node_indices.min().item(), 4)

    def test_batch_vector_shape(self):
        g0 = self._make_partitioned(5, seed=0)
        g1 = self._make_partitioned(7, seed=1)
        batch = Batch.from_data_list([g0, g1])
        self.assertEqual(batch.batch.shape, (12,))
        self.assertTrue((batch.batch[:5] == 0).all())
        self.assertTrue((batch.batch[5:] == 1).all())


class TestDatasetIntegration(unittest.TestCase):
    """End-to-end: dataset.get() with partition='random' returns PartitionedData."""

    def setUp(self):
        import csv
        import tempfile
        from pathlib import Path

        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

        # Build a small .pt file
        g = Data(
            x=torch.randn(8, 4),
            edge_index=torch.randint(0, 8, (2, 12)),
            edge_attr=torch.randn(12, 2),
            level=torch.randint(0, 5, (8, 1)).float(),
            pi_paths=torch.rand(8, 1),
            local_sp_sum=torch.rand(8, 1),
        )
        pt_path = self.root / "graph.pt"
        torch.save(g, pt_path)

        csv_path = self.root / "data.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "unoptimized_graph_path",
                    "design",
                    "algorithm",
                    "tier_id",
                    "optimizability",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "unoptimized_graph_path": str(pt_path),
                    "design": "test",
                    "algorithm": "Orchestrate",
                    "tier_id": "1",
                    "optimizability": "0.5",
                }
            )
        self.csv_path = csv_path

    def tearDown(self):
        self.tmp.cleanup()

    def _make_ds(self, **kwargs):
        from data.dataset import AIGGraphRegressionDataset

        return AIGGraphRegressionDataset(self.csv_path, **kwargs)

    def test_partition_none_returns_plain_data(self):
        ds = self._make_ds(partition=None)
        item = ds[0]
        self.assertNotIsInstance(item, PartitionedData)

    def test_partition_random_returns_partitioned_data(self):
        ds = self._make_ds(partition="random")
        item = ds[0]
        self.assertIsInstance(item, PartitionedData)

    def test_partition_random_has_partition_id(self):
        ds = self._make_ds(partition="random")
        item = ds[0]
        self.assertTrue(hasattr(item, "partition_id"))
        self.assertEqual(item.partition_id.shape, (item.x.shape[0],))

    def test_partition_random_has_num_partitions(self):
        ds = self._make_ds(partition="random")
        item = ds[0]
        self.assertTrue(hasattr(item, "num_partitions"))
        self.assertEqual(item.num_partitions.item(), 2)

    def test_partition_random_all_nodes_retained(self):
        ds_no_part = self._make_ds(partition=None)
        ds_part = self._make_ds(partition="random")
        self.assertEqual(ds_part[0].x.shape[0], ds_no_part[0].x.shape[0])

    def test_partition_random_batch_collates_correctly(self):
        ds = self._make_ds(partition="random")
        items = [ds[0], ds[0]]  # two items from same graph
        batch = Batch.from_data_list(items)
        # Second graph's partition_ids should be offset by 2
        g1_mask = batch.batch == 1
        self.assertTrue((batch.partition_id[g1_mask] >= 2).all())


if __name__ == "__main__":
    unittest.main()
