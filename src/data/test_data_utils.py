import unittest
from pathlib import Path

from src.data.data_utils import aig_to_pytorch_geometric


class TestAigToPytorchGeometric(unittest.TestCase):
    def test_with_adder_aig(self) -> None:
        aig_path = Path(__file__).with_name("adder.aig")
        self.assertTrue(aig_path.exists(), f"Missing fixture AIG file: {aig_path}")

        data = aig_to_pytorch_geometric(aig_path)

        self.assertGreater(data.num_nodes, 0)

        self.assertEqual(data.x.shape, (data.num_nodes, 4))
        self.assertEqual(data.level.shape, (data.num_nodes, 1))
        self.assertEqual(data.pi_paths.shape, (data.num_nodes, 1))
        self.assertEqual(data.local_sp_sum.shape, (data.num_nodes, 1))

        self.assertEqual(data.edge_index.ndim, 2)
        self.assertEqual(data.edge_index.shape[0], 2)
        self.assertEqual(data.edge_attr.ndim, 2)
        self.assertEqual(data.edge_attr.shape[1], 2)
        self.assertEqual(data.edge_attr.shape[0], data.edge_index.shape[1])

        self.assertEqual(data.rel_edge_index.ndim, 2)
        self.assertEqual(data.rel_edge_index.shape[0], 2)
        self.assertEqual(data.edge_rel_dist.ndim, 2)
        self.assertEqual(data.edge_rel_dist.shape[1], 1)
        self.assertEqual(data.edge_rel_dist.shape[0], data.rel_edge_index.shape[1])

        self.assertTrue(hasattr(data, "aig_meta"))
        counts = data.aig_meta["counts"]
        self.assertEqual(counts["num_nodes"], data.num_nodes)
        self.assertEqual(counts["num_edges"], data.edge_index.shape[1])
        self.assertGreaterEqual(counts["num_inputs"], 0)
        self.assertGreaterEqual(counts["num_outputs"], 0)

        for row_sum in data.x.sum(dim=1).tolist():
            self.assertAlmostEqual(row_sum, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
