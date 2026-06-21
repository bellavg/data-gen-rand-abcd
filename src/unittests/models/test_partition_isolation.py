import unittest
import torch
from torch_geometric.data import Batch, Data
from data.partition_utils import random_partitioning
from models.base_model import UnifiedGraphBaseModel


def _make_dummy_graph(seed: int = 42) -> Data:
    g = torch.Generator().manual_seed(seed)
    num_nodes = 12
    num_edges = 16
    x = torch.randn(num_nodes, 4, generator=g)
    src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=g)
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.randn(num_edges, 2, generator=g)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)


class TestPartitionIsolation(unittest.TestCase):
    def test_partition_isolation_all_encoders(self):
        """Verify partition isolation across the GCN GNN encoder and local/graph normalization types."""
        encoders = [
            "gcn",
        ]
        norm_types = ["none", "graph", "layer"]
        modes = ["eval", "train"]

        for encoder_name in encoders:
            for norm_type in norm_types:
                for mode in modes:
                    with self.subTest(encoder=encoder_name, norm=norm_type, mode=mode):
                        # Create model
                        encoder_kwargs = {
                            "num_layers": 2,
                            "hid_dim": 8,
                            "norm_type": norm_type,
                        }

                        model = UnifiedGraphBaseModel(
                            encoder_name=encoder_name,
                            hidden_dim=8,
                            encoder_kwargs=encoder_kwargs,
                            pe_type="none",
                            node_input_dim=4,
                            edge_attr_dim=2,
                            task_out_dim=1,
                        )

                        if mode == "eval":
                            model.eval()
                        else:
                            model.train()

                        # Partition the graph
                        data = _make_dummy_graph(seed=42)
                        part_data = random_partitioning(data, num_partitions=2)

                        mask_p0 = part_data.partition_id == 0
                        mask_p1 = part_data.partition_id == 1

                        self.assertTrue(mask_p0.any())
                        self.assertTrue(mask_p1.any())

                        # Setup forward hook to capture node embeddings
                        node_embs = []
                        def hook_fn(module, input, output):
                            node_embs.append(output)

                        hook = model.encoder.register_forward_hook(hook_fn)

                        # 1. Original forward pass
                        batch_orig = Batch.from_data_list([part_data])
                        _ = model.forward_batch(batch_orig)
                        emb_orig = node_embs[-1].clone()

                        # 2. Modify Partition 1 only (just one node to prevent GraphNorm cancellation)
                        part_data_a = part_data.clone()
                        part_data_a.x = part_data.x.clone()
                        idx_p1 = mask_p1.nonzero()[0].item()
                        part_data_a.x[idx_p1] += 10.0
                        batch_a = Batch.from_data_list([part_data_a])
                        _ = model.forward_batch(batch_a)
                        emb_a = node_embs[-1].clone()

                        # 3. Modify Partition 0 only (just one node)
                        part_data_b = part_data.clone()
                        part_data_b.x = part_data.x.clone()
                        idx_p0 = mask_p0.nonzero()[0].item()
                        part_data_b.x[idx_p0] += 10.0
                        batch_b = Batch.from_data_list([part_data_b])
                        _ = model.forward_batch(batch_b)
                        emb_b = node_embs[-1].clone()

                        hook.remove()

                        # Check Partition 0: Modifying Partition 1 must have 0 effect
                        p0_diff = torch.abs(emb_orig[mask_p0] - emb_a[mask_p0]).max().item()
                        self.assertAlmostEqual(
                            p0_diff, 0.0, places=4,
                            msg=f"Pollution detected in {encoder_name} ({norm_type} norm, {mode} mode): "
                                f"modifying Partition 1 changed Partition 0 by {p0_diff}"
                        )

                        # Check Partition 1: Modifying Partition 0 must have 0 effect
                        p1_diff = torch.abs(emb_orig[mask_p1] - emb_b[mask_p1]).max().item()
                        self.assertAlmostEqual(
                            p1_diff, 0.0, places=4,
                            msg=f"Pollution detected in {encoder_name} ({norm_type} norm, {mode} mode): "
                                f"modifying Partition 0 changed Partition 1 by {p1_diff}"
                        )

    def test_batch_norm_violates_isolation_in_training(self):
        """Demonstrate and verify that BatchNorm violates partition isolation in training mode."""
        encoder_kwargs = {
            "num_layers": 2,
            "hid_dim": 8,
            "norm_type": "batch",
        }

        model = UnifiedGraphBaseModel(
            encoder_name="gcn",
            hidden_dim=8,
            encoder_kwargs=encoder_kwargs,
            pe_type="none",
            node_input_dim=4,
            edge_attr_dim=2,
            task_out_dim=1,
        )
        model.train()

        data = _make_dummy_graph(seed=42)
        part_data = random_partitioning(data, num_partitions=2)

        mask_p0 = part_data.partition_id == 0
        mask_p1 = part_data.partition_id == 1

        node_embs = []
        def hook_fn(module, input, output):
            node_embs.append(output)

        hook = model.encoder.register_forward_hook(hook_fn)

        # 1. Original forward pass
        batch_orig = Batch.from_data_list([part_data])
        _ = model.forward_batch(batch_orig)
        emb_orig = node_embs[-1].clone()

        # 2. Modify Partition 1 only (single node)
        part_data_a = part_data.clone()
        part_data_a.x = part_data.x.clone()
        idx_p1 = mask_p1.nonzero()[0].item()
        part_data_a.x[idx_p1] += 10.0
        batch_a = Batch.from_data_list([part_data_a])
        _ = model.forward_batch(batch_a)
        emb_a = node_embs[-1].clone()

        hook.remove()

        # Partition 0 embeddings MUST change because BatchNorm normalizes across all nodes in the batch.
        p0_diff = torch.abs(emb_orig[mask_p0] - emb_a[mask_p0]).max().item()
        self.assertGreater(
            p0_diff, 1e-3,
            msg=f"Expected BatchNorm to violate partition isolation, but difference was only {p0_diff}"
        )


if __name__ == "__main__":
    unittest.main()

