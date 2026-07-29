from __future__ import annotations

import unittest

import pytorch_lightning as pl
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from baselines.common.lightning_wrapper import BaselineRegressionLightningModule
from baselines.openabc_synthnet.regressor import (
    SynthNetGraphRegressor,
    derive_node_type_index,
    derive_num_inverted_predecessors,
)

NUM_NODES = 12
NUM_EDGES = 16
NODE_INPUT_DIM = 4  # [constant, pi, and_gate, po]
EDGE_ATTR_DIM = 2  # [regular, inverted]


def _make_aig_data(seed: int = 42) -> Data:
    g = torch.Generator().manual_seed(seed)

    # One-hot node types, matching data/data_utils.py's [const, pi, and, po] layout.
    type_idx = torch.randint(0, NODE_INPUT_DIM, (NUM_NODES,), generator=g)
    x = torch.zeros(NUM_NODES, NODE_INPUT_DIM)
    x[torch.arange(NUM_NODES), type_idx] = 1.0

    src = torch.randint(0, NUM_NODES, (NUM_EDGES,), generator=g)
    dst = torch.randint(0, NUM_NODES, (NUM_EDGES,), generator=g)
    edge_index = torch.stack([src, dst], dim=0)

    inv = (torch.rand(NUM_EDGES, generator=g) > 0.5).float()
    edge_attr = torch.stack([1.0 - inv, inv], dim=1)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.rand(1, 1, generator=g),
    )


class TestFeatureAdapter(unittest.TestCase):
    def test_derive_node_type_index_matches_one_hot_argmax(self):
        x = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],  # constant
                [0.0, 1.0, 0.0, 0.0],  # pi
                [0.0, 0.0, 1.0, 0.0],  # and
                [0.0, 0.0, 0.0, 1.0],  # po
            ]
        )
        idx = derive_node_type_index(x)
        self.assertTrue(torch.equal(idx, torch.tensor([0, 1, 2, 3])))

    def test_derive_num_inverted_predecessors_counts_incoming_inverted_edges(self):
        # Edges: 0->2 (regular), 1->2 (inverted), 1->3 (inverted). Node 2 gets
        # exactly 1 inverted incoming edge, node 3 gets 1, nodes 0/1 get 0.
        edge_index = torch.tensor([[0, 1, 1], [2, 2, 3]])
        edge_attr = torch.tensor(
            [
                [1.0, 0.0],  # 0->2 regular
                [0.0, 1.0],  # 1->2 inverted
                [0.0, 1.0],  # 1->3 inverted
            ]
        )
        counts = derive_num_inverted_predecessors(edge_index, edge_attr, num_nodes=4)
        self.assertEqual(counts.dtype, torch.long)
        self.assertTrue(torch.equal(counts, torch.tensor([0, 0, 1, 1])))

    def test_derive_num_inverted_predecessors_handles_no_incoming_edges(self):
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2))
        counts = derive_num_inverted_predecessors(edge_index, edge_attr, num_nodes=3)
        self.assertTrue(torch.equal(counts, torch.zeros(3, dtype=torch.long)))


class TestSynthNetGraphRegressor(unittest.TestCase):
    def test_forward_pass_shape_and_range(self):
        model = SynthNetGraphRegressor(
            node_emb_dim=3, gnn_hidden_dim=8, num_fc_layer=2, fc_hidden_dim=16
        )
        model.eval()
        data = _make_aig_data()
        batch = Batch.from_data_list([data])

        out = model(batch)
        self.assertEqual(out.shape, (1, 1))
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0))

    def test_batch_independence(self):
        model = SynthNetGraphRegressor(
            node_emb_dim=3, gnn_hidden_dim=8, num_fc_layer=2, fc_hidden_dim=16
        )
        model.eval()
        data1 = _make_aig_data(seed=1)
        data2 = _make_aig_data(seed=2)

        out1_alone = model(Batch.from_data_list([data1]))
        out2_alone = model(Batch.from_data_list([data2]))
        out_combined = model(Batch.from_data_list([data1, data2]))

        self.assertTrue(torch.allclose(out1_alone[0], out_combined[0], atol=1e-4))
        self.assertTrue(torch.allclose(out2_alone[0], out_combined[1], atol=1e-4))

    def test_gradient_flows(self):
        model = SynthNetGraphRegressor(
            node_emb_dim=3, gnn_hidden_dim=8, num_fc_layer=2, fc_hidden_dim=16
        )
        data = _make_aig_data()
        out = model(Batch.from_data_list([data]))
        out.mean().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"Broken graph at {name}")

    def test_invalid_num_fc_layer_raises(self):
        with self.assertRaises(ValueError):
            SynthNetGraphRegressor(num_fc_layer=1)


class TestSynthNetLightningTraining(unittest.TestCase):
    def setUp(self):
        self.dataset = [_make_aig_data(seed=i) for i in range(10)]
        self.train_loader = DataLoader(self.dataset[:6], batch_size=2)
        self.val_loader = DataLoader(self.dataset[6:8], batch_size=2)
        self.test_loader = DataLoader(self.dataset[8:], batch_size=2)

    def test_training_and_testing_loop(self):
        base_model = SynthNetGraphRegressor(
            node_emb_dim=3, gnn_hidden_dim=8, num_fc_layer=2, fc_hidden_dim=16
        )
        model = BaselineRegressionLightningModule(
            base_model, lr=1e-3, loss_fn=torch.nn.MSELoss()
        )

        trainer = pl.Trainer(
            fast_dev_run=True,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(
            model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
        )
        trainer.test(model, dataloaders=self.test_loader)


if __name__ == "__main__":
    unittest.main()
