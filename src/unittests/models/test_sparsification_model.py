import pytest
import torch
from torch_geometric.data import Data, Batch

from data.sparsification import (
    random_edge_dropout,
    spanning_forest_sparsification,
    pagerank_sparsification,
    and_gate_only_sparsification,
    precomputed_sparsification,
)
from models.lightning_model import AIGRegressionLightningModule

@pytest.fixture
def dummy_graphs():
    """Create a list of two dummy graphs with attributes required by the model."""
    graphs = []
    for _ in range(2):
        x = torch.eye(4).repeat(3, 1)[:10]  # 10 nodes, 4 features
        edge_index = torch.randint(0, 10, (2, 20), dtype=torch.long)
        edge_attr = torch.rand(20, 2)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
    return graphs

@pytest.fixture
def aig_graphs():
    """Create AIG-like graphs specifically for and_gate_only testing."""
    graphs = []
    for _ in range(2):
        x = torch.tensor(
            [
                [0, 1, 0, 0],  # PI
                [0, 1, 0, 0],  # PI
                [0, 0, 1, 0],  # AND
                [0, 0, 1, 0],  # AND
                [0, 0, 0, 1],  # PO
            ],
            dtype=torch.float32,
        )
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [2, 2, 3, 4]], dtype=torch.long
        )
        edge_attr = torch.tensor(
            [[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float32
        )
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
    return graphs

@pytest.fixture
def model():
    encoder_kwargs = {
        "node_input_dim": 4,
        "edge_attr_dim": 2,
        "hid_dim": 16,
        "num_layers": 2,
        "dropout": 0.0,
        "jk_mode": "cat",
        "norm_type": "layer",
        "normalize_edges": False,
    }
    return AIGRegressionLightningModule(
        encoder_name="gcn",
        hidden_dim=16,
        encoder_kwargs=encoder_kwargs,
        node_input_dim=4,
        edge_attr_dim=2,
        task_out_dim=1,
        compile_model=False,
    )

def test_random_edge_dropout_forward(dummy_graphs, model):
    sparsified_graphs = []
    for g in dummy_graphs:
        mask = random_edge_dropout(g, dropout_rate=0.5, seed=42)
        g.random_edge_dropout_sparsification_mask = mask
        sg = precomputed_sparsification(g, "random_edge_dropout")

        # Verify continuous node IDs check
        assert sg.edge_index.max() < sg.num_nodes, "Edge index out of bounds!"
        assert sg.num_nodes == g.num_nodes, "Edge dropout should not remove nodes"

        sparsified_graphs.append(sg)

    batch = Batch.from_data_list(sparsified_graphs)
    out = model(batch)
    assert out.shape == (2, 1), f"Expected shape (2, 1), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaNs"

def test_spanning_forest_forward(dummy_graphs, model):
    sparsified_graphs = []
    for g in dummy_graphs:
        mask = spanning_forest_sparsification(g, seed=42)
        g.spanning_forest_sparsification_mask = mask
        sg = precomputed_sparsification(g, "spanning_forest")

        assert sg.edge_index.max() < sg.num_nodes if sg.edge_index.numel() > 0 else True
        assert sg.num_nodes == g.num_nodes

        sparsified_graphs.append(sg)

    batch = Batch.from_data_list(sparsified_graphs)
    out = model(batch)
    assert out.shape == (2, 1)
    assert not torch.isnan(out).any()

def test_pagerank_forward(dummy_graphs, model):
    sparsified_graphs = []
    for g in dummy_graphs:
        mask = pagerank_sparsification(g, keep_ratio=0.5)
        g.pagerank_sparsification_mask = mask
        sg = precomputed_sparsification(g, "pagerank")

        # Node removal!
        # subgraph() reindexes automatically
        assert sg.num_nodes <= g.num_nodes
        if sg.edge_index.numel() > 0:
            assert sg.edge_index.max() < sg.num_nodes, "Node IDs are NOT continuous!"

        sparsified_graphs.append(sg)

    batch = Batch.from_data_list(sparsified_graphs)
    out = model(batch)
    assert out.shape == (2, 1)
    assert not torch.isnan(out).any()

def test_and_gate_only_forward(aig_graphs, model):
    sparsified_graphs = []
    for g in aig_graphs:
        orig_num_nodes = g.num_nodes
        mask = and_gate_only_sparsification(g)
        g.and_gate_only_sparsification_mask = mask
        sg = precomputed_sparsification(g, "and_gate_only")

        # Node removal AND specific logic (precomputed_sparsification mutates
        # in place, so compare against the count captured before the call).
        assert sg.num_nodes < orig_num_nodes
        if sg.edge_index.numel() > 0:
            assert sg.edge_index.max() < sg.num_nodes, "Node IDs are NOT continuous!"

        sparsified_graphs.append(sg)

    batch = Batch.from_data_list(sparsified_graphs)
    out = model(batch)
    assert out.shape == (2, 1)
    assert not torch.isnan(out).any()
