import torch
from torch_geometric.data import Data
from pathlib import Path
import os
import shutil

# Create dummy directories
base_dir = Path("/tmp/dummy_graphs")
if base_dir.exists():
    shutil.rmtree(base_dir)

tier0 = base_dir / "tier0"
tier1 = base_dir / "tier1"
tier0.mkdir(parents=True)
tier1.mkdir(parents=True)

# Create some dummy graphs
def make_dummy_graph():
    edge_index = torch.tensor([
        [0, 1, 0, 2, 0],
        [1, 3, 2, 3, 3]
    ], dtype=torch.long)
    # Give them dummy x features so n_nodes is inferred correctly
    x = torch.zeros((4, 4))
    x[:, 1] = 0.0 # PI
    x[:, 3] = 0.0 # PO
    return Data(edge_index=edge_index, x=x)

for i in range(5):
    torch.save(make_dummy_graph(), tier0 / f"graph_tier0_{i}.pt")
    torch.save(make_dummy_graph(), tier1 / f"graph_tier1_{i}.pt")

print("Dummy graphs created. Running measure_sparsity.py...")
