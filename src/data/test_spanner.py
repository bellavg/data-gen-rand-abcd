import torch
from torch_geometric.data import Data
from data.sparsification import spanning_forest_sparsification

# Create a simple DAG with some redundant paths
# 0 -> 1 -> 3
# 0 -> 2 -> 3
# 0 -> 3 (shortcut)
edge_index = torch.tensor([
    [0, 1, 0, 2, 0],
    [1, 3, 2, 3, 3]
], dtype=torch.long)

data = Data(edge_index=edge_index)
print("Original edges:", data.edge_index.size(1))

mask = spanning_forest_sparsification(data, seed=42)
print("Spanning Forest mask:", mask)
print("Kept edges:", mask.sum().item())
