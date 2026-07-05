import torch
from torch_geometric.data import Data

# 3 nodes: 0, 1, 2
x = torch.tensor([[0.0], [1.0], [2.0]])
# edges: 0->1, 1->2, 0->2
edge_index = torch.tensor([[0, 1, 0],
                           [1, 2, 2]])

data = Data(x=x, edge_index=edge_index)
data.num_nodes = 3

print("Original:")
print("x:\n", data.x)
print("edge_index:\n", data.edge_index)

print("\n--- Testing Pagerank style (Node Mask) ---")
# Keep nodes 0 and 2. Drop node 1.
node_mask = torch.tensor([True, False, True])
sub_data = data.subgraph(node_mask)
print("x:\n", sub_data.x)
print("edge_index:\n", sub_data.edge_index)

print("\n--- Testing Dropout style (Edge Mask) ---")
# Keep edges 0->1 and 0->2. Drop edge 1->2.
edge_mask = torch.tensor([True, False, True])
data.edge_index = data.edge_index[:, edge_mask]
print("edge_index:\n", data.edge_index)
