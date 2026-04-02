import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
from src.arch import DAGConv
from model_utils import compute_gso_from_adj, dense_pool

class DAGConvEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, bias=True, dropout=0.0, readout='mean'):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.bias = bias
        
        self.dropout = nn.Dropout(dropout)
        self.readout = readout
        self.encoder = None 
        
    def forward(self, x, adj=None, edge_index=None, num_nodes=None, batch=None):
        device = x.device
        
        if adj is not None:
            adj_np = adj.cpu().detach().numpy()
            if len(adj_np.shape) == 3: 
                adj_np = adj_np[0] 
        elif edge_index is not None:
            adj_dense = to_dense_adj(edge_index, max_num_nodes=num_nodes)
            adj_np = adj_dense[0].cpu().detach().numpy()
        else:
            raise ValueError("Must provide either 'adj' or 'edge_index'.")

        N = adj_np.shape[0]
        gso = compute_gso_from_adj(adj_np).to(device)
        K = gso.shape[0]

        if self.encoder is None:
            self.encoder = DAGConv(
                in_dim=self.in_dim, 
                hid_dim=self.hid_dim, 
                out_dim=self.out_dim, 
                K=K, 
                n_layers=self.n_layers, 
                bias=self.bias
            ).to(device)

        node_embs = self.encoder(x, gso)
        node_embs = self.dropout(node_embs)
        
        graph_emb = dense_pool(node_embs, self.readout)
            
        return graph_emb