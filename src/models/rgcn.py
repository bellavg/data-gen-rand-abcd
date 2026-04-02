import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from model_utils import get_norm_layer, get_pyg_pool

class RGCNEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, num_relations, dropout=0.0, norm_type='batch', readout='mean'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.pool_fn = get_pyg_pool(readout)
        
        for i in range(num_layers):
            in_channels = in_dim if i == 0 else hid_dim
            out_channels = out_dim if i == num_layers - 1 else hid_dim
            self.convs.append(RGCNConv(in_channels, out_channels, num_relations))
            self.norms.append(get_norm_layer(norm_type, out_channels))

    def forward(self, x, edge_index, edge_type, batch):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_type)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            
        return self.pool_fn(x, batch)