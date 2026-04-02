import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from model_utils import get_pyg_pool, get_norm_layer

class VanillaMPNNConv(MessagePassing):
    """
    A vanilla Message Passing Neural Network layer.
    Message: MLP(concat(source_node, target_node, edge_feature))
    Update: MLP(concat(node, aggregated_messages))
    """
    def __init__(self, hid_dim):
        super(VanillaMPNNConv, self).__init__(aggr="add")  # Sum aggregation is standard for MPNN
        
        # Message MLP
        self.msg_mlp = nn.Sequential(
            nn.Linear(3 * hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU()
        )
        
        # Node Update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i represents the target node, x_j represents the source node
        # Concatenate source, target, and edge features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.msg_mlp(msg_input)

    def update(self, aggr_out, x):
        # aggr_out is the aggregated messages for each node
        # Concatenate the original node features with the aggregated message
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)


class MPNNEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, num_layers, edge_dim, dropout=0.0, norm_type='batch', readout='mean', jk='last'):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.jk = jk.lower()
        self.pool_fn = get_pyg_pool(readout)
        
        # Project initial node and edge features to hidden dimension
        self.node_encoder = nn.Linear(in_dim, hid_dim)
        self.edge_encoder = nn.Linear(edge_dim, hid_dim)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(VanillaMPNNConv(hid_dim))
            self.norms.append(get_norm_layer(norm_type, hid_dim))
            
        if self.jk == 'cat':
            self.out_dim = hid_dim * (num_layers + 1)
        else:
            self.out_dim = hid_dim

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        h_list = [x]
        
        for i in range(self.num_layers):
            h = self.convs[i](h_list[i], edge_index, edge_attr)
            h = self.norms[i](h)
            
            if i != self.num_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
                
            h_list.append(h)
            
        pooled_h_list = [self.pool_fn(h, batch) for h in h_list]
        
        if self.jk == 'cat':
            graph_emb = torch.cat(pooled_h_list, dim=1)
        elif self.jk == 'sum':
            graph_emb = sum(pooled_h_list)
        else:
            graph_emb = pooled_h_list[-1]
            
        return graph_emb