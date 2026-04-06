import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

try:
    from model_utils import get_pyg_pool, get_norm_layer
except ImportError:  # pragma: no cover - fallback for package-style imports
    from models.model_utils import get_pyg_pool, get_norm_layer

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
    def __init__(
        self,
        in_dim,
        hid_dim,
        num_layers,
        edge_dim,
        pos_enc_dim=0,
        pos_enc_mode='none',
        use_input_proj=True,
        use_edge_proj=True,
        dropout=0.0,
        norm_type='batch',
        readout='mean',
        jk='last',
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.jk = jk.lower()
        self.pos_enc_dim = pos_enc_dim
        self.pos_enc_mode = pos_enc_mode.lower()
        self.use_input_proj = use_input_proj
        self.use_edge_proj = use_edge_proj

        if self.pos_enc_mode not in {'none', 'concat', 'add'}:
            raise ValueError(f"Unknown pos_enc_mode: {pos_enc_mode}")

        effective_in_dim = in_dim
        if self.pos_enc_mode == 'concat' and pos_enc_dim > 0:
            effective_in_dim += pos_enc_dim
        
        if self.use_input_proj:
            self.node_encoder = nn.Linear(effective_in_dim, hid_dim)
        else:
            if effective_in_dim != hid_dim:
                raise ValueError(
                    f"use_input_proj=False requires effective_in_dim ({effective_in_dim}) == hid_dim ({hid_dim})"
                )
            self.node_encoder = nn.Identity()

        if self.use_edge_proj:
            self.edge_encoder = nn.Linear(edge_dim, hid_dim)
        else:
            if edge_dim != hid_dim:
                raise ValueError(
                    f"use_edge_proj=False requires edge_dim ({edge_dim}) == hid_dim ({hid_dim})"
                )
            self.edge_encoder = nn.Identity()
        self.pos_add_proj = (
            nn.Linear(pos_enc_dim, hid_dim, bias=False)
            if self.pos_enc_mode == 'add' and pos_enc_dim > 0
            else None
        )
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(VanillaMPNNConv(hid_dim))
            self.norms.append(get_norm_layer(norm_type, hid_dim))
            
        if self.jk == 'cat':
            self.out_dim = hid_dim * (num_layers + 1)
        else:
            self.out_dim = hid_dim

    def _validate_positional_encoding(self, pos_enc):
        if pos_enc is None or self.pos_enc_dim == 0 or self.pos_enc_mode == 'none':
            return
        if pos_enc.size(-1) != self.pos_enc_dim:
            raise ValueError(
                f"Expected pos_enc with feature size {self.pos_enc_dim}, got {pos_enc.size(-1)}"
            )

    def _integrate_positional_encoding_input(self, x, pos_enc):
        if pos_enc is None or self.pos_enc_dim == 0 or self.pos_enc_mode != 'concat':
            return x
        return torch.cat([x, pos_enc], dim=-1)

    def forward(self, x, edge_index, batch, edge_attr=None, edge_type=None, pos_enc=None):
        # edge_type is accepted for interface consistency and is unused in MPNN.
        _ = edge_type
        self._validate_positional_encoding(pos_enc)
        if edge_attr is None:
            raise ValueError('MPNNEncoder requires edge_attr tensor.')

        x = self._integrate_positional_encoding_input(x, pos_enc)
        x = self.node_encoder(x)
        if pos_enc is not None and self.pos_add_proj is not None:
            x = x + self.pos_add_proj(pos_enc)

        edge_attr = self.edge_encoder(edge_attr)
        
        h_list = [x]
        
        for i in range(self.num_layers):
            h = self.convs[i](h_list[i], edge_index, edge_attr)
            h = self.norms[i](h)
            
            if i != self.num_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
                
            h_list.append(h)
            
        if self.jk == 'cat':
            node_emb = torch.cat(h_list, dim=1)
        elif self.jk == 'sum':
            node_emb = sum(h_list)
        else:
            node_emb = h_list[-1]
            
        return node_emb