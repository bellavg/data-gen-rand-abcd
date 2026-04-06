import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
try:
    from model_utils import get_norm_layer
except ImportError:  # pragma: no cover - fallback for package-style imports
    from models.model_utils import get_norm_layer

class RGCNEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        num_layers,
        num_relations,
        pos_enc_dim=0,
        pos_enc_mode='none',
        use_input_proj=True,
        dropout=0.0,
        norm_type='batch',
        readout='mean',
    ):
        super().__init__()
        self.pos_enc_dim = pos_enc_dim
        self.pos_enc_mode = pos_enc_mode.lower()
        self.use_input_proj = use_input_proj

        if self.pos_enc_mode not in {'none', 'concat', 'add'}:
            raise ValueError(f"Unknown pos_enc_mode: {pos_enc_mode}")

        effective_in_dim = in_dim + pos_enc_dim if self.pos_enc_mode == 'concat' and pos_enc_dim > 0 else in_dim
        if self.use_input_proj:
            self.input_proj = nn.Linear(effective_in_dim, hid_dim)
        else:
            if effective_in_dim != hid_dim:
                raise ValueError(
                    f"use_input_proj=False requires effective_in_dim ({effective_in_dim}) == hid_dim ({hid_dim})"
                )
            self.input_proj = nn.Identity()

        self.pos_add_proj = (
            nn.Linear(pos_enc_dim, hid_dim, bias=False)
            if self.pos_enc_mode == 'add' and pos_enc_dim > 0
            else None
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        for i in range(num_layers):
            in_channels = hid_dim if i == 0 else hid_dim
            out_channels = out_dim if i == num_layers - 1 else hid_dim
            self.convs.append(RGCNConv(in_channels, out_channels, num_relations))
            self.norms.append(get_norm_layer(norm_type, out_channels))

    def _validate_positional_encoding(self, pos_enc):
        if pos_enc is None or self.pos_enc_dim == 0 or self.pos_enc_mode == 'none':
            return
        if pos_enc.size(-1) != self.pos_enc_dim:
            raise ValueError(
                f"Expected pos_enc with feature size {self.pos_enc_dim}, got {pos_enc.size(-1)}"
            )

    def _integrate_positional_encoding(self, x, pos_enc):
        if pos_enc is None or self.pos_enc_dim == 0 or self.pos_enc_mode != 'concat':
            return x
        return torch.cat([x, pos_enc], dim=-1)

    @staticmethod
    def _edge_type_from_edge_attr(edge_attr):
        if edge_attr is None:
            return None
        if edge_attr.dim() == 1:
            return edge_attr.long()
        if edge_attr.size(-1) == 1:
            return edge_attr.view(-1).long()
        return edge_attr.argmax(dim=-1).long()

    def forward(self, x, edge_index, batch=None, edge_attr=None, edge_type=None, pos_enc=None):
        self._validate_positional_encoding(pos_enc)
        x = self._integrate_positional_encoding(x, pos_enc)
        x = self.input_proj(x)

        if pos_enc is not None and self.pos_add_proj is not None:
            x = x + self.pos_add_proj(pos_enc)

        if edge_type is None:
            edge_type = self._edge_type_from_edge_attr(edge_attr)
        if edge_type is None:
            raise ValueError('RGCNEncoder requires edge_type or derivable edge_attr.')

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_type)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            
        return x