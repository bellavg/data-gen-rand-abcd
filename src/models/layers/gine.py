import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GINEConv
from torch_geometric.typing import Adj, OptTensor

try:
    from model_utils import get_norm_layer
except ImportError:  # pragma: no cover - fallback for package-style imports
    try:
        from models.model_utils import get_norm_layer
    except ImportError:
        from src.models.model_utils import get_norm_layer


class GINEConvLayer(nn.Module):
    """
    Graph Isomorphism Network with Edge features (GINE) layer,
    enhanced with GNN+ architectural components (Residuals, FFN).
    """

    def __init__(
        self, hid_dim: int, edge_dim: int | None, dropout: float, norm_type: str
    ):
        super().__init__()
        self.dropout = dropout

        # GINE Core MLP (Inner MLP for the GIN update)
        gin_nn = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            get_norm_layer(norm_type, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        )

        # PyG's GINEConv automatically handles edge projection if edge_dim is passed
        self.model = GINEConv(nn=gin_nn, edge_dim=edge_dim)

        self.norm_node = get_norm_layer(norm_type, hid_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        # Feed Forward Network (FFN) - Hardcoded to True per GNN+ paper
        self.norm1_local = get_norm_layer(norm_type, hid_dim)
        self.ff_linear1 = nn.Linear(hid_dim, hid_dim * 2)
        self.ff_linear2 = nn.Linear(hid_dim * 2, hid_dim)
        self.ff_act = nn.ReLU()
        self.norm2 = get_norm_layer(norm_type, hid_dim)

    def _ff_block(self, x: Tensor) -> Tensor:
        """Feed Forward block."""
        x = self.ff_act(self.ff_linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ff_linear2(x)
        return F.dropout(x, p=self.dropout, training=self.training)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor) -> Tensor:
        x_in = x

        # 1. Message Passing
        x = self.model(x, edge_index, edge_attr=edge_attr)
        x = self.norm_node(x)
        x = self.act(x)
        x = self.drop(x)

        # 2. Residual Connection - Hardcoded to True per GNN+ paper
        if x_in.shape == x.shape:
            x = x_in + x

        # 3. FFN Block - Hardcoded to True per GNN+ paper
        x = self.norm1_local(x)
        x = x + self._ff_block(x)
        x = self.norm2(x)

        return x


class GINEEncoder(nn.Module):
    """Edge-aware GINE+ encoder with hardcoded PE concatenation and Jumping Knowledge (cat)."""

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        num_layers: int,
        edge_dim: int | None = None,
        pos_enc_dim: int = 0,
        use_input_proj: bool = True,
        dropout: float = 0.0,
        norm_type: str = "batch",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_layers = num_layers
        self.pos_enc_dim = pos_enc_dim
        self.use_input_proj = use_input_proj

        # Hardcoded to 'concat' PE
        effective_in_dim = in_dim + pos_enc_dim if pos_enc_dim > 0 else in_dim

        if self.use_input_proj:
            self.input_proj = nn.Linear(effective_in_dim, hid_dim)
        else:
            if effective_in_dim != hid_dim:
                raise ValueError(
                    f"use_input_proj=False requires effective_in_dim ({effective_in_dim}) == hid_dim ({hid_dim})"
                )
            self.input_proj = nn.Identity()

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                GINEConvLayer(
                    hid_dim=hid_dim,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    norm_type=norm_type,
                )
            )

        # Hardcoded Jumping Knowledge = 'cat' output dimension
        self.out_dim = hid_dim * (num_layers + 1)

    def _validate_positional_encoding(self, pos_enc: OptTensor) -> None:
        if pos_enc is None or self.pos_enc_dim == 0:
            return
        if pos_enc.size(-1) != self.pos_enc_dim:
            raise ValueError(
                f"Expected pos_enc with feature size {self.pos_enc_dim}, got {pos_enc.size(-1)}"
            )

    def _integrate_positional_encoding(self, x: Tensor, pos_enc: OptTensor) -> Tensor:
        if pos_enc is None or self.pos_enc_dim == 0:
            return x
        return torch.cat([x, pos_enc], dim=-1)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Tensor,
        edge_attr: OptTensor = None,
        pos_enc: OptTensor = None,
    ) -> Tensor:
        # edge_type removed; encoders receive `edge_attr` when needed

        self._validate_positional_encoding(pos_enc)
        x = self._integrate_positional_encoding(x, pos_enc)

        x = self.input_proj(x)

        # Track hidden states for Jumping Knowledge
        h_list = [x]

        for layer in self.layers:
            x_out = layer(x=h_list[-1], edge_index=edge_index, edge_attr=edge_attr)
            h_list.append(x_out)

        # Hardcoded Jumping Knowledge = 'cat'
        return torch.cat(h_list, dim=-1)
