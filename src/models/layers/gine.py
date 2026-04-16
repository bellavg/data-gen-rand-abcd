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
        hid_dim: int,
        num_layers: int,
        node_input_dim: int,
        edge_attr_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        norm_type: str = "batch",
        **kwargs,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_layers = num_layers
        # Optional input projection when node input dim != hidden dim
        self.node_input_dim = node_input_dim
        self.hid_dim = hid_dim
        self.input_proj = (
            nn.Linear(node_input_dim, hid_dim)
            if node_input_dim != hid_dim
            else nn.Identity()
        )

        # Hardcoded to 'concat' PE

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                GINEConvLayer(
                    hid_dim=hid_dim,
                    edge_dim=edge_attr_dim,
                    dropout=dropout,
                    norm_type=norm_type,
                )
            )

        # Hardcoded Jumping Knowledge = 'cat' output dimension
        self.out_dim = output_dim

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Tensor,
        edge_attr: OptTensor = None,
        pos_enc: OptTensor = None,
    ) -> Tensor:

        # Apply optional input projection then track hidden states for Jumping Knowledge
        x0 = self.input_proj(x)
        h_list = [x0]

        for layer in self.layers:
            x_out = layer(x=h_list[-1], edge_index=edge_index, edge_attr=edge_attr)
            h_list.append(x_out)

        if self.jk_mode == "last":
            node_emb = h_list[-1]
        elif self.jk_mode == "max":
            node_emb = torch.stack(h_list, dim=-1).max(dim=-1)[0]
        elif self.jk_mode == "sum":
            node_emb = torch.stack(h_list, dim=-1).sum(dim=-1)
        elif self.jk_mode == "cat":
            node_emb = torch.cat(h_list, dim=1)

        return node_emb
