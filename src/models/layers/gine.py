import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GINEConv
from torch_geometric.typing import Adj

from models.model_utils import get_norm_layer


class GINEConvLayer(nn.Module):
    """
    Graph Isomorphism Network with Edge features (GINE) layer,
    enhanced with GNN+ architectural components (Residuals, FFN).
    """

    def __init__(
        self,
        dim_in: int,
        hid_dim: int,
        edge_dim: int | None,
        dropout: float,
        norm_type: str,
    ):
        super().__init__()
        self.dropout = dropout

        # GINE Core MLP (Inner MLP for the GIN update)
        gin_nn = nn.Sequential(
            nn.Linear(dim_in, hid_dim),
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

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        x_in = x

        # 1. Message Passing
        x = self.model(x, edge_index, edge_attr=edge_attr)
        from models.model_utils import apply_norm

        x = apply_norm(self.norm_node, x, batch)
        x = self.act(x)
        x = self.drop(x)

        # 2. Residual Connection - Hardcoded to True per GNN+ paper
        if x_in.shape == x.shape:
            x = x_in + x

        # 3. FFN Block - Hardcoded to True per GNN+ paper
        x = apply_norm(self.norm1_local, x, batch)
        x = x + self._ff_block(x)
        x = apply_norm(self.norm2, x, batch)

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
        self.jk_mode = kwargs.get("jk_mode", "cat")  # Default to 'cat' if not provided
        # Project initial embedding purely for Jumping Knowledge uniformity
        if node_input_dim != hid_dim:
            self.jk_proj = nn.Linear(node_input_dim, hid_dim)
        else:
            self.jk_proj = nn.Identity()

        # First layer expects `node_input_dim`; subsequent layers expect `hid_dim`.

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            dim_in = node_input_dim if i == 0 else hid_dim
            self.layers.append(
                GINEConvLayer(
                    dim_in=dim_in,
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
        edge_attr: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:

        # Project to a uniform hidden dimension for Jumping Knowledge
        x_jk = self.jk_proj(x)

        if self.jk_mode == "cat":
            h_list = [x_jk]
            for layer in self.layers:
                x = layer(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
                h_list.append(x)
            return torch.cat(h_list, dim=1)

        elif self.jk_mode == "max":
            res = x_jk
            for layer in self.layers:
                x = layer(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
                res = torch.max(res, x)
            return res

        elif self.jk_mode == "sum":
            res = x_jk
            for layer in self.layers:
                x = layer(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
                res = res + x
            return res

        elif self.jk_mode == "last":
            for layer in self.layers:
                x = layer(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            return x
