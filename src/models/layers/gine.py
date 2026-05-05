import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn  # Standard alias for PyG layers
from torch import Tensor
from torch_geometric.nn import GINEConv
from torch_geometric.typing import Adj

from models.model_utils import apply_norm, get_norm_layer


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

        # 1. GINE Core MLP using gnn.Linear for GNN-optimized weight init.
        gin_nn = nn.Sequential(
            gnn.Linear(dim_in, hid_dim),
            nn.LeakyReLU(),  # Robust for [-1, 1] range
            gnn.Linear(hid_dim, hid_dim),
        )

        # PyG's GINEConv handles message passing.
        self.model = GINEConv(nn=gin_nn, edge_dim=edge_dim)

        self.norm_node = get_norm_layer(norm_type, hid_dim)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

        # 2. Feed Forward Network (FFN) using gnn.Linear.
        self.norm1_local = get_norm_layer(norm_type, hid_dim)
        self.ff_linear1 = gnn.Linear(hid_dim, hid_dim * 2)
        self.ff_linear2 = gnn.Linear(hid_dim * 2, hid_dim)
        self.ff_act = nn.LeakyReLU()
        self.norm2 = get_norm_layer(norm_type, hid_dim)

    def _ff_block(self, x: Tensor) -> Tensor:
        """Feed Forward block with LeakyReLU for symmetric signal propagation."""
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

        # 2. Graph-aware normalization via batch tensor
        x = apply_norm(self.norm_node, x, batch)
        x = self.act(x)
        x = self.drop(x)

        # 3. Residual Connection
        if x_in.shape == x.shape:
            x = x_in + x

        # 4. FFN Block - Maintains per-graph statistics during normalization.
        x = apply_norm(self.norm1_local, x, batch)
        x = x + self._ff_block(x)
        x = apply_norm(self.norm2, x, batch)

        return x


class GINEEncoder(nn.Module):
    """Edge-aware GINE+ encoder with hardcoded PE concatenation and Jumping Knowledge."""

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
        self.jk_mode = kwargs.get("jk_mode", "cat")

        # 3. Initial projection using gnn.Linear.
        if node_input_dim != hid_dim:
            self.jk_proj = gnn.Linear(node_input_dim, hid_dim)
        else:
            self.jk_proj = nn.Identity()

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

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass passing the batch tensor through each layer block to
        ensure graph-aware normalization.
        """
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
