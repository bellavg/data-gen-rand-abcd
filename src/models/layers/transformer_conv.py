import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn  # Standard alias for PyG layers
from torch import Tensor
from torch_geometric.nn import TransformerConv
from torch_geometric.typing import Adj

# Project Imports
from models.model_utils import apply_norm, get_norm_layer


class TransformerConvLayer(nn.Module):
    """
    TransformerConv block unified with GNN+ enhancements.
    Hardcodes Residuals and FFNs to True for robust feature learning.
    """

    def __init__(
        self,
        dim_in: int,
        hid_dim: int,
        edge_dim: int | None = None,
        heads: int = 4,
        concat: bool = False,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        norm_type: str = "layer",
        beta: bool = False,
        root_weight: bool = True,
    ):
        super().__init__()
        self.dropout = dropout

        # Prevent dimension explosion and residual crashes when concat=True
        if concat:
            if hid_dim % heads != 0:
                raise ValueError("hid_dim must be divisible by heads if concat=True")
            conv_out_channels = hid_dim // heads
        else:
            conv_out_channels = hid_dim

        # 1. Residual Projection
        # Ensures the residual highway flows even if input dim != hidden dim
        if dim_in != hid_dim:
            self.res_proj = gnn.Linear(dim_in, hid_dim)
        else:
            self.res_proj = nn.Identity()

        self.norm_attn = get_norm_layer(norm_type, dim_in)
        self.conv = TransformerConv(
            in_channels=dim_in,
            out_channels=conv_out_channels,
            heads=heads,
            concat=concat,
            beta=beta,
            dropout=attn_dropout,
            edge_dim=edge_dim,
            root_weight=root_weight,
        )
        self.act = nn.LeakyReLU()  # Robust for [-1, 1] range
        self.drop = nn.Dropout(dropout)

        # 1. Swap to gnn.Linear for GNN-specific weight initialization.
        self.norm_ffn = get_norm_layer(norm_type, hid_dim)
        self.ff_linear1 = gnn.Linear(hid_dim, hid_dim * 2)
        self.ff_linear2 = gnn.Linear(hid_dim * 2, hid_dim)
        self.ff_act = nn.LeakyReLU()

    def _ff_block(self, x: Tensor) -> Tensor:
        """Feed Forward block with graph-aware dropout."""
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

        # 1. Pre-Norm: Normalize a copy of the state before attention
        x_norm_attn = apply_norm(self.norm_attn, x, batch)

        # 2. Sublayer: Attention Mechanism
        attn_out = self.conv(x=x_norm_attn, edge_index=edge_index, edge_attr=edge_attr)
        attn_out = self.act(attn_out)
        attn_out = self.drop(attn_out)

        # 3. Residual Connection (Clean Highway)
        x = self.res_proj(x_in) + attn_out

        # 4. Pre-Norm: Normalize a copy of the updated state before FFN
        x_norm_ffn = apply_norm(self.norm_ffn, x, batch)

        # 5. Sublayer: FFN Block
        ffn_out = self._ff_block(x_norm_ffn)

        # 6. Residual Connection (Clean Highway)
        x = x + ffn_out
        return x


class TransformerConvEncoder(nn.Module):
    """
    Graph-level TransformerConv encoder.
    Hardcoded PE concatenation and Jumping Knowledge (cat) for AIG tasks.
    """

    def __init__(
        self,
        hid_dim: int,
        num_layers: int,
        node_input_dim: int,
        edge_attr_dim: int,
        output_dim: int,
        heads: int = 4,
        concat: bool = False,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        norm_type: str = "layer",
        beta: bool = False,
        root_weight: bool = True,
        **kwargs,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_layers = num_layers
        self.jk_mode = kwargs.get("jk_mode", "cat")

        # 5. Use gnn.Linear for initial projection to hidden dimension.
        if node_input_dim != hid_dim:
            self.jk_proj = gnn.Linear(node_input_dim, hid_dim)
        else:
            self.jk_proj = nn.Identity()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dim_in = node_input_dim if i == 0 else hid_dim
            self.layers.append(
                TransformerConvLayer(
                    dim_in=dim_in,
                    hid_dim=hid_dim,
                    edge_dim=edge_attr_dim,
                    heads=heads,
                    concat=concat,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    norm_type=norm_type,
                    beta=beta,
                    root_weight=root_weight,
                )
            )
        # 3. Final Output Projection
        # Reconciles the Jumping Knowledge dimension with the required output_dim
        jk_out_dim = hid_dim * (num_layers + 1) if self.jk_mode == "cat" else hid_dim
        self.final_proj = gnn.Linear(jk_out_dim, output_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        """
        Supports Jumping Knowledge (JK) by passing the batch tensor
        through each layer for proper normalization.
        """
        x_jk = self.jk_proj(x)

        if self.jk_mode == "cat":
            h_list = [x_jk]
            for layer in self.layers:
                x = layer(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
                h_list.append(x)
            res = torch.cat(h_list, dim=1)

        elif self.jk_mode == "max":
            res = x_jk
            for layer in self.layers:
                x = layer(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
                res = torch.max(res, x)

        elif self.jk_mode == "sum":
            res = x_jk
            for layer in self.layers:
                x = layer(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
                res = res + x

        elif self.jk_mode == "last":
            for layer in self.layers:
                x = layer(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            res = x

        else:
            raise ValueError(f"Unknown jk_mode: {self.jk_mode}")

        # Final mapping from the aggregated JK tensor down to the required output_dim
        return self.final_proj(res)
