import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.utils import degree

from models.model_utils import get_norm_layer

# Adapted from: https://github.com/LUOyk1999/GNNPlus/blob/main/GNNPlus/layer/gcn_conv_layer_e.py


class GCNConvWithEdges(MessagePassing):
    """
    Edge-aware GCN message-passing layer per the GNN+ paper.
    Now includes proper GCN symmetric degree normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int | None = None,
        bias: bool = True,
    ):
        # Use add aggregation, we will manually normalize the weights
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels, bias=False)

        self.edge_encoder = (
            nn.Linear(edge_dim, out_channels, bias=False)
            if edge_dim is not None
            else None
        )
        self.bias_param = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self._edge_attr = None

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        # 1. Apply edge attributes
        ea = self._edge_attr
        if ea is not None:
            if self.edge_encoder is not None:
                edge_msg = self.edge_encoder(ea)
            else:
                edge_msg = ea
            msg = (x_j + edge_msg).relu()
        else:
            msg = x_j.relu()

        # 2. Scale the message by the GCN normalization weight
        return msg * edge_weight.view(-1, 1)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: Tensor,
    ) -> Tensor:
        # Calculate GCN normalization weights (1 / sqrt(deg(i) * deg(j)))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Temporarily attach edge_attr to self
        self._edge_attr = edge_attr

        # Apply linear transformation to node features
        x = self.lin(x)

        # Propagate messages passing the calculated edge_weight
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        # Clean up
        self._edge_attr = None

        # Add bias if it exists
        if self.bias_param is not None:
            out = out + self.bias_param

        return out


class GCNConvLayer(nn.Module):
    """
    Single GCN layer block combining Convolution, Normalization,
    Residuals, and Feed-Forward Network (FFN) per the GNN+ paper.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        edge_dim: int | None,
        dropout: float,
        norm_type: str,
    ):
        super().__init__()
        self.dropout = dropout

        self.model = GCNConvWithEdges(dim_in, dim_out, edge_dim, bias=True)
        self.norm_node = get_norm_layer(norm_type, dim_out)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        # Feed Forward Network (FFN) - Hardcoded to True per GNN+ paper
        self.norm1_local = get_norm_layer(norm_type, dim_out)
        self.ff_linear1 = nn.Linear(dim_out, dim_out * 2)
        self.ff_linear2 = nn.Linear(dim_out * 2, dim_out)
        self.ff_act = nn.ReLU()
        self.norm2 = get_norm_layer(norm_type, dim_out)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.ff_act(self.ff_linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ff_linear2(x)
        return F.dropout(x, p=self.dropout, training=self.training)

    def forward(
        self, x: Tensor, edge_index: Adj, edge_attr: Tensor, batch: Tensor | None = None
    ) -> Tensor:
        x_in = x

        # 1. Message Passing
        x = self.model(x, edge_index=edge_index, edge_attr=edge_attr)
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


class GCNEncoder(nn.Module):
    """Edge-aware GCN+ encoder with iterative Jumping Knowledge accumulation."""

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

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dim_in = node_input_dim if i == 0 else hid_dim
            self.layers.append(
                GCNConvLayer(
                    dim_in=dim_in,
                    dim_out=hid_dim,
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
