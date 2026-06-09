import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn  # 1. Standard alias for PyG layers
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.utils import degree

# Corrected Import
from models.model_utils import apply_norm, get_norm_layer


class GCNConvWithEdges(MessagePassing):
    """
    Edge-aware GCN message-passing layer per the GNN+ paper.
    Now includes proper GCN symmetric degree normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,  # Always required: AIG graphs always carry edge attributes.
        bias: bool = True,
    ):
        super().__init__(aggr="add")
        # 2. Use gnn.Linear for GNN-optimized weight initialization
        self.lin = gnn.Linear(in_channels, out_channels, bias=False)
        self.edge_encoder = gnn.Linear(edge_dim, out_channels, bias=False)
        self.bias_param = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def message(self, x_j: Tensor, edge_weight: Tensor, edge_attr: Tensor) -> Tensor:
        # Encode edge attributes and fuse with neighbour features before GCN scaling.
        # edge_attr is passed directly through propagate() — no instance-variable side
        # channel so this is thread-safe and works correctly with DataLoader workers.
        edge_msg = self.edge_encoder(edge_attr)
        msg = F.leaky_relu(x_j + edge_msg)
        return msg * edge_weight.view(-1, 1)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        if edge_weight is None:
            # Calculate GCN normalization weights (1 / sqrt(deg(i) * deg(j)))
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt.nan_to_num_(posinf=0.0)
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.lin(x)
        # Pass edge_attr through propagate so message() receives it as a named
        # argument — safe with multi-process DataLoader workers.
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, edge_attr=edge_attr, size=None)

        if self.bias_param is not None:
            out = out + self.bias_param

        return out


class GCNConvLayer(nn.Module):
    """
    Single GCN layer block with Residuals and Feed-Forward Network (FFN).
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        edge_dim: int,  # Always required: matches GCNConvWithEdges.
        dropout: float,
        norm_type: str,
    ):
        super().__init__()
        self.dropout = dropout

        self.model = GCNConvWithEdges(dim_in, dim_out, edge_dim, bias=True)
        self.norm_node = get_norm_layer(norm_type, dim_out)
        self.act = nn.LeakyReLU()  # Maintain LeakyReLU for symmetric target range
        self.drop = nn.Dropout(dropout)

        # 3. Swap FFN to gnn.Linear
        self.norm1_local = get_norm_layer(norm_type, dim_out)
        self.ff_linear1 = gnn.Linear(dim_out, dim_out * 2)
        self.ff_linear2 = gnn.Linear(dim_out * 2, dim_out)
        self.ff_act = nn.LeakyReLU()
        self.norm2 = get_norm_layer(norm_type, dim_out)

    def _ff_block(self, x: Tensor) -> Tensor:
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
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        x_in = x

        # 1. Message Passing
        x = self.model(x, edge_index=edge_index, edge_attr=edge_attr, edge_weight=edge_weight)

        # 2. Graph-Aware Normalization
        x = apply_norm(self.norm_node, x, batch)
        x = self.act(x)
        x = self.drop(x)

        # 3. Residual Connection
        if x_in.shape == x.shape:
            x = x_in + x

        # 4. FFN Block
        x = apply_norm(self.norm1_local, x, batch)
        x = x + self._ff_block(x)
        x = apply_norm(self.norm2, x, batch)

        return x


class GCNEncoder(nn.Module):
    """Edge-aware GCN+ encoder with Jumping Knowledge support."""

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

        # 4. Use gnn.Linear for initial projection
        if node_input_dim != hid_dim:
            self.jk_proj = gnn.Linear(node_input_dim, hid_dim)
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

    def forward(self, x, edge_index, edge_attr, batch=None, edge_weight=None):
        x_jk = self.jk_proj(x)

        if self.jk_mode == "cat":
            h_list = [x_jk]
            for layer in self.layers:
                x = layer(
                    x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch,
                    edge_weight=edge_weight,
                )
                h_list.append(x)
            return torch.cat(h_list, dim=1)

        elif self.jk_mode == "max":
            res = x_jk
            for layer in self.layers:
                x = layer(
                    x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch,
                    edge_weight=edge_weight,
                )
                res = torch.max(res, x)
            return res

        elif self.jk_mode == "sum":
            res = x_jk
            for layer in self.layers:
                x = layer(
                    x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch,
                    edge_weight=edge_weight,
                )
                res = res + x
            return res

        elif self.jk_mode == "last":
            for layer in self.layers:
                x = layer(
                    x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch,
                    edge_weight=edge_weight,
                )
            return x
