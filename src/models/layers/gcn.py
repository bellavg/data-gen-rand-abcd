import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj

try:
    from model_utils import get_norm_layer
except ImportError:  # pragma: no cover - fallback for package-style imports
    try:
        from models.model_utils import get_norm_layer
    except ImportError:
        from src.models.model_utils import get_norm_layer


# Adapted from: https://github.com/LUOyk1999/GNNPlus/blob/main/GNNPlus/layer/gcn_conv_layer_e.py


class GCNConvWithEdges(MessagePassing):
    """
    Edge-aware GCN message-passing layer per the GNN+ paper.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int | None = None,
        bias: bool = True,
    ):
        # 1. Inherit from MessagePassing and set aggregation to "add"
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels, bias=False)

        # 2. Re-enable the edge encoder projection
        self.edge_encoder = (
            nn.Linear(edge_dim, out_channels, bias=False)
            if edge_dim is not None
            else None
        )
        self.bias_param = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # 3. Use an instance variable to bypass strict PyG inspector dropping kwargs
        self._edge_attr = None

    def message(self, x_j: Tensor) -> Tensor:
        # Read the edge attributes stored during forward()
        ea = self._edge_attr
        if ea is not None:
            if self.edge_encoder is None:
                if ea.size(-1) != x_j.size(-1):
                    raise ValueError(
                        "edge_attr feature size does not match node hidden size. "
                        "Provide edge_dim to enable projection."
                    )
                edge_msg = ea
            else:
                edge_msg = self.edge_encoder(ea)
            # Add edge features to node features and apply ReLU
            return (x_j + edge_msg).relu()
        return x_j.relu()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        # Temporarily attach edge_attr to self
        self._edge_attr = edge_attr

        # Apply linear transformation to node features
        x = self.lin(x)

        # Propagate messages (cleanly, without edge_attr in kwargs)
        out = self.propagate(edge_index, x=x, size=None)

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

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor | None = None) -> Tensor:
        x_in = x

        # 1. Message Passing
        x = self.model(x, edge_index, edge_attr)
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


class GCNEncoder(nn.Module):
    """Edge-aware GCN+ encoder with hardcoded PE concatenation and Jumping Knowledge (cat)."""

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
                GCNConvLayer(
                    dim_in=hid_dim,
                    dim_out=hid_dim,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    norm_type=norm_type,
                )
            )

        # Hardcoded Jumping Knowledge = 'cat' output dimension
        self.out_dim = hid_dim * (num_layers + 1)

    def _validate_positional_encoding(self, pos_enc: Tensor | None) -> None:
        if pos_enc is None or self.pos_enc_dim == 0:
            return
        if pos_enc.size(-1) != self.pos_enc_dim:
            raise ValueError(
                f"Expected pos_enc with feature size {self.pos_enc_dim}, got {pos_enc.size(-1)}"
            )

    def _integrate_positional_encoding(self, x: Tensor, pos_enc: Tensor | None) -> Tensor:
        if pos_enc is None or self.pos_enc_dim == 0:
            return x
        return torch.cat([x, pos_enc], dim=-1)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Tensor,
        edge_attr: Tensor | None = None,
        pos_enc: Tensor | None = None,
    ) -> Tensor:
        self._validate_positional_encoding(pos_enc)
        x = self._integrate_positional_encoding(x, pos_enc)
        x = self.input_proj(x)

        h_list = [x]
        for layer in self.layers:
            h_list.append(layer(x=h_list[-1], edge_index=edge_index, edge_attr=edge_attr))

        # Hardcoded Jumping Knowledge = 'cat'
        return torch.cat(h_list, dim=-1)

