import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import TransformerConv
from torch_geometric.typing import Adj

try:
    from model_utils import get_norm_layer
except ImportError:  # pragma: no cover - fallback for package-style imports
    try:
        from models.model_utils import get_norm_layer
    except ImportError:
        from src.models.model_utils import get_norm_layer


class TransformerConvLayer(nn.Module):
    """
    TransformerConv block unified with GNN+ enhancements.
    Hardcodes Residuals and FFNs to True.
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
        norm_type: str = "batch",
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

        self.norm_node = get_norm_layer(norm_type, hid_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        # Feed Forward Network (FFN) - Hardcoded to True
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
        x = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
        from models.model_utils import apply_norm

        x = apply_norm(self.norm_node, x, batch)
        x = self.act(x)
        x = self.drop(x)

        # 2. Residual Connection - Hardcoded to True
        if x.shape == x_in.shape:
            x = x + x_in

        # 3. FFN Block - Hardcoded to True
        x = apply_norm(self.norm1_local, x, batch)
        x = x + self._ff_block(x)
        x = apply_norm(self.norm2, x, batch)

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
        norm_type: str = "batch",
        beta: bool = False,
        root_weight: bool = True,
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

        # First layer expects `node_input_dim`; subsequent layers expect `hid_dim`.

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
