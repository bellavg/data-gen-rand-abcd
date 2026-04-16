import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import TransformerConv
from torch_geometric.typing import Adj, OptTensor

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
            in_channels=hid_dim,
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
        self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None
    ) -> Tensor:
        x_in = x

        # 1. Message Passing
        x = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.norm_node(x)
        x = self.act(x)
        x = self.drop(x)

        # 2. Residual Connection - Hardcoded to True
        if x.shape == x_in.shape:
            x = x + x_in

        # 3. FFN Block - Hardcoded to True
        x = self.norm1_local(x)
        x = x + self._ff_block(x)
        x = self.norm2(x)

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

        self.layers = nn.ModuleList()

        # Optional input projection when node input dim != hidden dim
        self.node_input_dim = node_input_dim
        self.hid_dim = hid_dim
        self.input_proj = (
            nn.Linear(node_input_dim, hid_dim)
            if node_input_dim != hid_dim
            else nn.Identity()
        )

        for _ in range(num_layers):
            self.layers.append(
                TransformerConvLayer(
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
