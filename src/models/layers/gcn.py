import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, SparseTensor

# Adopted from: https://github.com/LUOyk1999/GNNPlus/blob/main/GNNPlus/layer/gcn_conv_layer.py 

try:
    from model_utils import (
        get_norm_layer, 
        validate_positional_encoding, 
        integrate_positional_encoding
    )
except ImportError:  # pragma: no cover - fallback for package-style imports
    from models.model_utils import (
        get_norm_layer, 
        validate_positional_encoding, 
        integrate_positional_encoding
    )


class GCNConvWithEdges(GCNConv):
    """Edge-aware GCNConv with message rule ReLU(x_j + edge_attr_projected)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int | None = None,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = False,
        normalize: bool = True,
        bias: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            improved=improved,
            cached=cached,
            add_self_loops=add_self_loops,
            normalize=normalize,
            bias=bias,
        )
        self.edge_dim = edge_dim
        self.edge_encoder = (
            nn.Linear(edge_dim, out_channels, bias=False) if edge_dim is not None else None
        )

    def message(self, x_j: Tensor, edge_weight: OptTensor, edge_attr: OptTensor = None) -> Tensor:
        if edge_attr is not None:
            if self.edge_encoder is None:
                if edge_attr.size(-1) != x_j.size(-1):
                    raise ValueError(
                        "edge_attr feature size does not match node hidden size. "
                        "Provide edge_dim in GCNConvWithEdges to enable projection."
                    )
                edge_msg = edge_attr
            else:
                edge_msg = self.edge_encoder(edge_attr)

            msg = (x_j + edge_msg).relu()
        else:
            msg = x_j.relu()

        if edge_weight is not None:
            msg = edge_weight.view(-1, 1) * msg
        return msg

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                        self.flow,
                        x.dtype,
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]
            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                        self.flow,
                        x.dtype,
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        if edge_attr is not None and isinstance(edge_index, Tensor):
            edge_count = edge_index.size(1)
            if edge_attr.size(0) != edge_count:
                if self.add_self_loops and edge_attr.size(0) < edge_count:
                    pad_count = edge_count - edge_attr.size(0)
                    pad = torch.zeros(
                        pad_count,
                        edge_attr.size(-1),
                        device=edge_attr.device,
                        dtype=edge_attr.dtype,
                    )
                    edge_attr = torch.cat([edge_attr, pad], dim=0)
                else:
                    raise ValueError(
                        f"edge_attr rows ({edge_attr.size(0)}) do not match number of edges ({edge_count})."
                    )

        out = self.propagate(
            edge_index,
            x=x,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            size=None,
        )

        if self.bias is not None:
            out = out + self.bias
        return out


class GCNEncoder(nn.Module):
    """Edge-aware GCN encoder with optional positional encodings and configurable normalization."""

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        num_layers: int,
        edge_dim: int | None = None,
        pos_enc_dim: int = 0,
        pos_enc_mode: str = "concat",
        use_input_proj: bool = True,
        dropout: float = 0.0,
        norm_type: str = "batch",
        readout: str = "mean",
        residual: bool = False,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = False,
        normalize: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.dropout = dropout
        self.residual = residual
        self.pos_enc_dim = pos_enc_dim
        self.pos_enc_mode = pos_enc_mode.lower()
        self.use_input_proj = use_input_proj

        if self.pos_enc_mode not in {"none", "concat", "add"}:
            raise ValueError(f"Unknown pos_enc_mode: {pos_enc_mode}")

        effective_in_dim = in_dim + pos_enc_dim if self.pos_enc_mode == "concat" and pos_enc_dim > 0 else in_dim
        if self.use_input_proj:
            self.input_proj = nn.Linear(effective_in_dim, hid_dim)
        else:
            if effective_in_dim != hid_dim:
                raise ValueError(
                    f"use_input_proj=False requires effective_in_dim ({effective_in_dim}) == hid_dim ({hid_dim})"
                )
            self.input_proj = nn.Identity()
        self.pos_add_proj = (
            nn.Linear(pos_enc_dim, hid_dim, bias=False)
            if self.pos_enc_mode == "add" and pos_enc_dim > 0
            else None
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer_idx in range(num_layers):
            layer_in_dim = hid_dim if layer_idx == 0 else hid_dim
            layer_out_dim = out_dim if layer_idx == num_layers - 1 else hid_dim
            self.convs.append(
                GCNConvWithEdges(
                    in_channels=layer_in_dim,
                    out_channels=layer_out_dim,
                    edge_dim=edge_dim,
                    improved=improved,
                    cached=cached,
                    add_self_loops=add_self_loops,
                    normalize=normalize,
                    bias=True,
                )
            )
            self.norms.append(get_norm_layer(norm_type, layer_out_dim))

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Tensor,
        edge_attr: OptTensor = None,
        edge_type: OptTensor = None,
        pos_enc: OptTensor = None,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        # edge_type is accepted for interface consistency and is unused in GCN.
        _ = edge_type
        
        # Use our external helper functions!
        validate_positional_encoding(pos_enc, self.pos_enc_dim, self.pos_enc_mode)
        x = integrate_positional_encoding(x, pos_enc, self.pos_enc_dim, self.pos_enc_mode)
        
        x = self.input_proj(x)

        if pos_enc is not None and self.pos_add_proj is not None:
            x = x + self.pos_add_proj(pos_enc)

        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_in = x
            x = conv(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_weight=edge_weight,
            )
            x = norm(x)

            if layer_idx < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            if self.residual and x.shape == x_in.shape:
                x = x + x_in

        return x