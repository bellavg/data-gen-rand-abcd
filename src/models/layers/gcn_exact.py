from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj


class GCNConvExact(MessagePassing):
    """Edge-attribute-free message passing with a multiplicity weight.

    Companion to ``models.layers.gcn.GCNConvWithEdges`` for the S2
    exact-compression track (``data.exact_graph``). Polarity lives on node
    features there (``fold_inversions_into_x``), not edges, so there is no
    edge encoder here. ``edge_weight`` is required and multiplies the
    message *after* its nonlinearity — the only way "k identical incoming
    messages" can be represented exactly by one aggregated computation
    (scaling a value fed *into* the nonlinearity, as ``GCNConvWithEdges``
    does via ``edge_attr``, cannot be decomposed back into k separate terms).
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr="add")
        self.lin = gnn.Linear(in_channels, out_channels, bias=False)
        self.bias_param = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return F.leaky_relu(x_j) * edge_weight.view(-1, 1)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: Tensor) -> Tensor:
        x = self.lin(x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        if self.bias_param is not None:
            out = out + self.bias_param
        return out


class GCNConvLayerExact(nn.Module):
    """Residual + FFN block around ``GCNConvExact``.

    Mirrors ``models.layers.gcn.GCNConvLayer`` (bias, FFN, residual, dropout
    all unchanged) with one deliberate omission: no normalization anywhere.
    GraphNorm and PyG's per-layer norm options are graph/batch-scoped
    statistics that a coarsened super-node cannot reproduce, which breaks
    exact quotient recovery (confirmed empirically; no exactness-preserving
    replacement was found). Since there is no norm, there is also no need to
    thread ``batch`` through this layer at all.

    The residual connection and FFN block need no special handling for the
    same reason: once ``GCNConvExact`` gives every member of a WL class an
    identical output (which is what the ``edge_weight`` multiplicity fix is
    for), any deterministic function applied identically and elementwise —
    residual add, FFN, real trained biases included — preserves that
    equality trivially. It is not a property that needs verifying per
    operation; it only breaks for something that reads a cross-node or
    cross-graph statistic, which is exactly what normalization does and
    nothing here does.
    """

    def __init__(self, dim_in: int, dim_out: int, dropout: float):
        super().__init__()
        self.dropout = dropout

        self.model = GCNConvExact(dim_in, dim_out, bias=True)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

        self.ff_linear1 = gnn.Linear(dim_out, dim_out * 2)
        self.ff_linear2 = gnn.Linear(dim_out * 2, dim_out)
        self.ff_act = nn.LeakyReLU()

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.ff_act(self.ff_linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ff_linear2(x)
        return F.dropout(x, p=self.dropout, training=self.training)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: Tensor) -> Tensor:
        x_in = x

        x = self.model(x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.act(x)
        x = self.drop(x)

        if x_in.shape == x.shape:
            x = x_in + x
        # First layer may have dim_in != dim_out; skip residual in that case.

        x = x + self._ff_block(x)

        return x


class GCNEncoderExact(nn.Module):
    """Edge-attribute-free GCN+ encoder for the exact-compression track.

    Structurally mirrors ``models.layers.gcn.GCNEncoder`` (same JK modes,
    same layer-count/hidden-dim knobs) so a tuned hyperparameter set can be
    reused as-is; the only differences are the ones exactness requires — no
    ``edge_attr``/``edge_attr_dim``, ``edge_weight`` mandatory, no norm_type.
    """

    def __init__(
        self,
        hid_dim: int,
        num_layers: int,
        node_input_dim: int,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = num_layers
        self.jk_mode = kwargs.get("jk_mode", "cat")

        self.use_input_jk = self.jk_mode == "cat"
        if self.use_input_jk and node_input_dim != hid_dim:
            self.jk_proj = gnn.Linear(node_input_dim, hid_dim)
        else:
            self.jk_proj = nn.Identity()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dim_in = node_input_dim if i == 0 else hid_dim
            self.layers.append(
                GCNConvLayerExact(dim_in=dim_in, dim_out=hid_dim, dropout=dropout)
            )

    def forward(self, x, edge_index, edge_weight):
        if self.jk_mode == "cat":
            x_jk = self.jk_proj(x)
            h_list = [x_jk]
            for layer in self.layers:
                x = layer(x, edge_index=edge_index, edge_weight=edge_weight)
                h_list.append(x)
            return torch.cat(h_list, dim=1)

        elif self.jk_mode == "max":
            x = self.layers[0](x, edge_index=edge_index, edge_weight=edge_weight)
            res = x
            for layer in self.layers[1:]:
                x = layer(x, edge_index=edge_index, edge_weight=edge_weight)
                res = torch.max(res, x)
            return res

        elif self.jk_mode == "sum":
            x = self.layers[0](x, edge_index=edge_index, edge_weight=edge_weight)
            res = x
            for layer in self.layers[1:]:
                x = layer(x, edge_index=edge_index, edge_weight=edge_weight)
                res = res + x
            return res

        elif self.jk_mode == "last":
            for layer in self.layers:
                x = layer(x, edge_index=edge_index, edge_weight=edge_weight)
            return x

        raise ValueError(f"Unknown jk_mode: {self.jk_mode}")
