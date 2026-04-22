import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn  # 1. Standard alias for PyG layers
from torch import Tensor
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter

# Project Imports
from models.model_utils import apply_norm, get_norm_layer


class MLP(nn.Module):
    """
    MLP used by EGIN blocks.
    Applies norm -> leaky_relu -> dropout on hidden layers.
    """

    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        norm_type: str = "batch",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            # 2. Swap to gnn.Linear
            self.linears.append(gnn.Linear(input_dim, output_dim))
        else:
            self.linears.append(gnn.Linear(input_dim, hidden_dim))
            self.norms.append(get_norm_layer(norm_type, hidden_dim))

            for _ in range(num_layers - 2):
                self.linears.append(gnn.Linear(hidden_dim, hidden_dim))
                self.norms.append(get_norm_layer(norm_type, hidden_dim))

            self.linears.append(gnn.Linear(hidden_dim, output_dim))

    def forward(self, x: Tensor, batch: Tensor = None) -> Tensor:
        if self.num_layers == 1:
            return self.dropout(self.linears[0](x))

        h = x
        for i in range(self.num_layers - 1):
            h = self.linears[i](h)
            # 3. Graph-aware normalization via batch tensor
            h = apply_norm(self.norms[i], h, batch)
            h = F.leaky_relu(h)
            h = self.dropout(h)

        return self.linears[-1](h)


class GraphEGIN(nn.Module):
    """
    PyG-only implementation of EGIN-style graph model.
    Optimized for symmetric target ranges like [-1, 1].
    """

    def __init__(
        self,
        node_input_dim: int,
        hid_dim: int,
        num_layers: int,
        edge_attr_dim: int,
        output_dim: int,
        num_mlp_layers: int = 2,
        dot_update: bool = False,
        edge_mlp: bool = True,
        edge_hidden_dim: int | None = None,
        dropout: float = 0.0,
        norm_type: str = "batch",
        **kwargs,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        if edge_attr_dim < 1:
            raise ValueError("edge_attr_dim must be >= 1 for EGIN")

        self.num_layers = num_layers
        self.jk_mode = kwargs.get("jk_mode", "cat")
        self.edge_attr_dim = edge_attr_dim
        self.dot_update = dot_update
        self.edge_mlp = edge_mlp
        self.dropout = dropout

        self.edge_hidden_dim = (
            edge_hidden_dim if edge_hidden_dim is not None else edge_attr_dim
        )

        # 4. Using gnn.Linear for GNN-optimized weight init
        if self.edge_mlp and not self.dot_update:
            self.edge_mlps = nn.ModuleList(
                [
                    gnn.Linear(edge_attr_dim, self.edge_hidden_dim)
                    for _ in range(self.num_layers - 1)
                ]
            )
        else:
            self.edge_mlps = None

        self.mlps = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer in range(self.num_layers - 1):
            node_dim = node_input_dim if layer == 0 else hid_dim

            if self.dot_update:
                mlp_in_dim = node_dim * edge_attr_dim
            elif self.edge_mlp:
                mlp_in_dim = node_dim + self.edge_hidden_dim
            else:
                mlp_in_dim = node_dim + edge_attr_dim

            self.mlps.append(
                MLP(
                    num_layers=num_mlp_layers,
                    input_dim=mlp_in_dim,
                    hidden_dim=hid_dim,
                    output_dim=hid_dim,
                    dropout=dropout,
                    norm_type=norm_type,
                )
            )
            self.norms.append(get_norm_layer(norm_type, hid_dim))

        self.linears_prediction = nn.ModuleList()
        for layer in range(num_layers):
            dim_in = node_input_dim if layer == 0 else hid_dim
            self.linears_prediction.append(gnn.Linear(dim_in, output_dim))

    @staticmethod
    def _to_2d_edge_attr(edge_attr: Tensor) -> Tensor:
        return edge_attr.view(-1, 1) if edge_attr.dim() == 1 else edge_attr

    def _dot_update_aggregate(
        self, h: Tensor, edge_index: Tensor, edge_attr: Tensor, layer: int
    ) -> Tensor:
        src, dst = edge_index
        num_nodes = h.size(0)
        msg = h[src].unsqueeze(-1) * edge_attr.unsqueeze(1)
        return scatter(
            msg.reshape(msg.size(0), -1), dst, dim=0, dim_size=num_nodes, reduce="sum"
        )

    def _concat_update_aggregate(
        self, h: Tensor, edge_index: Tensor, edge_attr: Tensor, layer: int
    ) -> Tensor:
        src, dst = edge_index
        num_nodes = h.size(0)
        node_agg = scatter(h[src], dst, dim=0, dim_size=num_nodes, reduce="sum")
        edge_rep = scatter(edge_attr, dst, dim=0, dim_size=num_nodes, reduce="sum")

        edge_part = (
            self.edge_mlps[layer](edge_rep)
            if self.edge_mlp and self.edge_mlps is not None
            else edge_rep
        )
        return torch.cat((node_agg, edge_part), dim=-1)

    def _egin_next_layer(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        layer: int,
        batch: Tensor,
    ) -> Tensor:
        pooled = (
            self._dot_update_aggregate(h, edge_index, edge_attr, layer)
            if self.dot_update
            else self._concat_update_aggregate(h, edge_index, edge_attr, layer)
        )

        h = self.mlps[layer](pooled, batch)
        h = apply_norm(self.norms[layer], h, batch)
        return F.leaky_relu(h)

    def egin_forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        if edge_attr is None:
            raise ValueError("EGIN requires edge_attr tensor.")

        edge_attr = self._to_2d_edge_attr(edge_attr)
        h = x
        hidden_rep = [x]

        for layer in range(self.num_layers - 1):
            h = self._egin_next_layer(h, edge_index, edge_attr, layer, batch)
            hidden_rep.append(h)

        score_over_layer = 0.0
        for layer, h_layer in enumerate(hidden_rep):
            # 5. Correct global pooling for graph-level regression
            pooled_h = global_add_pool(h_layer, batch)
            pooled_h = F.dropout(pooled_h, p=self.dropout, training=self.training)
            logits = self.linears_prediction[layer](pooled_h)
            score_over_layer = score_over_layer + logits

        return score_over_layer

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor = None,
        batch: Tensor | None = None,
    ) -> Tensor:
        return self.egin_forward(
            x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch
        )
