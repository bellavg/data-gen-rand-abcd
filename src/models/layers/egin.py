import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter

from models.model_utils import get_norm_layer

# Adapted from: https://github.com/YxRicardo/EGIN/blob/main/models/graphegin.py


class MLP(nn.Module):
    """
    MLP used by EGIN blocks.
    Aligned with original implementation: applies norm -> relu -> dropout
    on hidden layers.
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
            self.linears.append(nn.Linear(input_dim, output_dim))
        else:
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            self.norms.append(get_norm_layer(norm_type, hidden_dim))

            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
                self.norms.append(get_norm_layer(norm_type, hidden_dim))

            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x: Tensor, batch: Tensor = None) -> Tensor:
        if self.num_layers == 1:
            return self.dropout(self.linears[0](x))

        h = x
        from models.model_utils import apply_norm
        for i in range(self.num_layers - 1):
            h = self.linears[i](h)
            h = apply_norm(self.norms[i], h, batch)
            h = F.relu(h)
            h = self.dropout(h)

        return self.linears[-1](h)


class GraphEGIN(nn.Module):
    """
    PyG-only implementation of EGIN-style graph model.
    Hardcoded optimizations based on paper and pipeline needs:
    - Activation: ReLU
    - Readout/Pooling: Sum (global_add_pool)
    - Learnable Epsilon: Removed
    - Positional Encoding: Hardcoded to Concat
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
            raise ValueError(
                "num_layers must be >= 2 (input layer + at least one EGIN block)"
            )
        if edge_attr_dim < 1:
            raise ValueError("edge_attr_dim must be >= 1 for EGIN")

        self.num_layers = num_layers
        # Accept jk_mode from unified encoder kwargs (ignored by EGIN but kept for API compatibility)
        self.jk_mode = kwargs.get("jk_mode", "cat")  # Default to 'cat' if not provided
        self.edge_attr_dim = edge_attr_dim
        self.dot_update = dot_update
        self.edge_mlp = edge_mlp
        self.dropout = dropout

        self.edge_hidden_dim = (
            edge_hidden_dim if edge_hidden_dim is not None else edge_attr_dim
        )

        self.mlps = nn.ModuleList()
        self.norms = nn.ModuleList()

        if self.edge_mlp and not self.dot_update:
            self.edge_mlps = nn.ModuleList(
                [
                    nn.Linear(edge_attr_dim, self.edge_hidden_dim)
                    for _ in range(self.num_layers - 1)
                ]
            )
        else:
            self.edge_mlps = None

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
            if layer == 0:
                self.linears_prediction.append(nn.Linear(node_input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hid_dim, output_dim))

    @staticmethod
    def _to_2d_edge_attr(edge_attr: Tensor) -> Tensor:
        if edge_attr.dim() == 1:
            return edge_attr.view(-1, 1)
        return edge_attr

    def _dot_update_aggregate(
        self, h: Tensor, edge_index: Tensor, edge_attr: Tensor, layer: int
    ) -> Tensor:
        src, dst = edge_index
        num_nodes = h.size(0)

        # CHANGE 1: Use h[src] instead of h[dst] to grab the upstream node's features
        msg = h[src].unsqueeze(-1) * edge_attr.unsqueeze(1)

        # CHANGE 2: Scatter into 'dst' instead of 'src' to push features downstream
        pooled = scatter(
            msg.reshape(msg.size(0), -1), dst, dim=0, dim_size=num_nodes, reduce="sum"
        )

        return pooled

    def _concat_update_aggregate(
        self, h: Tensor, edge_index: Tensor, edge_attr: Tensor, layer: int
    ) -> Tensor:
        src, dst = edge_index
        num_nodes = h.size(0)

        # CHANGE 3: Use h[src] and scatter into 'dst'
        node_agg = scatter(h[src], dst, dim=0, dim_size=num_nodes, reduce="sum")

        # CHANGE 4: Scatter the edge features into 'dst'
        edge_rep = scatter(edge_attr, dst, dim=0, dim_size=num_nodes, reduce="sum")

        if self.edge_mlp and self.edge_mlps is not None:
            edge_part = self.edge_mlps[layer](edge_rep)
        else:
            edge_part = edge_rep

        pooled = torch.cat((node_agg, edge_part), dim=-1)
        return pooled

    def _egin_next_layer(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        layer: int,
        batch: Tensor,
    ) -> Tensor:
        if self.dot_update:
            pooled = self._dot_update_aggregate(h, edge_index, edge_attr, layer)
        else:
            pooled = self._concat_update_aggregate(h, edge_index, edge_attr, layer)

        h = self.mlps[layer](pooled, batch)
        from models.model_utils import apply_norm
        h = apply_norm(self.norms[layer], h, batch)
        h = F.relu(h)
        return h

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

        if edge_attr.size(-1) != self.edge_attr_dim:
            raise ValueError(
                f"Expected edge_attr feature size {self.edge_attr_dim}, got {edge_attr.size(-1)}"
            )

        hidden_rep = [x]
        h = x

        for layer in range(self.num_layers - 1):
            h = self._egin_next_layer(h, edge_index, edge_attr, layer, batch)
            hidden_rep.append(h)

        score_over_layer = 0.0
        for layer, h_layer in enumerate(hidden_rep):
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
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
        )
