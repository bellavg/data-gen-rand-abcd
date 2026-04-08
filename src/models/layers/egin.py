import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter

try:
    from model_utils import get_norm_layer
except ImportError:  # pragma: no cover - fallback for package-style imports
    try:
        from models.model_utils import get_norm_layer
    except ImportError:
        from src.models.model_utils import get_norm_layer

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

    def forward(self, x: Tensor) -> Tensor:
        if self.num_layers == 1:
            return self.dropout(self.linears[0](x))

        h = x
        for i in range(self.num_layers - 1):
            h = self.linears[i](h)
            h = self.norms[i](h)
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
        in_dim: int,
        hid_dim: int,
        num_layers: int,
        edge_dim: int,
        output_dim: int,
        num_mlp_layers: int = 2,
        dot_update: bool = False,
        edge_mlp: bool = False,
        edge_hidden_dim: int | None = None,
        dropout: float = 0.0,
        norm_type: str = "batch",
        pos_enc_dim: int = 0,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError(
                "num_layers must be >= 2 (input layer + at least one EGIN block)"
            )
        if edge_dim < 1:
            raise ValueError("edge_dim must be >= 1 for EGIN")

        self.num_layers = num_layers
        self.edge_dim = edge_dim
        self.dot_update = dot_update
        self.edge_mlp = edge_mlp
        self.dropout = dropout
        self.pos_enc_dim = pos_enc_dim

        # Hardcoded to 'concat' PE
        effective_input_dim = in_dim + pos_enc_dim if pos_enc_dim > 0 else in_dim

        self.edge_hidden_dim = (
            edge_hidden_dim if edge_hidden_dim is not None else edge_dim
        )

        self.mlps = nn.ModuleList()
        self.norms = nn.ModuleList()

        if self.edge_mlp and not self.dot_update:
            self.edge_mlps = nn.ModuleList(
                [
                    nn.Linear(edge_dim, self.edge_hidden_dim)
                    for _ in range(self.num_layers - 1)
                ]
            )
        else:
            self.edge_mlps = None

        for layer in range(self.num_layers - 1):
            node_dim = effective_input_dim if layer == 0 else hid_dim

            if self.dot_update:
                mlp_in_dim = node_dim * edge_dim
            elif self.edge_mlp:
                mlp_in_dim = node_dim + self.edge_hidden_dim
            else:
                mlp_in_dim = node_dim + edge_dim

            self.mlps.append(
                MLP(
                    num_layers=2, # HARDCODED: Standard expressive depth from paper
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
                self.linears_prediction.append(
                    nn.Linear(effective_input_dim, output_dim)
                )
            else:
                self.linears_prediction.append(nn.Linear(hid_dim, output_dim))

    def _validate_positional_encoding(self, pos_enc: Tensor | None) -> None:
        if pos_enc is None or self.pos_enc_dim == 0:
            return
        if pos_enc.size(-1) != self.pos_enc_dim:
            raise ValueError(
                f"Expected pos_enc with feature size {self.pos_enc_dim}, got {pos_enc.size(-1)}"
            )

    def _integrate_positional_encoding(
        self, x: Tensor, pos_enc: Tensor | None
    ) -> Tensor:
        if pos_enc is None or self.pos_enc_dim == 0:
            return x
        return torch.cat([x, pos_enc], dim=-1)

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

        msg = h[dst].unsqueeze(-1) * edge_attr.unsqueeze(1)
        pooled = scatter(
            msg.reshape(msg.size(0), -1), src, dim=0, dim_size=num_nodes, reduce="sum"
        )

        return pooled

    def _concat_update_aggregate(
        self, h: Tensor, edge_index: Tensor, edge_attr: Tensor, layer: int
    ) -> Tensor:
        src, dst = edge_index
        num_nodes = h.size(0)

        node_agg = scatter(h[dst], src, dim=0, dim_size=num_nodes, reduce="sum")
        edge_rep = scatter(edge_attr, src, dim=0, dim_size=num_nodes, reduce="sum")

        if self.edge_mlp and self.edge_mlps is not None:
            edge_part = self.edge_mlps[layer](edge_rep)
        else:
            edge_part = edge_rep

        pooled = torch.cat((node_agg, edge_part), dim=-1)
        return pooled

    def _egin_next_layer(
        self, h: Tensor, edge_index: Tensor, edge_attr: Tensor, layer: int
    ) -> Tensor:
        if self.dot_update:
            pooled = self._dot_update_aggregate(h, edge_index, edge_attr, layer)
        else:
            pooled = self._concat_update_aggregate(h, edge_index, edge_attr, layer)

        h = self.mlps[layer](pooled)
        h = self.norms[layer](h)
        h = F.relu(h)
        return h

    def egin_forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Tensor,
        pos_enc: Tensor | None = None,
    ) -> Tensor:
        if edge_attr is None:
            raise ValueError("EGIN requires edge_attr tensor.")

        edge_attr = self._to_2d_edge_attr(edge_attr)

        if edge_attr.size(-1) != self.edge_dim:
            raise ValueError(
                f"Expected edge_attr feature size {self.edge_dim}, got {edge_attr.size(-1)}"
            )

        self._validate_positional_encoding(pos_enc)
        x = self._integrate_positional_encoding(x, pos_enc)

        hidden_rep = [x]
        h = x

        for layer in range(self.num_layers - 1):
            h = self._egin_next_layer(h, edge_index, edge_attr, layer)
            hidden_rep.append(h)

        score_over_layer = 0.0
        for layer, h_layer in enumerate(hidden_rep):
            pooled_h = global_add_pool(h_layer, batch)
            logits = self.linears_prediction[layer](pooled_h)
            score_over_layer = score_over_layer + F.dropout(
                logits,
                p=self.dropout,
                training=self.training,
            )

        return score_over_layer

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Tensor = None,
        pos_enc: Tensor | None = None,
    ) -> Tensor:
        return self.egin_forward(
            x=x,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr,
            pos_enc=pos_enc,
        )
