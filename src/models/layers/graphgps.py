from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GINEConv, GPSConv
from torch_geometric.typing import Adj, OptTensor

try:
    from torch_geometric.nn.attention import PerformerAttention
except Exception:  # pragma: no cover
    PerformerAttention = None

try:
    from model_utils import get_norm_layer
except ImportError:  # pragma: no cover - fallback for package-style imports
    try:
        from models.model_utils import get_norm_layer
    except ImportError:
        from src.models.model_utils import get_norm_layer

# Adapted from: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py


class RedrawProjection:
    """Helper for periodically redrawing Performer projection matrices."""

    def __init__(self, model: nn.Module, redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return

        if self.num_last_redraw >= self.redraw_interval:
            if PerformerAttention is not None:
                modules = [
                    m for m in self.model.modules() if isinstance(m, PerformerAttention)
                ]
            else:
                modules = [
                    m
                    for m in self.model.modules()
                    if hasattr(m, "redraw_projection_matrix")
                ]

            for module in modules:
                module.redraw_projection_matrix()

            self.num_last_redraw = 0
            return

        self.num_last_redraw += 1


class GraphGPSEncoder(nn.Module):
    """
    GraphGPS encoder aligned with the GNN+ and GPS papers.
    Hardcoded PE concatenation and Jumping Knowledge (cat) for AIG tasks.
    """

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        use_input_proj: bool = True,
        use_edge_proj: bool = True,
        pos_enc_dim: int = 0,
        dropout: float = 0.0,
        norm_type: str = "batch",
        heads: int = 4,
        attn_type: str = "multihead",
        attn_kwargs: Optional[Dict[str, Any]] = None,
        performer_redraw_interval: Optional[int] = None,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.use_input_proj = use_input_proj
        self.use_edge_proj = use_edge_proj
        self.hidden_dim = hidden_dim
        self.pos_enc_dim = pos_enc_dim

        # Hardcoded to 'concat' PE per GPS paper specifications
        effective_in_dim = in_dim + pos_enc_dim if pos_enc_dim > 0 else in_dim

        if self.use_input_proj:
            self.node_encoder = nn.Linear(effective_in_dim, hidden_dim)
        else:
            if effective_in_dim != hidden_dim:
                raise ValueError(
                    f"use_input_proj=False requires in_dim ({effective_in_dim}) == hidden_dim ({hidden_dim})"
                )
            self.node_encoder = nn.Identity()

        if self.use_edge_proj:
            self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        else:
            if edge_dim != hidden_dim:
                raise ValueError(
                    f"use_edge_proj=False requires edge_dim ({edge_dim}) == hidden_dim ({hidden_dim})"
                )
            self.edge_encoder = nn.Identity()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # Inner MPNN (GINE) for the GPS block
            local_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                get_norm_layer(norm_type, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            local_conv = GINEConv(local_nn, edge_dim=hidden_dim)

            # PyG's GPSConv natively handles the parallel Attn, MLPs, Norms, and Residuals
            self.layers.append(
                GPSConv(
                    channels=hidden_dim,
                    conv=local_conv,
                    heads=4,
                    dropout=dropout,
                    # FORCED HARDCODE: Use Performer for O(N) scalability
                    attn_type="performer",
                    attn_kwargs={"dropout": 0.5},
                    norm="batch",
                )
            )

        # Ensure redraw_interval is set for Performer stability
        self.redraw_projection = RedrawProjection(self.layers, redraw_interval=1000)
        # Hardcoded Jumping Knowledge = 'cat' output dimension
        self.out_dim = hidden_dim * (num_layers + 1)

    def _validate_positional_encoding(self, pos_enc: OptTensor) -> None:
        if pos_enc is None or self.pos_enc_dim == 0:
            return
        if pos_enc.size(-1) != self.pos_enc_dim:
            raise ValueError(
                f"Expected pos_enc with feature size {self.pos_enc_dim}, got {pos_enc.size(-1)}"
            )

    def _integrate_positional_encoding(self, x: Tensor, pos_enc: OptTensor) -> Tensor:
        if pos_enc is None or self.pos_enc_dim == 0:
            return x
        return torch.cat([x, pos_enc], dim=-1)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Tensor,
        edge_attr: OptTensor = None,
        pos_enc: OptTensor = None,
    ) -> Tensor:
        # edge_type removed; encoders receive `edge_attr` when needed

        self._validate_positional_encoding(pos_enc)
        x = self._integrate_positional_encoding(x, pos_enc)

        x = self.node_encoder(x)

        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        # Track hidden states for Jumping Knowledge
        h_list = [x]

        for layer in self.layers:
            # GraphGPS propagates edge_attr automatically into the local GINEConv only
            x = layer(
                h_list[-1], edge_index=edge_index, batch=batch, edge_attr=edge_attr
            )
            h_list.append(x)

        # Hardcoded Jumping Knowledge = 'cat'
        return torch.cat(h_list, dim=-1)
