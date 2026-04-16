from typing import Optional

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
        node_input_dim: int,
        edge_attr_dim: int,
        hid_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.0,
        norm_type: str = "batch",
        heads: int = 4,
        performer_redraw_interval: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        # Expose num_layers consistently across encoders
        self.num_layers = num_layers
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            # Inner MPNN (GINE) for the GPS block
            local_nn = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                get_norm_layer(norm_type, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, hid_dim),
            )
            local_conv = GINEConv(local_nn, edge_dim=hid_dim)

            self.layers.append(
                GPSConv(
                    channels=hid_dim,
                    conv=local_conv,
                    heads=heads,
                    dropout=dropout,
                    attn_type="performer",
                    attn_kwargs={"dropout": 0.5},
                    norm=norm_type,
                )
            )

        # Ensure redraw_interval is set for Performer stability
        self.redraw_projection = RedrawProjection(self.layers, redraw_interval=1000)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Tensor,
        edge_attr: OptTensor = None,
        pos_enc: OptTensor = None,
    ) -> Tensor:

        # Track hidden states for Jumping Knowledge
        h_list = [x]

        for layer in self.layers:
            # GraphGPS propagates edge_attr automatically into the local GINEConv only
            x = layer(
                h_list[-1], edge_index=edge_index, batch=batch, edge_attr=edge_attr
            )
            h_list.append(x)

        if self.jk_mode == "last":
            node_emb = h_list[-1]
        elif self.jk_mode == "max":
            node_emb = torch.stack(h_list, dim=-1).max(dim=-1)[0]
        elif self.jk_mode == "sum":
            node_emb = torch.stack(h_list, dim=-1).sum(dim=-1)
        elif self.jk_mode == "cat":
            node_emb = torch.cat(h_list, dim=1)

        return node_emb
