from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GINEConv, GPSConv
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.typing import Adj

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


# Map short norm names used project-wide to the full names expected by PyG's
# normalization_resolver inside GPSConv.
_GPS_NORM_MAP: dict[str, str] = {
    "batch": "batch_norm",
    "layer": "layer_norm",
    "graph": "graph_norm",
    "instance": "instance_norm",
}


class GraphGPSEncoder(nn.Module):
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
        self.num_layers = num_layers
        # GPS already performs global attention every layer; JK cat multiplies
        # output size by (num_layers+1) for little gain.  Default to "last".
        self.jk_mode = kwargs.get("jk_mode", "last")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        # Added projection to fix GPSConv's strict dimensionality requirement
        if node_input_dim != hid_dim:
            self.in_proj = nn.Linear(node_input_dim, hid_dim)
        else:
            self.in_proj = nn.Identity()

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            dim_in = hid_dim  # All layers now strictly take hid_dim

            # Use a simple per-node normalization for the local MLP; GPSConv
            # will handle graph-level normalization with the chosen `norm`.
            local_nn = nn.Sequential(
                nn.Linear(dim_in, hid_dim),
                nn.LayerNorm(hid_dim),
                nn.LeakyReLU(),
                nn.Linear(hid_dim, hid_dim),
            )
            local_conv = GINEConv(nn=local_nn, edge_dim=edge_attr_dim)

            # Map project-wide short norm name to PyG resolver-compatible name.
            gps_norm = _GPS_NORM_MAP.get(str(norm_type).lower(), norm_type)

            self.layers.append(
                GPSConv(
                    channels=hid_dim,
                    conv=local_conv,
                    heads=heads,
                    dropout=dropout,
                    attn_type="performer",
                    # Tie performer feature dropout to the model dropout instead
                    # of a hardcoded 0.5 which was excessively high.
                    attn_kwargs={"dropout": dropout},
                    norm=gps_norm,
                )
            )

        self.redraw_projection = RedrawProjection(
            self.layers, redraw_interval=performer_redraw_interval
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        # Project once for Jumping Knowledge bookkeeping
        x_jk = self.in_proj(x)
        x_proj = x_jk

        if self.jk_mode == "cat":
            h_list = [x_jk]
            for layer in self.layers:
                x_proj = layer(
                    x_proj, edge_index=edge_index, edge_attr=edge_attr, batch=batch
                )
                h_list.append(x_proj)
            return torch.cat(h_list, dim=1)

        elif self.jk_mode == "max":
            res = x_jk
            for layer in self.layers:
                x_proj = layer(
                    x_proj, edge_index=edge_index, edge_attr=edge_attr, batch=batch
                )
                res = torch.max(res, x_proj)
            return res

        elif self.jk_mode == "sum":
            res = x_jk
            for layer in self.layers:
                x_proj = layer(
                    x_proj, edge_index=edge_index, edge_attr=edge_attr, batch=batch
                )
                res = res + x_proj
            return res

        elif self.jk_mode == "last":
            for layer in self.layers:
                x_proj = layer(
                    x_proj, edge_index=edge_index, edge_attr=edge_attr, batch=batch
                )
            return x_proj
