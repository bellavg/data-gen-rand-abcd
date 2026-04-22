import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GraphNorm, InstanceNorm, LayerNorm


def get_norm_layer(norm_type, dim):
    """Return a normalization layer from a string identifier using PyTorch Geometric.

    Supported options:
    - None/'none' -> `nn.Identity()`
    - 'batch' -> `torch_geometric.nn.norm.BatchNorm(dim)`
    - 'layer' -> `torch_geometric.nn.norm.LayerNorm(dim)`
    - 'graph'/'graphnorm'/'gn' -> `torch_geometric.nn.norm.GraphNorm(dim)`
    - 'instance'/'instancenorm'/'in' -> `torch_geometric.nn.norm.InstanceNorm(dim)`
    """
    if norm_type is None or str(norm_type).lower() == "none":
        return nn.Identity()

    nt = str(norm_type).lower()

    if nt == "batch":
        # PyG BatchNorm: standard node-wise normalization
        return BatchNorm(dim)

    if nt == "layer":
        # PyG LayerNorm: standard node-wise normalization
        return LayerNorm(dim)

    if nt in ("graph", "graphnorm", "gn"):
        # GraphNorm: requires 'batch' tensor in forward() to normalize per-graph
        return GraphNorm(dim)

    if nt in ("instance", "instancenorm", "in"):
        # PyG InstanceNorm: requires 'batch' tensor in forward() to normalize per-graph
        return InstanceNorm(dim)

    raise ValueError(f"Unknown norm_type: {norm_type}")


def get_batch_positional_encoding(batch: object) -> torch.Tensor | None:
    pe = getattr(batch, "pos_enc", None)
    # Check if pe is specifically a Tensor before calling .dim()
    if pe is None or not isinstance(pe, torch.Tensor):
        return None
    if pe.dim() == 1:
        pe = pe.unsqueeze(-1)
    return pe.to(torch.float32)


def apply_norm(
    norm_layer: nn.Module, x: torch.Tensor, batch: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Safely apply normalization.
    Passes 'batch' only to layers that support/require it (GraphNorm, InstanceNorm, LayerNorm).
    """
    if batch is not None and isinstance(
        norm_layer, (GraphNorm, InstanceNorm, LayerNorm)
    ):
        return norm_layer(x, batch)

    # BatchNorm and Identity only take x
    return norm_layer(x)
