from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GraphNorm, InstanceNorm, LayerNorm


def get_norm_layer(norm_type: Optional[str], dim: int) -> nn.Module:
    """Return a normalization layer from a string identifier using PyTorch Geometric.

    Args:
        norm_type: String identifier (e.g., 'batch', 'layer', 'graph').
        dim: Feature dimension for the normalization layer.

    Returns:
        Instantiated normalization module.
    """
    if norm_type is None or str(norm_type).lower() == "none":
        return nn.Identity()

    nt = str(norm_type).lower()

    if nt == "batch":
        return BatchNorm(dim)
    if nt == "layer":
        return LayerNorm(dim)
    if nt in ("graph", "graphnorm", "gn"):
        return GraphNorm(dim)
    if nt in ("instance", "instancenorm", "in"):
        return InstanceNorm(dim)

    raise ValueError(f"Unknown norm_type: {norm_type}")


def get_batch_positional_encoding(batch: object) -> Optional[torch.Tensor]:
    """Extract and format positional encoding from a PyG batch.

    Args:
        batch: PyG Data or Batch object.

    Returns:
        Formatted positional encoding tensor, or None if unavailable.
    """
    pe = getattr(batch, "pos_enc", None)
    if pe is None or not isinstance(pe, torch.Tensor):
        return None

    if pe.dim() == 1:
        pe = pe.unsqueeze(-1)
    return pe.to(torch.float32)


def apply_norm(
    norm_layer: nn.Module, x: torch.Tensor, batch: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Safely apply normalization.

    Passes 'batch' only to layers that support/require it (GraphNorm,
    InstanceNorm, LayerNorm).

    Args:
        norm_layer: Normalization module.
        x: Input tensor.
        batch: Graph assignment vector for per-graph normalization.

    Returns:
        Normalized tensor.
    """
    if batch is not None and isinstance(
        norm_layer, (GraphNorm, InstanceNorm, LayerNorm)
    ):
        return norm_layer(x, batch)

    return norm_layer(x)
