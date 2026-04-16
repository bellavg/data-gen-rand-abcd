import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm


def get_norm_layer(norm_type, dim):
    """Return a normalization layer from a string identifier.

    Supported options:
    - None/'none' -> `nn.Identity()`
    - 'batch' -> `nn.BatchNorm1d(dim)`
    - 'layer' -> `nn.LayerNorm(dim)`
    - 'graph'/'graphnorm'/'gn' -> `torch_geometric.nn.GraphNorm(dim)`
    """
    if norm_type is None or str(norm_type).lower() == "none":
        return nn.Identity()

    nt = str(norm_type).lower()
    if nt == "batch":
        return nn.BatchNorm1d(dim)
    if nt == "layer":
        return nn.LayerNorm(dim)
    if nt in ("graph", "graphnorm", "gn"):
        return GraphNorm(dim)

    raise ValueError(f"Unknown norm_type: {norm_type}")


def get_batch_positional_encoding(batch: object) -> torch.Tensor | None:
    """Return the collated `pos_enc` tensor from a PyG Batch (or Data) object.

    - If `batch` has no `pos_enc` attribute, returns `None`.
    - If `pos_enc` is 1D, it will be unsqueezed to shape `(N, 1)` and cast to float.
    - Otherwise returns `pos_enc.float()`.
    """
    pe = getattr(batch, "pos_enc", None)
    if pe is None:
        return None
    if pe.dim() == 1:
        pe = pe.unsqueeze(-1)
    # Preserve integer dtypes for learned/discrete positional encodings
    # (e.g., depth indices for LearnedDepthEmbedding). Normalize any
    # floating inputs to `torch.float32` to avoid dtype mismatches.
    if pe.is_floating_point():
        return pe.to(torch.float32)
    return pe
