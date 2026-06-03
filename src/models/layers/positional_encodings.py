import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as gnn  # Standard alias for PyG layers
from torch_geometric.data import Data

# ==============================================================================
# Scalable Encodings for Massive DAGs / AIGs (O(1) transforms using precomputed data)
# ==============================================================================


class ExtractPrecomputedPE:
    """
    Generic transform that extracts a precomputed feature from the Data object
    (like 'level', 'pi_paths', 'local_sp_sum').

    If discrete=True, it casts the tensor to .long() so it can be fed into nn.Embedding.
    If discrete=False, it ensures it is a .float() for nn.Linear.
    """

    def __init__(self, source_key: str, attr_name: str, discrete: bool = False):
        self.source_key = source_key
        self.attr_name = attr_name
        self.discrete = discrete

    def __call__(self, data: Data) -> Data:
        # Grab the tensor from the Data object
        val = getattr(data, self.source_key, None)
        if val is not None:
            # Release the original attribute early to reduce duplicate tensor
            # lifetime when generating pos_enc on very large graphs.
            delattr(data, self.source_key)
            if self.discrete:
                val = val.long()
            else:
                # Handles 'pi_paths' and 'local_sp_sum' while avoiding an extra
                # out-of-place log1p tensor allocation on very large graphs.
                val = val.float()
                val.log1p_()

            # Map it to the target attribute name expected by _get_pe
            setattr(data, self.attr_name, val)
        return data


class AddSinusoidalPE:
    """
    Normal sinusoidal positional encoding based on an ordered node index.
    Updated with device-awareness to prevent runtime errors.
    """

    def __init__(self, dim: int = 16, attr_name: str = "pos_enc"):
        if dim % 2 != 0:
            raise ValueError("Sinusoidal PE dimension should be even.")
        self.dim = dim
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        # Safety: Ensure tensors are created on the correct device
        device = (
            data.x.device
            if hasattr(data, "x") and data.x is not None
            else torch.device("cpu")
        )

        position = torch.arange(data.num_nodes, device=device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device).float()
            * -(np.log(10000.0) / self.dim)
        )

        pe = torch.zeros(data.num_nodes, self.dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        setattr(data, self.attr_name, pe)
        return data


# ==============================================================================
# PyTorch Modules for Learned Positional Encodings
# ==============================================================================


class LearnedDepthEmbedding(nn.Module):
    """
    Maps discrete integer logic depths ('level') to continuous learned vectors.
    """

    def __init__(self, max_depth: int, embed_dim: int):
        super().__init__()
        self.max_depth = max_depth
        self.embed = nn.Embedding(max_depth, embed_dim)

    def forward(self, depth_indices: torch.Tensor) -> torch.Tensor:
        # Clamp bounds to prevent crashes on out-of-distribution graphs
        clamped_indices = (
            depth_indices.long().squeeze(-1).clamp(min=0, max=self.max_depth - 1)
        )
        return self.embed(clamped_indices)


# ==============================================================================
# Factory Functions for Dynamic Configuration
# ==============================================================================


def identity_transform(data: Data) -> Data:
    return data


def get_pe_transform(pe_type: str, attr_name: str = "pos_enc", **kwargs):
    """
    Routes your requested pe_type to extract the correct pre-computed attribute
    from your Data object.
    """
    if pe_type is None or pe_type.lower() == "none":
        return identity_transform

    pe_type = pe_type.lower()

    # Strip 'learned_' prefix if present so we can just grab the raw data key
    source_key = pe_type.replace("learned_", "")

    if source_key in ["pi_paths", "local_sp_sum", "level"]:
        # Continuous pre-computed features (must be cast to float for Linear layers)
        return ExtractPrecomputedPE(
            source_key=source_key, attr_name=attr_name, discrete=False
        )

    elif pe_type in ["sinusoidal", "sine"]:
        return AddSinusoidalPE(attr_name=attr_name, **kwargs)

    else:
        raise ValueError(f"Unknown positional encoding transform: {pe_type}")


def get_pos_enc_layer(
    pe_type: str | None,
    pos_enc_dim: int = 16,
    max_depth: int = 1000,
) -> nn.Module:
    """
    Returns the learnable PE layer. Now using Sequential for better
    signal flow in [-1, 1] regression.
    """
    if pe_type is None or pe_type.lower() == "none":
        return nn.Identity()

    pe_type = pe_type.lower()

    if pe_type in [
        "learned_pi_paths",
        "pi_paths",
        "learned_local_sp_sum",
        "local_sp_sum",
        "level",
        "learned_level",
    ]:
        # Refactored to Sequential to support LeakyReLU
        return nn.Sequential(gnn.Linear(1, pos_enc_dim), nn.LeakyReLU())

    elif pe_type in ["sinusoidal", "sine"]:
        # Project sinusoidal fixed vectors into the latent space
        return nn.Sequential(gnn.Linear(pos_enc_dim, pos_enc_dim), nn.LeakyReLU())

    else:
        raise ValueError(f"Unknown pos_enc layer type: {pe_type}")
