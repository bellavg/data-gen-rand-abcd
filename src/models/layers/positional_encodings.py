import numpy as np
import torch
import torch.nn as nn
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
            if self.discrete:
                val = val.long()
            else:
                val = val.float()

            # Map it to the target attribute name expected by _get_pe
            setattr(data, self.attr_name, val)
        return data


class AddSinusoidalPE:
    """
    Normal sinusoidal positional encoding based on an ordered node index.
    Useful mostly if nodes have a strict sequential or temporal ordering.
    """

    def __init__(self, dim: int = 16, attr_name: str = "pos_enc"):
        self.dim = dim
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        position = torch.arange(data.num_nodes).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.dim, 2).float() * -(np.log(10000.0) / self.dim)
        )

        pe = torch.zeros(data.num_nodes, self.dim)
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


def get_pe_transform(pe_type: str, attr_name: str = "pos_enc", **kwargs):
    """
    Routes your requested pe_type to extract the correct pre-computed attribute
    from your data_utils.py Data object.
    """
    if pe_type is None or pe_type.lower() == "none":
        return lambda data: data

    pe_type = pe_type.lower()

    # Strip 'learned_' prefix if present so we can just grab the raw data key
    source_key = pe_type.replace("learned_", "")

    if source_key in ["level"]:
        # Discrete pre-computed features (must be cast to long for Embeddings)
        return ExtractPrecomputedPE(
            source_key=source_key, attr_name=attr_name, discrete=True
        )

    elif source_key in ["pi_paths", "local_sp_sum"]:
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
    max_hops: int = 10,
) -> nn.Module:
    """
    Routes your requested pe_type to the correct learnable PyTorch Layer.
    """
    if pe_type is None or pe_type.lower() == "none":
        return nn.Identity()

    pe_type = pe_type.lower()

    if pe_type in ["learned_level", "level"]:
        return LearnedDepthEmbedding(max_depth=max_depth, embed_dim=pos_enc_dim)

    elif pe_type in [
        "learned_pi_paths",
        "pi_paths",
        "learned_local_sp_sum",
        "local_sp_sum",
    ]:
        # Continuous values use a simple linear projection instead of an Embedding
        return nn.Linear(1, pos_enc_dim)

    elif pe_type in ["sinusoidal", "sine"]:
        # Sinusoidal is usually passed through a linear projection as well
        return nn.Linear(pos_enc_dim, pos_enc_dim)

    else:
        raise ValueError(f"Unknown pos_enc layer type: {pe_type}")


# ==============================================================================
# Integration Helpers
# ==============================================================================


def validate_positional_encoding(
    pos_enc: torch.Tensor, pos_enc_dim: int, pos_enc_mode: str
):
    """Common helper to validate PE dimensions across all encoders."""
    if pos_enc is None or pos_enc_dim == 0 or pos_enc_mode.lower() == "none":
        return
    if pos_enc.size(-1) != pos_enc_dim:
        raise ValueError(
            f"Expected pos_enc with feature size {pos_enc_dim}, got {pos_enc.size(-1)}"
        )


def integrate_positional_encoding(
    x: torch.Tensor, pos_enc: torch.Tensor, pos_enc_dim: int, pos_enc_mode: str
):
    """Common helper to concatenate or add PE to node features."""
    if pos_enc is None or pos_enc_dim == 0 or pos_enc_mode.lower() == "none":
        return x
    if pos_enc_mode.lower() == "concat":
        return torch.cat([x, pos_enc], dim=-1)
    elif pos_enc_mode.lower() == "add":
        return x + pos_enc
    return x
