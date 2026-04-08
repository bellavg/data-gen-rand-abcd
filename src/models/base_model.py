from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool

try:
    from model_utils import get_batch_positional_encoding
    from positional_encodings import get_pos_enc_layer
except ImportError:  # pragma: no cover - fallback for package-style imports
    from models.layers.positional_encodings import get_pos_enc_layer

    from models.model_utils import get_batch_positional_encoding

from src.constants import ENCODER_REGISTRY


class UnifiedGraphBaseModel(nn.Module):
    """
    Base model that standardizes inputs for all encoders.

    Standardized tensors:
    - x: [N, D]
    - edge_attr: [E, D_e]
    - pos_enc: [N, D_p]
    """

    def __init__(
        self,
        encoder_name: str,
        embed_dim: int,
        node_input_dim: int = 4,  # Default to 4 AIG node types [Const, PI, Gate, PO]
        edge_attr_dim: int | None = None,
        task_out_dim: int = 2,  # Default to Dual Regression [Node Opt, Depth Opt]
        encoder_kwargs: Optional[Dict] = None,
        pe_type: str | None = "none",
        pos_enc_dim: int = 0,
        max_depth: int = 1000,  # Safeguard bounds for Depth PE Embeddings
        max_hops: int = 10,  # Safeguard bounds for Relative Hop Embeddings
    ):
        super().__init__()
        encoder_key = encoder_name.lower()
        if encoder_key not in ENCODER_REGISTRY:
            raise ValueError(f"Unknown encoder_name: {encoder_name}")

        self.encoder_name = encoder_key

        # Projects the one-hot node features into the continuous embed_dim
        self.node_embed = nn.Linear(node_input_dim, embed_dim)

        # Optionally project incoming continuous edge features into `embed_dim`
        self.edge_attr_dim = edge_attr_dim
        self.edge_attr_proj = (
            nn.Linear(edge_attr_dim, embed_dim) if edge_attr_dim is not None else None
        )

        # 1. Instantiate the Learned Positional Encoding Projection Layer
        self.pe_encoder = get_pos_enc_layer(
            pe_type=pe_type,
            pos_enc_dim=pos_enc_dim,
            max_depth=max_depth,
            max_hops=max_hops,
        )

        kwargs = {} if encoder_kwargs is None else dict(encoder_kwargs)

        # Provide sensible defaults for encoder input dims and hidden sizes.
        kwargs.setdefault("in_dim", embed_dim)
        kwargs.setdefault("num_layers", 2)

        # GraphGPS uses `hidden_dim` while most other encoders use `hid_dim`.
        if self.encoder_name == "graphgps":
            if "hidden_dim" not in kwargs:
                kwargs["hidden_dim"] = kwargs.get("hid_dim", embed_dim)
            kwargs.pop("hid_dim", None)
        else:
            kwargs.setdefault("hid_dim", embed_dim)

        if "edge_dim" not in kwargs:
            if self.edge_attr_proj is not None:
                kwargs["edge_dim"] = embed_dim
            elif self.edge_attr_dim is not None:
                kwargs["edge_dim"] = self.edge_attr_dim

        # EGIN bypasses the shared prediction head and needs output_dim on encoder.
        if self.encoder_name == "egin":
            kwargs.setdefault("output_dim", task_out_dim)

        self.encoder = ENCODER_REGISTRY[encoder_key](**kwargs)

        # EGIN already outputs graph-level scores; others output node embeddings.
        # LazyLinear perfectly handles the unknown input size created by jk='cat' in the encoders
        self.head = (
            nn.Identity()
            if self.encoder_name == "egin"
            else nn.LazyLinear(task_out_dim)
        )

    # Note: edge_type-based derivation removed; edge attributes are supplied

    def _encode_with_selected_encoder(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        pos_enc: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Passes the fully mapped tensors directly into the selected architecture."""
        return self.encoder(
            x=x,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr,
            pos_enc=pos_enc,
        )

    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        pos_enc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Projects base graph features and positional encodings into continuous space."""
        x = self.node_embed(x.float())
        # Map edge_attr into model embedding space when a projection is configured
        if edge_attr is None:
            edge_attr_emb = None
        else:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            edge_attr = edge_attr.float()
            if self.edge_attr_proj is not None:
                edge_attr_emb = self.edge_attr_proj(edge_attr)
            else:
                edge_attr_emb = edge_attr

        # 3. Apply the Positional Encoding Projection
        if pos_enc is not None and not isinstance(self.pe_encoder, nn.Identity):
            if pos_enc.dim() == 1:
                pos_enc = pos_enc.unsqueeze(-1)

            pos_enc = self.pe_encoder(pos_enc)

            # Squeeze extra sequence dimensions from embeddings if they appear
            if pos_enc.dim() == 3 and pos_enc.size(1) == 1:
                pos_enc = pos_enc.squeeze(1)
        else:
            pos_enc = None

        return self._encode_with_selected_encoder(
            x=x,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr_emb,
            pos_enc=pos_enc,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        pos_enc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Complete forward pass: Node/Edge Feature Encoding -> Message Passing -> Pooling -> Linear Head."""
        enc_out = self.encode_nodes(
            x=x,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr,
            pos_enc=pos_enc,
        )

        # EGIN already incorporates the final linear projection and pooling internally
        if self.encoder_name == "egin":
            return enc_out

        # Hardcoded sum pooling (global_add_pool) for optimal theoretical expressivity
        graph_emb = global_add_pool(enc_out, batch)
        return self.head(graph_emb)

    def forward_batch(self, batch) -> torch.Tensor:
        """Convenience wrapper for PyTorch Geometric Batch objects."""
        pos_enc = get_batch_positional_encoding(batch)
        return self.forward(
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
            edge_attr=getattr(batch, "edge_attr", None),
            pos_enc=pos_enc,
        )
