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

from src.constants import ENCODER_REGISTRY, MAX_DEPTH, get_output_dim_for_encoder


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
        encoder_kwargs: Dict,
        pe_type: str,
        node_input_dim: int = 4,
        edge_attr_dim: int = 2,
        task_out_dim: int = 1,
        max_depth: int = MAX_DEPTH,
        embed_node_input: bool = True,
        embed_edge_input: bool = True,
        project_with_pos_enc: bool = True,
    ):
        super().__init__()  # Don't forget this if it's not somewhere else!

        # 1. Save all instance variables FIRST
        self.kwargs = encoder_kwargs.copy()  # Good practice to copy dicts
        self.encoder_name = encoder_name
        self.pe_type = pe_type
        self.project_with_pos_enc = project_with_pos_enc
        self.embed_node_input = embed_node_input
        self.embed_edge_input = embed_edge_input

        # 2. Calculate Positional Encoding Dimension
        self.pos_enc_dim = max(16, embed_dim // 4) if self.pe_type != "none" else 0

        # 3. Calculate the dimension after concatenation
        base_node_dim = embed_dim if embed_node_input else node_input_dim
        concat_dim = (
            base_node_dim + self.pos_enc_dim
            if (self.pe_type != "none" and self.pos_enc_dim > 0)
            else base_node_dim
        )

        # 4. Determine final dimension and initialize projection layer if needed
        if (
            self.pe_type != "none"
            and self.pos_enc_dim > 0
            and self.project_with_pos_enc
        ):
            final_node_input_dim = embed_dim
            self.post_pe_proj = nn.Linear(concat_dim, embed_dim)
        else:
            final_node_input_dim = concat_dim
            self.post_pe_proj = None

        # Provide node input dim to the encoder
        self.kwargs["node_input_dim"] = final_node_input_dim
        self.kwargs["edge_attr_dim"] = embed_dim if embed_edge_input else edge_attr_dim

        # 5. Initialize embeddings
        if self.embed_node_input:
            self.node_embed = nn.Linear(node_input_dim, embed_dim)

        if self.embed_edge_input:
            self.edge_attr_proj = nn.Linear(edge_attr_dim, embed_dim)

        # 6. Instantiate Positional Encoding Projection Layer
        if self.pe_type != "none":
            self.pe_encoder = get_pos_enc_layer(
                pe_type=self.pe_type,
                pos_enc_dim=self.pos_enc_dim,
                max_depth=max_depth,
            )
        else:
            self.pe_encoder = nn.Identity()

        # EGIN bypasses the shared prediction head and needs output_dim on encoder.
        self.kwargs["output_dim"] = get_output_dim_for_encoder(
            self.encoder_name, self.kwargs
        )

        # 7. Instantiate the selected encoder
        self.encoder = ENCODER_REGISTRY[self.encoder_name](**self.kwargs)

        self.head = nn.LazyLinear(int(task_out_dim))

    def _integrate_positional_encoding(self, x, pos_enc):
        if pos_enc is None or getattr(self, "pos_enc_dim", 0) == 0:
            return x

        pos_enc = pos_enc.unsqueeze(-1) if pos_enc.dim() == 1 else pos_enc
        pos_enc = self.pe_encoder(pos_enc)

        pos_enc = (
            pos_enc.squeeze(1)
            if pos_enc.dim() == 3 and pos_enc.size(1) == 1
            else pos_enc
        )

        # Concatenate base features with positional encoding
        out = torch.cat([x, pos_enc], dim=-1)

        # Apply post-integration projection if enabled
        if self.project_with_pos_enc and self.post_pe_proj is not None:
            out = self.post_pe_proj(out)

        return out

    def _encode_with_selected_encoder(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Passes the fully mapped tensors directly into the selected architecture."""
        return self.encoder(
            x=x,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr,
        )

    def encode_and_integrate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        pos_enc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Projects base graph features and positional encodings into continuous space."""
        x = self.node_embed(x.float()) if self.embed_node_input else x.float()
        assert (
            edge_attr is not None
            and edge_attr.dim() == 2
            and edge_attr.size(0) == edge_index.size(1)
        ), (
            f"edge_attr must be 2D with rows equal to number of edges (expected {edge_index.size(1)})"
        )

        edge_attr = (
            self.edge_attr_proj(edge_attr.float())
            if self.embed_edge_input
            else edge_attr.float()
        )
        # Integrate positional encoding (projection + concat) into node features
        x = self._integrate_positional_encoding(x, pos_enc)

        return x, edge_attr

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor,
        pos_enc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Complete forward pass: Node/Edge Feature Encoding -> Message Passing -> Pooling -> Linear Head."""

        x, edge_attr = self.encode_and_integrate(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos_enc=pos_enc,
        )

        enc_out = self._encode_with_selected_encoder(
            x=x,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr,
        )

        # EGIN already incorporates the final linear projection and pooling internally
        if self.encoder_name == "egin":
            return enc_out
        else:
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
            edge_attr=batch.edge_attr,
            pos_enc=pos_enc,
        )
