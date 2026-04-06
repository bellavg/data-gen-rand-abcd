from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    from model_utils import get_batch_positional_encoding, get_pyg_pool
except ImportError:  # pragma: no cover - fallback for package-style imports
    from models.model_utils import get_batch_positional_encoding, get_pyg_pool

from src.constants import ENCODER_REGISTRY


class UnifiedGraphBaseModel(nn.Module):
    """
    Base model that standardizes inputs for all encoders.

    Standardized tensors:
    - x: [N, D]
    - edge_attr: [E, D_e] (constructed from edge_type embedding)
    - edge_type: [E]
    - pos_enc: [N, D_p]
    """

    def __init__(
        self,
        encoder_name: str,
        node_input_dim: int,
        embed_dim: int,
        num_edge_types: int,
        encoder_kwargs: Optional[Dict] = None,
        task_out_dim: int = 1,
        pool_type: str = "mean",
    ):
        super().__init__()
        encoder_key = encoder_name.lower()
        if encoder_key not in ENCODER_REGISTRY:
            raise ValueError(f"Unknown encoder_name: {encoder_name}")

        self.encoder_name = encoder_key
        self.num_edge_types = num_edge_types
        self.pool_fn = get_pyg_pool(pool_type)

        self.node_embed = nn.Linear(node_input_dim, embed_dim)
        self.edge_type_embed = nn.Embedding(self.num_edge_types, embed_dim)

        kwargs = {} if encoder_kwargs is None else dict(encoder_kwargs)
        self.encoder = ENCODER_REGISTRY[encoder_key](**kwargs)

        # EGIN already outputs graph-level scores; others output node embeddings.
        self.head = (
            nn.Identity()
            if self.encoder_name == "egin"
            else nn.LazyLinear(task_out_dim)
        )

    def _derive_edge_type(
        self, edge_attr: Optional[torch.Tensor], edge_type: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if edge_type is not None:
            return edge_type.long()

        if edge_attr is None:
            raise ValueError(
                "Provide edge_type or edge_attr so edge_type can be derived."
            )

        if edge_attr.dim() == 1:
            # Handles values like 0/1/2 or -1/1 by remapping to contiguous ids.
            _, edge_type_idx = torch.unique(edge_attr, sorted=True, return_inverse=True)
            return edge_type_idx.long()

        if edge_attr.size(-1) == 1:
            _, edge_type_idx = torch.unique(
                edge_attr.view(-1), sorted=True, return_inverse=True
            )
            return edge_type_idx.long()

        return edge_attr.argmax(dim=-1).long()

    def _encode_with_selected_encoder(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_type: torch.Tensor,
        pos_enc: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.encoder(
            x=x,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr,
            edge_type=edge_type,
            pos_enc=pos_enc,
        )

    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
        pos_enc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        edge_type = self._derive_edge_type(edge_attr=edge_attr, edge_type=edge_type)
        edge_type = edge_type.clamp(min=0, max=self.num_edge_types - 1)

        x = self.node_embed(x.float())
        edge_attr_emb = self.edge_type_embed(edge_type)

        return self._encode_with_selected_encoder(
            x=x,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr_emb,
            edge_type=edge_type,
            pos_enc=pos_enc,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
        pos_enc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enc_out = self.encode_nodes(
            x=x,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr,
            edge_type=edge_type,
            pos_enc=pos_enc,
        )

        if self.encoder_name == "egin":
            return enc_out

        graph_emb = self.pool_fn(enc_out, batch)
        return self.head(graph_emb)

    def forward_batch(self, batch) -> torch.Tensor:
        pos_enc = get_batch_positional_encoding(batch)
        return self.forward(
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
            edge_attr=getattr(batch, "edge_attr", None),
            edge_type=getattr(batch, "edge_type", None),
            pos_enc=pos_enc,
        )
