from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool

from config import get_output_dim_for_encoder
from models.layers.gcn_exact import GCNEncoderExact


class ExactGraphBaseModel(nn.Module):
    """Size-weighted, normalization-free base model for the S2
    exact-compression track.

    A deliberately separate class rather than a flag on
    ``models.base_model.UnifiedGraphBaseModel``: the production model's
    forward pass stays completely untouched, and this model's constraints
    (no GraphNorm, size-weighted pooling, mandatory ``edge_weight``, no
    ``edge_attr``) are structural requirements for the exactness proof, not
    configurable knobs — a future change to the shared model can't silently
    reintroduce them here.

    Every hyperparameter surface that isn't one of those requirements
    (``hidden_dim``, ``num_layers``, ``dropout``, ``jk_mode``, head shape)
    matches ``UnifiedGraphBaseModel`` so a tuned config can be reused as-is.

    No positional encoding support: this first version targets ``pe_type=
    "none"`` only. Level/pos_enc would need re-adding to
    ``data.exact_graph.apply_exact_merge_map`` (min-pooled, same as the
    general primitive) before this model could consume it.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        node_input_dim: int,
        task_out_dim: int = 1,
        dropout: float = 0.0,
        jk_mode: str = "cat",
        head_dropout: Optional[float] = None,
    ):
        super().__init__()

        self.node_embed = nn.Linear(node_input_dim, hidden_dim)

        self.encoder = GCNEncoderExact(
            hid_dim=hidden_dim,
            num_layers=num_layers,
            node_input_dim=hidden_dim,
            dropout=dropout,
            jk_mode=jk_mode,
        )

        encoder_out_dim = get_output_dim_for_encoder(
            "gcn_exact", {"hid_dim": hidden_dim, "num_layers": num_layers, "jk_mode": jk_mode}
        )

        head_in_dim = encoder_out_dim
        head_hidden_dim = head_in_dim // 2
        head_dropout = float(head_dropout) if head_dropout is not None else float(dropout)

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=head_dropout),
            nn.Linear(head_hidden_dim, int(task_out_dim)),
            nn.Sigmoid(),
        )

    def encode_and_integrate(self, x: torch.Tensor) -> torch.Tensor:
        return self.node_embed(x.float())

    @staticmethod
    def _pool_size_weighted(
        emb: torch.Tensor,
        batch: torch.Tensor,
        node_size: torch.Tensor,
        size: int,
    ) -> torch.Tensor:
        """Mean pool weighted by how many original nodes each row stands for.

        A plain mean would count a super-node once regardless of how many
        nodes it represents, which breaks exact quotient recovery at the
        readout step; this is exact for node_size == 1 everywhere (a
        non-coarsened graph), matching plain mean pooling exactly.
        """
        weight = node_size.to(emb.dtype).view(-1, 1)
        weighted_sum = global_add_pool(emb * weight, batch, size=size)
        total_weight = global_add_pool(weight, batch, size=size)
        return weighted_sum / total_weight

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_graphs: int,
        node_size: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.encode_and_integrate(x)
        enc_out = self.encoder(x, edge_index=edge_index, edge_weight=edge_weight)

        if batch is None:
            batch = torch.zeros(enc_out.size(0), dtype=torch.long, device=enc_out.device)

        graph_emb = self._pool_size_weighted(enc_out, batch, node_size, size=num_graphs)
        return self.head(graph_emb)

    def forward_batch(self, batch) -> torch.Tensor:
        edge_weight = getattr(batch, "edge_weight", None)
        if edge_weight is None:
            raise ValueError(
                "ExactGraphBaseModel requires edge_weight on every graph "
                "(see data.exact_graph.fold_inversions_into_x)."
            )
        node_size = getattr(batch, "node_size", None)
        if node_size is None:
            # Deliberately not defaulted to all-ones: Batch.from_data_list
            # drops an attribute from the WHOLE batch if even one graph in
            # it lacks it, so a silent default here would treat a coarsened
            # graph's super-nodes as size-1 whenever it happens to share a
            # batch with a graph that wasn't produced by fold_inversions_
            # into_x / apply_exact_merge_map (both of which always set it).
            raise ValueError(
                "ExactGraphBaseModel requires node_size on every graph (see "
                "data.exact_graph.fold_inversions_into_x / "
                "apply_exact_merge_map)."
            )
        num_graphs = getattr(batch, "num_graphs", None)
        if num_graphs is None:
            raise ValueError("Batch.num_graphs is required for explicit pooling size.")

        return self.forward(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_weight=edge_weight,
            batch=batch.batch,
            node_size=node_size,
            num_graphs=int(num_graphs),
        )
