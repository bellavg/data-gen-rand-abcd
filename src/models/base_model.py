from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import (
    GraphNorm,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from constants import ENCODER_REGISTRY, MAX_DEPTH
from models.layers.positional_encodings import get_pos_enc_layer
from models.model_utils import get_batch_positional_encoding


class UnifiedGraphBaseModel(nn.Module):
    """
    Base model that standardizes inputs for all encoders.
    Now fully relies on Encoders for First-Layer Ingestion of Positional Encodings.
    """

    def __init__(
        self,
        encoder_name: str,
        hidden_dim: int,
        encoder_kwargs: Dict,
        pe_type: str,
        head_dropout: Optional[float] = None,
        pos_enc_dim: Optional[int] = 0,
        node_input_dim: int = 4,
        edge_attr_dim: int = 2,
        task_out_dim: int = 1,
        pooling_type: str = "max",
        max_depth: int = MAX_DEPTH,
    ):
        super().__init__()

        # 1. Save instance variables
        self.kwargs = encoder_kwargs.copy()
        self.encoder_name = encoder_name
        self.hidden_dim = hidden_dim
        self.pe_type = pe_type

        self.pooling_type = pooling_type

        # 2. Positional Encoding Setup
        self.pos_enc_dim = pos_enc_dim
        # 3. Latent Dimensions
        base_node_dim = hidden_dim

        # This is what gets passed to the first layer of the Encoders!
        concat_dim = base_node_dim + self.pos_enc_dim

        # Standardize kwargs for the GNN Encoder
        self.kwargs["hid_dim"] = hidden_dim  # Subsequent layers use this
        self.kwargs["node_input_dim"] = concat_dim  # First layer uses this!
        if self.encoder_name == "gcn":
            self.kwargs["edge_attr_dim"] = edge_attr_dim
            self.edge_attr_proj = nn.Identity()
        else:
            self.kwargs["edge_attr_dim"] = hidden_dim
            self.edge_attr_proj = nn.Linear(edge_attr_dim, hidden_dim)

        # 4. Feature Projections (Keep these to lift raw 4D/2D inputs to hidden_dim)

        self.node_embed = nn.Linear(node_input_dim, hidden_dim)
        self.input_node_norm = GraphNorm(concat_dim)

        # 5. Positional Encoding Layer
        if self.pe_type != "none":
            self.pe_encoder = get_pos_enc_layer(
                pe_type=self.pe_type,
                pos_enc_dim=self.pos_enc_dim,
                max_depth=max_depth,
            )
        else:
            self.pe_encoder = nn.Identity()

        # Update output_dim for Jumping Knowledge/Pooling based on jk_mode
        from constants import get_output_dim_for_encoder

        encoder_out_dim = get_output_dim_for_encoder(self.encoder_name, self.kwargs)
        self.kwargs["output_dim"] = encoder_out_dim

        # 6. Encoder and Head
        if self.encoder_name not in ENCODER_REGISTRY:
            raise KeyError(
                f"Unknown encoder_name '{self.encoder_name}'. "
                f"Valid options: {sorted(ENCODER_REGISTRY.keys())}"
            )
        encoder_cls = ENCODER_REGISTRY[self.encoder_name]
        self.encoder = encoder_cls(**self.kwargs)

        head_in_dim = encoder_out_dim
        head_hidden_dim = head_in_dim // 2
        head_dropout = (
            float(head_dropout)
            if head_dropout is not None
            else float(encoder_kwargs.get("dropout", 0.0))
        )

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=head_dropout),
            nn.Linear(head_hidden_dim, int(task_out_dim)),
            nn.Sigmoid(),
        )

    def _integrate_positional_encoding(
        self, x: torch.Tensor, pos_enc: Optional[torch.Tensor]
    ):
        if pos_enc is None or self.pe_type == "none":
            return x
        pos_enc = pos_enc.unsqueeze(-1) if pos_enc.dim() == 1 else pos_enc
        pos_enc = self.pe_encoder(pos_enc)
        pos_enc = (
            pos_enc.squeeze(1)
            if pos_enc.dim() == 3 and pos_enc.size(1) == 1
            else pos_enc
        )

        # Just concatenate and return! The encoders will handle the dimension.
        return torch.cat([x, pos_enc], dim=-1)

    def encode_and_integrate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        pos_enc: Optional[torch.Tensor] = None,
        partition_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project nodes to latent hidden_dim
        x = self.node_embed(x.float())

        edge_attr = self.edge_attr_proj(edge_attr.float())

        x = self._integrate_positional_encoding(x, pos_enc)

        # If partition_id is provided, normalize per-partition to prevent pollution
        norm_batch = partition_id if partition_id is not None else batch
        x = self.input_node_norm(x, norm_batch)

        return x, edge_attr

    def _encode(self, x, edge_index, edge_attr, batch, edge_weight=None):
        encoder_kwargs = {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "batch": batch,
        }
        if self.encoder_name == "gcn" and edge_weight is not None:
            encoder_kwargs["edge_weight"] = edge_weight
        return self.encoder(**encoder_kwargs)

    def _pool_graph_embeddings(
        self,
        emb: torch.Tensor,
        batch: torch.Tensor,
        size: int,
    ):
        if self.pooling_type == "mean":
            return global_mean_pool(emb, batch, size=size)
        if self.pooling_type == "sum":
            return global_add_pool(emb, batch, size=size)
        if self.pooling_type == "max":
            return global_max_pool(emb, batch, size=size)
        raise ValueError(f"Unknown pooling type: {self.pooling_type}")

    def _pool_partitioned_graph_embeddings(
        self,
        node_emb: torch.Tensor,
        partition_id: torch.Tensor,
        num_partitions: torch.Tensor,
        size: int,
    ) -> torch.Tensor:
        """Two-level hierarchical pooling: nodes → partitions → graph.

        Args:
            node_emb:       Node embeddings ``[total_nodes, hidden_dim]``.
            partition_id:   Globally-unique partition assignment per node
                            ``[total_nodes]`` (already offset by PyG Batch).
            num_partitions: Number of partitions per graph in the batch
                            ``[num_graphs]``.

        Returns:
            Graph-level embeddings ``[num_graphs, hidden_dim]``.
        """
        # 1. Pool node embeddings → one embedding per partition.
        #    partition_id is globally unique across the batch, so this
        #    produces shape [total_partitions, hidden_dim].
        partition_emb = self._pool_graph_embeddings(
            node_emb,
            partition_id,
            size=int(num_partitions.sum().item()),
        )

        # 2. Build a partition→graph assignment vector so we know which
        #    graph each partition belongs to.
        num_graphs = num_partitions.size(0)
        partition_batch = torch.repeat_interleave(
            torch.arange(num_graphs, device=node_emb.device),
            num_partitions,
        )

        # 3. Pool partition embeddings → one embedding per graph.
        return self._pool_graph_embeddings(partition_emb, partition_batch, size=size)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_graphs: int,
        batch: torch.Tensor | None = None,
        pos_enc: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        partition_id: Optional[torch.Tensor] = None,
        num_partitions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x, edge_attr = self.encode_and_integrate(
            x, edge_index, edge_attr, batch, pos_enc, partition_id=partition_id
        )

        enc_batch = partition_id if partition_id is not None else batch
        enc_out = self._encode(
            x, edge_index, edge_attr, enc_batch, edge_weight=edge_weight
        )

        if batch is None:
            batch = torch.zeros(
                enc_out.size(0), dtype=torch.long, device=enc_out.device
            )

        if partition_id is not None and num_partitions is not None:
            # Hierarchical pooling: nodes → partitions → graph.
            graph_emb = self._pool_partitioned_graph_embeddings(
                enc_out, partition_id, num_partitions, size=num_graphs
            )
        else:
            graph_emb = self._pool_graph_embeddings(enc_out, batch, size=num_graphs)

        return self.head(graph_emb)

    def forward_batch(self, batch) -> torch.Tensor:
        pos_enc = get_batch_positional_encoding(batch)
        edge_weight = getattr(batch, "edge_weight", None)
        partition_id = getattr(batch, "partition_id", None)
        num_partitions = getattr(batch, "num_partitions", None)
        num_graphs = getattr(batch, "num_graphs", None)
        if num_graphs is None:
            raise ValueError("Batch.num_graphs is required for explicit pooling size.")

        return self.forward(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
            pos_enc=pos_enc,
            edge_weight=edge_weight,
            partition_id=partition_id,
            num_partitions=num_partitions,
            num_graphs=int(num_graphs),
        )
