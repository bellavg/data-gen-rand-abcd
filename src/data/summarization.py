from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

# Node-level attributes pooled by taking the member minimum.  Both are
# level-like: with pe_type="level" the cached graphs carry pos_enc =
# log1p(level) instead of level, and log1p is strictly monotone, so the
# minimum commutes with it.  Taking the earliest level also preserves the
# causal ordering of the DAG.
_MIN_POOLED_ATTRS = ("level", "pos_enc")


# =====================================================================
# CLUSTERING METHODS
# =====================================================================

def identity_clustering(data: Data) -> torch.Tensor:
    """Assign every node to its own super-node (zero compression).

    Not a real summarizer.  It exercises the whole precompute/staging
    pipeline before any method exists, and doubles as an experimental
    control: a run at zero compression must reproduce the unsummarized
    baseline exactly.
    """
    return torch.arange(data.x.size(0), dtype=torch.long)


def _initial_colors(data: Data, pe_aware: bool) -> torch.Tensor:
    """Colour nodes by their features, optionally including the encoding."""
    parts = [data.x]
    if pe_aware:
        # The model consumes level/pos_enc, and apply_merge_map pools it by
        # minimum.  Folding it into the colour keeps nodes of different depth
        # apart so the pooled encoding is exact rather than an approximation.
        parts += [
            getattr(data, key).reshape(data.x.size(0), -1).to(data.x.dtype)
            for key in _MIN_POOLED_ATTRS
            if getattr(data, key, None) is not None
        ]
    return torch.unique(torch.cat(parts, dim=1), dim=0, return_inverse=True)[1]


def _refine_once(
    colors: torch.Tensor,
    owners: np.ndarray,
    tokens: np.ndarray,
    num_nodes: int,
    count_cap: int | None,
) -> torch.Tensor:
    """One refinement round: recolour by (own colour, neighbour multiset)."""
    order = np.lexsort((tokens, owners))
    owners, tokens = owners[order], tokens[order]

    if count_cap is not None and owners.size:
        # Keep at most count_cap copies of each distinct neighbour token, so
        # count_cap=1 compares neighbour *sets* (bisimulation) while an
        # uncapped run compares multisets (exact colour refinement).
        starts_run = np.empty(owners.size, dtype=bool)
        starts_run[0] = True
        starts_run[1:] = (owners[1:] != owners[:-1]) | (tokens[1:] != tokens[:-1])
        positions = np.arange(owners.size)
        keep = positions - np.maximum.accumulate(np.where(starts_run, positions, 0))
        owners, tokens = owners[keep < count_cap], tokens[keep < count_cap]

    bounds = np.searchsorted(owners, np.arange(num_nodes + 1))
    own = colors.numpy()
    seen: dict[tuple[int, bytes], int] = {}
    refined = np.empty(num_nodes, dtype=np.int64)
    for node in range(num_nodes):
        signature = (int(own[node]), tokens[bounds[node] : bounds[node + 1]].tobytes())
        refined[node] = seen.setdefault(signature, len(seen))
    return torch.from_numpy(refined)


def color_refinement(
    data: Data,
    depth: int = 4,
    count_cap: int | None = None,
    direction: str = "backward",
    pe_aware: bool = True,
) -> torch.Tensor:
    """Cluster nodes by graded Weisfeiler-Leman colour refinement.

    Runs *depth* rounds in which each node is recoloured by its own colour
    plus the multiset of ``(neighbour colour, edge type)`` pairs, then merges
    nodes that end up in the same class.

    *count_cap* grades between the two named endpoints: ``None`` compares full
    multisets (exact colour refinement / 1-WL), while ``1`` compares only sets
    (bisimulation), which ignores how *many* neighbours of a kind a node has
    and therefore merges more aggressively.

    *direction* selects which edges are followed: ``"backward"`` (the default)
    walks fanin edges toward the primary inputs, matching the direction
    messages actually flow in during message passing; ``"forward"`` walks
    fanouts toward the primary outputs; ``"both"`` uses each, tagged so the
    two are never confused.

    *depth* should track the number of encoder layers, since that is how far
    information travels in the model.  ``depth=0`` classifies by node
    features alone.

    If *data* has no ``edge_attr`` (the exact-compression track folds
    polarity into node features instead — see ``data.exact_graph``), every
    edge is treated as a single relation.
    """
    if direction not in ("backward", "forward", "both"):
        raise ValueError(
            f"direction must be 'backward', 'forward' or 'both', got {direction!r}"
        )
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    if count_cap is not None and count_cap < 1:
        raise ValueError(f"count_cap must be >= 1 or None, got {count_cap}")

    num_nodes = data.x.size(0)
    colors = _initial_colors(data, pe_aware)
    if depth == 0 or data.edge_index.numel() == 0:
        return colors

    src, dst = data.edge_index
    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is not None:
        # Edge type, so an inverted edge is never confused with a regular one.
        rel = torch.unique(edge_attr, dim=0, return_inverse=True)[1]
        num_rel = int(rel.max()) + 1
    else:
        rel = torch.zeros(src.size(0), dtype=torch.long)
        num_rel = 1

    # (owner, neighbour) per followed direction, with a tag for "both".
    walks = {"backward": [(dst, src)], "forward": [(src, dst)]}.get(
        direction, [(dst, src), (src, dst)]
    )
    owners = torch.cat([owner for owner, _ in walks]).numpy()
    neighbours = torch.cat([neighbour for _, neighbour in walks])
    relations = rel.repeat(len(walks))
    tags = torch.cat(
        [torch.full_like(owner, i) for i, (owner, _) in enumerate(walks)]
    )

    for _ in range(depth):
        tokens = ((colors[neighbours] * num_rel + relations) * len(walks) + tags).numpy()
        refined = _refine_once(colors, owners, tokens, num_nodes, count_cap)
        if int(refined.max()) == int(colors.max()):
            # The partition stopped changing; further rounds cannot split it.
            colors = refined
            break
        colors = refined

    return colors


# A method maps a graph to a cluster vector ``LongTensor[num_nodes]``
# assigning each node a super-node id in ``[0, num_clusters)``.  Adding a
# summarization method means adding an entry here; nothing downstream of
# apply_merge_map changes.
SUMMARIZATION_REGISTRY: dict[str, Callable[..., torch.Tensor]] = {
    "identity": identity_clustering,
    "wl": color_refinement,
}


# =====================================================================
# MERGE-MAP REWRITE
# =====================================================================

def _validate_and_bincount(
    cluster: torch.Tensor, num_nodes: int, num_clusters: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and validate a cluster vector; return it with per-cluster sizes.

    Shared by apply_merge_map and exact_graph.apply_exact_merge_map. Bounds
    must be checked before bincount: it raises on negative values, and
    returns a tensor of length max(num_clusters, cluster.max() + 1), which
    would make the empty-cluster check below fire for the wrong reason.
    """
    cluster = cluster.reshape(-1).long()

    if cluster.numel() != num_nodes:
        raise ValueError(
            f"cluster has {cluster.numel()} entries but the graph has "
            f"{num_nodes} nodes"
        )
    if num_clusters < 1:
        raise ValueError(f"num_clusters must be >= 1, got {num_clusters}")
    if int(cluster.min()) < 0 or int(cluster.max()) >= num_clusters:
        raise ValueError(
            f"cluster ids must lie in [0, {num_clusters}), got range "
            f"[{int(cluster.min())}, {int(cluster.max())}]"
        )

    node_size = torch.bincount(cluster, minlength=num_clusters)
    if int(node_size.min()) == 0:
        raise ValueError(
            f"{int((node_size == 0).sum())} of {num_clusters} cluster ids are "
            "unused; cluster ids must be contiguous"
        )
    return cluster, node_size


def _pool_min(values: torch.Tensor, cluster: torch.Tensor, num_clusters: int) -> torch.Tensor:
    """Per-super-node minimum over member rows, preserving trailing dims."""
    flat = values.reshape(values.size(0), -1)
    out = torch.zeros(num_clusters, flat.size(1), dtype=flat.dtype)
    out.scatter_reduce_(
        0, cluster.view(-1, 1).expand_as(flat), flat, reduce="amin", include_self=False
    )
    return out.reshape(num_clusters, *values.shape[1:])


def apply_merge_map(data: Data, cluster: torch.Tensor, num_clusters: int) -> Data:
    """Collapse each cluster of nodes into one super-node and return a new graph.

    This is the single rewrite shared by every summarization method: a method
    only has to produce *cluster*, and this does the rest.

    Expects the AIG schema built by ``data_utils.aig_to_pytorch_geometric``:
    ``x`` [N,4] one-hot node type, ``edge_attr`` [E,2] one-hot inverter
    polarity, and either ``level`` or ``pos_enc`` [N,1].  The result uses the
    same dimensions with counts in place of one-hots, so a size-1 super-node
    reproduces its original row exactly and the model needs no changes.

    Merging is applied as follows:
      - ``x``          → member type counts ``[#const, #PI, #AND, #PO]``
      - ``edge_attr``  → super-edge polarity counts ``[#normal, #inverted]``
      - ``level`` / ``pos_enc`` → member minimum
      - intra-cluster edges are dropped (they would be self-loops) and their
        number recorded as ``internal_edges``
      - ``pi_paths``, ``local_sp_sum`` and ``edge_weight`` are dropped: the
        first two are unused, and ``edge_weight`` is degree-derived, so it is
        stale after a merge and is recomputed by the dataset when needed.

    *cluster* must assign every node an id in ``[0, num_clusters)`` and use
    every id at least once; methods own contiguity, this never relabels.

    Note that parallel edges in *data* are summed rather than preserved, so
    an identity merge is only exactly lossless on a graph that has none.
    ABC structurally hashes every graph the dataset is built from, so this
    does not arise in practice and is not checked — validating it would cost
    an extra coalesce on each of ~700k graphs.

    Pure: *data* is never mutated and no input tensor is returned by
    reference.  This runs offline in a multiprocessing pool over the whole
    corpus, and ``ExtractPrecomputedPE`` later mutates its input in place.
    """
    num_nodes = data.x.size(0)
    cluster, node_size = _validate_and_bincount(cluster, num_nodes, num_clusters)

    # x: member type counts.  A size-1 super-node reproduces the one-hot.
    x = torch.zeros(num_clusters, data.x.size(1), dtype=data.x.dtype)
    x.index_add_(0, cluster, data.x)

    # Edges: remap endpoints, drop intra-cluster self-loops, then sum the
    # polarity counts of parallel super-edges.
    merged = cluster[data.edge_index]
    keep = merged[0] != merged[1]
    internal_edges = int(data.edge_index.size(1) - int(keep.sum()))
    edge_index, edge_attr = coalesce(
        merged[:, keep],
        data.edge_attr[keep],
        num_nodes=num_clusters,
        reduce="sum",
    )

    out = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    pooled = [key for key in _MIN_POOLED_ATTRS if getattr(data, key, None) is not None]
    if not pooled:
        # Without one of these the dataset re-runs pe_transform, finds nothing,
        # leaves pos_enc unset, and then rebuilds the cache file on every run.
        raise ValueError(
            "graph carries neither 'level' nor 'pos_enc'; one is required to "
            "derive the positional encoding of the merged graph"
        )
    for key in pooled:
        out[key] = _pool_min(getattr(data, key), cluster, num_clusters)

    out.internal_edges = torch.tensor([internal_edges], dtype=torch.long)
    # PyG can infer num_nodes from x, but the dataset's edge-normalization
    # path reads it explicitly, so store it.
    out.num_nodes = num_clusters
    out.num_edges = edge_index.size(1)
    # Member counts are merge-invariant, so these stay exact.
    out.num_pis = int(x[:, 1].sum())
    out.num_pos = int(x[:, 3].sum())

    return out


def summarize_graph(data: Data, method: str, **params) -> Data:
    """Run a registered summarization method and rewrite the graph."""
    if method not in SUMMARIZATION_REGISTRY:
        raise ValueError(
            f"Unknown summarization method: {method!r}. "
            f"Valid: {sorted(SUMMARIZATION_REGISTRY)}"
        )
    cluster = SUMMARIZATION_REGISTRY[method](data, **params).reshape(-1).long()
    return apply_merge_map(data, cluster, int(cluster.max()) + 1)


__all__ = [
    "SUMMARIZATION_REGISTRY",
    "apply_merge_map",
    "color_refinement",
    "identity_clustering",
    "summarize_graph",
]
