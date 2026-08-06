from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

from data.summarization import _validate_and_bincount

# Column index of the "inverted" one-hot component in the standard [E,2]
# edge_attr = [normal, inverted] built by data_utils.aig_to_pytorch_geometric.
_INVERTED_COLUMN = 1


# =====================================================================
# EXACT-SCHEMA CONVERSION
# =====================================================================

def fold_inversions_into_x(data: Data) -> Data:
    """Replace edge-level inverter polarity with a per-node fanin count.

    AIGs have a fixed fan-in per node type (0 for const/PI, 2 for AND, 1 for
    PO), so "how many of this node's incoming edges are inverted", plus its
    type, fully describes its polarity pattern for graph-level optimizability
    prediction — at the cost of no longer knowing *which* specific fanin is
    inverted. That trade-off is scoped to this exact-compression track only
    (see models/layers/gcn_exact.py); the general pipeline keeps polarity as
    a per-edge relation.

    This is what makes S2 exact colour refinement (count_cap=None) genuinely
    lossless for a trained GNN: with no edge_attr to fold multiplicity into,
    the merge (apply_exact_merge_map) can put multiplicity entirely on
    edge_weight, applied *after* the message function's nonlinearity —
    the only way "k identical incoming messages" can be represented exactly
    by one aggregated computation. Folding it into edge_attr instead (the
    general apply_merge_map's approach) scales a value *fed into* the
    nonlinearity, which it cannot decompose over sums.

    Attaches ``edge_weight = 1`` and ``node_size = 1`` on every node, so the
    output is a complete, ready-to-train exact-schema graph whether or not
    it is later coarsened.  Setting ``node_size`` here too (not just on
    coarsened output) matters beyond consistency: ``Batch.from_data_list``
    silently *drops* an attribute from the whole batch if even one graph in
    it lacks that attribute, so a batch mixing an uncoarsened and a
    coarsened graph would otherwise lose ``node_size`` entirely and
    ``ExactGraphBaseModel`` would silently pool the coarsened graph's
    super-nodes as if each stood for a single node.

    Drops ``edge_attr`` entirely and appends the inversion count as an extra
    column of ``x``.  Pure: returns a new Data, never mutates *data*.
    """
    num_nodes = data.x.size(0)
    num_edges = data.edge_index.size(1)
    inverted = torch.zeros(num_nodes, 1, dtype=data.x.dtype)
    if num_edges > 0:
        inverted.index_add_(
            0,
            data.edge_index[1],
            data.edge_attr[:, _INVERTED_COLUMN : _INVERTED_COLUMN + 1].to(data.x.dtype),
        )

    out = Data(
        x=torch.cat([data.x, inverted], dim=1),
        edge_index=data.edge_index.clone(),
        edge_weight=torch.ones(num_edges, dtype=data.x.dtype),
    )
    out.node_size = torch.ones(num_nodes, 1, dtype=torch.long)
    out.num_nodes = num_nodes
    out.num_edges = num_edges
    return out


# =====================================================================
# EXACT MERGE-MAP REWRITE
# =====================================================================

def apply_exact_merge_map(data: Data, cluster: torch.Tensor, num_clusters: int) -> Data:
    """Collapse each cluster into one super-node for the exact-compression track.

    Companion to ``summarization.apply_merge_map``, specialised for graphs
    already produced by ``fold_inversions_into_x``. Unlike the general
    primitive this keeps ``x`` as the class **representative**, not the
    member sum: every original member of an exact-WL cluster shares an
    identical ``x`` row by construction (WL seeds on ``x``), so any
    reduction gives the same answer, and only the representative is
    consistent with a bias-carrying, sum-aggregating message-passing net —
    see fold_inversions_into_x's docstring for the underlying argument.

    Multiplicity moves entirely onto ``edge_weight``, and specifically the
    **per-target-member** value: how many source-class neighbours does
    *each* member of the target class have (guaranteed uniform across
    members by the WL equitable-partition property). This is *not* the raw
    total of coalesced original edges — that total scales with the target
    class's size too, which is a different, wrong quantity. It is computed
    as ``coalesced_count / target_class_size``.

    Returns a Data carrying ``x`` (representative), ``edge_index``,
    ``edge_weight`` (the corrected multiplicity), and ``node_size`` (needed
    for size-weighted pooling downstream — see models/base_model_exact.py).
    No ``edge_attr``: this schema never has one.

    Intra-cluster edges (a real edge between two members of the same class)
    are kept, as a weighted self-loop, and need no special case. Bollen's
    Def 3.6 builds the reduct's edge relation over *all* pairs of classes
    including ``v == w``, so a self-loop is part of the definition rather
    than a degenerate case of it, and ``coalesced_count / target_class_size``
    already computes its weight correctly: each member of the class has the
    same number of same-class fanins, and that count is what the self-loop
    carries. This is not rare — a cluster of structurally repetitive AIG
    regions (duplicated bit-slices) can easily contain adjacent members
    without any literal cycle, and the coarser the reduct the more common it
    gets. Note this is where the two rewrites genuinely differ: the lossy
    ``apply_merge_map`` *drops* these edges, which is what makes it lossy.

    Same preconditions as apply_merge_map: *cluster* assigns every node an
    id in [0, num_clusters) and uses each at least once; pure (never mutates
    *data*, never returns an input tensor by reference).
    """
    num_nodes = data.x.size(0)
    cluster, node_size = _validate_and_bincount(cluster, num_nodes, num_clusters)

    # x: class representative. Sum-then-divide rather than picking a single
    # member's row so a non-WL-homogeneous cluster degrades to an average
    # instead of an arbitrary pick.
    x_sum = torch.zeros(num_clusters, data.x.size(1), dtype=data.x.dtype)
    x_sum.index_add_(0, cluster, data.x)
    x = x_sum / node_size.unsqueeze(1).to(data.x.dtype)

    # Edges: remap.  Intra-cluster edges survive as self-loops rather than
    # being dropped (see the docstring above); coalesce collapses them onto
    # the ``(c, c)`` entry like any other parallel super-edge.
    merged = cluster[data.edge_index]
    ones = torch.ones(merged.size(1), 1, dtype=data.x.dtype)
    edge_index, counts = coalesce(
        merged, ones, num_nodes=num_clusters, reduce="sum"
    )

    target_size = node_size[edge_index[1]].to(data.x.dtype)
    edge_weight = counts.squeeze(1) / target_size

    out = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    out.node_size = node_size.unsqueeze(1)
    out.num_nodes = num_clusters
    out.num_edges = edge_index.size(1)

    return out


__all__ = ["apply_exact_merge_map", "fold_inversions_into_x"]
