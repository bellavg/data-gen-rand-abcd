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

# Ceiling on the matching sub-passes of one agglomerative round.  Set well
# above what real graphs need (12-238 measured up to 366k nodes) so it only
# ever catches a pathological cost ordering; see _greedy_disjoint_pairs.
_MAX_MATCHING_PASSES = 512


# =====================================================================
# SHARED HELPERS
# =====================================================================

def _relabel(labels: torch.Tensor) -> torch.Tensor:
    """Map arbitrary integer labels onto contiguous ids in ``[0, num_classes)``.

    Every method has to return such a vector, because ``apply_merge_map``
    rejects gaps rather than relabelling on its behalf.
    """
    return torch.unique(labels.reshape(-1), return_inverse=True)[1]


def _relabel_rows(key: torch.Tensor) -> torch.Tensor:
    """Contiguous ids for the distinct *rows* of an integer key matrix."""
    return torch.unique(key, dim=0, return_inverse=True)[1]


def _node_levels(data: Data) -> torch.Tensor:
    """Integer topological level per node, from whichever attribute is present.

    Raw graphs carry ``level``; cached ones carry ``pos_enc = log1p(level)``
    instead, since ``ExtractPrecomputedPE`` consumes ``level`` and deletes it.
    Levels are integers below ``config.MAX_DEPTH``, comfortably inside the range
    where a float32 log1p round-trips exactly after rounding.
    """
    level = getattr(data, "level", None)
    if level is not None:
        return level.reshape(-1).round().long()
    pos_enc = getattr(data, "pos_enc", None)
    if pos_enc is None:
        raise ValueError(
            "graph carries neither 'level' nor 'pos_enc'; one is required to "
            "recover node levels"
        )

    # pos_enc is log1p of whichever attribute PE_TYPE selected, so inverting
    # it as a level is only valid for pe_type="level".  Under "pi_paths" it
    # would return path counts — which reach values that saturate int64 once
    # exponentiated — as if they were topological depths, silently.
    import config

    if config.PE_TYPE != "level":
        raise ValueError(
            f"cannot recover levels from pos_enc under PE_TYPE={config.PE_TYPE!r}; "
            "it holds log1p of that attribute, not of the level"
        )
    return torch.expm1(pos_enc.reshape(-1).double()).round().long()


def _edge_polarity(data: Data) -> torch.Tensor:
    """``[E, 2]`` one-hot inverter polarity, defaulting to all-normal.

    The exact-compression track folds polarity into node features and drops
    ``edge_attr`` entirely (see ``data.exact_graph``), so a descriptor built
    from it must not assume the attribute exists.  Note this only keeps the
    *clustering* callable on such a graph: ``summarize_graph`` would still
    fail in ``apply_merge_map``, which needs ``edge_attr`` to build the
    super-edge polarity counts.  The exact track has its own rewrite.
    """
    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is not None:
        return edge_attr
    polarity = torch.zeros(data.edge_index.size(1), 2)
    polarity[:, 0] = 1.0
    return polarity


def _undirected_simple(
    edge_index: torch.Tensor,
    num_nodes: int,
    weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Undirected simple edge list ``(i < j)`` carrying edge multiplicities.

    The direction-blind methods (S3, S4) are defined on an undirected graph,
    so this drops orientation, discards self-loops, and folds parallel edges
    into one edge whose weight is how many originals it stands for.
    """
    lower = torch.minimum(edge_index[0], edge_index[1])
    upper = torch.maximum(edge_index[0], edge_index[1])
    keep = lower != upper
    pairs = torch.stack([lower[keep], upper[keep]])
    if weight is None:
        weight = torch.ones(edge_index.size(1), dtype=torch.float32)
    weight = weight[keep]
    if pairs.size(1) == 0:
        return pairs, weight
    return coalesce(pairs, weight, num_nodes=num_nodes, reduce="sum")


def _mutual_best_pairs(
    pairs: torch.Tensor,
    cost: torch.Tensor,
    num_nodes: int,
    budget: int,
) -> torch.Tensor:
    """Select disjoint node pairs that are each other's cheapest partner.

    Both agglomerative methods want "merge the cheapest non-overlapping
    candidates".  Doing that exactly means a sequential scan over every sorted
    candidate, which is far too slow in Python at ~10^6 edges across ~10^5
    graphs.  Mutual-best matching is the standard parallel stand-in: it selects
    a subset of what the greedy scan would select, runs in a few vectorised
    passes, and is deterministic (ties broken by node id).  Callers use it
    through ``_greedy_disjoint_pairs``, which repeats it to recover most of
    what one pass leaves behind.

    Returns a ``[2, K]`` tensor of pairs with ``K <= budget``.
    """
    empty = torch.empty((2, 0), dtype=torch.long)
    if budget <= 0 or pairs.size(1) == 0:
        return empty

    # Each undirected candidate is looked at from both endpoints.
    src = torch.cat([pairs[0], pairs[1]]).numpy()
    dst = torch.cat([pairs[1], pairs[0]]).numpy()
    both = torch.cat([cost, cost]).double().numpy()

    order = np.lexsort((dst, both, src))
    src, dst, both = src[order], dst[order], both[order]
    is_first = np.empty(src.size, dtype=bool)
    is_first[0] = True
    is_first[1:] = src[1:] != src[:-1]

    best = np.full(num_nodes, -1, dtype=np.int64)
    best_cost = np.zeros(num_nodes)
    best[src[is_first]] = dst[is_first]
    best_cost[src[is_first]] = both[is_first]

    matched = np.flatnonzero(best >= 0)
    matched = matched[(best[best[matched]] == matched) & (matched < best[matched])]
    if matched.size == 0:
        return empty

    keep = matched[np.lexsort((matched, best_cost[matched]))][:budget]
    return torch.from_numpy(np.stack([keep, best[keep]]))


def _greedy_disjoint_pairs(
    pairs: torch.Tensor,
    cost: torch.Tensor,
    num_nodes: int,
    budget: int,
) -> torch.Tensor:
    """Cheapest-first disjoint pairs, by repeated mutual-best sub-passes.

    One mutual-best pass alone matches very little on these graphs: primary
    inputs and, later, super-nodes are hubs that many nodes name as their
    cheapest partner while only one can be reciprocated.  Repeating the pass
    over the nodes left unmatched is far cheaper than the caller's per-round
    work (rebuilding the coarse graph, re-coalescing every edge), so several
    passes per round is what collapses the round count — measured at 2 rounds
    for graphs from 24k to 366k nodes, against 150-330 for a single pass.  It
    also lands closer to the sequential greedy scan the published algorithms
    specify than one pass does, and still selects a subset of it.

    Each pass is O(num_nodes) whatever is left to match, and a pathological
    cost ordering (monotone along a path) matches one pair per pass, so the
    passes are capped.  Real inputs need 12-238; hitting the cap only defers
    merges to the caller's next round, it does not change the target.
    """
    selected: list[torch.Tensor] = []
    taken = 0
    for _ in range(_MAX_MATCHING_PASSES):
        if taken >= budget or pairs.size(1) == 0:
            break
        matched = _mutual_best_pairs(pairs, cost, num_nodes, budget - taken)
        if matched.size(1) == 0:
            break
        selected.append(matched)
        taken += matched.size(1)

        # Drop every candidate touching a node just matched.  The rest are
        # reused with the costs they already have: exact for spectral, where
        # a merge elsewhere changes no other node's degree or subspace row,
        # and a close approximation for ConvMatch, where a survivor next to a
        # merged node drifts by a few percent (rank correlation 0.99).
        used = torch.zeros(num_nodes, dtype=torch.bool)
        used[matched.reshape(-1)] = True
        keep = ~(used[pairs[0]] | used[pairs[1]])
        pairs, cost = pairs[:, keep], cost[keep]

    if not selected:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.cat(selected, dim=1)


def _apply_pairs(label: torch.Tensor, matched: torch.Tensor, num_clusters: int):
    """Fold matched cluster pairs into their partner and return the new labels.

    Returns ``(label, level_map, num_clusters)``: the updated per-node cluster
    vector, the old-cluster to new-cluster map (needed to rebuild the coarse
    graph without touching the node level), and the new cluster count.
    """
    level_map = torch.arange(num_clusters, dtype=torch.long)
    level_map[matched[0]] = matched[1]
    level_map = _relabel(level_map)
    return level_map[label], level_map, int(level_map.max()) + 1


# =====================================================================
# CLUSTERING METHODS
# =====================================================================

def identity_clustering(data: Data) -> torch.Tensor:
    """Assign every node to its own super-node (zero compression).

    A test fixture, not an experiment.  Its output compares equal to the
    input on everything the production encoder reads, so training on it only
    reproduces the unsummarized baseline — and it cannot serve as a control
    for the staging path either, since a run that silently fell back to raw
    graphs would produce exactly this.  Useful for telling a broken pipeline
    apart from a broken method, and as the identity case of apply_merge_map.
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


def _immediate_postdominators(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Immediate post-dominator of every node, as ids in ``[0, num_nodes]``.

    Node *p* post-dominates *v* when every path from *v* to an output passes
    through *p*, so the immediate post-dominator is the gate at which *v*'s
    fanout cone reconverges.  Two gates sharing one are parallel branches of
    the same cone — which is exactly the relation the width axis merges on.

    Post-dominance, not dominance.  Forward dominators were measured on this
    corpus first and are degenerate here: ~99% of AND gates have the virtual
    source as their immediate dominator, because a gate deep in an AIG is
    reachable from the input frontier along many independent paths.  Grouping
    on that collapses to "merge everything on a level", which is a different
    (and much blunter) method than the one intended.

    An AIG has many outputs rather than one, so a virtual sink is added over
    every node of out-degree zero.  A node whose immediate post-dominator is
    that sink is **given a private id**, not the sink's: its fanout reaches
    several outputs without passing through any one gate, so it has no
    reconvergence point and nothing to be a parallel branch *of*.  Grouping
    those together would merge gates that share nothing — two of them can sit
    in different connected components — and there are a lot of them: 29% of
    AND gates on an adder-like circuit, 73% on a graph with no reconvergent
    structure at all.  The same private id is given to nodes
    that reach no output at all, which a well-formed netlist does not contain.
    """
    import networkx as nx

    # Post-dominators of a graph are the dominators of its reverse.
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edge_index.flip(0).t().tolist())
    sink = -1
    graph.add_edges_from(
        (sink, node) for node in range(num_nodes) if graph.in_degree(node) == 0
    )

    # Private ids start past every real node id so they cannot collide.
    postdominators = torch.arange(num_nodes, dtype=torch.long) + num_nodes
    for node, postdominator in nx.immediate_dominators(graph, sink).items():
        if node != sink and postdominator != sink:
            postdominators[node] = postdominator
    return postdominators


def _contract_chains(
    edge_index: torch.Tensor,
    label: torch.Tensor,
    is_and: torch.Tensor,
    levels: torch.Tensor,
    max_chain_length: int,
) -> torch.Tensor:
    """Contract fanout-free chains of AND clusters in the quotient of *label*.

    A cluster whose only outgoing edge goes to cluster *b* is absorbed into
    *b*.  This is what shortens the circuit: it removes a level from every
    path through the chain, which is the axis that matters for over-squashing
    on a graph far deeper than the encoder has layers.

    Acyclicity is preserved.  Every edge leaving the merged cluster leaves from
    *b*, so a cycle would need a path ``b -> ... -> a``; together with the
    ``a -> b`` edge that is a cycle in the input, which a netlist does not have.
    The argument only assumes the input is a DAG, so it still holds when this
    runs on the quotient left by the width axis.
    """
    num_clusters = int(label.max()) + 1
    quotient = label[edge_index]
    external = quotient[:, quotient[0] != quotient[1]]
    if external.size(1) == 0:
        return label

    candidates = torch.unique(external.t(), dim=0)
    out_degree = torch.bincount(candidates[:, 0], minlength=num_clusters)

    # A cluster may only be absorbed if all of its members are AND gates:
    # primary inputs and outputs are the circuit's fixed interface and are
    # never merged away.
    cluster_is_and = torch.ones(num_clusters, dtype=torch.bool)
    cluster_is_and.scatter_reduce_(0, label, is_and, reduce="amin", include_self=False)

    candidates = candidates[
        (out_degree[candidates[:, 0]] == 1)
        & cluster_is_and[candidates[:, 0]]
        & cluster_is_and[candidates[:, 1]]
    ]
    if candidates.size(0) == 0:
        return label

    # Bottom-up, so a chain grows from the inputs toward the outputs and the
    # length cap truncates the far end rather than an arbitrary middle.
    cluster_level = torch.zeros(num_clusters, dtype=torch.long)
    cluster_level.scatter_reduce_(0, label, levels, reduce="amin", include_self=False)
    order = torch.argsort(cluster_level[candidates[:, 0]], stable=True)

    parent = list(range(num_clusters))
    chain = [1] * num_clusters

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for tail, head in candidates[order].tolist():
        tail_root, head_root = find(tail), find(head)
        if tail_root == head_root:
            continue
        if chain[tail_root] + chain[head_root] > max_chain_length:
            continue
        parent[tail_root] = head_root
        chain[head_root] += chain[tail_root]

    roots = torch.tensor([find(c) for c in range(num_clusters)], dtype=torch.long)
    return _relabel(roots[label])


def cone_coarsening(
    data: Data,
    max_chain_length: int = 4,
    level_band: int | None = 0,
) -> torch.Tensor:
    """Cluster nodes by level-bounded cone coarsening (the AIG-native method).

    Merges along two axes, both restricted to AND gates so the primary
    input/output interface survives intact:

    *Width* — AND gates on the same topological level whose fanout cones
    reconverge at the same gate (they share an immediate post-dominator) are
    merged.  These are the parallel branches of one cone, so this compresses
    circuit width while leaving every path length untouched; because merged
    nodes share a level, the level positional encoding of the super-node is
    the exact level of its members rather than a pooled approximation.  A gate
    whose fanout reaches several outputs without reconverging on any single
    gate has no cone to be a branch of and is left alone, which is most of a
    multi-output AIG.

    *Depth* — maximal fanout-free chains of AND clusters are contracted, up to
    *max_chain_length* clusters per chain.  This is the axis that shortens
    paths, and so the one that acts on over-squashing.

    Both axes preserve acyclicity, and composing them does too: the width axis
    only merges nodes of equal level, and every edge of a netlist runs from a
    lower level to a strictly higher one, so no edge can exist inside a merged
    group and no cycle can form between two of them; the depth axis then runs
    on that quotient, which is still a DAG (see ``_contract_chains``).

    *level_band* widens the width axis to bands of ``level_band + 1``
    consecutive levels: more compression, at the cost of both the exact level
    encoding and the acyclicity guarantee.  Bands are fixed windows rather
    than a sliding ±k, since "within k levels of each other" is not an
    equivalence relation and so cannot define a partition at all; one
    consequence is that widening the band does not monotonically increase
    compression, because two adjacent levels may fall either side of a window
    boundary.  ``None`` disables the width axis,
    ``max_chain_length <= 1`` disables the depth axis; either alone is the
    single-axis ablation.
    """
    if level_band is not None and level_band < 0:
        raise ValueError(f"level_band must be >= 0 or None, got {level_band}")

    num_nodes = data.x.size(0)
    is_and = data.x[:, 2] == 1.0
    levels = _node_levels(data)
    label = torch.arange(num_nodes, dtype=torch.long)

    if level_band is not None:
        group = _relabel_rows(
            torch.stack(
                [
                    _immediate_postdominators(data.edge_index, num_nodes),
                    levels // (level_band + 1),
                ],
                dim=1,
            )
        )
        # Offset past the identity labels so a group id cannot collide with
        # the private id of a node that does not take part.
        label = _relabel(torch.where(is_and, group + num_nodes, label))

    if max_chain_length > 1 and data.edge_index.numel():
        label = _contract_chains(
            data.edge_index, label, is_and, levels, max_chain_length
        )

    return label


def convmatch_coarsening(
    data: Data,
    reduction_ratio: float = 0.5,
    sgc_depth: int = 4,
    num_probes: int = 2,
    seed: int = 42,
) -> torch.Tensor:
    """Cluster nodes by convolution matching (A-ConvMatch, Dickens et al. 2024).

    Merges the nodes whose merging perturbs the *output of a graph convolution*
    least, rather than preserving a property of the graph itself.  Each round
    scores candidate pairs with the closed-form upper bound of the paper's
    Theorem 1 — the change in the two nodes' own convolved representations plus
    the change they force on their neighbours' — then merges the cheapest
    disjoint pairs, repeating until *reduction_ratio* of the nodes are gone.
    The ratio is a target, not a guarantee: a graph can run out of candidate
    pairs first, which leaves it less compressed than asked for.

    Candidates come from the paper's step 1: nearest neighbours in the
    embedding of an unparameterised *sgc_depth*-layer SGC network.  Exact
    nearest neighbours are quadratic, so neighbours in that space are
    approximated by pairing nodes adjacent along *num_probes* random
    projections of it, together with every graph edge.

    Deliberately domain-blind: it uses the paper's symmetrically normalised
    undirected convolution, ignoring both edge direction and inverter polarity.
    That is what makes it the general SOTA bar — a method with no knowledge
    that these graphs are circuits.

    Worth knowing when reading its results: the published objective is an
    *unweighted* L1 sum over the convolution output, so it favours merging
    low-degree nodes with small representations independently of how alike
    they are.  Convolution equivalence decides between pairs of comparable
    degree, not across them.

    Two documented deviations from the published algorithm, both for scale:
    disjoint pairs are chosen by repeated mutual-best matching rather than a
    sequential greedy scan (see ``_greedy_disjoint_pairs``), and the
    ``|C|`` self-loops a merged super-node inherits from its members are left
    contributing their pre-merge values, which is part of what makes the cost
    O(1) per pair.  The larger approximation — an adjacent pair's shared edge
    — is corrected exactly.
    """
    if not 0.0 <= reduction_ratio < 1.0:
        raise ValueError(f"reduction_ratio must be in [0, 1), got {reduction_ratio}")
    if sgc_depth < 0:
        raise ValueError(f"sgc_depth must be >= 0, got {sgc_depth}")

    num_nodes = data.x.size(0)
    target = max(1, int(round(num_nodes * (1.0 - reduction_ratio))))
    pairs, weight = _undirected_simple(data.edge_index, num_nodes)
    if target >= num_nodes or pairs.size(1) == 0:
        return torch.arange(num_nodes, dtype=torch.long)

    # The graph signal the model actually convolves: node type plus the level
    # encoding, on a comparable scale so neither dominates the L1 cost.
    levels = _node_levels(data).to(torch.float32).reshape(-1, 1)
    features = torch.cat([data.x, torch.log1p(levels)], dim=1)

    embedding = _sgc_embedding(pairs, weight, features, num_nodes, sgc_depth)
    candidates = torch.cat(
        [pairs, _projection_neighbour_pairs(embedding, num_probes, seed)], dim=1
    )

    label = torch.arange(num_nodes, dtype=torch.long)
    size = torch.ones(num_nodes)
    num_clusters = num_nodes

    while num_clusters > target and candidates.size(1):
        cost = _convmatch_costs(candidates, pairs, weight, features, size, num_clusters)
        matched = _greedy_disjoint_pairs(
            candidates, cost, num_clusters, num_clusters - target
        )
        if matched.size(1) == 0:
            break

        label, cluster_map, num_clusters = _apply_pairs(label, matched, num_clusters)
        pairs, weight = _undirected_simple(cluster_map[pairs], num_clusters, weight)
        features = _pool_sum(features * size.unsqueeze(1), cluster_map, num_clusters)
        size = _pool_sum(size, cluster_map, num_clusters)
        features = features / size.unsqueeze(1)
        candidates = _remap_pairs(candidates, cluster_map)

    return label


def _pool_sum(
    values: torch.Tensor, cluster: torch.Tensor, num_clusters: int
) -> torch.Tensor:
    """Per-cluster sum over member rows."""
    out = torch.zeros(num_clusters, *values.shape[1:], dtype=values.dtype)
    return out.index_add_(0, cluster, values)


def _remap_pairs(pairs: torch.Tensor, cluster_map: torch.Tensor) -> torch.Tensor:
    """Push candidate pairs through a merge, dropping collapsed duplicates."""
    mapped = cluster_map[pairs]
    lower = torch.minimum(mapped[0], mapped[1])
    upper = torch.maximum(mapped[0], mapped[1])
    keep = lower != upper
    return torch.unique(torch.stack([lower[keep], upper[keep]]).t(), dim=0).t()


def _coarse_degree(
    pairs: torch.Tensor,
    weight: torch.Tensor,
    size: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Degrees under ConvMatch's convention for the coarse graph.

    A super-node standing for ``|C|`` original nodes carries ``|C|``
    self-loops, so ``d_i = sum_j a_ij + |C_i|``.
    """
    return torch.zeros(num_nodes).index_add_(
        0, pairs.reshape(-1), weight.repeat(2)
    ) + size


def _candidate_edge_weight(
    candidates: torch.Tensor, pairs: torch.Tensor, weight: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Weight of the edge joining each candidate pair, or zero if not adjacent.

    ``pairs`` comes out of ``coalesce`` sorted row-major and both tensors hold
    ``(lower, upper)``, so a single searchsorted over the flattened keys
    answers the whole candidate set at once — keeping the merge cost O(1) per
    candidate even though it now needs to know whether the pair is an edge.
    """
    if pairs.size(1) == 0 or candidates.size(1) == 0:
        return torch.zeros(candidates.size(1))

    keys = pairs[0] * num_nodes + pairs[1]
    wanted = candidates[0] * num_nodes + candidates[1]
    position = torch.searchsorted(keys, wanted).clamp_max(keys.numel() - 1)
    return torch.where(keys[position] == wanted, weight[position], 0.0)


def _sgc_embedding(
    pairs: torch.Tensor,
    weight: torch.Tensor,
    features: torch.Tensor,
    num_nodes: int,
    depth: int,
) -> torch.Tensor:
    """``(D^-1/2 A D^-1/2)^depth X`` — the unparameterised SGC embedding."""
    degree = _coarse_degree(pairs, weight, torch.ones(num_nodes), num_nodes)
    inverse_sqrt = degree.clamp_min(1e-12).rsqrt()
    normalized = weight * inverse_sqrt[pairs[0]] * inverse_sqrt[pairs[1]]
    self_loop = 1.0 / degree
    embedding = features
    for _ in range(depth):
        propagated = self_loop.unsqueeze(1) * embedding
        message = normalized.unsqueeze(1) * embedding[pairs[1]]
        propagated.index_add_(0, pairs[0], message)
        message = normalized.unsqueeze(1) * embedding[pairs[0]]
        propagated.index_add_(0, pairs[1], message)
        embedding = propagated
    return embedding


def _projection_neighbour_pairs(
    embedding: torch.Tensor, num_probes: int, seed: int
) -> torch.Tensor:
    """Approximate nearest-neighbour pairs by sorting random projections."""
    if num_probes < 1 or embedding.size(0) < 2:
        return torch.empty((2, 0), dtype=torch.long)

    generator = torch.Generator().manual_seed(seed)
    projections = embedding @ torch.randn(
        embedding.size(1), num_probes, generator=generator
    )
    neighbours = []
    for probe in range(num_probes):
        order = torch.argsort(projections[:, probe], stable=True)
        neighbours.append(torch.stack([order[:-1], order[1:]]))
    return _remap_pairs(
        torch.cat(neighbours, dim=1), torch.arange(embedding.size(0), dtype=torch.long)
    )


def _convmatch_representation(
    pairs: torch.Tensor,
    weight: torch.Tensor,
    features: torch.Tensor,
    size: torch.Tensor,
    num_clusters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One coarse graph convolution, ``D^-1/2 (A + I) D^-1/2 X``.

    Returns ``(aggregate, representation, scaled, degree)``.  *aggregate* is
    the convolution held back one step, ``sum_j a_ij x_j / sqrt(d_j)`` without
    the owner's own ``1/sqrt(d_i)``.  Keeping it in that form is what makes
    the merge cost O(1) per candidate: over *disjoint* neighbourhoods it is
    additive, so a merged super-node's aggregate is the sum of its members'
    and needs no rebuild of the coarse graph.  When the two members are
    themselves adjacent the shared edge appears in both terms and has to be
    taken back out — see ``_convmatch_costs``.
    """
    degree = _coarse_degree(pairs, weight, size, num_clusters)
    inverse_sqrt = degree.clamp_min(1e-12).rsqrt()

    scaled = features * inverse_sqrt.unsqueeze(1)
    aggregate = size.unsqueeze(1) * scaled
    aggregate.index_add_(0, pairs[0], weight.unsqueeze(1) * scaled[pairs[1]])
    aggregate.index_add_(0, pairs[1], weight.unsqueeze(1) * scaled[pairs[0]])
    return aggregate, aggregate * inverse_sqrt.unsqueeze(1), scaled, degree


def _convmatch_costs(
    candidates: torch.Tensor,
    pairs: torch.Tensor,
    weight: torch.Tensor,
    features: torch.Tensor,
    size: torch.Tensor,
    num_clusters: int,
) -> torch.Tensor:
    """A-ConvMatch merge cost (Theorem 1) for every candidate pair."""
    aggregate, representation, scaled, degree = _convmatch_representation(
        pairs, weight, features, size, num_clusters
    )
    inverse_sqrt = degree.clamp_min(1e-12).rsqrt()

    # How much a node's *neighbours* weight it when they aggregate — the
    # `sum over i in N({u})` of Theorem 1, which ranges over neighbours only
    # and so carries no term for the node itself.
    influence = torch.zeros(num_clusters)
    influence.index_add_(0, pairs[0], weight * inverse_sqrt[pairs[1]])
    influence.index_add_(0, pairs[1], weight * inverse_sqrt[pairs[0]])

    left, right = candidates[0], candidates[1]

    # Merging two *adjacent* nodes turns the edge between them into an
    # internal one, so it leaves both degrees, both neighbour sums and both
    # influences.  Most candidates are graph edges, so skipping this is not a
    # rounding detail — it reorders the costs rather than perturbing them.
    shared = _candidate_edge_weight(candidates, pairs, weight, num_clusters)

    merged_size = size[left] + size[right]
    merged_degree = degree[left] + degree[right] - 2.0 * shared
    merged_inverse_sqrt = merged_degree.clamp_min(1e-12).rsqrt()
    merged_features = (
        size[left].unsqueeze(1) * features[left]
        + size[right].unsqueeze(1) * features[right]
    ) / merged_size.unsqueeze(1)
    merged_scaled = merged_features * merged_inverse_sqrt.unsqueeze(1)

    # Neighbour sums are additive once the shared edge is removed; the super
    # node's own self term is then rebuilt from its merged feature, so the
    # representation is exact rather than approximated.
    merged_neighbours = (
        aggregate[left]
        - size[left].unsqueeze(1) * scaled[left]
        + aggregate[right]
        - size[right].unsqueeze(1) * scaled[right]
        - shared.unsqueeze(1) * (scaled[left] + scaled[right])
    )
    merged_representation = (
        merged_neighbours + merged_size.unsqueeze(1) * merged_scaled
    ) * merged_inverse_sqrt.unsqueeze(1)

    return (
        (representation[left] - merged_representation).abs().sum(1)
        + (representation[right] - merged_representation).abs().sum(1)
        + (merged_scaled - scaled[left]).abs().sum(1)
        * (influence[left] - shared * inverse_sqrt[right])
        + (merged_scaled - scaled[right]).abs().sum(1)
        * (influence[right] - shared * inverse_sqrt[left])
    )


def spectral_coarsening(
    data: Data,
    reduction_ratio: float = 0.5,
    variant: str = "local_variation",
    num_eigenvectors: int = 4,
    max_spectral_nodes: int = 5_000,
) -> torch.Tensor:
    """Cluster nodes by spectral local variation or heavy-edge matching.

    The generic, logic-blind control: it contracts the edges whose contraction
    disturbs the graph's *spectrum* least, knowing nothing about gates, levels
    or polarity.  This is the coarsening counterpart of random edge dropout in
    the sparsification family — principled, and principled about the wrong
    thing.

    ``variant="local_variation"`` scores an edge by Loukas' local variation,
    the movement it forces on the subspace spanned by the *num_eigenvectors*
    smallest non-trivial Laplacian eigenvectors, normalised by the volume of
    the pair.  Because that costs an eigendecomposition per graph, graphs above
    *max_spectral_nodes* fall back to ``variant="heavy_edge"``, which scores by
    the classic ``w_ij / sqrt(d_i d_j)`` matching weight and needs no
    eigensolver at all.  The same fallback catches an eigensolver that fails to
    converge.  Which path a graph took is not recorded, so the cap should be
    read as part of the method's definition rather than as an implementation
    detail.

    As for ConvMatch, *reduction_ratio* is a target rather than a guarantee:
    a graph with too few contractible edges stops short of it.
    """
    if variant not in ("local_variation", "heavy_edge"):
        raise ValueError(
            f"variant must be 'local_variation' or 'heavy_edge', got {variant!r}"
        )
    if not 0.0 <= reduction_ratio < 1.0:
        raise ValueError(f"reduction_ratio must be in [0, 1), got {reduction_ratio}")

    num_nodes = data.x.size(0)
    target = max(1, int(round(num_nodes * (1.0 - reduction_ratio))))
    pairs, weight = _undirected_simple(data.edge_index, num_nodes)
    if target >= num_nodes or pairs.size(1) == 0:
        return torch.arange(num_nodes, dtype=torch.long)

    subspace = None
    if variant == "local_variation" and num_nodes <= max_spectral_nodes:
        subspace = _laplacian_subspace(pairs, weight, num_nodes, num_eigenvectors)

    label = torch.arange(num_nodes, dtype=torch.long)
    size = torch.ones(num_nodes)
    num_clusters = num_nodes

    while num_clusters > target and pairs.size(1):
        cost = _spectral_edge_costs(pairs, weight, subspace, num_clusters)
        matched = _greedy_disjoint_pairs(
            pairs, cost, num_clusters, num_clusters - target
        )
        if matched.size(1) == 0:
            break

        label, cluster_map, num_clusters = _apply_pairs(label, matched, num_clusters)
        pairs, weight = _undirected_simple(cluster_map[pairs], num_clusters, weight)
        new_size = _pool_sum(size, cluster_map, num_clusters)
        if subspace is not None:
            subspace = _pool_sum(
                subspace * size.unsqueeze(1), cluster_map, num_clusters
            ) / new_size.unsqueeze(1)
        size = new_size

    return label


def _spectral_edge_costs(
    pairs: torch.Tensor,
    weight: torch.Tensor,
    subspace: torch.Tensor | None,
    num_clusters: int,
) -> torch.Tensor:
    """Score each edge for contraction; lower is better.

    With a *subspace*, this is Loukas' edge-family local variation.  For a
    two-node contraction set his general form ``||B' L_e B||_F`` collapses to
    the movement the contraction forces on the preserved subspace scaled by
    the volume of the pair — the edge weight cancels out of ``L_e``, and the
    degrees **multiply**.  Without one it is his heavy-edge proximity,
    ``w_ij`` over the heaviest edge incident to either endpoint, negated so
    that lower stays better.
    """
    degree = torch.zeros(num_clusters).index_add_(
        0, pairs.reshape(-1), weight.repeat(2)
    )
    if subspace is None:
        heaviest = torch.zeros(num_clusters).scatter_reduce_(
            0, pairs.reshape(-1), weight.repeat(2), reduce="amax", include_self=False
        )
        return -weight / torch.maximum(
            heaviest[pairs[0]], heaviest[pairs[1]]
        ).clamp_min(1e-12)

    movement = (subspace[pairs[0]] - subspace[pairs[1]]).pow(2).sum(1)
    return (degree[pairs[0]] + degree[pairs[1]]) * movement


def _laplacian_subspace(
    pairs: torch.Tensor,
    weight: torch.Tensor,
    num_nodes: int,
    num_eigenvectors: int,
) -> torch.Tensor | None:
    """Loukas' preserved subspace, or None if the eigensolver is unavailable.

    The subspace is the ``num_eigenvectors`` lowest Laplacian eigenvectors
    scaled by the inverse square root of their eigenvalues, as the reference
    implementation builds it — an eigenvector is worth preserving in inverse
    proportion to its frequency, and the flat nullspace directions are given
    unit weight rather than infinite.

    Returning None is not an error path: it is how a graph too small or too
    awkward for the eigensolver hands the caller back to heavy-edge scoring.
    """
    wanted = min(num_eigenvectors, num_nodes - 2)
    if wanted < 1:
        return None

    import numpy as _np
    from scipy.sparse import coo_matrix, diags, eye
    from scipy.sparse.linalg import eigsh

    rows = torch.cat([pairs[0], pairs[1]]).numpy()
    cols = torch.cat([pairs[1], pairs[0]]).numpy()
    values = weight.double().repeat(2).numpy()
    adjacency = coo_matrix(
        (values, (rows, cols)), shape=(num_nodes, num_nodes)
    ).tocsr()
    degrees = _np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = diags(degrees) - adjacency

    try:
        # Reflecting the spectrum turns the smallest eigenvalues into the
        # largest, which converges without the sparse factorization that
        # shift-invert needs and that "SM" on a singular Laplacian does not
        # reach reliably.  A fixed v0 keeps the result reproducible.
        offset = 2.0 * float(degrees.max()) if degrees.size else 1.0
        reflected = offset * eye(num_nodes, format="csc") - laplacian
        eigenvalues, eigenvectors = eigsh(
            reflected, k=wanted + 1, which="LM", tol=1e-5, v0=_np.ones(num_nodes)
        )
        eigenvalues = offset - eigenvalues
    except Exception:
        return None

    order = _np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    # A flat direction carries no frequency information to trade off, so it
    # is weighted 1 instead of blowing up as lambda^-1/2.
    eigenvalues[eigenvalues < 1e-10] = 1.0
    subspace = eigenvectors[:, order] * (eigenvalues ** -0.5)
    return torch.from_numpy(_np.ascontiguousarray(subspace)).float()


def lsh_coarsening(
    data: Data,
    bin_width: float = 2.0,
    num_projections: int = 8,
    seed: int = 42,
) -> torch.Tensor:
    """Cluster nodes by locality-sensitive hashing (UGC-style).

    The cheap tier: one pass to build a per-node descriptor, one matrix
    multiply to hash it, and nodes landing in the same bucket merge.  No
    iteration, no eigensolver, no neighbour search — linear in the size of the
    graph, which makes it the method most obviously affordable across millions
    of AIGs, and the naive control the principled methods have to beat.

    The descriptor is the light AIG adaptation named in the design notes: node
    type, level, the fan-in and fan-out counts split by inverter polarity, and
    the type census of the immediate fanins and fanouts.  Columns are
    standardised per graph, so *bin_width* is in standard deviations rather
    than raw units.  Node type is an exact part of the bucket key rather than
    a hashed feature, so primary inputs and outputs never dissolve into AND
    super-nodes.

    Hashing is p-stable (Euclidean) LSH.  Offsets are drawn independently of
    *bin_width*, which makes doubling it produce a coarser partition — a true
    refinement of the previous one — rather than merely a differently-shaped
    one: compression responds to the knob monotonically, not just on average.

    *bin_width* fixes the bucket width, **not** the compression.  The number
    of occupied buckets saturates while the node count keeps growing, so
    retention falls sharply with graph size: at the settings in ``config`` it
    measures ~0.83 on an 88-node graph and ~0.001 on a 366k-node one.  That is
    inherent to hashing rather than a defect, but it means S5 cannot be
    compared against the ratio-driven methods at a matched compression point
    without calibrating *bin_width* per graph first.
    """
    if bin_width <= 0:
        raise ValueError(f"bin_width must be > 0, got {bin_width}")
    if num_projections < 1:
        raise ValueError(f"num_projections must be >= 1, got {num_projections}")

    num_nodes = data.x.size(0)
    features = _hash_descriptor(data)
    features = (features - features.mean(dim=0)) / features.std(
        dim=0, unbiased=False
    ).clamp_min(1e-6)

    generator = torch.Generator().manual_seed(seed)
    projections = torch.randn(
        features.size(1), num_projections, generator=generator, dtype=torch.float64
    )
    offsets = torch.rand(num_projections, generator=generator, dtype=torch.float64)
    codes = torch.floor((features.double() @ projections + offsets) / bin_width).long()

    types = data.x[:, :4].argmax(dim=1).reshape(num_nodes, 1)
    return _relabel_rows(torch.cat([types, codes], dim=1))


def _hash_descriptor(data: Data) -> torch.Tensor:
    """Per-node feature + connectivity descriptor hashed by ``lsh_coarsening``."""
    num_nodes = data.x.size(0)
    source, destination = data.edge_index
    polarity = _edge_polarity(data)

    fanin_polarity = torch.zeros(num_nodes, 2).index_add_(0, destination, polarity)
    fanout_polarity = torch.zeros(num_nodes, 2).index_add_(0, source, polarity)
    fanin_types = torch.zeros(num_nodes, data.x.size(1)).index_add_(
        0, destination, data.x[source]
    )
    fanout_types = torch.zeros(num_nodes, data.x.size(1)).index_add_(
        0, source, data.x[destination]
    )

    return torch.cat(
        [
            data.x,
            _node_levels(data).to(torch.float32).reshape(-1, 1),
            fanin_polarity,
            fanout_polarity,
            fanin_types,
            fanout_types,
        ],
        dim=1,
    )


# A method maps a graph to a cluster vector ``LongTensor[num_nodes]``
# assigning each node a super-node id in ``[0, num_clusters)``.  Adding a
# summarization method means adding an entry here; nothing downstream of
# apply_merge_map changes.
SUMMARIZATION_REGISTRY: dict[str, Callable[..., torch.Tensor]] = {
    "identity": identity_clustering,
    "cone": cone_coarsening,
    "wl": color_refinement,
    "convmatch": convmatch_coarsening,
    "spectral": spectral_coarsening,
    "lsh": lsh_coarsening,
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
    "cone_coarsening",
    "convmatch_coarsening",
    "identity_clustering",
    "lsh_coarsening",
    "spectral_coarsening",
    "summarize_graph",
]
