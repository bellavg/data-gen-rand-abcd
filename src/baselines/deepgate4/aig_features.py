"""Turn this project's AIGs into the graph format DeepGate4's model code expects.

This module replaces upstream's `src/dg_datasets/data_preparation.py`. That
file cannot be reused, for two independent reasons:

1. **It computes pretraining labels we do not need.** Most of it produces
   supervision for DeepGate4's self-supervised objectives -- logic-1
   probabilities and truth-table similarity from `prepare_dg2_labels_cpp`
   (which shells out to a C++ simulator that has to be built separately),
   connectivity pairs, hop-level truth tables, graph edit distances. A
   supervised regression baseline uses none of it. Notably `prob` is not
   needed even for the encoder: `DeepGate2.forward` accepts `PI_prob` and then
   never reads it (see dg2.py's docstring), so dropping simulation costs the
   encoder nothing.

2. **Its structural preprocessing does not run at this project's scale.**
   `get_area` + `get_fanin_fanout_cone` (data_preparation.py:95 and :168)
   build, for every cone, a dense `[max_no_nodes, max_no_nodes]` reachability
   matrix inside a double Python loop with `in`-tests against per-node PI/PO
   cover *lists*. Upstream runs this over the ITC99/EPFL benchmarks -- tens of
   circuits. This project has ~788k graphs averaging ~40k nodes. The
   difference is not "slow", it is several orders of magnitude past feasible.
   `virtual_edges()` below computes the same relation with sparse boolean
   matrix powers instead.

WHAT IS FAITHFUL AND WHAT IS NOT
--------------------------------
Faithful: the virtual edge set. Paper Section 3.5 defines it as
`Ē = {(u, v) : u ≼_k v, u ∈ cone_i}`, where `u ≼_k v` means "there is a path
from u to v of length at most k". Because the cones of Algorithm 1 tile the
circuit, the union of the per-cone edge sets is (up to cone-boundary effects,
which only ever *drop* pairs) exactly `{(u, v) : dist(u, v) ≤ k}` over the
whole graph. `virtual_edges()` computes that global set directly, which is
why skipping the partitioning does not change what the sparse transformer
attends over. See regressor.py's docstring for the partitioning question
itself.

Also faithful: the edge DIRECTION. Virtual edges run one way, ancestor ->
descendant, which is what BOTH the paper and the released code do. Worth
spelling out, because `get_fanin_fanout_cone` looks symmetric at a glance and
is not:

    fanin_fanout_cones[i][j] = 1   # j is in i's FANIN cone   (line 222)
    fanin_fanout_cones[i][j] = 2   # j is in i's FANOUT cone  (line 233)

and the consumer keeps only the 1s (data_preparation.py:523):

    ff_cone = ff_cone + torch.eye(n).int()      # diagonal 0 -> 1
    global_virtual_edge = torch.argwhere(ff_cone.T == 1)

The fanout entries are marked 2 and are therefore structurally excluded; the
`+ eye` only ever touches the diagonal. So the released edge set is
ancestor->descendant plus self-loops, i.e. exactly Section 3.5's one-way
`Ē = {(u, v) : u ≼_k v}`. `symmetric=False` (the default) reproduces that.

`symmetric=True` remains available but is a genuine deviation from both paper
and code, and it doubles the edge count -- which is this baseline's dominant
memory term. Do not enable it casually.

Deviation, forced: NOT-node expansion, see `expand_not_nodes()`.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch_geometric.data import Data

# Upstream's gate vocabulary (data_preparation.py:20, minus DFF -- this
# project's AIGs are combinational, and upstream rewrites DFFs to PIs anyway).
GATE_PI = 0
GATE_AND = 1
GATE_NOT = 2

# Width of dg4.py's `out_and`/`out_not` embedding tables (`nn.Embedding(5000,
# hidden)`, dg4.py:174-175). Out-degrees are clamped to the last row rather
# than allowed to index out of bounds -- see `node_out_degrees()`.
OUT_DEGREE_TABLE_SIZE = 5000

# Paper Section 4.1: "In Algorithm 1, we set k to 8 and delta to 6." k is the
# cone depth, and therefore the virtual-edge radius of Section 3.5. delta is
# the partition stride, which has no meaning without partitioning.
DEFAULT_NUM_HOPS = 8


class DeepGateData(Data):
    """`Data` subclass that batches DeepGate4's extra attributes correctly.

    Mirrors the parts of upstream's `AreaData.__inc__`/`__cat_dim__`
    (data_preparation.py:48-92) that apply to the attributes this port actually
    carries. Without it, `Batch.from_data_list` would concatenate
    `global_virtual_edge` along the wrong axis and would not offset `nodes`
    into the batch's global node numbering.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("global_virtual_edge", "nodes", "forward_index"):
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "global_virtual_edge":
            return 1
        if key in ("nodes", "forward_index"):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


def check_topological(
    edge_index: torch.Tensor, level: np.ndarray, num_nodes: int
) -> None:
    """Fail loudly if `level` is not a topological stratification of `edge_index`.

    Used by `to_deepgate_graph` as a POSTcondition on the level it computed for
    the expanded graph, which is the property `DeepGate2.forward` depends on --
    it walks levels in order and assumes a node's fanins were finalised at a
    strictly lower level. One vectorised comparison per graph, negligible
    against the rest of the conversion, and it turns a silent wrong-answer into
    an immediate error.

    Deliberately NOT applied to the cached input level: that one legitimately
    fails this test for dangling logic, because `DepthAig.level()` reports 0
    outside every primary output's fanin cone. `forward_levels` handles that
    case rather than rejecting it.
    """
    if edge_index.numel() == 0:
        return
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    if level.shape[0] != num_nodes:
        raise ValueError(
            f"level has {level.shape[0]} entries for a {num_nodes}-node graph"
        )
    bad = level[src] >= level[dst]
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"per-node level is not a topological stratification: edge "
            f"{int(src[i])} -> {int(dst[i])} has level "
            f"{int(level[src[i]])} >= {int(level[dst[i]])} "
            f"({int(bad.sum())} of {bad.shape[0]} edges violate this). "
            "The level must be the raw integer logic level; a rescaled one "
            "(e.g. log1p, as stored in pos_enc) will trigger this."
        )


def inverted_mask(edge_attr: torch.Tensor | None, num_edges: int) -> np.ndarray:
    """Boolean mask of inverted edges from this project's `[regular, inverted]`
    edge attribute. Absent `edge_attr` means no inversions."""
    if edge_attr is None or edge_attr.numel() == 0:
        return np.zeros(num_edges, dtype=bool)
    return edge_attr[:, 1].numpy() > 0.5


def expand_not_nodes(
    x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Rewrite edge-attribute inversions as the explicit NOT nodes DeepGate uses.

    This project stores inversion on the edge (`edge_attr = [regular,
    inverted]`, data/data_utils.py). DeepGate stores it as a node: its AIGs
    have three node types and it identifies them by in-degree (paper Section
    3.1, "the in-degree of a NOT gate is 1"). The tokenizer is built around
    that -- `DeepGate2` keeps *separate* aggregators and GRUs for AND and NOT
    (`aggr_and_*`/`aggr_not_*`, `update_and_*`/`update_not_*`), and
    `get_slices` dispatches on `gate == 1` vs `gate == 2`. Feed it a graph with
    no `gate == 2` nodes and half the tokenizer never fires: the NOT pathway
    receives no gradient and inversion becomes invisible to the model, since
    nothing else in DeepGate4 reads `edge_attr`. Expansion is therefore not a
    stylistic choice, it is what makes the vendored model well-defined.

    Follows upstream's `aiger_utils.aig_to_xdata` (aiger_utils.py:149) exactly,
    including its sharing rule: **one NOT node per inverted source**, memoised
    in upstream through `has_not[fanin_index]`, reused by every consumer of
    that inverted signal rather than created per edge.

    Node type mapping for the original nodes, from this project's 4-way
    one-hot `[constant, PI, AND, PO]`:

    - constant -> `GATE_PI`. DeepGate has no constant type; a constant has
      in-degree 0, which is exactly its PI test.
    - PI -> `GATE_PI`.
    - AND -> `GATE_AND`.
    - PO -> `GATE_NOT`. These are synthetic single-fanin sink nodes this
      project adds (`_extract_topology` in data/data_utils.py); upstream has no
      PO node at all, it just marks the driving gate. `GATE_NOT` is the
      in-degree-1 type, so a PO is routed through the same single-input update
      path, which is the closest well-defined behaviour -- DeepGate's vocabulary
      has no identity/buffer type to map a PO onto. The alternative --
      dropping PO nodes -- would change the node population that
      `global_mean_pool` averages over and so change the regression target's
      meaning, which is worse.

      Consequence worth knowing: an INVERTED primary output ends up as a
      two-hop `driver -> NOT -> PO` chain, both hops typed `GATE_NOT`, because
      the driver->PO edge is an ordinary edge and gets the generic inversion
      expansion below on top of the PO node itself. Upstream instead emits a
      single terminal NOT node for an inverted PO and no PO node at all
      (aiger_utils.py:206-220), so its NOT update runs once where ours runs
      twice. Logically the first hop is the real inversion and the second is
      the PO marker; structurally both have in-degree 1 and levels still
      stratify, so nothing breaks -- but it is a second, smaller distortion
      stacked on the PO->NOT mapping, and it is deliberate rather than
      overlooked.

    Returns:
        `(gate, expanded_edge_index, num_nodes, not_of)`. `gate` is
        `[num_nodes]` int64 over `{GATE_PI, GATE_AND, GATE_NOT}`;
        `expanded_edge_index` is `[2, num_expanded_edges]` int64; `not_of` is
        `[num_base]` int64 mapping each original node to its shared NOT node,
        or -1 if it never drives an inverted edge. All numpy. Original nodes
        keep their indices; NOT nodes are appended.
    """
    num_base = int(x.shape[0])
    src = edge_index[0].numpy().astype(np.int64)
    dst = edge_index[1].numpy().astype(np.int64)

    base_type = x.argmax(dim=1).numpy()
    gate_base = np.empty(num_base, dtype=np.int64)
    gate_base[base_type == 0] = GATE_PI  # constant
    gate_base[base_type == 1] = GATE_PI  # PI
    gate_base[base_type == 2] = GATE_AND
    gate_base[base_type == 3] = GATE_NOT  # synthetic PO, in-degree 1

    inverted = inverted_mask(edge_attr, src.shape[0])
    inv_sources = np.unique(src[inverted])
    num_not = int(inv_sources.shape[0])
    num_nodes = num_base + num_not

    gate = np.concatenate(
        [gate_base, np.full(num_not, GATE_NOT, dtype=np.int64)]
    )

    # source node id -> its shared NOT node id (upstream's `has_not` table).
    not_of = np.full(num_base, -1, dtype=np.int64)
    not_of[inv_sources] = np.arange(num_base, num_base + num_not, dtype=np.int64)

    if num_not == 0:
        return gate, np.stack([src, dst]), num_nodes, not_of

    # Inverted edge (u, v) becomes (not_u, v); the (u, not_u) edges are added
    # once per distinct inverted source, matching the sharing rule above.
    new_src = np.where(inverted, not_of[src], src)
    feed_src = inv_sources
    feed_dst = not_of[inv_sources]

    expanded = np.stack(
        [
            np.concatenate([new_src, feed_src]),
            np.concatenate([dst, feed_dst]),
        ]
    )
    return gate, expanded, num_nodes, not_of


def _frontier_edge_slots(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Indices of the out-edges of a whole frontier, without a Python loop.

    Given each frontier node's first out-edge position and out-degree, expand
    the variable-length ranges `[start, start+count)` into one flat index array.
    """
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    offsets = np.cumsum(counts) - counts
    return np.repeat(starts, counts) + (np.arange(total) - np.repeat(offsets, counts))


def forward_levels(
    edge_index: torch.Tensor,
    inverted: np.ndarray,
    not_of: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    """Longest-path logic level on the expanded graph (paper Eq. 1).

    `level(v) = 0` for a PI, else `1 + max level(u)` over fanins u.

    Levels must be recomputed rather than reused from this project's cached
    `pos_enc`: inserting a NOT node lengthens every inverted path by one, so
    the cached levels are no longer a valid topological stratification of the
    expanded graph, and `DeepGate2.forward` relies on one -- it walks levels in
    order and assumes a node's fanins were finalised at a strictly lower level.

    Works on the ORIGINAL edge list rather than the expanded one, which keeps
    the recurrence one-dimensional: an inverted edge `u -> v` becomes the
    two-hop path `u -> not_u -> v`, so it simply carries weight 2 instead of 1,
    giving

        level(v) = max over original fanins u of (level(u) + 1 + inverted(u, v))

    over original nodes, after which every NOT node takes `level(u) + 1` from
    its single fanin `u`.

    Computed by level-synchronous Kahn's algorithm, which needs NO precomputed
    level as input. That is deliberate: this project's cached level is not a
    reliable topological order. It comes from `aigverse`'s `DepthAig.level()`
    (data/data_utils.py), which returns **0 for every node outside the fanin
    cone of some primary output** -- i.e. for dangling logic. A dangling AND
    gate and its dangling fanins then all report level 0, so `level[src] <
    level[dst]` fails on those edges. Verified directly against aigverse 0.1.1.

    Using it merely as a scheduling hint and relaxing to a fixpoint also works
    but degrades badly: each extra pass advances the mis-hinted frontier by
    only one node, measured at ~15 s for a single graph with a 500-deep
    dangling chain. Kahn's does O(E) total work regardless -- a node's
    out-edges are relaxed exactly once, when its last in-edge is consumed, so
    every source embedding is final at the moment it is read.

    Works on the ORIGINAL edge list rather than the expanded one, which keeps
    the recurrence one-dimensional: an inverted edge `u -> v` becomes the
    two-hop path `u -> not_u -> v`, so it simply carries weight 2 instead of 1,
    giving

        level(v) = max over original fanins u of (level(u) + 1 + inverted(u, v))

    over original nodes, after which every NOT node takes `level(u) + 1` from
    its single fanin `u`.
    """
    src = edge_index[0].numpy().astype(np.int64)
    dst = edge_index[1].numpy().astype(np.int64)

    level = np.zeros(num_nodes, dtype=np.int64)
    if src.shape[0] == 0:
        return level

    weight = 1 + inverted.astype(np.int64)

    # Group out-edges by source so a frontier's edges are contiguous slices.
    order = np.argsort(src, kind="stable")
    src_s, dst_s, w_s = src[order], dst[order], weight[order]
    out_degree = np.bincount(src, minlength=num_nodes)
    starts = np.concatenate([[0], np.cumsum(out_degree)])[:num_nodes]

    remaining_in = np.bincount(dst, minlength=num_nodes)
    frontier = np.flatnonzero(remaining_in == 0)
    settled = int(frontier.size)

    while frontier.size:
        slots = _frontier_edge_slots(starts[frontier], out_degree[frontier])
        if slots.size == 0:
            break
        s, d, w = src_s[slots], dst_s[slots], w_s[slots]
        # Every s here is settled, so its level is final.
        np.maximum.at(level, d, level[s] + w)
        np.subtract.at(remaining_in, d, 1)
        frontier = np.unique(d[remaining_in[d] == 0])
        settled += int(frontier.size)

    if settled < num_nodes and np.any(remaining_in > 0):
        raise ValueError(
            "forward_levels found a cycle: the AIG edge list is not acyclic "
            f"({int((remaining_in > 0).sum())} nodes never reached in-degree 0)"
        )

    has_not = not_of >= 0
    if has_not.any():
        level[not_of[has_not]] = level[np.flatnonzero(has_not)] + 1

    return level


def node_out_degrees(
    gate: np.ndarray, edge_index: np.ndarray, num_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """`OutAND(v)` and `OutNOT(v)` for the structural encoding (paper Eq. 2).

    Counts, among a node's immediate successors, how many are AND gates and how
    many are NOT gates. Same quantity as upstream (data_preparation.py:465-475),
    computed with `np.bincount` over the edge list instead of upstream's
    `for i in range(num_nodes)` loop with a full `edge_index[0] == i` scan per
    node -- O(E) rather than O(n * E), identical result.

    Counts are clamped to `OUT_DEGREE_TABLE_SIZE - 1`. Upstream's embedding
    tables have 5000 rows, which its benchmark circuits never exhaust; a
    366k-gate AIG here can carry a higher-fanout net, and an unclamped lookup
    would be an out-of-bounds index rather than a graceful degradation.
    """
    src, dst = edge_index[0], edge_index[1]
    succ_gate = gate[dst]
    out_and = np.bincount(
        src, weights=(succ_gate == GATE_AND), minlength=num_nodes
    ).astype(np.int64)
    out_not = np.bincount(
        src, weights=(succ_gate == GATE_NOT), minlength=num_nodes
    ).astype(np.int64)
    cap = OUT_DEGREE_TABLE_SIZE - 1
    return np.clip(out_and, 0, cap), np.clip(out_not, 0, cap)


def virtual_edges(
    edge_index: np.ndarray,
    num_nodes: int,
    num_hops: int = DEFAULT_NUM_HOPS,
    symmetric: bool = False,
) -> torch.Tensor:
    """`Ē = {(u, v) : u ≼_k v}` -- paper Section 3.5's virtual edges.

    Computed as the union of boolean sparse matrix powers `A^1 .. A^k`, where
    `A[u, v] = 1` for a real edge `u -> v`. `A^j` is exactly "there is a walk of
    length j from u to v", so the union is the `≼_k` relation.

    Self-loops are NOT emitted, although upstream adds them (`ff_cone +
    torch.eye(...)`, data_preparation.py:522). They would be redundant:
    `GATConv` defaults to `add_self_loops=True`, and its implementation calls
    `remove_self_loops` before adding, so an explicitly supplied self-loop and
    an absent one produce the identical edge set inside the layer.

    Cost note: this is the dominant memory term of the whole baseline. The
    result has one edge per ancestor-descendant pair within k hops, so it is
    far denser than the circuit itself -- see train_baseline.py's docstring for
    the measured numbers and what they imply for the node budget.
    """
    src, dst = edge_index[0], edge_index[1]
    if src.shape[0] == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    adj = csr_matrix(
        (np.ones(src.shape[0], dtype=bool), (src, dst)),
        shape=(num_nodes, num_nodes),
        dtype=bool,
    )
    reach = adj.copy()
    power = adj
    for _ in range(max(0, num_hops - 1)):
        power = power @ adj
        power.eliminate_zeros()
        if power.nnz == 0:
            break
        reach = reach + power

    coo = reach.tocoo()
    u = coo.row.astype(np.int64)
    v = coo.col.astype(np.int64)
    if symmetric:
        stacked = np.stack([np.concatenate([u, v]), np.concatenate([v, u])])
    else:
        stacked = np.stack([u, v])
    return torch.from_numpy(stacked)


def to_deepgate_graph(
    data: Data, num_hops: int = DEFAULT_NUM_HOPS, symmetric: bool = False
) -> DeepGateData:
    """Convert one of this project's AIG `Data` objects into DeepGate4 form.

    Reads `x`, `edge_index`, `edge_attr` and `y` -- and deliberately NOT
    `pos_enc`. Levels are recomputed from the edge list by `forward_levels`
    rather than taken from the cache, for two independent reasons: the cached
    value is `log1p(level)` rather than the level (`get_pe_transform('level')`
    builds `ExtractPrecomputedPE(discrete=False)`, whose branch applies
    `log1p_()`), and even un-scaled it is not a topological order on circuits
    with dangling logic. Do not "optimise" this by reading `pos_enc`.
    Produces the attribute set
    `dg2.DeepGate2.forward` and `plain_tf_linear.Sparse_Transformer.forward`
    read: `gate`, `edge_index`, `forward_level`, `forward_index`, `out_and`,
    `out_not`, `global_virtual_edge`, `nodes`.

    `forward_index` is `arange(num_nodes)`, which is what upstream produces too
    rather than an approximation of it: `deepgate.utils.dag_utils`'
    `return_order_info` sets `forward_index = torch.LongTensor(range(num_nodes))`,
    and `data_preparation.py:254` likewise builds it as `torch.tensor(range(...))`
    per cone. It is an identity, not a level-ordered permutation.

    `nodes` is `arange(num_nodes)`, and `DeepGate4GraphRegressor` pairs it with
    an all-zero `mk`. Upstream uses those two to skip gates whose embedding was
    already computed in an earlier cone; with no partitioning nothing is ever
    pre-computed, so every gate reads as "needs updating" and the three
    `mk[...]` filters in the vendored files become no-ops. That is what lets
    dg2.py and plain_tf_linear.py stay byte-identical to upstream.
    """
    edge_attr = getattr(data, "edge_attr", None)
    gate, expanded_edge_index, num_nodes, not_of = expand_not_nodes(
        data.x, data.edge_index, edge_attr
    )

    level = forward_levels(
        data.edge_index,
        inverted_mask(edge_attr, int(data.edge_index.shape[1])),
        not_of,
        num_nodes,
    )
    # Postcondition, not a precondition. The contract `DeepGate2.forward`
    # actually relies on is that the EXPANDED graph's level strictly increases
    # along every edge; checking the computed result asserts exactly that,
    # whereas checking the cached input level would reject the legitimate
    # dangling-logic case that forward_levels now handles. One vectorised
    # comparison per graph.
    check_topological(
        torch.from_numpy(expanded_edge_index), level, num_nodes
    )
    out_and, out_not = node_out_degrees(gate, expanded_edge_index, num_nodes)

    out = DeepGateData(
        edge_index=torch.from_numpy(expanded_edge_index),
        gate=torch.from_numpy(gate).unsqueeze(1),
        forward_level=torch.from_numpy(level),
        forward_index=torch.arange(num_nodes, dtype=torch.long),
        nodes=torch.arange(num_nodes, dtype=torch.long),
        out_and=torch.from_numpy(out_and),
        out_not=torch.from_numpy(out_not),
        global_virtual_edge=virtual_edges(
            expanded_edge_index, num_nodes, num_hops, symmetric
        ),
        y=data.y,
    )
    out.num_nodes = num_nodes
    return out


class DeepGateGraphAdapter(torch.utils.data.Dataset):
    """Index-aligned wrapper converting an AIG split to DeepGate4 form on access.

    Mirrors `baselines/hoga/hop_features.HopFeatureCache`, minus the optional
    on-disk cache. HOGA's class documents why caching its hop features is not
    size-viable here (~3.1 TB across ~788k files, against an already-full
    scratch quota); the same arithmetic rules it out more decisively for this
    baseline, whose per-graph payload is the virtual edge list -- denser than
    HOGA's `[num_nodes, 6, 4]` tensor by a wide margin. Conversion therefore
    happens in the dataloader workers, overlapped with GPU compute.

    Index alignment with the wrapped dataset is relied on by
    `train_baseline.py`, which builds a node-budget batch plan from the
    *wrapped* dataset's node counts and uses it to index this one.
    """

    def __init__(
        self,
        base_dataset,
        num_hops: int = DEFAULT_NUM_HOPS,
        symmetric: bool = False,
    ) -> None:
        self.base_dataset = base_dataset
        self.num_hops = num_hops
        self.symmetric = symmetric

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> DeepGateData:
        return to_deepgate_graph(
            self.base_dataset[idx], self.num_hops, self.symmetric
        )


def collate_deepgate_batch(data_list):
    """Collate `DeepGateData` samples, preserving the subclass's batching rules."""
    from torch_geometric.data import Batch

    return Batch.from_data_list(list(data_list))
