"""Graph-level optimizability regressor built on Gamora's GraphSAGE trunk.

Adapts Yu-Maryland/Gamora's `SAGE_MULT` (see model.py in this directory) from
its released task -- per-node classification of functional blocks (xor / maj /
adder-root) in AIGs -- to this project's task: a single scalar optimizability
prediction per whole AIG.

WHAT THIS BASELINE ACTUALLY MEASURES -- READ BEFORE PUTTING IT IN A TABLE
------------------------------------------------------------------------
Gamora is not an architecture paper. Its contribution is a *formulation*: that
symbolic reasoning over Boolean networks can be posed as multi-task per-node
classification, with the three task heads sharing one encoder, plus a
post-processing step (`gnn_multitask.py:199-255`) that reconciles the xor and
maj predictions into adder boundaries. The encoder underneath is stock
GraphSAGE (Hamilton et al.), which the paper says outright: "the specific
model employed is GraphSAGE [11]" (Section III.B.1), followed by Equation (1),
which is GraphSAGE's own update rule verbatim.

Removing the three classification heads -- which a graph-level regression task
has no use for -- therefore removes the multi-task formulation and, with it,
the post-processing that depends on the heads agreeing. What is left and what
this file trains is a 4-layer GraphSAGE encoder with Gamora's node
featurisation, its shared `Linear -> ReLU -> BatchNorm` neck, and its published
hyperparameters.

Be precise about what survives, in both directions. The paper's Figure 4
ablates TWO ingredients separately: the multi-task setting, and the functional
node features ("there is always a boost of accuracy when employing functional
information"). This port keeps the second and discards the first. So it is not
true that nothing of Gamora is left -- the featurisation below is theirs and is
a claimed contribution ("These compressed node features not only encapsulate
Boolean functionality of each node but also enable high compute and memory
efficiency", Section III.B.1). It is true that the formulation the paper is
named for is gone.

So: this row measures GAMORA'S ENCODER ADAPTED TO GRAPH-LEVEL REGRESSION. It
does not measure Gamora's published task, and its number cannot be compared to
any number in the DAC'23 paper (those are per-node classification accuracies on
CSA/Booth multipliers, trained on an 8-bit multiplier and evaluated on larger
ones). Label the row accordingly in the thesis, or a reader will take it as a
claim about Gamora's results. Calling it "GraphSAGE (Gamora)" is more honest
than calling it "Gamora", and the multi-task ablation in the paper's own
Figure 4 -- where the single-task setting is "conspicuously" worse -- is direct
evidence from the authors that the encoder alone is not the method.

THE SAMPLING QUESTION
---------------------
This project's baselines must not require neighbour sampling, graph
partitioning, or subgraph decomposition, because the thesis contribution is
graph reduction and a baseline that internally reduces the graph would make the
comparison circular. Gamora passes that bar at the ARCHITECTURE level and fails
it at the RELEASED-TRAINER level, and the distinction matters:

  - The architecture is sampling-free. A GraphSAGE stack is a plain
    message-passing model; nothing in it requires sampled neighbourhoods.
    Upstream wrote the full-graph path themselves --
    `SAGE_MULT.forward_nosampler` (`gnn_multitask.py:86-105`) loops
    `for conv in self.convs: x = conv(x, adj_t)` over the entire adjacency.

  - Upstream's released TRAINING path samples, and there is no alternative in
    the repository. `gnn_multitask.py:570-572` builds
    `NeighborSampler(data.adj_t, node_idx=train_idx, sizes=[8, 5, 5, 5],
    batch_size=20, shuffle=True)`, and `train()` (:141-197) iterates it.
    `forward_nosampler` is called from exactly one place, `test_nosampler`
    (:342-422), i.e. evaluation. There is no `train_nosampler`.

  - THIS PORT TRAINS FULL-GRAPH. That is a deviation from Gamora's published
    training procedure, stated plainly. It is not "an option upstream
    provides": upstream provides the forward computation, not the trainer that
    uses it for optimization.

The deviation is possible at all because this codebase supplies its own
Lightning training loop (train_baseline.py + baselines/common/lightning_wrapper.py),
so upstream's `train()` never enters this repository -- no `NeighborSampler`,
`ClusterLoader`, or subgraph call is imported, vendored, or reachable from any
code path here.

What the deviation changes: upstream's sampled minibatch is 20 root nodes with
[8, 5, 5, 5] fanout per layer, so each gradient step sees a truncated
neighbourhood and each node's representation is computed from a random subset
of its neighbours (sampling is also a regulariser, and its variance is part of
their optimization). Here every node sees all of its neighbours at every step.
For a fixed model this reduces estimator variance rather than changing what is
representable, but it is a different optimization problem and the published
lr=0.008 was tuned under the other one.

INPUT FEATURES -- HOW INVERSION REACHES THIS MODEL
--------------------------------------------------
Gamora specifies its own node features, and they are the reason this baseline
is not blind to inverted edges. Paper Section III.B.1: "For each node, there
are three node features represented in binary values denoting node types and
Boolean functionality. The first node feature indicates whether this node is a
PI/PO or intermediate node (i.e., AND gate). The second and the third node
features indicate whether each input edge is inverted or not, such that AIGs
can be represented as homogeneous graphs without additional edge features."

That last clause is the design decision: Gamora folds edge inversion into the
node features on purpose, so the model needs no `edge_attr` at all. This port
therefore reconstructs those features from this project's graphs
(`gamora_node_features` below) rather than passing `edge_attr`, which SAGEConv
could not consume anyway. This is the paper's own answer to the question, not
an improvisation. Against the sibling ports: HOGA's never passes `edge_attr`
and is genuinely blind to inversions, so this baseline and that one differ on
an input, not only an architecture. SynthNet's is NOT blind -- it feeds
`derive_num_inverted_predecessors` as a node feature
(openabc_synthnet/regressor.py:79) -- so of the three GNN baselines only HOGA
lacks inversion information.

The released code writes FOUR columns, not the paper's three
(`abc/src/proof/acec/acecXor.c:382-421`, function `Gia_edgelist`):
    CI  (primary input) : `0,0,0,0`
    AND                 : `1,1,<FaninC0>,<FaninC1>`
    CO  (primary output): `0,0,1,1`
and `dataset_generator.py:233-234` loads that file straight into `node_feat`,
so `data.num_features == 4` is what `SAGE_MULT` is actually constructed with
(:583). Column 0 and column 1 are identical by construction -- the extra
column is redundant, not informative. This port reproduces the released
4-column encoding rather than the paper's 3-column description, because the
released one is what the published hyperparameters were tuned against.

Three consequences, recorded so nobody mistakes them for porting bugs:
  - A CO's own edge inversion is DISCARDED. Upstream writes the constant
    `0,0,1,1` for every CO even though `Gia_ObjFaninC0` is available at that
    point. In this project's AIGs a PO edge does carry a real inversion bit
    (data/data_utils.py:166), so that bit is not visible to this baseline.
    Faithful to upstream.
  - This project has a `constant` node type that Gia has no CI/CO/AND slot for
    (Gia's const0 is object 0 and `Gia_ManForEachCi` skips it, so it never gets
    a feature row). It is mapped to `0,0,0,0`, the same vector as a PI, which
    is the natural reading: a source node with no fanins.
  - NOT faithful, deliberately: upstream's EDGE list is wrong for COs and this
    port does not reproduce the error. `Gia_edgelist` writes two edges per CO
    (acecXor.c:413-414), the second from `Gia_ObjFaninId1`. A CO has only one
    fanin; for a terminal object `iDiff1` stores the CIO index instead
    (gia.h:468, `Gia_ObjCioId` asserts `fTerm` and returns `iDiff1`), so
    `Gia_ObjFaninId1 = ObjId - CioId` is an unrelated node and upstream emits
    one spurious edge per primary output. This project's graphs give each PO
    exactly one incoming edge, from its real driver. Reproducing the bug would
    mean injecting garbage edges, so it is not reproduced; the divergence is
    noted here because everything else about the featurisation IS bug-for-bug
    faithful and the asymmetry should be visible.

READOUT
-------
Gamora has no readout: it predicts per node, so pooling never arises upstream
and there is nothing to be faithful to. Global MEAN pooling is used here
because that is what this project's primary model uses
(`config.POOLING_TYPE = "mean"`, applied at models/base_model.py:169), so the
baseline and the model it is compared against aggregate identically and any
difference is attributable to the encoder.

Be aware of what mean pooling costs, since it costs the primary model the same
thing: it is invariant to duplicating a graph's connected components, so it
cannot distinguish a graph from two disjoint copies of itself and carries no
information about |V| or |E|. On this dataset that is a real loss -- graph size
is a strong predictor of the target. It is NOT compensated for here. Feeding
size to the head was considered and removed: nothing else in this project
receives it, so it would have given this one row an input no other row has,
and a baseline that wins on an input the primary model never sees is worse
than useless. If size information is ever added, it has to be added to every
model at once, which is a change to models/base_model.py and not to a baseline.

ONE INTERACTION BETWEEN UPSTREAM'S `bn0` AND THIS PORT'S POOLING
----------------------------------------------------------------
`bn0` normalises over every node in the batch, so the batch-wide mean of its
output is exactly its bias. Mean pooling then averages over that SAME node set.
When a batch holds one graph the two reductions are over identical sets, so the
graph embedding is the BatchNorm bias regardless of the circuit, and that step
carries no information from the encoder at all. (Train mode only; eval uses
running statistics.)

Nothing to fix in the configured regime -- at the 3M-node budget a batch holds
~75 graphs and no graph reaches the budget, so singleton batches do not arise.
It matters if the budget is ever lowered below config.MAX_NUM_GATES, because a
graph larger than the budget forms a singleton batch that graph-level pooling
cannot split. `test_single_graph_train_batch_collapses_to_the_bn_bias` pins it
so the consequence surfaces in the suite rather than as a flat training curve.

HYPERPARAMETERS
---------------
All published, from Section IV.A and upstream's argparse defaults -- see the
constants below for which came from which. The paper gives two configurations
(a shallow 4-layer/32-channel model and a deep 8-layer/80-channel one) and this
port defaults to the shallow one, matching upstream's own defaults; the deep
one is reachable with `--gamora_num_layers 8 --gamora_hidden_dim 80`.

WATCH `lr` ON THE FIRST RUN. 0.008 is upstream's, but it was tuned for
`F.nll_loss` over 20-node sampled minibatches on an 8-bit multiplier. Here it
drives a regression loss over ~75 whole graphs per step on a different circuit
distribution, at ~27x the primary model's own LR (config.LR = 0.0003). Keeping
it is the faithful choice and this port keeps it, but a baseline in this repo
has already collapsed to a flat val_loss with R^2 near zero before -- see
diagnose_synthnet_baseline.py, which exists for that reason -- and an
8,193-parameter model is the likeliest of the four to repeat it. If the curve
is flat after one epoch, that is the first thing to check, and lowering the LR
must then be reported as a deviation from the published value rather than
quietly applied.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from baselines.gamora.model import SAGE_MULT

# Upstream argparse defaults (gnn_multitask.py:528-530), which also match the
# paper's shallow configuration: "a shallow 4-layer model with the hidden
# channel of 32 (for CSA multipliers w/ and w/o simple technology mapping)"
# (Section IV.A). The paper's other configuration is 8 layers / 80 channels,
# "for Booth multipliers and after complex technology mapping".
DEFAULT_NUM_LAYERS = 4
DEFAULT_HIDDEN_DIM = 32
DEFAULT_DROPOUT = 0.5

# Upstream training config: `--lr` default 0.008 (gnn_multitask.py:531),
# `weight_decay = 5e-5` and Adam (:595), `--epochs` default 100 (:532). The
# paper states none of these; they are the released code's values.
DEFAULT_LR = 0.008
DEFAULT_WEIGHT_DECAY = 5e-5
DEFAULT_NUM_EPOCHS = 100

# Gamora's released node featurisation is 4-wide (acecXor.c:392-415); see the
# module docstring for the per-node-type encoding and for why it is 4 and not
# the paper's 3.
GAMORA_NODE_FEATURE_DIM = 4


def gamora_node_features(
    x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
) -> torch.Tensor:
    """Rebuild Gamora's 4-column node features from this project's AIG tensors.

    Reproduces `Gia_edgelist`'s feature rows (acecXor.c:392, :399, :415):
        primary input / constant : `[0, 0, 0, 0]`
        AND gate                 : `[1, 1, inv_fanin0, inv_fanin1]`
        primary output           : `[0, 0, 1, 1]`

    Args:
        x: `[N, config.NODE_INPUT_DIM]` one-hot over
            `[constant, primary_input, and_gate, primary_output]`
            (data/data_utils.py:aig_to_pytorch_geometric).
        edge_index: `[2, E]`, row 0 the fanin, row 1 the node it feeds.
        edge_attr: `[E, 2]`, column 1 set to 1.0 for an inverted signal.

    Returns:
        `[N, 4]` float tensor in `x`'s dtype and on `x`'s device.

    ON FANIN ORDER: `(Gia_ObjFaninC0, Gia_ObjFaninC1)` is an ORDERED pair, and
    the order is not arbitrary. `Gia_ManAppendAnd` (abc/src/aig/gia/gia.h:670-682)
    branches on `if (iLit0 < iLit1)` and stores the smaller LITERAL as fanin 0,
    where a literal is `2 * node_id + complement`. So upstream's pair is sorted
    by (fanin node id, then complement flag), and `Gia_ObjIsAndReal` (gia.h:483)
    relies on exactly that invariant.

    This function reproduces that key rather than approximating it. The two
    incoming edges of an AND node are ranked by `2 * source_index + inverted`
    and their inversion bits emitted low-key-first, which is upstream's rule
    applied to this project's node indexing -- and `data/data_utils.py:97-112`
    asserts that indexing is the AIG's own contiguous ascending node ids, so
    "smaller index" means the same thing on both sides. Being a min/max over
    the incoming edges, it is also invariant to `edge_index` column order,
    which nothing in this pipeline promises to preserve.

    The one thing it cannot promise: these AIGs are built by `aigverse`, not by
    ABC's Gia, so whether a given gate's fanins carry the same relative ids in
    both tools is unknowable. The RULE is reproduced exactly; whether the two
    toolchains number a particular gate's fanins identically is not something
    this port can establish.
    """
    num_nodes = x.size(0)
    src, dst = edge_index[0], edge_index[1]

    # Upstream's literal, 2 * node_id + complement. Ranking edges by it and
    # taking the min/max per target node recovers (FaninC0, FaninC1).
    literal = src.to(torch.long) * 2 + edge_attr[:, 1].to(torch.long)
    # amin's identity must exceed every literal (max is 2 * (N - 1) + 1) and be
    # EVEN so an edgeless node reads back complement 0; amax's can be 0,
    # since every literal is non-negative. Both are masked by `is_and` below
    # anyway -- an AND always has fanins -- so these only keep the arithmetic
    # well-defined for PI / PO / constant rows.
    lo = torch.full((num_nodes,), 2 * num_nodes + 2, dtype=torch.long, device=x.device)
    lo.scatter_reduce_(0, dst, literal, reduce="amin")
    hi = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
    hi.scatter_reduce_(0, dst, literal, reduce="amax")

    is_and = x[:, 2]
    is_po = x[:, 3]
    fanin_c0 = (lo & 1).to(x.dtype) * is_and
    fanin_c1 = (hi & 1).to(x.dtype) * is_and

    # Columns 0 and 1 are identical upstream (`fprintf(f_feats, "1,1,%d,%d\n",
    # ...)`) -- the redundancy is upstream's, reproduced so in_channels stays 4.
    return torch.stack(
        [is_and, is_and, fanin_c0 + is_po, fanin_c1 + is_po], dim=1
    )


class GamoraGraphRegressor(SAGE_MULT):
    """Gamora's GraphSAGE trunk + graph-level pooling + a regression head."""

    def __init__(
        self,
        in_channels: int = GAMORA_NODE_FEATURE_DIM,
        hidden_channels: int = DEFAULT_HIDDEN_DIM,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        task_out_dim: int = 1,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=1,  # unused: the heads that consumed it are removed below.
            num_layers=num_layers,
            dropout=dropout,
        )
        # Drop the 3 per-node classification heads (xor / maj / adder-root).
        # self.linear[0] and self.bn0 stay: they are the shared neck, not
        # Gamora-specific output. The paper describes that neck as "a shared
        # linear layer with size of 32 and the ReLU activation function" and
        # does NOT mention a BatchNorm -- bn0 comes from the released code
        # (gnn_multitask.py:58, applied at :98), so the neck kept here is the
        # code's, one layer DEEPER than the paper's sentence -- same width
        # either way. Deleted from the back so the
        # surviving indices don't shift mid-loop.
        del self.linear[3]
        del self.linear[2]
        del self.linear[1]

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_channels // 2, task_out_dim),
        )

    def forward(self, batch) -> torch.Tensor:
        """Args:
            batch: a `torch_geometric.data.Batch` with `.x`, `.edge_index`,
                `.edge_attr` and `.batch` (graph-membership index). An
                uncollated `Data` also works and is treated as one graph.

        Returns:
            Tensor of shape `(num_graphs, task_out_dim)` in `[0, 1]`.
        """
        x = gamora_node_features(batch.x, batch.edge_index, batch.edge_attr)

        # --- Gamora's full-graph trunk, i.e. the body of
        # SAGE_MULT.forward_nosampler (gnn_multitask.py:91-98), re-inlined here
        # so pooling and the regression head can replace the three
        # classification heads that followed it. `edge_index` stands in for
        # upstream's `adj_t`: upstream applies T.ToSparseTensor() to the same
        # edge set, which changes the storage layout, not the message
        # direction. No neighbour sampling -- see the module docstring. ---
        for conv in self.convs:
            x = conv(x, batch.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear[0](x)
        x = self.bn0(F.relu(x))
        # --- end Gamora trunk ---

        graph_embed = global_mean_pool(x, batch.batch)
        return torch.sigmoid(self.regression_head(graph_embed))
