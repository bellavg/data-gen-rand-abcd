"""Graph-level optimizability regressor built on PolarGate's ambipolar signed message passing.

Adapts BUPT-GAMMA/PolarGate (see layers.py in this directory, vendored with
three documented deletions) from its released tasks -- per-node signal
probability and truth-table distance prediction -- to this project's task: a
single scalar optimizability prediction per whole AIG.

WHICH PolarGate THIS IS
=======================
This port implements the **ICCAD'24 conference version**, and that is the
version to cite:

    @inproceedings{PolarGate,
      author={Liu, Jiawei and Zhai, Jianwang and Zhao, Mingyu and Lin, Zhe and
              Yu, Bei and Shi, Chuan},
      booktitle={2024 IEEE/ACM International Conference on Computer-Aided
                 Design (ICCAD)},
      title={PolarGate: Breaking the Functionality Representation Bottleneck of
             And-Inverter Graph Neural Network},
      year={2024}}

(the bibtex upstream's own README publishes). The PDF in
~/Documents/AIG_ML_Papers/PolarGate.pdf is the **extended TODAES 2025 journal
version**, which adds two global modules on top of the conference model: a
structure-aware preprocessing module (SAP, Section 4.5) and an optimal global
attention module (OGA). NEITHER EXISTS IN THE RELEASED CODE. The repository's
last push is 2024-10-08, its layers.py contains exactly
`create_spectral_features`, `MLP`, `PolarGateConv` and `restPolarGateConv`, and
grepping layers.py and model.py for global/OGA/SAP/attention matches nothing
but the `create_spectral_features` import. Reimplementing SAP or OGA from the
journal prose would be inventing a baseline rather than running one, so this
port does not attempt it. Do not describe this baseline as "PolarGate
(TODAES'25)" and do not compare it against that paper's numbers.

THE MECHANISM BEING PORTED
==========================
Every node carries two half-width states -- a positive (logic-1) and a negative
(logic-0) embedding -- and message passing cross-propagates them: along a
non-inverted edge each half reads the fanin's matching half, and along an
inverted edge each half reads the fanin's *opposite* half. That swap is the
whole method; it is what makes an inversion a first-class operation rather than
an edge feature the network has to learn to interpret. `restPolarGateConv`
additionally aggregates with elementwise `min` (paper's OPAND, the
differentiable Boolean intersection) and negates the inverted branch (OPNOT,
the differentiable complement).

The two conv classes are NOT the same shape, and only the second matches that
description:

  - `PolarGateConv(first_aggr=True)`, the first layer only, has no halves to
    cross yet -- it reads the raw node features. It MEAN-aggregates them once
    over the non-inverted edges and once over the inverted edges, and builds
    two 2-slot concatenations, `lin_b([agg_pos(x), x])` and
    `lin_u([agg_neg(x), x])`, whose outputs become the positive and negative
    halves. No min, no negation.
  - `restPolarGateConv`, the remaining `layer_num - 1` layers, is the
    cross-propagating one. `out_b` (positive half) is built from
    `[agg_pos(pos_half), -agg_neg(neg_half), own_pos_half]` and `out_u` from
    `[agg_pos(neg_half), -agg_neg(pos_half), own_neg_half]`, with `agg` being
    elementwise min.

A node with no incoming edges of a given sign aggregates to exactly zero
(verified: PyG's `min`/`mean` scatter reductions fill empty rows with 0), which
reproduces the paper's `[0, 0, h]` form for primary inputs (Equation 6) without
any node-type branching.

DEVIATIONS FROM UPSTREAM, ALL DELIBERATE
========================================
1. **Graph-level readout replaces the per-node one.** Upstream's `model.py`
   ends with `readout_prob(z)` applied per node, one signal probability per
   gate. This project predicts one scalar per graph, so the trunk output is
   pooled to graph level first. The MLP itself is upstream's, at upstream's
   shape (`num_layer=3`, `p_drop=0.2`, `act_layer='relu'`), followed by the
   sigmoid upstream also applies.

2. **`in_dim` is 4, not upstream's 3.** Upstream's node features are a one-hot
   over {PI, AND, NOT}, because its graphs come from `.bench` files where an
   inverter is an explicit NODE (preprocess_data.py:153-158 emits a `NOT` node
   and gives its incoming edge sign -1). This project's AIGs have no NOT nodes
   at all: inversion lives on the edge, and the node one-hot is
   `[constant, pi, and_gate, po]` (config.NODE_INPUT_DIM = 4, see
   data/data_utils.py). The edge sign carries the same information in both
   codebases -- upstream's "-1" edge is the edge into an inverter, ours is an
   inverted fanin -- so the conv is unchanged; only the feature width differs.

   One consequence worth stating in the write-up: upstream, a node is either an
   AND (two sign-+1 fanins) or a NOT (one sign-(-1) fanin), never both, so its
   positive and negative aggregation branches are never simultaneously
   populated. Here an AND gate can have one inverted and one non-inverted
   fanin, so both branches fire at once. The conv handles that as the obvious
   generalisation, but it is a regime the released model never ran in.

3. **`create_spectral_features` is not reachable.** Upstream falls back to a
   `TruncatedSVD(n_components=64, n_iter=128)` spectral embedding whenever
   `init_emb is None`. That is infeasible at 366,040 nodes across ~707k train
   graphs (~788k with val). The function is deleted from the vendored layers.py, and `forward` raises if
   `batch.x` is missing, so the fallback cannot be silently re-entered.

4. **Size-aware readout, on by default.** Mean pooling is invariant to |V| and
   |E|, and on this dataset a two-parameter OLS on log node and edge count
   already outranks the primary encoder on Spearman -- so a size-blind baseline
   is handicapped against a trivial predictor rather than against the model it
   is meant to test. `size_covariates=True` (the default) concatenates
   `log1p(|V|)` and `log1p(|E|)` per graph onto the pooled embedding before the
   head. `pooling="sum"` is the alternative route to the same information and
   is available, but it is NOT the default: summing tanh-bounded rows over a
   366,040-node graph produces graph embeddings four orders of magnitude larger
   than a 40-node graph's, straight into a sigmoid. Whichever is used must be
   stated in the results caption, since neither is upstream's readout.

   Under bf16 autocast the covariates are computed in float32 and cast down,
   leaving ~0.4% relative precision, i.e. ~+/-0.05 on a log-count of 12.8. The
   between-graph spread of `log1p(|V|)` on this dataset runs from ~2 to ~12.8,
   so that rounding is far below the signal.

5. **No BatchNorm in the head by default.** Upstream passes
   `norm_layer='batchnorm'` to `readout_prob`, where it sees one row per NODE
   (thousands per call). Here the head sees one row per GRAPH, and node-budget
   batching yields a handful of graphs per micro-batch -- sometimes exactly
   one, since a graph larger than the budget cannot be split.
   `head_norm_layer='batchnorm'` therefore does not merely degrade, it RAISES:
   `nn.BatchNorm1d` in training mode rejects a 1-row input with "Expected more
   than 1 value per channel when training". Verified, and pinned by
   `test_batchnorm_head_raises_on_a_singleton_batch`.

   The DeepGate4 port's `--deepgate4_head_norm_layer` shares the default and
   the motivation but NOT the failure mode: its vendored `MLP.forward` pads a
   1-row input by repeating it, so BatchNorm there sees zero variance and emits
   a constant instead of raising. PolarGate's vendored `MLP` has no such
   padding. Do not set this flag unless every micro-batch is guaranteed to hold
   at least two graphs, which the node budget cannot guarantee while
   config.MAX_NUM_GATES-sized graphs exist.

HYPERPARAMETERS: WHAT IS PUBLISHED AND WHAT IS NOT
==================================================
Upstream's `train.sh` is the published configuration and it passes
`--layer_num 9 --in_dim 3 --feature_type 'one-hot' --batch_size 256
--eval_step 1 --split_file 0.05-0.05-0.9`. Everything else falls through to
`train.py`'s argparse defaults: `--out_dim 256`, `--lr 0.01`,
`--weight_decay 1e-3`, `--epochs 500`, `--patience 50`, `--loss_type 'mae'`.

CAREFUL WITH out_dim. Three different values are floating around and only one
of them is what running `train.sh` produces:
  - **256** -- `train.py`'s argparse default, and therefore the value
    `load_model` passes to `PolarGate(...)` when `train.sh` runs. This is what
    `DEFAULT_OUT_DIM` is set to below, because it is the published *run*.
  - 64 -- the `PolarGate.__init__` signature default in `model.py`. `train.sh`
    never reaches it: argparse always supplies `out_dim`, overriding the class
    default. Quoting 64 as "the published config" is a mistake.
  - 128 -- "The node hidden dimension is set to 128 across all models" in the
    TODAES paper, Section 6.2. That is the journal version's re-run, of the
    model this port is not implementing, and it is a cross-baseline
    normalisation rather than PolarGate's own tuned width.
Pass `--polargate_out_dim` to switch; report whichever was used.

`--batch_size 256` does NOT mean 256 graphs per forward. Upstream's training
loop iterates ONE graph at a time and calls `optimizer.step()` every
`batch_size` iterations (train.py:339), i.e. it is gradient accumulation over
256 single-graph forwards. The effective batch is 256 graphs per update, and
that is the number this port reproduces -- see train_baseline.py and
shell/train_baseline_polargate.sh for how the node budget and
`--accumulate_grad_batches` are paired to hit it.

Upstream's own training loop is otherwise not reused: it moves the entire
dataset onto the GPU at load time (`load_data_signed_parallel`, every graph
`.to(args.device)`), which is impossible for ~707k train graphs averaging 40k
nodes.
This port uses the project's `BaselineRegressionLightningModule` and the
existing node-budget loader.

SCALE CAVEAT -- must appear in the results caption, not only here. The paper's
evaluated dataset "includes circuits with up to 3214 nodes" (TODAES Section 7,
Table 2: total range [36-3,214] nodes over 10,824 subcircuits). This project's
average graph is ~40,000 nodes and the largest is config.MAX_NUM_GATES =
366,040, roughly 114x their largest evaluated circuit. Every result from this
baseline is an out-of-regime extrapolation of the published model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool

from baselines.polargate.layers import MLP, PolarGateConv, restPolarGateConv

# Published in upstream's train.sh / train.py argparse defaults -- see the
# module docstring for exactly which of these train.sh states explicitly.
DEFAULT_LAYER_NUM = 9  # train.sh: --layer_num 9
DEFAULT_OUT_DIM = 256  # train.py argparse default, reached by train.sh
DEFAULT_LR = 0.01  # train.py argparse default
DEFAULT_WEIGHT_DECAY = 1e-3  # train.py argparse default
DEFAULT_NUM_EPOCHS = 500  # train.py argparse default; TODAES Sec 6.2 confirms
DEFAULT_EFFECTIVE_BATCH_GRAPHS = 256  # train.sh --batch_size, = accumulation steps
# readout_prob = MLP(..., num_layer=3, p_drop=0.2, norm_layer='batchnorm',
# act_layer='relu') in model.py:46. The dropout and depth are kept; the norm is
# not (deviation 5 above).
DEFAULT_HEAD_DROPOUT = 0.2

_POOLINGS = {"mean": global_mean_pool, "sum": global_add_pool}


def split_signed_edge_index(
    edge_index: torch.Tensor, edge_attr: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split `edge_index` into non-inverted and inverted edge sets.

    Upstream expects two separate edge lists -- a "+1" list and a "-1" list --
    because its `.bench` preprocessing writes an explicit sign column
    (preprocess_data.py:150/157). This project instead encodes inversion in
    `edge_attr`, built as `e_type = [1.0 - inv, inv]` at
    data/data_utils.py:148, so column 1 is the inverted indicator.

    Note that config.py's comment on `EDGE_ATTR_DIM` describes the two columns
    as `[normal edge, primary output edge]`. That comment is stale and
    describes something this pipeline does not build; data_utils.py is the
    authority, and its column 1 is inversion. Nothing else in the codebase
    reads edge_attr as a PO indicator (see
    openabc_synthnet/regressor.derive_num_inverted_predecessors, which also
    treats column 1 as inversion).

    Args:
        edge_index: `[2, E]` LongTensor, `edge_index[0]` the fanin,
            `edge_index[1]` the node it feeds.
        edge_attr: `[E, 2]` float tensor, `[1 - inv, inv]` per edge.

    Returns:
        `(pos_edge_index, neg_edge_index)`, `[2, E_pos]` and `[2, E_neg]`,
        partitioning the columns of `edge_index` with no overlap and no loss.
    """
    if edge_attr is None:
        raise ValueError(
            "PolarGate requires edge_attr: inversion is the only thing that "
            "distinguishes its two message-passing channels, so a batch "
            "without it would reduce the model to an ordinary min-aggregating "
            "GNN. See data/data_utils.py:148."
        )
    if edge_attr.dim() != 2 or edge_attr.size(1) != 2:
        raise ValueError(
            f"edge_attr must be [E, 2] ([1 - inv, inv]); got "
            f"{tuple(edge_attr.shape)}."
        )
    if edge_attr.size(0) != edge_index.size(1):
        raise ValueError(
            f"edge_attr has {edge_attr.size(0)} rows but edge_index has "
            f"{edge_index.size(1)} edges."
        )

    inverted = edge_attr[:, 1] > 0.5
    return edge_index[:, ~inverted], edge_index[:, inverted]


class PolarGateGraphRegressor(nn.Module):
    """PolarGate's ambipolar trunk + graph-level pooling + regression head."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = DEFAULT_OUT_DIM,
        layer_num: int = DEFAULT_LAYER_NUM,
        norm_emb: bool = False,
        task_out_dim: int = 1,
        pooling: str = "mean",
        size_covariates: bool = True,
        head_dropout: float = DEFAULT_HEAD_DROPOUT,
        head_norm_layer: str | None = None,
    ) -> None:
        super().__init__()
        if layer_num < 1:
            raise ValueError(f"layer_num must be >= 1; got {layer_num}")
        if out_dim % 2 != 0:
            # The trunk splits every embedding into a positive and a negative
            # half of width out_dim // 2, so an odd width would silently drop a
            # channel at each slice.
            raise ValueError(f"out_dim must be even; got {out_dim}")
        if pooling not in _POOLINGS:
            raise ValueError(
                f"pooling must be one of {sorted(_POOLINGS)}; got {pooling!r}"
            )

        self.pooling = pooling
        self.size_covariates = size_covariates

        # model.py:38-45, verbatim in shape.
        self.conv1 = PolarGateConv(in_dim, out_dim // 2, first_aggr=True)
        self.convs = torch.nn.ModuleList()
        for _ in range(layer_num - 1):
            self.convs.append(
                restPolarGateConv(
                    out_dim // 2, out_dim // 2, first_aggr=False, norm_emb=norm_emb
                )
            )
        self.weight = torch.nn.Linear(out_dim, out_dim)

        head_in = out_dim + (2 if size_covariates else 0)
        self.readout = MLP(
            head_in,
            out_dim,
            task_out_dim,
            num_layer=3,
            p_drop=head_dropout,
            norm_layer=head_norm_layer,
            act_layer="relu",
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.weight.reset_parameters()

    def encode(self, batch) -> torch.Tensor:
        """Per-node ambipolar embedding, `(num_nodes, out_dim)`.

        Reproduces `PolarGate.forward`'s trunk (model.py:73-77) exactly: tanh
        after the first conv, tanh after each rest conv, tanh after the final
        linear.
        """
        x = getattr(batch, "x", None)
        if x is None:
            # Upstream would fall back to create_spectral_features here
            # (model.py:60). That path is deleted from the vendored layers.py
            # and is unreachable by construction; this raise is what keeps it
            # that way rather than leaving a latent NameError.
            raise ValueError(
                "PolarGate requires node features (batch.x). Upstream's "
                "init_emb=None fallback runs TruncatedSVD over the adjacency, "
                "which is not viable at this project's scale and is not "
                "ported -- see baselines/polargate/layers.py."
            )

        pos_edge_index, neg_edge_index = split_signed_edge_index(
            batch.edge_index, getattr(batch, "edge_attr", None)
        )

        z = torch.tanh(self.conv1(x, pos_edge_index, neg_edge_index))
        for conv in self.convs:
            z = torch.tanh(conv(z, pos_edge_index, neg_edge_index))
        return torch.tanh(self.weight(z))

    def forward(self, batch) -> torch.Tensor:
        """Args:
            batch: a `torch_geometric.data.Batch` with `.x`, `.edge_index`,
                `.edge_attr` and `.batch` (graph-membership index).

        Returns:
            Tensor of shape `(num_graphs, task_out_dim)` in `[0, 1]`.
        """
        z = self.encode(batch)

        node_batch = getattr(batch, "batch", None)
        if node_batch is None:
            node_batch = z.new_zeros(z.size(0), dtype=torch.long)
        num_graphs = getattr(batch, "num_graphs", None)
        if num_graphs is None:
            num_graphs = int(node_batch.max()) + 1 if node_batch.numel() else 1

        graph_embed = _POOLINGS[self.pooling](z, node_batch, size=num_graphs)
        if self.size_covariates:
            graph_embed = torch.cat(
                [
                    graph_embed,
                    _log_sizes(batch, node_batch, num_graphs).to(graph_embed.dtype),
                ],
                dim=1,
            )
        return torch.sigmoid(self.readout(graph_embed))


def _log_sizes(batch, node_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """`[num_graphs, 2]` of `log1p(|V|)`, `log1p(|E|)`, in float32.

    Computed in float32 rather than the autocast dtype: see deviation 4 in the
    module docstring.
    """
    nodes = torch.bincount(node_batch, minlength=num_graphs)
    # Edges are attributed to their source node's graph. Every edge is
    # intra-graph under PyG's block-diagonal batching, so either endpoint gives
    # the same answer.
    edges = torch.bincount(node_batch[batch.edge_index[0]], minlength=num_graphs)
    return torch.log1p(torch.stack([nodes, edges], dim=1).to(torch.float32))
