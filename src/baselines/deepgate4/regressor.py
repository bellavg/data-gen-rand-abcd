"""Graph-level optimizability regressor built on DeepGate4's encoder.

Adapts zyzheng17/DeepGate4-ICLR-25 (Zheng et al., ICLR'25) from its released
task -- self-supervised multi-task pretraining of gate embeddings -- to this
project's task: one scalar optimizability prediction per whole AIG.

WHAT IS REUSED
--------------
The encoder is upstream's, unmodified. `dg2.DeepGate2` (the tokenizer) and
`plain_tf_linear.Sparse_Transformer` (the GAT-based sparse transformer that
*is* DeepGate4, paper Section 3.5) are vendored byte-identical apart from
deleted unused imports -- see PROVENANCE.md. `forward()` below reproduces the
encoder path of `dg4.py:DeepGate4.forward` (upstream lines 379-392):

    abs_pe    = self.abs_pe_embedding(self.sinu_pe[g.forward_level])
    init_lhs  = abs_pe + self.out_not(g.out_not) + self.out_and(g.out_and)
    hs, hf    = self.tokenizer(g, ..., mk=self.mk, lhs=init_lhs)
    hf_tf, hs_tf = self.transformer(g, hf.clone(), hs.clone(), self.mk)
    hf = hf + hf_tf ; hs = hs + hs_tf

`init_lhs` is paper Eq. 2's structural encoding,
`SE(v) = Emb_l(level(v)) + Emb_and(OutAND(v)) + Emb_not(OutNOT(v))`.

Three textual differences from those upstream lines, all behaviour-preserving
and each verified against the original:
  - upstream computes `abs_pe` in two statements (index the table, then apply
    the embedding); this merges them into one nested call;
  - upstream guards the transformer call with `if self.tf_arch != 'baseline'`.
    This class has no `tf_arch`, and 'baseline' selects the tokenizer-only
    ablation (i.e. DeepGate2, not DeepGate4), so the guard is always true here
    and is dropped;
  - upstream assigns the residual sums back into `hf_tf`/`hs_tf`; this assigns
    into `hf`/`hs`. Same values -- upstream only needs the separate names for
    the History scatter that follows, which this port does not have. That
    scatter is also where upstream's `dg4.py:445-446` copy-paste bug lives
    (`all_update_hs` computed from `hf_tf`), so not reproducing it is
    deliberate.

The tokenizer call also omits upstream's `g.prob`, `hf_detach`, `hs_detach`
positional arguments. `DeepGate2.forward` never reads `PI_prob`, and the two
history tensors are only ever selected by `mk[...] == 1`, which is empty here
because `mk` is all-zero -- so `None` is equivalent, not merely convenient.

WHAT IS DROPPED, AND WHY
------------------------
1. **The pretraining heads.** `dg4.py`'s `init_MLP` builds fourteen task heads
   (probability, level, truth table, connectivity, GED, hop-level area/time/
   on-hop, pairwise similarities). All of them supervise the self-supervised
   objectives of paper Section 3.6 and need labels from a C++ logic simulator
   and truth-table enumeration. A supervised regression baseline uses none.
   They are replaced by mean pooling plus ONE head of the same construction:
   upstream's own `MLP` class at upstream's own shape (3 layers, width 128,
   ReLU, p_drop 0.5), followed by a sigmoid because this project's targets lie
   in [0, 1].

   The decision to pool to one vector per circuit is necessarily ours:
   DeepGate4 has no whole-circuit readout at all. Every "graph-level" task in
   the paper is per-*cone*, pooled through the Pooling Transformer over a
   cone's PI/PO embeddings. There is nothing to copy for "one number per
   circuit" -- but there is something to copy for "how DeepGate4 builds a
   head", and that is what `MLP` is. See `__init__` for the single documented
   deviation (BatchNorm, which degenerates at graph-level batch sizes).

2. **Partitioning and the History buffer** (paper Sections 3.2-3.4, upstream's
   `large_ckt=True` branch). These exist to bound *memory* when a circuit
   cannot be embedded in one pass: cones are encoded level by level and gates
   already embedded are pulled from a stale global `History` table instead of
   recomputed. Three reasons this port runs the whole graph in one pass
   instead:

   - The virtual edge set is unchanged by the decision. Cones tile the
     circuit, so the union of the per-cone `Ē` sets is the global
     `{(u, v) : dist(u, v) <= k}` that aig_features.virtual_edges() computes.
     Section 3.5 -- the actual architecture -- is therefore untouched.
   - Graph-level pooling needs every node of a graph in one forward pass
     anyway, so the memory that partitioning saves has to be spent regardless.
   - Fresh embeddings beat stale ones. Under the History scheme a pooled graph
     embedding would average vectors computed at different points in training.

   Be precise about what the paper says here, because it is easy to overclaim.
   The authors DO define a "w/o Partition" setting (Table 4, Section 4.5) --
   but they report it as **OOM on both ITC99 and EPFL**, concluding that this
   "highlight[s] the necessity of partitioning for memory usage reduction".
   Their benchmark circuits are smaller than this dataset's ~40k-node average.
   So the ablation names the setting; it does not vindicate it.

   What makes it viable here is not that the paper blessed it, but that this
   port pays for the memory a different way: gradient checkpointing over the
   12 GAT layers plus a node-budget batch sampler (see
   CheckpointedSparseTransformer below and train_baseline.py). Those substitute
   for what partitioning buys upstream. That is a defensible engineering
   substitution, not a reproduction of the authors' configuration.

   Two consequences for reporting, both of which should be stated plainly:
   this baseline does not exercise DeepGate4's scalability contribution -- it
   measures the representation, which is what a QoR-regression comparison
   needs -- and it reaches that representation through memory machinery the
   authors did not use.

   (Incidentally, that branch also carries a plain copy-paste bug upstream:
   dg4.py:445-446 computes `all_update_hs` from `hf_tf` rather than `hs_tf`,
   so the structural history is filled with functional embeddings. Not a
   factor here, since the branch is unused, but it is one more reason not to
   port it verbatim.)

3. **`prob`.** DeepGate's random-simulation logic-1 probabilities are never
   read by the encoder -- `DeepGate2.forward` takes `PI_prob` and ignores it
   (see dg2.py's docstring). Upstream needs them only for the `readout_prob`
   pretraining task and to seed `reset_history`, both dropped above. So no
   logic simulator is required.

PUBLISHED HYPERPARAMETERS
-------------------------
From paper Section 4.1: hidden dimension 128 ("The dimensions of both the
structural and functional embedding are set to 128"), sparse transformer depth
12 ("The depth of Sparse Transformer is 12"), cone depth / virtual-edge radius
k = 8 ("we set k to 8"), 3-layer MLP task heads, 200 epochs, Adam at lr 1e-4.
Upstream's argparse supplies the head's width and norm (`--mlp_hidden 128
--mlp_layer 3 --norm_layer batchnorm`). The MSE loss is upstream's `--loss l2`
default, mapped to `nn.MSELoss` in `trainer/dg4_trainer.py`.

On lr specifically: 1e-4 is the paper's stated value AND upstream's effective
one, but not via `--lr`. `config.py`'s `--lr` default is 5e-4 and both run
scripts override it to 1e-4 -- yet `main.py` never passes `lr=` into
`Trainer(...)`, so the flag is dead and the operative value is
`dg4_trainer.Trainer.__init__`'s own `lr=1e-4` default regardless. Same
number, different provenance than the flag suggests.

Upstream applies NO gradient clipping (`dg4_trainer.py` never calls a clipping
function) and NO LR scheduler in practice (`lr_step` defaults to -1 and
`main.py` never calls `set_training_args`). This project's shared baseline
defaults supply both; `train_baseline_deepgate4.sh` sets
`--gradient_clip_val 0` to match upstream, while the ReduceLROnPlateau
schedule is kept deliberately so the comparison against the primary model
differs by architecture rather than LR schedule (see
baselines/common/lightning_wrapper.py).

Published but inapplicable without partitioning: the stride delta = 6, and the
batch/mini-batch sizes of 1/128 -- upstream's batching unit is a cone, not a
circuit.

NOT published: the GAT's `heads=4`, `concat=True` and `dropout=0.1`. Those are
`Sparse_Transformer.__init__`'s own defaults, and upstream never overrides
them (`dg4.py` constructs it as `Sparse_Transformer(args, hidden=self.hidden)`,
so even `--TF_depth` does not reach it). They are upstream's effective
settings, but do not call them the paper's.

ON LOWERING k: Appendix A.3 (Table 8) ablates it, and k = 6 / delta = 4 is a
*published setting*, not an off-piste deviation -- it records the best
functional loss of any row (L_func 0.4629 vs 0.4863 at k=8/delta=6), a
comparable overall loss (3.1192 vs 3.1646), and roughly half the training
memory (6.59 GB vs 12.62 GB). The paper's own reading is that "larger k will
degrade structural task performance ... structural tasks rely more heavily on
local information". So if memory forces k down, k = 6 is defensible on the
authors' own evidence; still report which k was used.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch_geometric.nn import global_mean_pool

import config
from baselines.deepgate4.aig_features import OUT_DEGREE_TABLE_SIZE
from baselines.deepgate4.dg2 import DeepGate2
from baselines.deepgate4.mlp import MLP
from baselines.deepgate4.plain_tf_linear import Sparse_Transformer

# Paper Section 4.1.
DEFAULT_HIDDEN_DIM = 128
DEFAULT_NUM_TF_LAYERS = 12
DEFAULT_LR = 1e-4
DEFAULT_NUM_EPOCHS = 200  # "We train all models for 200 epochs to ensure convergence."

# Task-head shape. "All training task heads are 3-layer multilayer perceptrons
# (MLPs)" (Section 4.1); the width and activation come from upstream's argparse
# defaults, `--mlp_hidden 128 --mlp_layer 3 --norm_layer batchnorm`, and
# dg4.py's `init_MLP` passes `act_layer='relu'` with `p_drop` left at MLP's own
# 0.5 default.
DEFAULT_MLP_HIDDEN = 128
DEFAULT_MLP_LAYER = 3
DEFAULT_HEAD_DROPOUT = 0.5

# Upstream constructor defaults, not published -- see this module's docstring.
DEFAULT_HEADS = 4
DEFAULT_TF_DROPOUT = 0.1

# Upstream sizes its sinusoidal level table at 10,000 rows (dg4.py:172). This
# project's AIGs reach config.MAX_DEPTH = 24,972 levels *before* NOT-node
# expansion, and expansion can nearly double a path's length, so the table has
# to be sized from the data rather than left at upstream's constant or every
# deep circuit would index out of bounds.
DEFAULT_MAX_LEVEL = 2 * config.MAX_DEPTH + 1


class CheckpointedSparseTransformer(Sparse_Transformer):
    """`Sparse_Transformer` with gradient checkpointing over its 12 layers.

    Overrides `forward()` only, in the same way `HOGAGraphRegressor` subclasses
    upstream's `HOGA` -- the vendored file stays byte-identical and the layers,
    their weights, and the arithmetic are untouched. Activations are recomputed
    during the backward pass instead of being retained, which is a
    memory/compute trade with no effect on the result or the gradients.

    Not an optimisation for its own sake -- without it this baseline does not
    run at all. Each `GATConv` materialises an `[E, heads, out_channels]`
    message tensor, and DeepGate4's published radius k = 8 makes E enormous:
    measured on synthetic AIGs matching this dataset's shape, an average
    40k-node graph expands to ~66k nodes carrying **7.36M virtual edges**, i.e.
    ~112 per expanded node (~182 per pre-expansion node). At `heads=4`,
    `out_channels=64` and bf16 that is ~3.8 GB per layer, so retaining all 12
    costs ~45 GB for a *single average graph* -- over half an 80 GB H100 before
    the batch has a second graph in it. Checkpointing collapses the 12x to 1x
    plus recompute.

    The peak that remains is set by the largest graph, which cannot be split
    (graph-level pooling needs all its nodes at once): at config.MAX_NUM_GATES
    = 366,040 nodes and ~182 virtual edges per *original* node, one layer alone
    is ~34 GB. Such a graph is tight even checkpointed. If it OOMs, lower
    `--deepgate4_num_hops` to 6 -- Appendix A.3 ablates exactly that and
    reports the best functional loss of any setting at roughly half the memory,
    so it is a published configuration rather than a departure. Report
    whichever k was used either way.

    (These figures are for the default one-way virtual edges, matching paper
    and code. `--deepgate4_symmetric_virtual_edges true` doubles every number
    here.)
    """

    def forward(self, g, hf, hs, mk):
        virtual_edge = g.global_virtual_edge

        virtual_edge = virtual_edge.T
        virtual_edge = virtual_edge[mk[g.nodes[virtual_edge[:, 1].cpu()]] == 0]
        virtual_edge = virtual_edge.T

        if virtual_edge.shape[1] == 0:
            return hf, hs

        h = torch.cat([hf, hs], dim=-1)
        for i in range(self.num_layers):
            if self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(
                    self.tf_layers[i], h, virtual_edge, use_reentrant=False
                )
            else:
                h = self.tf_layers[i](h, virtual_edge)

        hf, hs = torch.chunk(h, 2, dim=-1)

        return hf, hs


class DeepGate4GraphRegressor(nn.Module):
    """DeepGate4's tokenizer + sparse transformer, pooled to one scalar per graph."""

    def __init__(
        self,
        hidden: int = DEFAULT_HIDDEN_DIM,
        num_tf_layers: int = DEFAULT_NUM_TF_LAYERS,
        heads: int = DEFAULT_HEADS,
        tf_dropout: float = DEFAULT_TF_DROPOUT,
        max_level: int = DEFAULT_MAX_LEVEL,
        task_out_dim: int = 1,
        mlp_hidden: int = DEFAULT_MLP_HIDDEN,
        mlp_layer: int = DEFAULT_MLP_LAYER,
        head_dropout: float = DEFAULT_HEAD_DROPOUT,
        head_norm_layer: str | None = None,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.max_level = max_level

        self.tokenizer = DeepGate2(dim_hidden=hidden)
        # `Sparse_Transformer.__init__(self, args, hidden, ...)` accepts `args`
        # and never reads it (see plain_tf_linear.py); upstream passes dg4's
        # argparse namespace purely positionally. None is therefore faithful.
        transformer_cls = (
            CheckpointedSparseTransformer
            if gradient_checkpointing
            else Sparse_Transformer
        )
        self.transformer = transformer_cls(
            None,
            hidden=hidden,
            num_layers=num_tf_layers,
            heads=heads,
            dropout=tf_dropout,
        )

        # Structural encoding, paper Eq. 2 (dg4.py:172-175).
        self.register_buffer(
            "sinu_pe",
            self.sinuous_positional_encoding(max_level + 1, hidden),
            persistent=False,
        )
        self.abs_pe_embedding = nn.Linear(hidden, hidden)
        self.out_and = nn.Embedding(OUT_DEGREE_TABLE_SIZE, hidden)
        self.out_not = nn.Embedding(OUT_DEGREE_TABLE_SIZE, hidden)

        # WHICH readout is ours; WHAT it is made of is upstream's. DeepGate4 has
        # no whole-circuit head to copy -- every "graph-level" task in the paper
        # is per-cone -- so pooling to one vector per circuit is this port's
        # decision. The head itself is then upstream's own `MLP` class at
        # upstream's own shape: 3 layers, width 128, ReLU, p_drop 0.5 (Section
        # 4.1's "All training task heads are 3-layer multilayer perceptrons",
        # plus `--mlp_hidden 128 --mlp_layer 3` and dg4.py's `init_MLP`).
        #
        # Consumes cat([hf, hs]) at width 2*hidden, matching upstream's own
        # two-embedding heads (`connect_head`, `on_hop_head`, `proj_*` are all
        # `dim_in=hidden*2`). The functional and structural embeddings are
        # disentangled by construction in DeepGate2 and the sparse transformer
        # operates on their concatenation, so both are carried through.
        #
        # ONE DEVIATION: `head_norm_layer` defaults to None, where upstream's
        # `--norm_layer` is 'batchnorm'. BatchNorm1d here would be actively
        # harmful, not merely different. Upstream's heads run over thousands of
        # gates or cones at once, so their batch axis is large. Ours runs over
        # GRAPHS, and the node budget yields ~1 graph per micro-batch. MLP.forward
        # handles a 1-row input by `x.repeat(2, 1)`, which makes BatchNorm see
        # two identical rows: variance 0, so every input normalises to exactly 0
        # and the layer emits its bias regardless of the circuit. The head would
        # predict a constant. Pass 'batchnorm' explicitly if the budget ever
        # allows genuinely large graph batches.
        self.regression_head = MLP(
            dim_in=hidden * 2,
            dim_hidden=mlp_hidden,
            dim_pred=task_out_dim,
            num_layer=mlp_layer,
            norm_layer=head_norm_layer,
            act_layer="relu",
            p_drop=head_dropout,
        )

    @staticmethod
    def sinuous_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
        """Verbatim from dg4.py:220 (`DeepGate4.sinuous_positional_encoding`)."""
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def load_pretrained_tokenizer(self, path: str) -> None:
        """Initialise the tokenizer from an upstream DeepGate2 checkpoint.

        Upstream's released `trained/model_last*.pth` hold tokenizer weights
        only (see PROVENANCE.md), which is exactly what `DeepGate2.load` reads.
        Requires `hidden=128` to match the released dimension; `DeepGate2.load`
        skips shape-mismatched tensors with a printed warning rather than
        failing, so a mismatch would silently train from scratch.
        """
        if self.hidden != DEFAULT_HIDDEN_DIM:
            raise ValueError(
                f"Pretrained DeepGate2 weights are dim_hidden={DEFAULT_HIDDEN_DIM}; "
                f"this model was built with hidden={self.hidden}. Loading would "
                "silently skip every mismatched tensor and train from scratch."
            )
        self.tokenizer.load(path)

    def forward(self, batch) -> torch.Tensor:
        """Args:
            batch: a `Batch` of `aig_features.DeepGateData`, carrying `gate`,
                `edge_index`, `forward_level`, `forward_index`, `nodes`,
                `out_and`, `out_not` and `global_virtual_edge`.

        Returns:
            Tensor of shape `(num_graphs, task_out_dim)` in `[0, 1]`.
        """
        device = batch.gate.device
        num_nodes = int(batch.num_nodes)

        # Upstream's "already embedded" bitmap. With no History there is
        # nothing pre-embedded, so an all-zero mask makes every `mk[...]` test
        # in the vendored tokenizer and transformer a no-op -- which is what
        # lets those two files stay byte-identical to upstream. Kept on the
        # model's device: `get_slices` indexes it with a device tensor, while
        # `DeepGate2.forward` indexes it with a CPU one, and only a device-side
        # `mk` satisfies both.
        mk = torch.zeros(num_nodes, device=device)

        # --- upstream dg4.py:379-392, encoder path ---
        # Upstream writes `self.sinu_pe[g.forward_level.cpu()].to(device)`,
        # keeping the table on the host. Indexing the device-side buffer
        # directly is equivalent and avoids a host sync per forward pass. The
        # clamp is a safety net only: DEFAULT_MAX_LEVEL is sized to cover the
        # deepest possible expanded graph, so it should never bind.
        level = batch.forward_level.clamp(max=self.max_level)
        abs_pe = self.abs_pe_embedding(self.sinu_pe[level])
        init_lhs = abs_pe + self.out_not(batch.out_not) + self.out_and(batch.out_and)

        hs, hf = self.tokenizer(batch, mk=mk, lhs=init_lhs)
        hf_tf, hs_tf = self.transformer(batch, hf.clone(), hs.clone(), mk)
        hf = hf + hf_tf
        hs = hs + hs_tf
        # --- end upstream encoder path ---

        node_embed = torch.cat([hf, hs], dim=-1)
        graph_embed = global_mean_pool(node_embed, batch.batch)
        return torch.sigmoid(self.regression_head(graph_embed))
