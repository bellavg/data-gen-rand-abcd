# SynthNet baseline: why the first run collapsed to a constant

Written up after the first full `train_baseline_synthnet.sh` run on Orchestrate
produced a flat `val_loss` and a negative `val_r2`. Sources compared: this
directory's vendored `model.py`/`regressor.py`, upstream
[NYU-MLDA/OpenABC](https://github.com/NYU-MLDA/OpenABC) `models/qor/SynthNetV3/`,
and the OpenABC-D paper (Chowdhury et al., arXiv:2110.11292).

Run `src/diagnose_synthnet_baseline.py` against a checkpoint to confirm the
mechanism on real data before citing any of this.

## What the run showed

| metric | value |
| --- | --- |
| `val_loss` | 4.4e-4, flat across epochs 2-4 |
| `val_rmse` | 0.0208, flat |
| `val_r2` | -0.167, flat |
| `train_loss_epoch` | 0.00207 -> 0.00205, then flat to ~5 significant figures |
| `train_rmse_epoch` | 0.0456 -> 0.0453, then flat |
| `train_rmse_step` | swings 0.02-0.075 |
| epochs completed | 5 (`--patience 4` early stopping fired) |

Per-step metrics moving while per-epoch metrics sit still is the signature of a
constant prediction: only the target variance in each batch is moving. Solving
`RMSE^2 = sigma^2 + (c - mu)^2` for a constant `c` gives an estimated
`sigma(val target) ~ 0.019`, with `c` landing within ~0.0085 of the val mean.
The train side is weaker: `sigma(train target) <= 0.045` is an upper bound, not
an estimate -- it equals `train_rmse_epoch` only if `c` sits exactly at the
train mean. Q3 of `diagnose_synthnet_baseline.py` prints the real per-split
mean and std; quote those rather than these once it has been run. The same
applies to the "targets cluster near 0.03" figure used in item 4 -- the metrics
above pin sigma and |c - mu|, not mu itself.

`val_loss` is flat rather than noisy because the val split is fixed and the
prediction never changes: once `c` stops moving, `val_loss` is pinned at
`sigma_val^2 + (c - mu_val)^2` by construction.

## The paper reports the same failure on this split

Table 6 (p.14) reports MSE on targets that upstream z-scores **per design**
(`models/qor/SynthNetV3/utils.py:addNormalizedTargets`). Unit-variance targets
give `R2 = 1 - MSE`, exactly for Variant 2 (whole held-out designs) and
approximately for Variants 1 and 3, whose test sets are recipe subsets within a
design and so have variance only near 1:

| split | Net1 | Net2 | Net3 |
| --- | --- | --- | --- |
| Variant 1 -- unseen **recipe**, all IPs in train | R2 +0.35 | +0.19 | +0.42 |
| Variant 2 -- unseen **IP** | **-9.59** | **-0.24** | **-0.47** |
| Variant 3 -- unseen IP-recipe pair | +0.41 | +0.46 | +0.46 |

`data/dataset.py` bakes `split_by=design` into the cache signature, so this
project's split is Variant 2.

**Do not put `val_r2 = -0.167` side by side with those numbers.** Per-design
z-scoring is not an affine rescaling of the label -- it removes between-design
variance from `SS_tot`, so upstream's R2 measures *within-design* explanatory
power while ours is dominated by *between-design* variance. The two denominators
are different quantities. Only the **sign** transfers, and that is the claim
worth making: on held-out designs neither the full published model nor this
trunk beats predicting the mean.

Held-out designs are hard for SynthNet even with the recipe branch present.
That is a published result, not something this port introduced.

## What actually differs here, ranked

### 1. The recipe branch is gone, and it was carrying the signal

`regressor.py` drops `SynthFlowEncoder` and the four parallel `SynthConv`
branches. The stated reason is sound for this task -- training targets one
fixed algorithm, so there is no recipe to condition on -- but the consequence
is that what runs here is not SynthNet. It is the half of SynthNet the paper
never evaluates in isolation.

Upstream's per-design z-scoring removes all between-design variance from the
label, so within a design the only thing left to predict is the recipe effect
-- which is exactly what the conv branch encodes. Every headline number in
Table 6 is a measurement of the recipe branch with the GCN trunk alongside it.
The trunk's standalone contribution is never reported.

Graph size does not explain the difference: averaging the node counts in the
paper's Table 1 gives ~45k nodes per design, the same scale as this project's
graphs. It is the recipe branch, not smaller graphs, that made their numbers
work.

### 2. Targets are not normalised -- and should not be

Upstream: `(x - mu_design) / sigma_design`, unit variance.
Here: the raw `(t0_nodes - t1_nodes) / t0_nodes` fraction from
`data/creation/generate_csv.py`.

**This is the correct choice and must not change.** The purpose of this
baseline is to swap the architecture while holding the task fixed, so the
target, loss, split, and metrics have to stay identical to the primary model's.
Adopting upstream's per-design z-score would make the baseline
incomparable to the very model it exists to be compared against.

The normalisation difference matters for exactly two things, neither of which
is a fix to apply here:

- Our MSE (~2e-3) and the paper's (0.579) are not comparable quantities. Only
  R2 and Spearman transfer across the two conventions. Never quote them side
  by side.
- It is the reason the paper's own results cannot tell us how strong the GCN
  trunk is; see item 1.

Separately, estimated `sigma_train ~ 0.045` vs `sigma_val ~ 0.019` is a ~2.4x
distribution shift across the design split, so a constant fitted on train is
miscalibrated on val. That is a property of design-level splitting, shared by
the primary model, and belongs in the write-up as a caveat -- not something to
patch in the baseline.

### 3. Pooling concentration at this graph scale

`model.py`'s second GCN layer output goes through `batch_norm2` with no ReLU,
standardising each channel over **every node in the batch**. `global_mean_pool`
then averages ~40k nodes per graph, so each graph's pooled vector concentrates
near the batch mean; `global_max_pool` over 40k standardised values
concentrates similarly. With the recipe branch removed there is nothing else
feeding the head.

This is upstream's own architecture, unmodified -- it is a property of running
the trunk alone on large graphs, not a porting error. The unit tests in
`src/unittests/baselines/test_openabc_synthnet.py` use 12-node graphs and
cannot observe it. `diagnose_synthnet_baseline.py` measures it directly.

### 4. Terminal sigmoid (keep it; amplifier, not cause)

`regressor.py` adds `torch.sigmoid` on the final FC output; upstream's head is
linear. Keeping it is right here: the primary model's regression head ends in a
Sigmoid too, so matching it holds one more variable fixed between the two
models, which is the whole point of the baseline.

The known cost is the operating point: targets cluster near 0.03, where
`sigma'(x) = y(1-y) ~ 0.029`, so gradients reaching the FC stack are attenuated
~34x -- for a bound the targets never come close to saturating. That makes
escaping a plateau harder; it does not by itself cause the collapse, and the
primary model pays the same cost on the same targets. Note it as a deviation
from upstream, do not change it.

### 5. Smaller deviations, none of them causes

- Trained 5 epochs against the paper's 80, because `--patience 4` early
  stopping fired on the flat `val_loss`.
- `gradient_clip_val=1.0`; upstream does not clip.
- `ReduceLROnPlateau(factor=0.5, patience=2)` from `config.py`; upstream uses
  PyTorch defaults (`0.1`, `10`).
- Edge direction was reversed relative to upstream, and is now configurable.
  **This one is not numerically neutral -- see the degree-normalisation note
  below.**
  `andAIG2Graphml.py:56` (and :71 for the PO buffer) adds edges node -> fanin
  and `pygDataFromNetworkx` passes `list(G.edges)` through unreversed, so under
  PyG's default `flow="source_to_target"` messages travel toward the PIs and
  each node summarises its fanout cone. `data/data_utils.py:150` adds
  fanin -> node, the opposite. `regressor.py` now defaults to
  `upstream_edge_direction=True` (reverses `edge_index` before the GCN, so the
  trunk sees exactly what OpenABC-D's does);
  `--synthnet_upstream_edge_direction false` restores this project's native
  direction, where each node summarises its fanin cone. Both are
  single-direction. Report the pair -- the native direction is arguably the
  better inductive bias for optimizability, since whether a node can be
  collapsed depends on the logic feeding it, not on what it drives.

  The direction also changes the GCN normalisation, which is easy to miss.
  `model.py:78` computes `deg = degree(row) + 1` on `row = edge_index[0]`, so
  `deg` is always an **out-degree**:
  - native fanin -> node: `row` is the fanin, so `deg` is that node's *fanout*
    -- thousands for a PI, and `norm` spans orders of magnitude.
  - upstream node -> fanin: `row` is the consumer, so `deg` is its *fanin*
    count, which in an AIG is 0, 1, or 2. After `add_self_loops` and the `+1`,
    `deg` only ever takes the values 2, 3, or 4 and `norm` collapses to a
    handful of distinct values.

  So upstream's direction makes the normalisation nearly degree-blind. That is
  faithful -- it is what OpenABC-D actually does -- but for the pooling
  concentration in item 3 it plausibly makes matters worse, not better. Treat
  `upstream_edge_direction=True` as the fidelity run and `false` as the one
  more likely to work, and report both.

## What is not wrong

The vendored `model.py` is faithful to upstream `SynthNetV3/model.py`:
`NodeEncoder`, `GCNConv`, `GNN`, and the FC stack all match, and the one
documented change (4 node types instead of 3) is correct for this project's
AIGs. `derive_num_inverted_predecessors` reproduces upstream's
`num_inverted_predecessors` semantics for AND nodes -- upstream computes it at
graph-build time as the count of inverted fanins; ours counts inverted edges
arriving at each node, the same quantity given this project's edge direction.
One nuance: upstream always creates its PO buffer node with
`num_inverted_predecessors: 0` and a `BUFF` edge (`andAIG2Graphml.py:64-71`),
whereas this project's PO edge carries the real `po_sig.complement`
(`data/data_utils.py:167-169`), so PO nodes here can carry a count of 1.
Arguably an improvement, but not identical -- declare it.

The Lightning plumbing is also fine: prediction/target shapes agree, `val_r2`
is an epoch-level `torchmetrics.R2Score`, and the split is the same one the
primary model uses.

## Is it still a usable baseline?

Yes, provided it is reported honestly.

Reporting "the canonical published architecture for AIG QoR regression collapses
to the mean on unseen designs" is a legitimate result, and it corroborates the
paper's own Variant 2. Two things must be fixed before the number is defensible:

1. Give it the published budget. Five epochs of an 80-epoch schedule is not a
   fair run, even if the flat loss suggests more epochs will not help.
2. Report R2 and Spearman against an explicit mean-predictor baseline. Raw MSE
   in this project's target units invites a false comparison to the paper's
   0.579.

One thing distinguishes this task from the paper's Variant 2 in our favour:
because the targets here are *not* z-scored per design, between-design variance
stays in the label, so optimizability genuinely is a function of the input
graph. The question is well posed for a graph-only model -- the SynthNet trunk
is simply too weak to answer it. That is a sharper and more useful claim than
"unseen designs are hard".

## Next experiment worth running

A recipe-level (same-design) split, i.e. the paper's Variant 1/3. If the trunk
scores positive R2 there and ~0 on the design split, the failure is
generalisation across designs. If it scores ~0 on both, the trunk cannot read
this label at all and the pooling collapse in item 3 is the whole story. That
single control determines which claim the thesis can make.
