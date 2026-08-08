# PolarGate vendored source — provenance

Upstream: <https://github.com/BUPT-GAMMA/PolarGate>
Commit: `4bbb23e965ec5e9b17835878f5c50e108f4a7df7` (2024-10-08, "Update
layers.py" — the repository's last push).
Paper (the version cited): Liu, Zhai, Zhao, Lin, Yu, Shi, *PolarGate: Breaking
the Functionality Representation Bottleneck of And-Inverter Graph Neural
Network*, **ICCAD 2024**.

## Licence

**Upstream declares no licence.** The PolarGate repository contains no
`LICENSE` file, no `COPYING`, and no licence header in any source file, so
there is no `LICENSE_UPSTREAM` here to sit alongside the ones in `../hoga/`
(BSD 3-Clause) and `../openabc_synthnet/` (BSD 3-Clause). This matches the
situation recorded in `../deepgate4/PROVENANCE.md`.

That is a statement of fact about the upstream repository, not a grant. Absent
an explicit licence, default copyright applies and no redistribution right is
conveyed. Before this code is published — thesis appendix, an artifact
release, or a public fork of this repository — either obtain permission from
the authors or replace `layers.py` with a clean-room reimplementation.
Vendoring it for private research use is a separate question from
redistributing it, and only the first has happened so far.

## Which version of PolarGate this is

**ICCAD'24, the conference version.** Cite the bibtex upstream's own README
publishes:

```bibtex
@inproceedings{PolarGate,
  author={Liu, Jiawei and Zhai, Jianwang and Zhao, Mingyu and Lin, Zhe and
          Yu, Bei and Shi, Chuan},
  booktitle={2024 IEEE/ACM International Conference on Computer-Aided Design (ICCAD)},
  title={PolarGate: Breaking the Functionality Representation Bottleneck of
         And-Inverter Graph Neural Network},
  year={2024}}
```

The PDF in `~/Documents/AIG_ML_Papers/PolarGate.pdf` is the **extended TODAES
2025 journal version**, which adds two global modules on top of the conference
model:

- a **structure-aware preprocessing module (SAP)**, Section 4.5, which
  tokenises each node as itself plus its up to two predecessors, and
- an **optimal global attention module (OGA)**, a linear-complexity global
  attention block sitting after the local message passing.

**Neither exists in the released code, so neither is in this port.** Verified
directly against the clone at the commit above:

- `layers.py` defines exactly `create_spectral_features`, `MLP`,
  `PolarGateConv` and `restPolarGateConv`, and nothing else.
- `grep -i -E 'global|oga|sap|attention|spectral' layers.py model.py` returns
  only the `create_spectral_features` definition and its import in `model.py`.
- `model.py`'s `PolarGate.forward` is `conv1 → convs → weight → readout_prob`,
  with no global stage.

Reimplementing SAP or OGA from the journal prose would be inventing a baseline
rather than running one, so it was not attempted. Do not label this baseline
"PolarGate (TODAES'25)" and do not compare its results against that paper's
numbers.

## Files vendored

| File | Upstream path | Change |
|---|---|---|
| `layers.py` | `layers.py` | three deletions + added header docstring; see below |

`model.py` is **not** vendored, following the precedent in `../deepgate4/`.
Its `PolarGate` class is inseparable from upstream's per-node signal-probability
readout and from the `create_spectral_features` fallback, neither of which
applies to graph-level regression. `regressor.py` reproduces the encoder path
of that `forward()` (model.py:73-77) line for line — tanh after the first conv,
tanh after each rest conv, tanh after the final linear — and cites the upstream
line numbers it mirrors. `train.py`, `load_data.py` and `preprocess_data.py`
are not vendored either; see "Upstream's training loop" below.

### Deletions from `layers.py`

Verified with a line diff against the upstream original. Everything from the
import block down is identical once line endings are normalised (upstream is
CRLF, this copy is LF) apart from the three deletions below — same import
order, same literals, same statements, including the trailing space inside
`MLP`'s `'''The basic structure is refered from '''` docstring. Outside that:
this repository's header docstring is added, and upstream's `# coding=utf-8`
first line and two trailing blank lines are dropped.

1. **`create_spectral_features`.** Upstream calls it only when the caller
   passes `init_emb=None` (model.py:60). It runs
   `TruncatedSVD(n_components=64, n_iter=128)` over the N x N signed adjacency.
   Infeasible here: `config.MAX_NUM_GATES` is 366,040 and there are ~707k train
   graphs (~788k with val), so a 128-iteration randomized SVD per graph per
   epoch is off by orders of magnitude in both time and memory. `regressor.py`
   always supplies
   real node features and raises `ValueError` if handed a batch without `x`, so
   the fallback is unreachable rather than merely unused; the raise is what
   keeps it that way. Removing the function also drops the `sklearn`,
   `scipy.sparse` and `torch_sparse.coalesce` imports, none of which are
   installed in this project's environment.

2. **`message_and_aggregate` on both conv classes.** Upstream's is
   `matmul(adj_t.set_value(None), x[0], reduce=self.aggr)` from `torch_sparse`,
   which is not a dependency of this project and is not installed, so keeping
   it would make the module unimportable. Behaviour-neutral on this port's call
   path: PyG dispatches to `message_and_aggregate` only when `propagate` is
   given a `SparseTensor` or a `torch.sparse` adjacency, and `regressor.py`
   always passes a dense `[2, E]` `LongTensor`, so upstream's own `message` +
   `aggregate` path is what runs in both codebases.

3. **The imports left unused by 1 and 2** — `SparseTensor`, `matmul`,
   `coalesce`, `scipy.sparse`, `TruncatedSVD`. Every remaining import is
   referenced, keeping `ruff check src` clean repo-wide.

Nothing else was touched. `MLP`, `PolarGateConv` and `restPolarGateConv` are
byte-for-byte upstream, including upstream's own commented-out
`# kwargs.setdefault('aggr', 'mean')` line in `restPolarGateConv`, its `min`
default, and its unconventional class name.

## Regimes this port enters that upstream never does

Not "discrepancies": the released code and the paper agree with each other on
upstream's own graphs. Both items below are cases this project's graph
encoding makes reachable and upstream's does not, so the paper never had to
define them. Recorded because they bear on any claim that this port is
faithful to the *paper* rather than to the *code* — where they diverge, the
code as released is what is vendored, since it is what produced the published
numbers.

- **The inverted branch's aggregation order.** `restPolarGateConv` sets
  `aggr='min'` on both propagate calls, so its inverted branch computes
  `-min_j(h_j)`. Paper Equation (5), which governs NOT nodes, is
  `σ(W [0, OPNOT_{j∈N_i} h_j, h_i])` — OPNOT written as an operator over the
  neighbour set, with **no aggregator specified**, because a NOT gate has
  exactly one fanin and none is needed. At |N_i| = 1 code and paper coincide,
  so there is nothing inconsistent upstream. It matters only here: with
  inversion on the edge, an AND gate can have two inverted fanins, and then
  `-min_j(h_j) = max_j(-h_j)` differs from "negate first, then OPAND"
  `min_j(-h_j) = -max_j(h_j)`. The paper does not say which is meant because
  the case cannot arise in its data. This port takes the code's order.
  (Contrast Equation (4), for AND nodes, which *does* specify OPAND over the
  fanin set — the code matches it directly.)
- **Both channels populated at once.** Upstream's graphs come from `.bench`
  files where an inverter is an explicit NODE (`preprocess_data.py:150-158`
  emits a `NOT` node and signs its incoming edge -1), so a node is either an
  AND with two `+1` fanins or a NOT with one `-1` fanin — never both, and its
  positive and negative aggregation branches are never simultaneously
  non-empty. This project's AIGs have no NOT nodes: inversion is an edge
  attribute (`data/data_utils.py:148`, `e_type = [1.0 - inv, inv]`), so an AND
  gate routinely has one inverted and one non-inverted fanin and both branches
  fire together. The conv handles this as the obvious generalisation, but it is
  a regime the released model never ran in.

## Upstream's training loop is not reused

`train.py` is not vendored, for two independent reasons:

- **It loads the entire dataset onto the GPU.** `load_data_signed_parallel`
  calls `.to(args.device)` on every graph's features, labels and edge tensors
  at parse time and keeps the whole list resident. That is fine for its ~10k
  subcircuits of at most 3,214 nodes; it is impossible for ~707k train graphs
  averaging ~40,000 nodes.
- **Its `--batch_size` is not a batch size.** The training loop iterates ONE
  graph at a time and calls `optimizer.step()` every `batch_size` iterations
  (`train.py:339`), i.e. it is gradient accumulation over single-graph
  forwards. `train.sh --batch_size 256` therefore means an effective batch of
  256 graphs per update, which this port reproduces through the pairing of
  `--polargate_max_nodes_per_batch` and `--accumulate_grad_batches` (see
  `src/shell/train_baseline_polargate.sh`).

This port uses the project's `BaselineRegressionLightningModule` and the
existing node-budget loader instead.

## Hyperparameters, and the three different `out_dim` values

Upstream's `train.sh` is the published configuration. It passes
`--layer_num 9 --in_dim 3 --feature_type 'one-hot' --batch_size 256
--eval_step 1 --split_file 0.05-0.05-0.9`; everything else falls through to
`train.py`'s argparse defaults (`--out_dim 256`, `--lr 0.01`,
`--weight_decay 1e-3`, `--epochs 500`, `--patience 50`, `--loss_type 'mae'`).

`out_dim` has three candidate values in circulation and only one of them is
what running `train.sh` produces:

- **256** — `train.py`'s argparse default, and therefore what `load_model`
  passes to `PolarGate(...)` when `train.sh` runs. This is the port's default.
- 64 — the `PolarGate.__init__` signature default in `model.py`. `train.sh`
  never reaches it, because argparse always supplies `out_dim` and overrides
  the class default. Quoting 64 as "the published config from train.sh" is a
  mistake.
- 128 — "The node hidden dimension is set to 128 across all models", TODAES
  Section 6.2. That is the journal version's re-run of the model this port does
  not implement, and it is a cross-baseline normalisation rather than
  PolarGate's own width.

`--polargate_out_dim` switches between them; report whichever was used.

`in_dim` is 4 here, not upstream's 3: upstream's one-hot is over
{PI, AND, NOT} because of the explicit inverter nodes described above, while
this project's node feature is `[constant, pi, and_gate, po]`
(`config.NODE_INPUT_DIM`). It is not exposed as a flag, because it describes
the dataset rather than the model.

## Two figures this port leans on that the repo does not record

Both appear throughout the PolarGate files and between them justify the two
biggest deviations below (the loss default and the size covariates):

- the label is **48.8% exactly zero, mean 0.020, SD 0.053**;
- a **two-parameter OLS on log node and edge count outranks the primary
  encoder on Spearman**.

Both were supplied by the project author as prior diagnostic findings. Neither
is computed or recorded by any script, notebook or log in this repository, so
neither can be re-derived from it as it stands. Before either is quoted in the
thesis, recompute it and commit the script that does.

## SCALE CAVEAT — belongs in the results caption, not only here

The paper's evaluated dataset "includes circuits with up to 3214 nodes"
(TODAES Section 7; Table 2 gives the full range as [36–3,214] nodes across
10,824 subcircuits from EPFL, ITC99, IWLS and Opencores). This project's
average graph is ~40,000 nodes and the largest is `config.MAX_NUM_GATES` =
**366,040, roughly 114x their largest evaluated circuit**. Every number this
baseline produces is an out-of-regime extrapolation of the published model, and
that must be stated wherever the numbers are, not only in this file.

## Deviations introduced by this port

Full rationale in `regressor.py`'s module docstring; summarised here because
they are the things a reader of the results table needs to know.

1. **Graph-level readout replaces upstream's per-node one.** Upstream predicts
   one signal probability per gate; this project predicts one scalar per graph.
   The readout MLP is upstream's, at upstream's shape (`num_layer=3`,
   `p_drop=0.2`, `act_layer='relu'`), preceded by pooling and followed by the
   sigmoid upstream also applies.
2. **Size-aware readout, available, OFF by default as of 2026-08-07 (was on).**
   Mean pooling is invariant to |V| and |E|, and on this dataset a
   two-parameter OLS on log node and edge count alone already outranks the
   primary encoder on Spearman — so a size-blind baseline loses to a trivial
   predictor before its architecture is tested. `--polargate_size_covariates`
   (default **false**, was true) concatenates `log1p(|V|)` and `log1p(|E|)`
   onto the pooled embedding when enabled. `--polargate_pooling sum` is the
   alternative encoding of the same information and is not the default
   either way, because summing tanh-bounded rows over 366,040 nodes produces
   embeddings four orders of magnitude larger than a 40-node graph's,
   straight into a sigmoid.

   Dropped to `false` per the project author: this covariate is not part of
   upstream's model and not required to get this port running on this
   hardware/dataset, so it doesn't meet the bar for a deliberate deviation —
   this is meant to be a baseline, not a customized comparison.

   ⚠️ **TURNING THIS ON MAKES PolarGate THE ONLY MODEL IN THE SUITE THAT SEES
   GRAPH SIZE EXPLICITLY.** HOGA, DeepGate4, SynthNet and the primary encoder
   all pool without any size covariate (`grep -rn size_covariate src/baselines
   src/models` returns only this baseline's files). So a PolarGate result that
   BEATS the others with covariates on cannot be attributed to ambipolar
   message passing — it may just be the size head-start. Run
   `--polargate_size_covariates true` as a paired ablation and report both
   numbers if that comparison is ever needed.
3. **No BatchNorm in the head.** Upstream passes `norm_layer='batchnorm'`,
   where the MLP sees one row per NODE. Here it sees one row per GRAPH — often
   a single row, since a graph larger than the node budget cannot be split.
   `--polargate_head_norm_layer batchnorm` therefore does not degrade, it
   **raises** (`nn.BatchNorm1d`: "Expected more than 1 value per channel when
   training") and kills the job on the first singleton micro-batch. Verified,
   and pinned by a test. `--deepgate4_head_norm_layer` shares the default and
   the motivation but not the failure mode: DeepGate4's vendored `MLP.forward`
   pads a 1-row input by repeating it, so it emits a constant instead of
   raising. PolarGate's vendored `MLP` has no such padding.
4. **Loss defaults to upstream's own `mae` as of 2026-08-07 (was SmoothL1(beta=0.01)).**
   The SmoothL1 default matched `train.py:151`, scoring PolarGate on the same
   objective as the primary model — a deliberate comparability choice, not
   something upstream does or something required to run this port. Dropped
   per the project author: this is a baseline, and fidelity to upstream's
   published config (`--loss_type 'mae'`) now wins over comparability with the
   primary model. `--loss smooth_l1` restores the comparability arm; `--loss
   mse` gives a like-for-like check against SynthNet/HOGA/DeepGate4 instead.
   The run label is suffixed for either non-default arm so runs cannot
   overwrite each other.
5. **Early stopping at patience 4, not upstream's 50.** Upstream's 500
   epochs / patience 50 (train.py argparse; TODAES Section 6.2) is unreachable
   in a 72h walltime and would let this baseline train far longer than the
   others. 4 matches the other baseline job scripts.

## Measured memory

All figures below are CPU, float32, at the port's defaults (`out_dim=256`,
`layer_num=9`), on a **synthetic** AIG: 2 fanins per AND node, 1% primary
inputs, ~50% of edges inverted. The node count 366,040 is
`config.MAX_NUM_GATES`, the corpus's real largest graph, and is therefore the
irreducible per-graph peak (graph-level pooling cannot split one graph across
batches). The paired edge count 724,760 is **not** a corpus measurement — it
is what the generator emits at that node count (`2 × (n − n//100)`, i.e. 1.98
edges/node). The real largest graph's edge count is not recorded anywhere in
this repository.

That distinction matters, because the "per-node" constant is really per node
*and* per edge, while the node budget is enforced on nodes only. Measured at
50,000 nodes, varying only the density:

| Edges/node | Retained per node |
|---|---|
| 1.80 | 82,509 B |
| 1.98 | 83,987 B |
| 2.20 | 85,792 B |

i.e. ~8,200 B per node per additional edge-per-node, on a ~68,000 B fixed
floor. The sensitivity is mild — a 22% swing in density moves the constant 4% —
so the budget is not fragile, but a corpus materially denser than ~2 edges/node
would push the figures below up in proportion, and nothing here has checked the
real density.

**Retained activations** (bytes autograd saves for the backward pass, counted
directly through `torch.autograd.graph.saved_tensors_hooks` over unique
storages). This is the figure that governs, and it is allocator-independent:

| Nodes | Unique saved storages | Retained | Per node |
|---|---|---|---|
| 25,000 | 156 | 1.957 GiB | 84,066 B |
| 50,000 | 156 | 3.911 GiB | 83,987 B |
| 100,000 | 156 | 7.818 GiB | 83,947 B |

Constant to three significant figures, over the same 156 storages — expected,
since the trunk is nine fixed-width convs with no attention, no virtual edges,
and no data-dependent structure beyond the edge count.

At 366,040 nodes that constant gives **28.7 GiB (30.8 GB) retained** in
float32. That number is arithmetic from the measured constant, **not a direct
measurement**:
the run needs ~29 GiB and the machine this port was written on has 8 GiB, with
the cluster unreachable from it. A full-scale attempt was made and abandoned
after it spent 20+ minutes paging without completing.

**Peak process RSS** (`resource.getrusage(RUSAGE_SELF).ru_maxrss`), at full
scale:

| What | Nodes | Peak RSS |
|---|---|---|
| Inference forward (`torch.no_grad`) | 366,040 | 2.85 GiB |
| Training step (forward + backward) | 366,040 | 3.05 GiB |

**Do not quote those two RSS numbers as the memory requirement.** They are
badly under-reported: the machine has 8 GiB of RAM and macOS's memory
compressor evicted most of the working set, which is exactly why the
forward+backward figure (3.05 GiB) is implausibly close to the inference one
(2.85 GiB) when the saved-tensor accounting above says the backward pass alone
retains an order of magnitude more. They are recorded only to document that
the full-scale run does complete. The saved-tensor table is the number to use.

Against an H100's 80-94 GiB (this repo's DeepGate4 script assumes the 80 GB
SKU; 94 GiB is the larger one) the conclusion is the same either way. A
largest-graph singleton batch retains 28.7 GiB in float32, roughly half that
under the bf16-mixed AMP the SLURM script selects, so it sits comfortably
inside the card — and unlike HOGA and DeepGate4, the largest graph is not what
sets this baseline's peak.

**The GPU figure now IS measured**, from a real run (2026-08-06/07, wandb run
`94bm63rj`, H100, bf16-mixed AMP): at the then-500,000-node budget,
`system.gpu.0.memoryAllocatedBytes` held flat at ~30.7-30.9 GB for the run's
full 6.5h (the CUDA caching allocator's high-water mark, not a leak) — ~61.9
KB/node end-to-end, including optimizer state and allocator overhead on top
of the isolated activation estimate above. Scaling that measured ratio, the
node budget was raised to **800,000** (2026-08-07), estimated ~49 GB —
matching DeepGate4's own calibrated ceiling on this card (the more aggressive
of HOGA's ~45 GB/56% and DeepGate4's ~49 GB/61%), leaving ~30 GB (38%)
headroom for allocator fragmentation and packing variance. Still an
extrapolation — verify the real peak at 800k from the next run's `nvidia-smi`
or wandb trace and update this figure again.
