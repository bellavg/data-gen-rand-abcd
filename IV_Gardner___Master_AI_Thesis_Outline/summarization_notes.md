# Summarization / Coarsening — working notes

Scratch pad for the summarization half of the reduction study. Not thesis prose —
decisions, open questions, and per-method notes. Feeds
`sections/3-methodology.tex` §`sec:method:reduction:summarization`.

Task reminder: target is **graph-level regression** (one optimizability scalar per
AIG), trained on Orchestrate. This matters a lot for R1 (see below).

---

## STATUS (2026-07-31) — all five methods built, read this box first

Supersedes the 2026-07-28 box below, which said S1/S3/S4/S5 were not started.
**All five are now implemented, registered and unit-tested** in
`src/data/summarization.py` (one module, one registry, one shared
`apply_merge_map`); `config.SUMMARIZATION_PARAMS` holds the parameters a
production run actually uses. Suite green, ruff clean.

**Submitting a method is an array range, not an environment variable.** A
bare `METHOD=wl sbatch ...` is *not propagated to the job on Snellius* — it
silently ran whatever the default was, which is why pairing a method with a
submission never worked. Both job scripts now read the method out of the
Slurm array index, using one shared ordered list
(`src/shell/summarization_methods.sh`) so the two can never disagree about
which index means which method. That list is **append-only**: ranges people
submit by hand are positional, so reordering it would silently repoint a
range at a different method.

| method | precompute range (32 shards each) | train task |
|---|---|---|
| identity  | `--array=0-31`    | `--array=0` |
| cone      | `--array=32-63`   | `--array=1` |
| wl        | `--array=64-95`   | `--array=2` |
| convmatch | `--array=96-127`  | `--array=3` |
| spectral  | `--array=128-159` | `--array=4` |
| lsh       | `--array=160-191` | `--array=5` |

Each job prints its own method and range on the first lines of its log.
Precompute defaults to `--array=0-31` (identity only), the cheapest
end-to-end check of the precompute → pack → stage → train path.

| id | name | registry key | role |
|----|------|--------------|------|
| — | identity | `identity` | zero-compression control |
| S1 | Level-bounded cone coarsening | `cone` | domain-specific (the contribution) |
| S2 | Graded WL / bisimulation | `wl` | adapted SOTA, lossless anchor |
| S3 | A-ConvMatch | `convmatch` | general SOTA bar |
| S4 | Spectral / local variation | `spectral` | domain-blind control |
| S5 | LSH / UGC hashing | `lsh` | cheap naive control |

### Findings from building them (these belong in the writeup, not just here)

1. **Forward dominators are degenerate on AIGs — use post-dominators, and
   require a *real* reconvergence gate.** S1's width axis was specified as
   "same level + common immediate dominator". Measured: **~99% of AND gates
   (2996/3000) have the virtual source as their immediate dominator**, because
   a gate deep in an AIG is reachable from the input frontier along many
   independent paths. Grouping on that collapses S1 to "merge everything on a
   level". Switched to the **immediate post-dominator** — the gate at which a
   node's fanout cone reconverges, which is what "reconvergence coarsening"
   means anyway.
   **The same degeneracy then reappeared on the output side** and had to be
   closed separately: any gate whose fanout reaches two outputs without
   passing through one common gate post-dominates to the *virtual sink*, and
   on a multi-output AIG that is **73% of AND gates (5821/8000)**. Grouping
   those by level merged gates in *different connected components*. They now
   get a private id and are left alone, so the axis merges only gates with a
   genuine shared reconvergence gate. Both halves of this are worth one
   sentence in §3 — the naive reading of "common immediate dominator" does not
   survive contact with an AIG in either direction.
2. **S1 is provably DAG-preserving, and the two axes compose.** Width at
   `level_band=0` merges only within a level, and every netlist edge runs to a
   strictly higher level, so no edge lies inside a group and no cycle can form
   between two. Depth contracts a cluster into the single target of its only
   outgoing edge, which cannot close a cycle in a DAG. The depth axis runs on
   the width quotient, still a DAG, so the composition is safe. `level_band>0`
   gives up both this guarantee and the exact level PE. This upgrades C7/CA4
   from "achievable" to "achieved and tested".
3. **Level bands are fixed windows, not ±k.** `level // (band+1)`, because
   "within ±1 level" is not an equivalence relation and so cannot define a
   partition. Consequence to state: widening the band does **not** monotonically
   increase compression — band=2 can merge less than band=1 when the window
   boundary falls between two otherwise-groupable levels.
4. **ConvMatch's objective favours low-degree nodes regardless of similarity.**
   Eq. 7 is an *unweighted* L1 sum over the convolution output, so merging two
   low-degree, small-representation nodes scores better than merging two
   genuine convolution twins of higher degree. Verified on a hand-built graph:
   twins cost less than a same-degree mismatched pair (1.876 vs 1.967), but a
   pair of degree-2 PIs beats both (1.257). Convolution equivalence decides
   *between* pairs of comparable degree, not across them. One sentence in the
   S3 writeup, and a reason its behaviour on a deep DAG differs from the
   node-classification setting it was published in.
5. **LSH's compression knob is monotone by construction, not on average.**
   Offsets are drawn independently of `bin_width`, so
   `floor(y/2r) = floor(floor(y/r)/2)` makes each doubling a strict
   coarsening. Node type is an *exact* part of the bucket key, so PI/PO never
   dissolve into AND super-nodes (C4 by construction).
6. **S4's spectral path is the CPU risk the plan predicted.** Measured
   local-variation cost: 4.4 ms at n=371, 73 ms at n=1.5k, **880 ms at n=4k**
   against 9 ms for heavy-edge. Hence `max_spectral_nodes=5000`, above which
   it falls back to heavy-edge. That cap is part of the method's *definition*,
   not an implementation detail — say so when reporting S4.

7. **S1's compression is entirely structure-dependent, and is NOT yet
   measured on the real corpus.** On uniform-random DAGs both axes do
   essentially nothing: such graphs contain **zero** fanout-free AND→AND
   chains (out-degree-1 gates exist, but every one of them feeds a PO) and
   almost no genuine reconvergence. On a generator that draws fanins from
   recently-created gates — which is how a synthesised netlist actually looks
   — the depth axis contracts 331 → 233 nodes (30%). **Do not quote a
   compression figure for S1 from synthetic graphs.** The first real number
   has to come from a precompute run on tier0/tier1. This is also the honest
   reading of the fix in finding 1: it removed compression that was largely
   spurious.

Measured per-graph cost and peak RSS, single core, after the fixes below, on a
**370,801-node / 723,600-edge** graph — the largest size in the corpus:
`identity` 0 s, `lsh` 0.6 s, `wl` 2.7 s, `cone` 8.9 s, `convmatch` 9.1 s,
`spectral` 11.4 s, all under 1 GB. The SLURM budget is ~126 s/graph
(700k graphs / 32 shards / 96 workers) and these are worst-case sizes, so all
five clear R3 with room. Memory is the thing to watch, not time: 96 workers
× ~1 GB on the largest graphs is ~100 GB, so shard by graph size if the first
`cone` run pressures the node.

### Checked against the authors' own code (do this before citing fidelity)

Both general-purpose methods have official reference implementations, and
comparing against them found divergences that reading the papers alone had
not. Cite the repositories alongside the papers.

- **S3 — `github.com/amazon-science/convolution-matching`** (Dickens et al.,
  the authors' own release; also vendors Loukas' coarsening code, so both
  references live in one place). Three divergences found:
  1. My influence term (`sum over i in N({u})` of Theorem 1) wrongly included
     a term for the node itself. It ranges over **neighbours only**.
  2. It also has to have the merge partner's contribution removed —
     `influence[u] - w_uv / sqrt(d_v)` — which I was not doing.
  3. **Deliberate divergence, keep and document:** the reference keeps an
     internal edge as a **self-loop** on the super-node (degree unchanged,
     `self_loop_weight += 2w`), whereas `apply_merge_map` **drops** internal
     edges. So our degree is `d_u + d_v - 2w` and our self-loop weight is
     always zero. Ours is the one that matches the graph the model will
     actually be trained on; say so rather than claiming a straight port.
  After fixing 1 and 2 the cost matches the reference to **9.5e-7**.
  One further difference is scale, not correctness: the reference finds
  candidates by exact KD-tree kNN on a PCA-reduced, standardised SGC
  embedding; we use random-projection sorting, because exact kNN per graph
  over 700k graphs is not affordable.
- **S4 — `github.com/loukasa/graph-coarsening`** (`contract_variation_edges`).
  My local-variation cost had **the degree dependence inverted**: the
  reference's `||B' L_e B||_F` works out to `(d_i + d_j) * ||a_i - a_j||^2`,
  and I was *dividing* by `(d_i + d_j)` and multiplying by the edge weight
  (which in fact cancels out of `L_e` entirely). That is the kind of error
  that still produces a plausible-looking coarsening, which is exactly why it
  survived the unit tests. Two more fidelity fixes came with it: the
  preserved subspace is `U_k diag(lambda_k^-1/2)` (eigenvalue-weighted, flat
  directions weighted 1), not raw eigenvectors; and heavy-edge proximity is
  `w_ij / max(heaviest edge at either endpoint)`, not `w_ij / sqrt(d_i d_j)`.
  The corrected cost is exactly **2x** the reference (the constant in his
  norm), so the ranking — all that drives the matching — is identical,
  rank correlation **1.0000**.
- Both are now pinned by **differential tests** that transcribe the reference
  formula in the reference's own variable names and assert equality. These
  replaced two behavioural tests that had been fixture-hunted, and they catch
  mutations those could not.
- **S5 (UGC/AH-UGC)** — no public code found for either paper; the
  implementation follows the papers' description only. Worth one sentence of
  disclosure in the writeup.
- **S1** is ours, so there is nothing to compare it against.

### Defects an adversarial review caught (all fixed; recorded so they stay fixed)

- **S3 was not computing the paper's operator.** `_convmatch_costs` carried an
  extra `1/sqrt(d_i)` on the neighbour term, so the "convolution" it preserved
  was not `D^-1/2(A+I)D^-1/2 X`. Partitions differed materially (size-2
  clusters 61 vs 143 on a 1400-node graph). Now checked against a dense
  reference in `test_representation_matches_the_gcn_operator` — the previous
  behavioural tests all passed under both the right and the wrong formula,
  which is why the regression test is a numeric one.
- **S3 would not have finished.** Mutual-best matching alone merged only
  ~50 pairs per round while each round paid a full coarse-graph rebuild, so
  round count grew linearly with n: 522 s at 104k nodes, extrapolating to
  **~2.2 h for one 366k-node graph** against a ~126 s/graph budget. Fixed by
  repeating the matching pass over the still-unmatched nodes within a round
  (`_greedy_disjoint_pairs`), which is also closer to the sequential greedy
  the papers specify. Now **3.4 s at 154k nodes**, ~150× faster and near
  linear.
- **S3 double-counted the edge between two adjacent candidates.** Merging two
  adjacent nodes makes the edge between them internal — it leaves both degrees
  and drops out of both aggregates — and almost every candidate pair *is* a
  graph edge. Ignoring it overestimated the merged degree by a median 25% and
  did not perturb the cost so much as reorder it: rank correlation against an
  exact coarse-graph recompute was **0.66**, and only 14 of the 50 cheapest
  pairs were the right ones. Corrected (still O(1), via a searchsorted lookup
  of the shared edge weight) it is **0.93** and 29/50. Two tests pin it, each
  a graph where the correction changes which pair merges.
- **S5's `bin_width` is not scale-invariant** despite the standardisation.
  Retention measured 0.83 at 88 nodes, 0.032 at 55k, **0.001 at 366k**. So S5
  cannot enter a matched-compression comparison (C8) without per-graph
  calibration of `bin_width` — currently it would simply be the most
  aggressive method on every large graph, for no principled reason. Decide
  before running RQ3: either calibrate per graph, or report S5 only on its own
  compression/retention curve.

### Still open after this pass

- [ ] **`measure_summarization.py` and the CA8 receptive-field metric are NOT
      built.** `summarize_graphs.py` reports node/edge retention and wall-clock
      per shard, which covers RQ2's compression table, but nothing yet measures
      effective receptive field before vs after — so the over-squashing
      hypothesis is still asserted, not evidenced. Note a DAG longest-path
      metric will not work across all five: only `cone` guarantees an acyclic
      quotient. A k-hop fanin-cone size averaged over sampled nodes (k = number
      of encoder layers) is well defined for every method and is the metric to
      build.
- [ ] Which graphs took S4's heavy-edge fallback is not recorded anywhere. If
      most of the corpus is above 5k nodes, S4 is really "heavy-edge with a
      spectral rule on small graphs" and must be described that way.
- [ ] The empirical S2 residual-redundancy probe (d=1..4 over ~1–10k graphs)
      from the original plan still has not been run. CA1 says d=1 should find
      nearly nothing; that is a reportable dataset statistic either way.
- [ ] FRAIG leakage negative control: still not started.
- [ ] S1's `level_band`, S3/S4's `reduction_ratio` and S5's `bin_width` are set
      to plausible defaults in `config.SUMMARIZATION_PARAMS`, not calibrated.
      A matched-compression comparison (C8) needs them tuned against measured
      retention first — and for S5 that calibration is not optional (see the
      scale-dependence defect above).
- [ ] **S1's compression on the real corpus is unknown** and is the first thing
      to measure. If tier0/tier1 AIGs turn out to have few fanout-free cones,
      S1 is a weak compressor and the contribution has to lean on retention at
      low compression rather than on the compression itself. Run one shard of
      cone (`sbatch --array=32 src/shell/precompute_summarization.sh`) and read
      `_summary_stats_cone_shard000.json` before committing to the full sweep.
- [ ] `_immediate_postdominators` materialises a networkx DiGraph: +0.35 GB and
      ~2.6 s on a 366k-node graph. With `--cpus-per-task=96` that is ~100 GB if
      many workers hit large graphs at once. Watch the first `cone` precompute
      run's memory; if it bites, shard by graph size rather than rewriting it.

---

## STATUS (2026-07-28) — branch `summarization`, superseded by the box above

Everything below this box was written before implementation started. Three places
in it are now **superseded by what building the code actually revealed** — each is
marked `⚠ SUPERSEDED` in place, original text kept (not deleted) with a pointer up
here. Skim this box, then treat the marked spots as historical, not current.

**Built and merged:**
- **Backbone** (`src/data/summarization.py`) — `apply_merge_map` (the shared
  rewrite primitive Phase 0 called for), `color_refinement` (S2, both `c=∞`
  exact-WL and `c=1` bisimulation via `count_cap`), a method registry
  (`SUMMARIZATION_REGISTRY`), plus `src/data/summarize_graphs.py` (sharded,
  resumable precompute driver) and the two SLURM scripts (below).
- **S2 exact-compression model track** — a from-scratch finding, not in the
  original plan at all (see the S2 section and "Theory foundation" section
  below, both marked superseded). `src/data/exact_graph.py`
  (`fold_inversions_into_x`, `apply_exact_merge_map`) +
  `src/models/layers/gcn_exact.py` + `src/models/base_model_exact.py`: a
  **separate** model, verified to float32 precision, that makes `c=∞` genuinely
  lossless — the shipped production GCN+ is not, and can't be made so without
  giving up something (see below).

**Not started:** S1 (level-bounded cone), S3 (ConvMatch), S4 (spectral), S5 (LSH).
All four use the **normal/production model**, not the exact one — that's a
deliberate scope decision, revisit only if the exact track ends up training/testing
much better than the others.
> ⚠ **SUPERSEDED — all four are now built; see the 2026-07-31 box at the top.**
> The "normal/production model, not the exact one" scope decision still stands.

### The exact-compression finding, briefly (full derivation lives in session history)

The original plan (see "Theory foundation" below) treated `c=∞` WL-coarsening as
provably lossless for our GCN+, citing Grohe et al.'s equitable-partition result.
**That result assumes an idealized sum-aggregating MPNN. GCN+ isn't one** — three
things break it independently:
1. **GraphNorm** (`base_model.py`) is a graph-level statistic; a coarsened graph's
   statistics differ from the original's, full stop.
2. **Mean pooling** counts a super-node once regardless of how many original nodes
   it stands for.
3. **Multiplicity representation** — coarsening `k` identical incoming edges into
   one super-edge and scaling `edge_attr` by `k` feeds a scaled value *into* the
   message nonlinearity, which cannot be decomposed back into `k` separate terms.
   The fix: `edge_weight` multiplying the message *after* the nonlinearity is the
   only mechanism that reproduces `k` identical messages exactly, and it needs the
   **per-target-member** multiplicity (`coalesced_count / target_class_size`), not
   the raw coalesced total.

Fixing all three (dedicated model, no norm, size-weighted pooling, edge_weight
multiplicity, node features folded to a class representative not a sum) gives
exactness verified to float32 precision (`atol=1e-5`, residual ~1e-7) through the
**real** forward pass — including default (non-zeroed, trained-style) weights,
dropout enabled at train time, alternate JK modes, and a case where a merge is
correctly *blocked* by an inversion-pattern difference. It does **not** hold for
the production model as shipped, and no exactness-preserving normalization
replacement was found (tried PyG's per-node `LayerNorm` too — it's graph/batch
scoped despite the name, breaks it the same way GraphNorm does).

One deliberate simplification specific to the exact track: **inverter polarity
moved from a per-edge relation to a per-node feature** (count of inverted incoming
edges — AIGs have fixed fan-in per node type, so this plus type fully determines
the polarity pattern, at the cost of losing *which* specific fanin was inverted).
This trades away something the notes elsewhere call out as important (PolarGate,
C2/CA3) — **scoped to the exact track only**, on your call; S1–S5 on the normal
model keep polarity as a per-edge relation.

### Storage architecture (see "CPU/GPU BUDGET STRATEGY" below, partly superseded)

The original plan's materialize-don't-rewrite decision was right, but its proposed
layout (`/scratch-shared/$USER/aig_summary_cache/<method>/`, one `.pt` per graph)
turned out to be **inode-fatal**, not just a resource cost: at ~700k graphs, one
file per coarsened graph is ~700k inodes against ~717k of remaining scratch inode
quota — a *single* method would eat 98% of what's left. Corrected design:
- Precompute (`src/shell/precompute_summarization.sh`, genoa array job) writes
  coarsened graphs to **node-local `$TMPDIR`**, then packs each cache directory
  into a `tar.zst` shard on `/scratch-shared` — no per-graph inode ever touches
  shared storage. ~320 inodes for all five methods combined, not 3.5M.
- Training (`src/shell/train_summarization.sh`) stages shards back to
  **node-local, job-independent** disk (`/scratch-node/$USER/aig_summary/<method>`,
  deliberately *not* `$TMPDIR` — `dataset.py`'s cache signature hashes the tier
  dir paths, so a per-job path would force a full cache rebuild every run) via a
  `flock`-guarded parallel untar, ~3–8 min against a 48 h job. Verified both tiers
  land non-empty before training starts — a missing tier used to silently fall
  back to caching the raw, uncoarsened graph.

### Open items this creates (not yet decided)

- [ ] Does the exact track eventually get compared against S1–S5-on-normal-model
      as a genuine competitor, or stay a validation/ablation artifact? User's call,
      contingent on how it trains.
- [ ] `internal_edges` (a real edge between two members of the same exact-WL
      class) currently makes `apply_exact_merge_map` **raise**, not silently drop
      the edge — this can legitimately happen for structurally-repetitive AIG
      regions (duplicated bit-slices), not just literal cycles. No fix attempted
      yet; needs its own derivation (fold into a weighted self-loop?) before S2
      exact can run unattended over the full corpus.
- [ ] PE (`pe_type="level"`) is unsupported on the exact track for now
      (`pe_type="none"` only) — deferred, not evaluated as infeasible.

---

## Key papers

- **Bollen, Steegmans, Van den Bussche, Vansummeren (2023)** — *Learning GNNs using
  Exact Compression.* Collapse nodes in the same **d-step color-refinement (1-WL)**
  class (d = #layers) → **provably identical** GNN output. Lossless. Output is a
  **multigraph** with edge **multiplicities**. Graded variant `cr_c`: `c=∞` = full
  color refinement, `c=1` = **bisimulation**. Compression is data-dependent (33–93%).
- **Generale, Blume, Cochez (2022)** — *Scaling R-GCN Training with Graph
  Summarization.* RQ4 precedent for **node classification**: train R-GCN on summary,
  transfer weights back via a **node→super-node mapping**, infer on full graph.
  Outperforms from-scratch baseline (jump-start). Uses Attributes/IO summary +
  **k-forward bisimulation** (FLUID, k=3). Super-nodes carry **weighted multi-label**
  (member type frequencies). Ablation: keeping node content was **critical**.
- **Hashemi et al. (2024)** — *Comprehensive Survey on Graph Reduction.* Taxonomy:
  sparsification / coarsening / condensation. Coarsening-for-GNNs: **SCAL** (train on
  coarsened, infer directly = shared weights), **CONVMATCH** (merge nodes equivalent
  w.r.t. the GCN convolution), **Buffelli** (match embeddings across coarsening
  ratios → fixes size-shift). Kron reduction extended to **directed graphs** (Sugiyama
  & Sato 2023).
- **Chen, Saad, Zhang** — *Graph Coarsening: Sci-Computing → ML.* Spectral / AMG
  lineage, Kron, Local Variation. The generic principled baseline family.
- **Shabani, Wu et al. (2023)** — *Survey on Graph Summarization with GNNs.*
  aggregation (→ supernodes) / selection / transformation; structural vs attribute.
- **CTS-Bench (2026, arXiv 2602.19330)** — nearest competitor; graph coarsening
  trade-offs for GNNs in **clock tree synthesis**. Generic clustering → 17.2× memory /
  3× speed but **negative R² zero-shot**; calls for domain-aware coarsening. See the
  gap section — narrows our claim, strengthens our motivation.
- **ConvMatch / A-ConvMatch** (Dickens et al., WWW'24) — coarsening by **convolution
  matching**; ~95% performance at **1%** graph size. Our SOTA bar (S3).
- **UGC / AH-UGC** (2024/2025) — **LSH-based, linear-time** universal coarsening. Our
  cheap/scalable tier (S5).
- **DeepGate3 / DeepGate4** (2024/2025) — AIG scaling via *architecture* (pooling
  transformer; GAT sparse transformer, sub-linear memory, −84% inference time vs DG3).
  Contrast class: they scale the **model**, we reduce the **input**.
- **PolarGate (2024)** — polarity/functionality bottleneck in AIG GNNs → supports
  treating inverter polarity as a first-class relation (C2/CA3).

---

## R1 correction — shared weights vs mapping (IMPORTANT)

Earlier framing ("R1 = a node→super-node mapping, not shared weights") was for
**node-level** tasks. For our **graph-level regression** it's the other way round:

- A GNN's learnable weights are **size- and identity-agnostic** (inductive). Training
  GCN+ on summarized graphs and running the **same weights** on full graphs is the
  natural RQ4 mechanism — **no mapping needed**. This is "shared weights."
- **Anyone done shared weights?** Yes — **SCAL** (Huang 2021): train on coarsened
  graph, "directly use this model to inference." **Buffelli** (2022): train so
  embeddings are consistent across coarsening ratios (size-shift). Plus the whole
  inductive-GNN line (GraphSAGE, etc.).
- A **mapping** (Generale) is only needed to recover **per-node** outputs — we don't
  have per-node targets, so we don't strictly need it. Its real use for us would be a
  **warm-start / pretraining** experiment (jump-start effect), which is a *different
  question* than pure cross-state generalization.

### → Proposed: run BOTH mechanisms as two experiments (they answer different Qs)
1. **Shared-weights direct transfer** — train on summary, test on full, same weights.
   The clean RQ4 test. **Requires the input feature schema to match** between summary
   and full graphs (enriched-superset schema; full graph = size-1 super-nodes).
2. **Summary-pretrain → full-finetune (warm-start)** — Generale-style jump-start.
   Tests whether summary pretraining helps, separate from generalization.

So **R1 (real form) = feature-space compatibility** so the shared weights can ingest
both summary and full graphs. Mapping is optional (experiment 2 only).

---

## A vs B — they're the SAME family (merge them)

I split them but you were right to push back. Both are structural equivalence by
neighbor classes, parameterized by a **count-cap c** (Bollen's graded refinement):

- **Color refinement / 1-WL (c = ∞):** distinguishes by the **multiset** (counts) of
  neighbor colors — exactly what a sum/mean GNN sees → **lossless** for our GCN+.
  Finer partition → **less** compression, but exact.
- **Bisimulation (c = 1):** distinguishes only by the **set** (presence) of neighbor
  colors — ignores counts. Coarser partition → **more** compression, but **lossy**
  for a counting GNN (merges nodes the GNN *can* tell apart).

Both also have: **depth** (d or k rounds; couple to #layers = 4) and **direction**
(forward toward PO / backward toward PI / both). `k`-bisimulation's `k` = #hops (a
depth), *not* higher-order k-WL — don't conflate.

**Decision: present as ONE method — "Graded WL–Bisimulation coarsening"** with knobs
{count-cap c, depth d, direction}. Named endpoints: **exact (c=∞, WL)** and
**bisimulation (c=1)**. Cleaner and matches the literature (Bollen unifies them).

---

## THE FIVE METHODS (locked set)

Spread mirrors the sparsification section: novel-domain → provable → SOTA → classic
control → naive control. Ordered by how domain-aware they are, not by expected rank.

### S1 — Level-Bounded Cone Coarsening  ★ FULLY DOMAIN-SPECIFIC (the contribution)
AIG-native, built on levels / dominators / fanout-free structure. Two merge axes:
- **Depth axis — cascade (fanout-free chain) contraction.** Contract maximal chains of
  single-fanout gates into one super-node. *Reduces circuit depth* → directly attacks
  over-squashing (see CA9); and **cascade coarsening provably preserves acyclicity**
  → DAG-safe by construction.
- **Width axis — level-band reconvergence merge.** Merge gates within a tight level
  band (±1) sharing a common immediate dominator. Compresses parallel width while
  *locking critical-path depth* → the 32-D level PE stays exact.
- Respects PI/PO boundaries; structural-only (never functional) → sits on the safe end
  of the optimization-overlap spectrum (CA2).
- Knobs: band width, max chain length, dominator strictness.
- Risk: compression may be modest if AIGs are fanout-rich (reconvergence is common).

### S2 — Graded WL / Bisimulation Coarsening  *(general SOTA, AIG-ADAPTED)*

> ⚠ **SUPERSEDED — see "STATUS" box at top of file.** The `c=∞` "provably lossless
> for our GCN+" claim below turned out false for the *production* model
> (GraphNorm + mean pooling + a multiplicity-representation bug all break it
> independently — verified, not assumed). It's genuinely lossless only for a
> **separate, dedicated model** (`gcn_exact.py`/`base_model_exact.py`), which also
> needed polarity moved off edges onto a node feature. Original text kept below;
> read the STATUS box for the corrected picture before acting on this section.

Bollen exact compression + FLUID k-bisimulation, unified by the count-cap `c`.
- Knobs: `c` (∞ = exact WL/lossless, 1 = bisimulation/lossy), depth `d` (=4, couple to
  #layers), direction (forward toward PO / backward toward PI / both).
- **AIG adaptation:** node colors = 4-D type; **polarity as distinct relations**
  (relational refinement, cf. PolarGate/R-GCN); super-edges carry polarity
  **multiplicities** → a full graph's edge is [1,0]/[0,1] (schema superset, satisfies R1).
- `c=∞` is **provably lossless** for our GCN+ and **orthogonal to optimization by
  construction** (removes only what the GNN cannot distinguish → cannot erase the label).
- The lossless anchor of the whole study. NB: subsumes IO/k-SNAP schema summary as its
  `d=1` case — cite k-SNAP, don't run it as a separate method (it's redundant).
- Risk: strash already removed easy equivalences (CA1) → measure *residual* redundancy.

### S3 — Convolution-Matching Coarsening (ConvMatch / A-ConvMatch)  *(general SOTA, GNN-aware)*
- Merges nodes that are equivalent/similar **w.r.t. the graph-convolution operation** —
  i.e. it directly preserves the *convolution output* rather than a graph property.
- Reported: up to **95% of GNN prediction performance at 1% of original size** (node
  classification). A-ConvMatch = scalable variant.
- Role: the strongest *general* competitor — GNN-aware but domain-blind. If S1/S2 beat
  it, the domain-aware claim is earned against a real SOTA bar, not a strawman.
- Risk: designed for node classification on homophilous undirected graphs; behaviour on
  a deep polarity-carrying DAG is untested (that's part of the finding).

### S4 — Spectral / Local-Variation Coarsening  *(general classic, DOMAIN-BLIND CONTROL)*
- Pairwise contraction scored by **Heavy-Edge / Local Variation** (Loukas), preserving
  the Laplacian **spectrum** (REE guarantee); **Kron/Schur** variant (directed extension,
  Sugiyama-Sato) preserves effective resistance.
- Role: the coarsening analogue of `random_edge_dropout` — principled but logic-blind.
- **Expected to hurt**, and we now have external evidence: CTS-Bench found generic
  clustering coarsening on EDA netlists gives big memory/speed wins but **negative R²
  under zero-shot evaluation**. Predict the same here; that contrast is the point.

### S5 — Hash-Based Universal Coarsening (UGC / AH-UGC, LSH)  *(general, cheap/naive tier)*
- **Locality-sensitive hashing** over node feature+connectivity → merge colliding nodes.
  **Linear time**, no eigendecomposition, no iterative refinement.
- Role: the cheap scalable tier — the one method certain to satisfy **R3** at 3.9M
  graphs, and a naive control for "does *any* principled merging beat hashing?"
- Light AIG adaptation: hash on (type, level, fan-in/out polarity profile).
- Risk: feature-similarity ≠ logical equivalence; expect weak retention, strong speed.

### Negative control (not a 6th method) — FRAIG / functional reduction
Run as a **leakage probe**, not a summarizer. SAT-sweeping merges *functionally*
equivalent nodes = it pre-performs part of Orchestrate → should visibly destroy/leak the
optimizability label. Demonstrates empirically where the optimization-overlap boundary
lies (CA2) and justifies why S1–S2 stay structural. Also answers the "unless we optimize
it but weight it somehow?" question with data.

### Optional add-on (fold into S1 if used) — rewrite-potential super-node features
Attach **structural** rewrite-potential features to super-nodes (MFFC size, reconvergence
count, fan-out). Inputs available on any graph, *never the label* → legitimate. Treat as
an ablation on S1, not a separate method.

---

## Family F — Condensation (GCond/DosCond) — EXCLUDED (reasons)

Kept out as a primary method; note as related-work / future-work only.
- **No node correspondence** — synthesizes a new graph from scratch → nothing to map
  back to full-graph nodes → breaks the warm-start path and interpretability.
- **Label-dependent** — needs Y for gradient/distribution matching; bakes the label
  in. (Survey Table 2: condensation = ✗ interpretability, ✓ label-reliance.)
- **Architecture-tied** — gradient-matching couples to the specific GNN; known to
  generalize poorly across architectures.
- **Scope** — a whole separate literature + expensive bi-level optimization; our
  three-family framing is partitioning / sparsification / summarization, not this.
- NB: shared-weights cross-state (RQ4) is impossible for condensation (synthetic
  nodes aren't real), so it can't answer our headline question anyway.

---

## Requirements (gates — a method must satisfy these)

- **R1 — Feature-space compatibility.** Summary and full graphs share one input
  schema so the **shared weights** ingest both (enriched superset; full = size-1
  super-nodes). Mapping-back optional (warm-start experiment only).
- **R2 — Shrink while preserving predictability.** Spectrum: provably lossless (M1
  c=∞) → empirically lossy (M3). Non-negotiable that the signal survives.
- **R3 — Offline-tractable at scale.** Cached merge-maps over millions of AIGs;
  color refinement is O((n+m) log n).

## Considerations (tunable axes — methods may satisfy different subsets)

- **C1 depth alignment** — set d/k to #layers (4). Open: is optimal k = #layers?
- **C2 edge polarity** — treat as distinct relations + carry multiplicities. Ablate.
- **C3 super-node features** — carry counts/distributions (type freq, size, level
  [min,max,mean,var], rewrite-potential). Content mattered in Generale's ablation.
- **C4 boundary (PI/PO) preservation** — test with/without; keep only if it helps.
- **C5 direction of refinement** — fwd/bwd/both; changes compression AND what
  survives. Real experiment.
- **C6 multigraph vs simple** — exact compression yields a multigraph (edge counts);
  decide whether encoder ingests multiplicities or re-simplifies (lossy).
- **C7 DAG preservation** — not required by GCN (levels computed pre-merge). Prefer
  acyclic, don't block.
- **C8 matched-compression comparability** — report best ratio + one matched point.
- **C9 determinism** — preferred; seed where possible.

---

## IMPLEMENTATION PLAN (for a coding-agent session)

Order: shared infrastructure → the four off-the-shelf methods (S2, S5, S4, S3) → S1 last
(deferred by choice; needs more thinking//discussion). Each phase states its **verify**
step, per CLAUDE.md §5 goal-driven execution.

### Architecture decision — read this first
**One primitive: the merge-map.** Every method reduces to a function
`graph -> cluster: LongTensor[num_nodes]` assigning each node a super-node id. A single
shared `apply_merge_map()` does the actual rewrite. Methods differ *only* in how they
produce the cluster vector — so implement the rewrite **once** (DRY; CLAUDE.md §3).

Precedents in-repo to mirror, not reinvent:
- **Caching/index pattern:** `src/data/sparsification.py` — chunked `_sparse_<algo>*.pt`
  index files, `_SPARSE_INDEX_CACHE` keyed `(cache_dir, algo)`, `preload_sparse_index`,
  `mmap=True` loads, `CHECKPOINT_EVERY`, `update_existing_cache_with_masks`,
  multiprocessing with `_worker_initializer`. Copy this shape for merge-maps.
- **Assignment-carrying Data:** `src/data/partition_utils.py::PartitionedData.__inc__`
  (offsets `partition_id` per graph in a batch). A merge-map needs the same treatment if
  it's carried on the Data object.
- **Offline profiling script:** `src/data/measure_sparsity.py` + `shell/measure_sparsity.sh`.
- **Precompute job:** `shell/precompute_sparsification_masks.sh`.

**Key difference from sparsification:** sparsification applies a *mask* (selection);
summarization *rewrites the graph* (coalesce edges, pool features). Masks can be applied
at `get()` time cheaply; a merge rewrite is heavier → strongly prefer **precompute the
cluster vector offline, apply at `get()`**, same as the existing mask flow.

### Phase 0 — shared infrastructure  ⟵ blocks everything, do first
New file `src/data/summarization.py` (mirror `sparsification.py` layout).
NB: `src/summarize.py` at repo root is **empty** — decide whether to delete it or make it
the CLI entrypoint; do not leave both.

1. `apply_merge_map(data, cluster, num_clusters) -> Data`
   - **edges:** map `edge_index` through `cluster`, drop intra-super-node self-loops
     (or record their count), coalesce duplicates.
   - **edge_attr:** super-edge carries polarity **counts** `[#normal, #inverted]`
     (C2/CA3). A single original edge → `[1,0]` or `[0,1]`.
   - **x:** super-node carries member **type counts** `[#const,#PI,#AND,#PO]`. A single
     node → its original one-hot. (Schema superset — R1 satisfied by construction.)
   - **level PE:** pool member levels → `[min,max,mean,var]` (C4); keep the existing
     32-D projection downstream.
   - **extra:** `node_size` (member count), optionally `log1p(node_size)`.
2. Config constants → `src/config.py` **only** (single source of truth; no second
   constants module — see CLAUDE.md).
3. Measurement harness `measure_summarization.py` + SLURM script: node/edge retention,
   wall-clock, **and** mean shortest-path or effective-receptive-field before/after
   (CA8 — needed to evidence the over-squashing claim).

**Verify:** unit test in `src/unittests/data/test_summarization.py` —
`apply_merge_map(g, identity_cluster)` returns a graph **equal to `g`** (up to the
enriched schema). This identity property *is* R1; if it fails, nothing downstream is
valid.

### Phase 1 — S2 Graded WL / Bisimulation  ⟵ flagship, and the easiest real method
`color_refinement(data, depth, count_cap, direction) -> cluster`
- Iterate `depth` times: `new_color[v] = hash(color[v], multiset{(color[u], polarity) for
  u in neighbours(v)})`, where the multiset is **capped at `count_cap`** copies per
  distinct element (`c=1` → set → bisimulation; `c=∞` → exact WL).
- `direction ∈ {forward, backward, both}` selects out-/in-/both-edges (C5).
- Complexity `O((n+m) log n)`; deterministic given a stable hash → satisfies C9 and R3.

**Verify (three tests, the third is the important one):**
1. On a hand-built symmetric graph, classes match by hand.
2. `depth=0` → classes == node types.
3. **Losslessness test:** build a small AIG, run the actual GCN+ encoder on it and on its
   `c=∞` coarsening, assert the pooled graph-level output matches within tolerance. This
   empirically confirms Bollen's theorem *for our architecture* and is the single most
   valuable test in the project.

**⚠ Do the empirical probe EARLY (before Phases 2–4):** run refinement at d=1..4 over a
sample (~1–10k graphs) and report class-count reduction. Because of **CA1 (strash already
applied)**, d=1 will likely find ~nothing; the question is d≥2. This de-risks S2 and is a
reportable dataset statistic either way (CA10). *If residual redundancy is near zero,
S2 becomes a lossless-baseline result rather than a compressor — and S1/S3 carry more
weight. Know this before investing in Phases 2–4.*

### Phase 2 — S5 Hash-based coarsening (UGC / LSH)
Cheapest method; good second because it exercises Phase 0 with a totally different
cluster source. Hash `(type, level, fan-in/out polarity profile)` via LSH; colliding
nodes merge. Linear time, tunable band/width for the compression knob.
**Verify:** compression ratio responds monotonically to the LSH parameter; deterministic
under fixed seed.

### Phase 3 — S4 Spectral / Local Variation (+ Kron)
Use existing implementations where possible (Loukas' `graph-coarsening`; heavy-edge
matching is already reachable via the METIS path used in `partition.py`). Domain-blind
control — do **not** spend effort adapting it; its job is to be generic.
**Verify:** matches reference implementation on a small graph; REE reported.

### Phase 4 — S3 ConvMatch
Most involved of the off-the-shelf set (reference implementation exists). Merges nodes
equivalent w.r.t. the convolution operation.
**Verify:** reproduces reported behaviour on a small standard graph before trusting it on
AIGs.

### Phase 5 — S1 Level-Bounded Cone Coarsening  ⟵ DEFERRED (thinking/discussion first)
Do not start until the merge rule is specified. Needed decisions: cascade (fanout-free
chain) contraction rule, level-band width, dominator strictness, tie-breaking, and the
interaction between the depth axis and the width axis. Phase 0 means S1 will only need a
cluster-producing function when it's ready.

### Cross-cutting integration (after Phase 1, before any training run)
- `dataset.py`: `get_num_nodes_list()` must special-case summarization (node counts
  change post-merge — same issue node-mask sparsification already has).
- `datamodule.py` / `sampler.py`: batch plan must be built from **post-merge** node counts
  or dynamic batching will mis-size batches.
- `config.py`: `SUMMARIZATION_*` constants; add a `--summarization` flag to `train.py`
  mirroring `--sparsification`.
- **CA18 blocker:** confirm the GCN+ edge encoder can ingest edge **counts** (or
  `log1p(count)`) before relying on S2 `c=∞`; if it silently re-simplifies, the method is
  no longer lossless.

### What "Phase 0" means (plain version)
Phase 0 contains **no coarsening algorithm at all**. It is only the shared plumbing that
all five methods need: *given* a merge-map (which node goes into which super-node),
rewrite the graph — coalesce edges, pool features, keep the feature schema valid. Every
method (S1–S5) then reduces to "produce a cluster vector," and this code does the rest.
Building it first means each method is a small, self-contained function rather than five
copies of the same rewrite logic.

Deliberately **not** in Phase 0 (YAGNI, CLAUDE.md §2): the on-disk merge-map cache/index.
There is nothing to cache until a method exists → it lands in **Phase 1** alongside S2,
mirroring `sparsification.py`'s chunked-index pattern then.

### Design decisions already made (hand these to the agent; don't re-litigate)
- **`x`** → member **type counts** `[#const,#PI,#AND,#PO]` (float). A size-1 super-node
  reproduces the original one-hot exactly.
- **`edge_attr`** → super-edge polarity **counts** `[#normal,#inverted]`. A single edge
  reproduces `[1,0]`/`[0,1]`.
- **`level`** → must stay an **integer scalar per node**, because `pe_type="level"` feeds
  `ExtractPrecomputedPE`/discrete level embedding (`models/layers/positional_encodings.py`).
  Use the member **minimum** level (earliest level preserves causal ordering). Store the
  richer `[min,max,mean,var]` separately as `level_stats` for the C4 ablation — do **not**
  replace `level` with a vector in Phase 0 or the existing PE path breaks.
- **`node_size`** → member count (plus `log1p` variant if useful downstream).
- **intra-super-node edges** → dropped (they'd be self-loops); record the count as
  `internal_edges` so the information is not silently lost.
- **`src/summarize.py`** (empty, repo root) → **delete it**. The new module is
  `src/data/summarization.py`.

### Opening prompt for the coding session
See the ready-to-paste prompt at the end of this document.

Reminders for that session: tests live **only** in `src/unittests/`; SLURM scripts are
prepared, not run; and CLAUDE.md requires an **adversarial sub-agent review** of the diff
before any commit/push.

---

## CPU/GPU BUDGET STRATEGY (GPU time is the scarce resource)

Goal: **zero summarization work on the GPU node.** All coarsening happens offline on
`genoa` (CPU); the GPU job should be unable to tell it's training on coarsened graphs.

### Decision: MATERIALIZE the coarsened graphs — do not rewrite at `get()` time

> ⚠ **PARTIALLY SUPERSEDED — see "STATUS" box at top of file.** The
> materialize-don't-rewrite call below is still right. The proposed *layout*
> (one `.pt` per graph directly on `/scratch-shared`) is not — it's ~700k inodes
> per method against ~717k of remaining scratch quota, discovered only once real
> numbers were pulled. Corrected: node-local `$TMPDIR` during precompute, packed
> into `tar.zst` shards for the shared filesystem, staged back to node-local
> (job-independent) disk at train time. Original reasoning kept below, still
> correct; only the "write directly to `/scratch-shared/.../<method>/`" part
> needs replacing.

This **differs from the sparsification pattern**, deliberately:
- Sparsification stores a boolean **mask** and applies it in `get()` — that's just an
  `index_select`, genuinely cheap.
- Summarization's rewrite (scatter/coalesce edges, pool features) is **much heavier**.
  Running it per-graph, per-epoch inside dataloader workers burns CPU **on the GPU node**
  and risks starving the H100 (`NUM_WORKERS=12`, `PREFETCH_FACTOR=4`).
- **The clincher: coarsened graphs are SMALLER than the originals.** Writing them out
  costs *less* disk than the existing graph cache, and loading them is *less* I/O. There
  is no storage penalty to pay — materializing is cheaper on every axis.

So: precompute job writes **materialized coarsened `.pt` graphs** into their own cache
dir (mirror `SPARSIFICATION_REPLACE_PATH`'s separate-cache idea, e.g.
`/scratch-shared/$USER/aig_summary_cache/<method>/`). Training then points at that dir and
uses the **normal unreduced code path** — no summarization logic in the hot loop at all.
Bonus: this makes the summarized training path nearly identical to the baseline path,
which removes a whole class of bugs.

`apply_merge_map()` is still the right primitive — it just runs **offline**, once.

### Cost ceiling: ~4 s/graph on one 96-core genoa node
Relevant corpus ≈ tier0 + tier1(Orchestrate) + tier2 ≈ **~700k graphs** (we train on
Orchestrate only — do **not** precompute all four algorithms; that's a free 4× saving).
On 96 cores with the existing 8-hour wall:

| per-graph cost | CPU-seconds | wall on 96 cores | verdict |
|---|---|---|---|
| 10 ms | 7.0k | ~1 min | trivial |
| 100 ms | 70k | ~12 min | fine |
| 1 s | 700k | ~2 h | fine |
| **~4 s** | 2.8M | **~8 h** | **at the wall limit** |
| 10 s | 7M | ~20 h | needs an array job / chunking |

Reference points from the measured sparsification table: `random_edge_dropout` 26 ms,
`and_gate_only` 92 ms, `pagerank` 1.83 s, `spanning_forest` 3.52 s per graph — a **130×**
spread. Assume the same spread here.

### Expected per-method CPU cost (drives method viability = R3)
- **S2 (WL/bisimulation)** — `O((n+m) log n)`, integer hashing. Should be **ms**. Cheap.
- **S5 (LSH)** — linear, cheapest of all.
- **S1 (level/dominator)** — dominator tree is near-linear (Lengauer–Tarjan). Cheap.
- **S3 (ConvMatch)** — iterative, moderate.
- **S4 (spectral / Local Variation)** — ⚠ **the CPU risk**: eigendecomposition per graph.
  Could blow the budget on large AIGs. Mitigations: cap by graph size, use Kron/heavy-edge
  instead of full spectral, or run S4 on a **stratified subsample** and report it as a
  reduced-scope control (it's only the domain-blind control — degrading its coverage is
  acceptable if disclosed).

### Rules for the precompute jobs
1. **Profile before committing.** Extend `measure_sparsity.py` into a summarization
   profiler; run each method on ~1k stratified graphs, extrapolate with the table above,
   *then* decide whether to launch the full job. Do this **before** implementing S3/S4.
2. **Make jobs resumable.** Reuse the `CHECKPOINT_EVERY = 50_000` atomic-index idea so a
   job killed at 7h59m doesn't restart from zero. Skip-if-exists sentinels like
   `warmup_train_cache.sh` already uses.
3. **Compute once, reuse everywhere.** Merge-maps are deterministic per
   `(method, params, seed)` → key by signature hash and reuse across all seeds, Optuna
   trials, and both the RQ3 (matched-state) and RQ4 (cross-state) experiments. Never
   recompute per training run.
4. **Chain, don't idle the GPU:**
   `PID=$(sbatch --parsable src/shell/precompute_summarization.sh)` then
   `sbatch --dependency=afterok:$PID src/shell/train.sh`.
5. **Keep the existing knobs:** `--partition=genoa`, `--cpus-per-task=96`,
   `--constraint=scratch-node`, `OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1` (correct — avoids
   thread thrash under multiprocessing), chunked + `mmap=True` index loads.
6. **Array-job the expensive methods** by shard if profiling says >4 s/graph, rather than
   raising `--time`.

### Where the GPU actually wins
Batching is **node-budgeted** (`MAX_TOTAL_NODES_PER_BATCH = 3e6`), so fewer nodes per
graph ⇒ more graphs per batch ⇒ **fewer steps per epoch** and lower peak VRAM. That is
precisely the RQ2 measurement — so the GPU saving *is* a result, not just an
optimization. Report GPU-hours per epoch alongside VRAM.

---

## Open questions / TODO

- [ ] Empirical: how much WL/bisimulation redundancy do AIGs actually have? (Decides
      whether M1 is a real compressor or mainly a lossless-baseline result.)
- [ ] Forward vs backward vs both — measure compression + retention per direction.
- [ ] Shared-weights vs warm-start — run both; report separately.
- [ ] M5: define the leakage boundary for rewrite-potential features.
- [ ] Confirm GCN+ edge encoder can ingest edge multiplicities (C6) or decide to
      re-simplify.
- [ ] Cite the two reference **implementations** as well as the papers:
      `github.com/amazon-science/convolution-matching` (S3, authors' own) and
      `github.com/loukasa/graph-coarsening` (S4). Both were used to verify
      fidelity; see the comparison section at the top. No public code exists
      for UGC/AH-UGC (S5) — disclose that S5 follows the paper description
      only.
- [ ] Citations to add: Bollen 2023, Generale 2022, Hashemi 2024, Chen-Saad-Zhang,
      Shabani 2023, Loukas 2019, Tian 2008 (SNAP/k-SNAP), Huang 2021 (SCAL),
      Buffelli 2022, Dickens 2023 (CONVMATCH), Dorfler-Bullo 2012 + Sugiyama-Sato
      2023 (Kron).

---

## AIG-native reduction — what already exists (domain machinery)

The generic graph-summarization papers don't know AIGs already ship with
equivalence-based reduction — and it's often **stronger** (functional, not just
structural). This is the bridge that makes the section AIG-specific rather than
"generic coarsening applied to circuits."

- **Structural hashing (strash).** Merges AND gates with identical (fan-in, polarity)
  pairs — the AIG-native 1-hop structural-equivalence merge. Lossless, cheap. **ABC
  already strashes**, so our dataset graphs are *already* strash-reduced.
  → Consequence: M1's trivial endpoint (d=0 / one-level) is basically strash, so some
  "free" equivalence is *already gone* → **must measure residual WL-redundancy** or M1
  may look like it barely compresses. (Real risk, flagged.)
- **FRAIG / SAT-sweeping / functional reduction** (Mishchenko et al., Berkeley).
  Random simulation + SAT proves nodes **functionally** equivalent and merges them;
  semi-canonical. This is the AIG-native *exact compression*, at the **function**
  level — stronger than WL/bisimulation (which are structural).
  → **This is the non-overlap tension made precise:** functional reduction removes the
  same redundancy that *synthesis optimization removes*. Fraiging **pre-does part of
  the Orchestrate optimization → leaks/erases the label.** So FRAIG is a *bad*
  summarizer for optimizability prediction — but a **perfect negative control**: it
  demonstrates that optimization-overlapping reduction destroys the target. Use it to
  draw the leakage boundary for M5.
- **Cut-based / windowing / supergate coarsening.** k-feasible cut enumeration induces
  logic cones; collapse a cut's cone into a super-node. Supergates (tech mapping) =
  precomputed small gate clusters. **MFFC contraction** (already in the thesis) is a
  special case. Domain-native, but cuts/MFFCs are prime **rewrite targets** → same
  optimization-overlap risk as FRAIG (softer).
- **DAG-aware handling.**
  - Coarsening a DAG **along cascades preserves acyclicity** → there exist
    guaranteed-acyclic DAG coarsenings (upgrades C7 from "don't block" to "achievable
    if wanted").
  - DAG-GNNs (DAGNN, D-VAE) respect topological order; **DCN/PDCN decouples model
    complexity from graph size** (size generalization = RQ4 flavour). Architecture
    alternative — note, but out of scope (we use GCN+).

## The research gap (positioning — use in intro/related work)

How the field scales GNNs on AIGs *today*:
1. **Extract small subcircuits** (30–3k gates) and train on those — crude sampling,
   loses global structure. (This is the de-facto "reduction" in most circuit-GNN
   papers, incl. DeepGate.)
2. **Scale the architecture** — DeepGate3 (pooling-transformer over subcircuits),
   DeepGate4 (sparse attention, sub-linear memory, fights over-squashing). Changes the
   *model*, not the *input*.

→ **Nobody has systematically applied graph summarization/coarsening as *input*
reduction to whole AIGs for GNN training, nor measured which reduction preserves a
downstream regression label.** Generic summarization papers (Bollen, Generale, Loukas)
ignore strash/FRAIG; EDA papers (FRAIG, DeepGate) don't frame their reductions as
GNN-input summarization. **The contribution is the bridge**: bring the generic
exactness framework (WL/bisimulation, *provable*) together with AIG-native equivalence
(strash/FRAIG, *functional*) and evaluate label retention across the spectrum.

### ⚠ Nearest competitor — CTS-Bench (arXiv 2602.19330, Feb 2026). READ BEFORE WRITING.
*"Benchmarking Graph Coarsening Trade-offs for GNNs in Clock Tree Synthesis."*
Khadka, Roxy, Ahmed. Submitted 22 Feb 2026, cs.LG (preprint).

**Correction to an earlier overstatement in these notes:** it narrows the gap **far
less** than first assumed. Having read the method section, it benchmarks **exactly one
bespoke, physical-design-specific clustering heuristic** — it does *not* survey or
compare the graph-coarsening literature at all.

**Their actual "coarsening" = a 3-step custom heuristic (~13.3× node compression):**
1. **Atomic cluster formation** — BFS outward from flip-flops
2. **High-spread filtering** — on **spatial** standard deviation (σ > 0.05)
3. **Gravity-vector-aligned merging** — cosine similarity > 0.9 + Manhattan-distance
   constraint

→ This is **spatial/placement clustering**: it consumes **physical XY coordinates**
(spatial variance, gravity vectors, Manhattan distance) from a **post-placement**
design. **It is not even applicable to an AIG**, which has no coordinates — a
logic-level AIG is pre-physical. Their "generic graph clustering" label is generous to
itself.

**What they did NOT do** (i.e. still open, and squarely ours): METIS, Louvain,
spectral/Local Variation, heavy-edge matching, Kron, ConvMatch, WL/color refinement,
bisimulation, LSH/UGC — **none** benchmarked. No exactness/lossless framing anywhere.
No structural-equivalence angle. One task, one label, one clustering method.

- **Different EDA stage/graph/task**: clock tree synthesis, **post-placement gate-level
  netlists**, clock-skew prediction. **Not AIGs, not logic synthesis, not
  optimizability.** Our claim must be scoped to *logic synthesis / AIG / optimizability*,
  not "EDA" generally.
- **Setup**: 4,860 converged PD solutions, 5 architectures; GNNs = **GCN, GraphSAGE,
  GATv2**; metrics = peak VRAM, training throughput, **MAE and R²**.
- **Findings that support our premise**: generic clustering coarsening gave **up to
  17.2× GPU memory reduction and 3× training speedup**, but degraded accuracy — with
  **negative R² under zero-shot evaluation**. It concludes generic clustering removes
  task-essential structure and explicitly **calls for domain-aware ("CTS-aware")
  coarsening strategies** — which is precisely our thesis, one stage earlier.
- **Consequences for us:**
  1. Cite as the strongest motivation that **domain-aware coarsening is needed** in EDA.
  2. Their **zero-shot negative R²** is a direct warning for **RQ4** — expect cross-state
     transfer to be hard. If our domain-aware methods transfer at all, that's a *result*,
     and we now have a citation showing the generic case fails.
  3. It's a **benchmark-paper template** — mirrors our RQ2/RQ3 trade-off framing; use its
     structure (and its memory/speed numbers) as a comparison point.
  4. Reframe our novelty precisely: *first systematic study of graph reduction for
     **AIG/logic-synthesis** GNN regression, and first to pair generic coarsening with
     **AIG-native equivalence** (strash/FRAIG) as an exactness spectrum.*
  5. **The MAE-vs-R² lesson (important, methodological).** Their skew MAE barely moved
     (0.16 → 0.17) while **R² fell below 0**. Global/absolute error looked fine while
     explained variance collapsed. → **Never judge a reduction on RMSE/MAE alone**; a
     method can preserve mean error and destroy all discriminative power. Reinforces
     reporting RMSE **and** R² **and** Spearman together (we already do — now we have a
     citation for *why*).

**Net position (what we can still honestly claim to be first at):**
- First to benchmark **multiple graph-coarsening families** (spectral, WL/bisimulation,
  convolution-matching, hashing, domain-specific) for GNN training in EDA at all —
  CTS-Bench benchmarks **one** bespoke spatial heuristic.
- First on **AIGs / logic synthesis / optimizability regression**.
- First to bring **provable (lossless) compression** into an EDA GNN setting.
- First to connect **AIG-native equivalence** (strash/FRAIG) to graph summarization.
CTS-Bench costs us only the unqualified phrase *"first coarsening study in EDA."*
Everything specific remains ours — and it now provides external evidence that the
problem is real and that domain-aware coarsening is the needed answer.

Papers to cite here: Mishchenko FRAIG (2005/2007), DeepGate (2021) / DeepGate3 (2024) /
DeepGate4 (2025), PolarGate (2024), HOGA, FuncGNN, DAGNN / D-VAE, DCN/PDCN (2025),
DAGNN-RE (2024).

---

## Why summarization could BEAT sparsification (the "save the thesis" argument)

This is the strongest angle and it's literature-backed. Summarization is **not just
memory reduction** — coarsening **contracts paths**, which mitigates **over-squashing**.

- **Over-squashing**: a fixed-size message-passing GNN cannot carry long-range signal;
  info from many/distant nodes is crushed into one vector. AIGs are **very deep**
  (`config.MAX_DEPTH ≈ 25k`), so a **4-layer** GCN+ sees only a tiny fraction of a
  deep circuit → chronic over-squashing / receptive-field starvation.
- **Hierarchical coarsening expands the receptive field** and improves long-range
  propagation (shown on the Long-Range Graph Benchmark). Contracting a chain of gates
  into a super-node **shortens the path** the signal must travel.
- **Therefore**: sparsification *removes edges* → can **worsen** propagation;
  summarization *contracts paths* → can **improve** it. That's a principled reason
  summarization can **raise** accuracy on a deep DAG, not merely trade it for memory —
  exactly the "it could perform best" hope. **Make this an explicit hypothesis (H:
  coarsening improves effective receptive field → better retention than sparsification
  at matched compression).**
- Framing bridge: this connects summarization to the **graph-rewiring / over-squashing**
  literature (Ricci-curvature rewiring, LRGB), which no AIG paper has used.

## Theory foundation for M1 losslessness (equitable partitions / orbits)

> ⚠ **SUPERSEDED (scope narrowed) — see "STATUS" box at top of file.** The
> permutation-equivariance argument below is correct for the class of GNN it
> assumes, but "GNNs are permutation-equivariant → lossless" quietly assumed a
> plain sum-aggregating MPNN. GraphNorm, mean pooling, and how multiplicity gets
> represented in message passing are all extra structure real architectures add
> that this argument doesn't cover — GCN+ as shipped has all three, and isn't
> lossless as a result. Still true, now demonstrated rather than assumed, for a
> dedicated bias-and-architecture-matched model built to actually satisfy the
> argument's preconditions (`gcn_exact.py`). Keep this section for the citation
> and the orbit-counting idea; don't cite it as applying to GCN+ directly.

Tightens *why* WL-coarsening is lossless, with citable formal basis:

- The **coarsest equitable partition** is exactly what **color refinement computes**.
  Automorphism **orbits** form an equitable partition. GNNs are
  **permutation-equivariant** → they output **identical** representations for nodes in
  the same orbit → quotient-by-equitable-partition is **lossless**. (Grohe et al.,
  *Dimension Reduction via Colour Refinement*; equitable-partition/orbit literature.)
- **AIG-specific symmetry sources** (→ why AIGs may compress well under M1):
  interchangeable AND-gate fan-ins (input symmetry), **replicated datapath bit-slices**
  (adders/multipliers/registers = many isomorphic cones), repeated standard sub-logic.
  These create real orbits a generic random graph lacks. Worth *measuring* — orbit
  count / equitable-partition size is a structural statistic of the dataset.

## Task grounding & baselines (feeds §baselines + related work)

- **OpenABC-D** (NYU-MLDA) — the reference large-scale ML4EDA dataset; graph-level
  labels incl. **"% of nodes optimized"** — essentially **our optimizability label**.
  Cite as precedent/positioning even though our dataset is custom. Also **OpenLS-DGF**,
  and *"Towards the Imagenets of ML4EDA"* for benchmark framing.
- **QoR-prediction precedents**: Transformer(recipe)+GraphSAGE(circuit) joint model
  (arXiv 2207.11437); **LOSTIN** (GNN + **super-node** to encode the synthesis
  sequence — note: super-node used for *temporal* recipe, orthogonal to our structural
  super-nodes but a nice terminological tie).
- **Standard GNN baselines** for graph-level circuit regression: **GCNConv,
  GraphSAGE, GINConv** — use as the naive-model baselines in
  §`sec:results:rq1:baselines` alongside mean/median/size-only predictors.
- Motivation stat: best synthesis recipes across designs overlap **< 30%** → optimal
  sequence is design-dependent → data-driven prediction is worthwhile.
- Recent representation-learning context: DeepGate2/3/4, PolarGate, **Masked Gate
  Modeling / Verilog-AIG Alignment** (2025) — the "what circuit embeddings exist" line.

---

## Considerations (consolidated, at the bottom as requested)

Generic (from earlier, C1–C9) **plus** AIG-specific:

- **CA1 — Strash already applied (VERIFIED in this repo).** `strash` runs in **every**
  optimization command template (`data/creation/automate_bulkOptimization.py:15,19,22,25`)
  and again when tier0 graphs are built (`automate_bulkSynthesis.py:115`). So our AIGs
  are **already structurally hashed**: no two AND gates share an identical
  (fanin-pair, polarity) signature. Structural hashing *is* the trivial/1-hop case of
  structural equivalence → **S2 at depth 1 will find almost nothing.** The open question
  is whether **deeper** refinement (d = 2,3,4) still finds mergeable classes. Measure
  before believing S2 compresses. (This is the single biggest risk to S2.)
- **CA1b — Possible label confound (LOW PRIORITY — document, do not fix).**
  `generate_csv.py:46-49` computes `optimizability = (t0_nodes - t1_nodes)/t0_nodes`
  where `t0_nodes` comes from `stats[0]` = the **first** `print_stats`, which runs
  **before** `strash`. If strash were non-trivial there, part of the label would be
  credit for structural hashing rather than for Orchestrate.
  **Decision: not regenerating the dataset — infeasible on time/compute, and correctly
  out of scope.** This is a *disclosure* item, not a blocker: the label is defined
  consistently across every graph and every method, so all comparisons in RQ2/RQ3/RQ4
  remain internally valid regardless.
  **If ever checked, it costs minutes, not compute:** the existing logs only hold two
  `print_stats` (pre-strash, post-opt), so it needs a handful of ABC calls on ~20 tier0
  files — `abc -c "read F; print_stats; strash; print_stats"` — comparing node counts.
  No rerun of the pipeline. Outcome either way is one sentence in Limitations, never a
  code change.
- **CA2 — Optimization overlap is a spectrum.** FRAIG (functional) > cut/MFFC
  (semi-functional) > WL/bisimulation (structural) > level-band (M4). The *more
  functional* the merge, the *more it leaks the label*. Design methods to sit on the
  structural end; use FRAIG as a negative control to prove the point.
- **CA3 — Relational polarity is native.** AIG edges already carry inverter polarity;
  treat as distinct relations everywhere (matches PolarGate's finding that polarity is
  a functionality bottleneck for AIG GNNs).
- **CA4 — DAG acyclicity is achievable**, not just tolerable (cascade coarsening).
  Decide whether to guarantee it or let M1 produce a multigraph with cycles.
- **CA5 — Level PE is domain-native.** Merges within tight level bands (M4) keep the
  level PE exact; cross-level merges (FRAIG/cut) blur it. Ties C4↔M4.
- **CA6 — Baseline framing.** Consider whether to contrast *input reduction* (this
  thesis) against *architectural scaling* (DeepGate4) as related work — likely a
  paragraph, not an experiment, but it sharpens the "why input reduction" argument.
- **CA7 — Negative-result value.** Even if domain-aware summarization *loses* to
  sparsification on retention, the finding "functional overlap destroys the label,
  structural coarsening preserves it" is a publishable, thesis-carrying result. The
  spectrum is the contribution, not any single winner.
- **CA8 — Receptive-field metric.** Because the headline claim is "coarsening improves
  long-range propagation," *measure* it: effective receptive field / mean
  shortest-path before-vs-after, or commute time. Report alongside compression so the
  over-squashing argument is evidenced, not asserted.
- **CA9 — Depth vs layers mismatch is the motivation.** `MAX_DEPTH ≈ 25k` vs 4 layers
  is the concrete over-squashing gap. Summarization that shrinks *depth* (path
  contraction) matters more here than one that shrinks *width* (parallel merges).
  Prefer/measure depth-reducing merges (chains) over width-only merges.
- **CA10 — Orbit/equitable-partition statistics.** Report the dataset's
  equitable-partition size / orbit count — it upper-bounds M1's lossless compression
  and is itself a structural finding about AIG regularity (datapath repetition).
- **CA11 — Label parity with OpenABC-D.** Our "% node reduction" ≈ OpenABC-D's "% nodes
  optimized" — state the correspondence so results are comparable/positioned, and so a
  reviewer sees the task is established, not invented.
- **CA12 — Super-node term collision.** "Super-node" already means the *recipe*
  encoder in LOSTIN; we use it structurally. Disambiguate in the writeup to avoid
  confusion.
- **CA13 — Scope the novelty claim tightly.** After CTS-Bench, never claim "first
  coarsening study in EDA." Claim: *first for AIG / logic synthesis / optimizability
  regression, and first to bridge generic coarsening with AIG-native equivalence.*
- **CA14 — Expect RQ4 to be hard.** CTS-Bench reports **negative R² zero-shot** for
  generic coarsening. Plan for the possibility that cross-state fails for S3–S5 and
  succeeds only for S1–S2 — that asymmetry *is* the headline result, so instrument for
  it (report matched-state and cross-state side by side per method).
- **CA15 — Two transfer mechanisms, two experiments.** Shared-weights direct transfer
  (clean RQ4) vs summary-pretrain→full-finetune warm-start (Generale jump-start). Don't
  conflate; they answer different questions.
- **CA16 — Method-family balance is deliberate.** S1 domain-specific / S2 adapted-SOTA /
  S3 SOTA / S4 classic control / S5 naive control. If a method is dropped for time,
  drop from the control end (S5), never S1/S2 — the contribution lives there.
- **CA17 — Beat a real bar.** S3 (ConvMatch) is the honest SOTA competitor. Domain-aware
  claims only count if measured against it at matched compression, not against S4/S5.
- **CA18 — Multigraph support is a prerequisite, not a detail.** S2 at c=∞ *requires*
  edge multiplicities. Confirm the GCN+ edge encoder can ingest a count (or a log-count)
  before committing to S2 — otherwise re-simplification silently makes it lossy.
- **CA19 — Report MAE/RMSE *and* R² *and* Spearman, always.** CTS-Bench: MAE 0.16→0.17
  (looks fine) while R² went **negative**. A reduction can preserve average error and
  destroy explained variance. Our Pareto front must plot an explained-variance metric,
  not just error, or it will flatter bad methods.
- **CA20 — Coordinate-free is a feature.** CTS-Bench's clustering needs placement XY;
  AIGs have none. Every method we pick must be **purely topological/functional** — which
  they are. Worth one sentence in the writeup: reduction at the *logic* level generalises
  across physical implementations, since it never sees placement.
- **CA21 — Watch for follow-ups.** CTS-Bench (Feb 2026, cs.LG preprint, not obviously
  peer-reviewed yet) explicitly calls for "CTS-aware coarsening." Someone will answer
  that call. Re-check arXiv before submission for (a) CTS-aware coarsening follow-ups and
  (b) anyone extending it to logic synthesis / AIGs.

---

## READY-TO-PASTE PROMPT — next coding session (Phase 0)

> **Context.** This repo trains a GNN (GCN+) to regress AIG "optimizability". I am adding a
> third graph-reduction family — **summarization/coarsening** — alongside the existing
> partitioning and sparsification. Before writing this code, read:
> - `IV_Gardner___Master_AI_Thesis_Outline/summarization_notes.md` — specifically the
>   **IMPLEMENTATION PLAN** section (Phase 0, and the "Design decisions already made" list).
> - `src/data/sparsification.py` and `src/data/partition_utils.py` — match their style,
>   naming, and structure. Do not invent a new pattern.
>
> **Scope: Phase 0 ONLY. Do not implement any coarsening algorithm** (no WL, no
> bisimulation, no spectral, no hashing). Phase 0 is purely the shared rewrite primitive
> that all five future methods will call. Do **not** build the on-disk cache/index yet —
> that lands in Phase 1 when the first real method exists.
>
> **Important execution context:** `apply_merge_map()` will run **offline on a CPU-only
> SLURM node** (`genoa`) inside a multiprocessing pool, over ~700k graphs — never on the
> GPU node and never in the training hot loop (GPU time is the scarce resource here). So
> it must be: a **pure function** (no global/module state), **CPU-only** (no `.cuda()`, no
> device assumptions), **picklable-friendly**, and allocation-light. Prefer vectorised
> `torch` scatter/`coalesce`-style ops over Python loops over nodes/edges.
>
> **Tasks**
> 1. Delete `src/summarize.py` (it is empty and would shadow/confuse the new module).
> 2. Create `src/data/summarization.py` with:
>    `apply_merge_map(data: Data, cluster: LongTensor, num_clusters: int) -> Data`
>    - map `edge_index` through `cluster`; drop intra-super-node self-loops; coalesce
>      duplicate super-edges
>    - `x` → member type counts `[#const,#PI,#AND,#PO]` (float)
>    - `edge_attr` → super-edge polarity counts `[#normal,#inverted]`
>    - `level` → **integer scalar** per super-node = member **min** level (must stay an int
>      tensor: `pe_type="level"` feeds a discrete level embedding via
>      `ExtractPrecomputedPE`; do not turn it into a vector)
>    - `level_stats` → `[min,max,mean,var]` stored separately for a later ablation
>    - `node_size` → member count; `internal_edges` → count of dropped intra-cluster edges
> 3. Add any needed `SUMMARIZATION_*` constants to `src/config.py` **only** (single source
>    of truth — there is no `constants.py`).
> 4. Add `src/unittests/data/test_summarization.py`. The **critical** test:
>    `apply_merge_map(g, torch.arange(g.num_nodes), g.num_nodes)` must return a graph
>    equivalent to `g` — same edges, `x` equal to the original one-hot, `edge_attr` equal to
>    the original one-hot, `node_size` all ones, `level` unchanged. This identity property
>    is the foundation of cross-state inference (RQ4); if it fails nothing downstream is
>    valid. Also test: a simple hand-built merge (2 nodes → 1) produces the expected counts,
>    coalesced edges, and `internal_edges`.
>
> **Verify**
> ```
> PYTHONPATH=src pytest src/unittests
> ruff check src
> ```
> Suite is currently green (256 passed, 5 skipped) and ruff is clean — keep both true.
>
> **Constraints**
> - Tests live **only** in `src/unittests/` (mirroring the `src/` layout). Never create a
>   root-level `tests/` dir or root-level `test_*.py`.
> - Everything runs on a SLURM cluster: prepare/edit job scripts, do not execute training.
> - Follow CLAUDE.md: YAGNI (no speculative abstraction), DRY, surgical changes only.
> - Before any `git commit`/`git push`, spawn a fresh adversarial sub-agent to review the
>   diff with no prior context, and address what it flags.
