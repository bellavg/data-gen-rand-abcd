# Summarization / Coarsening — working notes

Scratch pad for the summarization half of the reduction study. Not thesis prose —
decisions, open questions, and per-method notes. Feeds
`sections/3-methodology.tex` §`sec:method:reduction:summarization`.

Task reminder: target is **graph-level regression** (one optimizability scalar per
AIG), trained on Orchestrate. This matters a lot for R1 (see below).

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

## Candidate methods (lossless → lossy spectrum)

**M1 — Graded WL–Bisimulation coarsening**  *(flagship; Bollen, Generale/FLUID)*
- Merge nodes with equal d-step relational neighborhood; knobs c (∞=exact WL,
  1=bisimulation), d (=4), direction (fwd/bwd/both).
- AIG: node color = 4-D type; relations = 2 polarities (relational refinement).
  Super-edges carry polarity **multiplicities** → full-graph edge = [1,0]/[0,1]
  (schema superset). Level PE preserved (WL-equiv nodes share level structure).
- c=∞ is **orthogonal to optimization by construction** (removes only
  GNN-indistinguishable redundancy → cannot erase the label signal).
- Risk: compression is data-dependent; if AIGs have little WL-redundancy it barely
  shrinks — but *measuring that redundancy is itself a finding.*
- Open: does forward or backward refinement compress AIGs more? (arxiv fwd 33% vs
  inv 66% shows direction dominates.)

**M2 — IO / Attribute-schema summary**  *(cheapest; Tian SNAP/k-SNAP, Campinas)*
- Group by (node type, fan-in polarity multiset, fan-out polarity multiset). 1-hop.
- Trivially offline-tractable; k-SNAP gives a resolution knob.
- Risk: 1-hop, structure-blind beyond neighbors; expect it to trail M1.

**M3 — Spectral / Local-Variation coarsening**  *(the generic control; Loukas, Kron)*
- Pairwise contraction scored by Heavy-Edge / Local Variation; preserves Laplacian
  spectrum (REE). Kron/Schur variant (directed, Sugiyama-Sato) preserves eff.
  resistance.
- This is the domain-BLIND control (coarsening analogue of random_edge_dropout).
- Risk (intended): severs causal cones; expected to hurt → makes domain-aware look
  strong.

**M4 — Level-bounded reconvergence coarsening**  *(custom / domain-aware contribution)*
- Merge only within tight level bands sharing a common dominator; preserves level-PE
  and causal cones. Position explicitly against M3.

**M5 — Optimization-aware weighted coarsening**  *(your "weight it somehow")*
- Merge more aggressively but attach **structural rewrite-potential** features to
  super-nodes (MFFC size, reconvergence count, fan-out). These are *inputs available
  on any graph, never the label* → legitimate, not leakage.
- Open: where's the line between a useful structural feature and label leakage?

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

## Open questions / TODO

- [ ] Empirical: how much WL/bisimulation redundancy do AIGs actually have? (Decides
      whether M1 is a real compressor or mainly a lossless-baseline result.)
- [ ] Forward vs backward vs both — measure compression + retention per direction.
- [ ] Shared-weights vs warm-start — run both; report separately.
- [ ] M5: define the leakage boundary for rewrite-potential features.
- [ ] Confirm GCN+ edge encoder can ingest edge multiplicities (C6) or decide to
      re-simplify.
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
(strash/FRAIG, *functional*) and evaluate label retention across the spectrum. That
positioning is novel and defensible.

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

- **CA1 — Strash already applied.** Measure *residual* WL/functional redundancy before
  claiming M1 compresses; the easy equivalences are gone.
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
