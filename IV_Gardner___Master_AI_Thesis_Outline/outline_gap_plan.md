# Outline gap plan — what to fill, what drifted

> **STATUS: sections A–G have been applied to the `.tex` files** on branch
> `claude/latex-outline-review-46587d` (uncommitted). Everything writable without run results
> is written; everything blocked carries an inline `% BLOCKED` or `% TODO` marker naming what
> is needed. The document builds clean with all citations and cross-references resolving.
>
> What remains is listed at the bottom under **H. Still open after the pass**. Sections A–G
> below are kept as the record of what was found and why each decision was made.


Review of `sections/*.tex` against `main`, `origin/summarization`,
`origin/baselines/openabc-synthnet-hoga`, and
`claude/aig-graph-summarization-coarsening-8a9ea6` (S6 mffc).

Two separate problems:

- **A. Empty stubs** — headings with no prose. Mostly known and marked.
- **B. Drift** — prose that is *there* but no longer matches the code on your branches.
  This is the dangerous half, because it reads as finished.

---

## A. Where text needs filling

### A1. Chapter 2 is 100% empty — the largest single hole

`sections/2-related-work.tex` is 168 lines of headings and zero sentences: ~15
Preliminaries subsubsections, ~15 Related Work subsubsections. `bibliographies/references.bib`
has **2 entries**, both leftover template refs (Gruber 1995, Viola 1997) — neither cited.

Every term the Introduction leans on is promised to a Preliminaries label that has no text
behind it. `sec:intro:terms` is an explicit map of 13 terms → 13 target labels; all 13 targets
are empty.

**The citations already exist in prose form** in `summarization_notes.md` (§Key papers,
§The research gap, §Task grounding & baselines) — ~20 papers with one-to-three-line
characterisations. That is a lit-review draft sitting outside the thesis.

### A2. Chapters 5 and 6 are 100% empty

`5-discussion.tex` and `6-conclusion.tex` are structure-only. That is correct for now —
both are genuinely blocked on results — but Discussion §Limitations and §Ethical
Considerations are *not* blocked and can be written today.

### A3. Empty stubs in Ch. 1 and 3 (blocked vs writable)

**Writable now:**

| Section | Label |
|---|---|
| Introduction (context) | `sec:intro:context` |
| Problem Statement & Research Gap | `sec:intro:gap` |
| Scope & Delimitations + 6 subsubsections | `sec:intro:rqs:scope` |
| Thesis Outline | `sec:intro:outline` |
| Source Circuits | `sec:method:data:sources` |
| Random Structural Transformation | `sec:method:data:generation:transform` |
| Target Synthesis Algorithms | `sec:method:data:generation:algorithms` |
| PE Precomputation | `sec:method:data:representation:pe` |
| Research Design | `sec:method:experiment:design` |
| Hyperparameter Tuning | `sec:method:experiment:hp` |
| Training Configuration | `sec:method:experiment:training` |
| Hardware & Software Environment | `sec:method:experiment:hardware` |
| Evaluation Metrics + 4 subsubsections | `sec:method:experiment:metrics` |
| Reproducibility | `sec:method:experiment:reproducibility` |

**Blocked on runs:** `sec:intro:contributions:results`, `sec:method:data:label:distribution`,
`sec:method:data:statistics`, all of Ch. 4.

### A4. Front matter is still the UvA template

`msc_thesis.tex` ships with placeholder title ("Master AI thesis template: A tale of
computers"), author "Student van Naam", student number 314159265, `\examiner{Pr. E. Xaminer}`,
`\dailysupervisor{M. Entor}`, and a lorem ipsum abstract. `sections/7-appendix.tex` is lorem
ipsum. `sections/0-list-symbols.tex` is fully commented out.

---

## B. Drift — prose that no longer matches your branches

Ranked by how wrong it is.

### B1. The whole summarization section is obsolete ★ biggest

`sec:method:reduction:summarization` (methodology lines 137–212) lists **SNAP**, approximate
$k$-bisimulation, **Heavy Edge Contraction**, Level-Bounded Reconvergence, then a "wider
candidate pool" of 10 more, then MFFC and $K$-feasible cuts. It is framed as undecided.

`origin/summarization` has a **locked, implemented, unit-tested set of 6** in one registry
(`src/data/summarization.py`, `SUMMARIZATION_REGISTRY`), with production parameters in
`config.SUMMARIZATION_PARAMS` and a **scope decision: run three, cite five**.

| notes id | registry key | role | in the .tex? |
|---|---|---|---|
| — | `identity` | zero-compression control / test fixture | no |
| S1 | `cone` | domain-specific — **the contribution** | yes, but described wrongly (B2) |
| S2 | `wl` | graded WL/bisimulation, **lossless anchor** | split across 3 unrelated bullets |
| S3 | `convmatch` | A-ConvMatch, the general SOTA bar | **absent** |
| S4 | `spectral` | local variation, domain-blind control | one line in the "wider pool" |
| S5 | `lsh` | UGC hashing, cheap naive control | **absent** |

Decisions the .tex does not reflect:

- **Only `wl`, `cone`, `convmatch` get runs.** `spectral` and `lsh` move to Related Work,
  staying implemented and registered. Reason on record: the claim is about coarsening *for
  GNN training*, so the bar must be the GNN-aware SOTA, not the spectrum-preserving classic.
- **SNAP is explicitly ruled out** — subsumed by S2 at `d=1`. Cite k-SNAP, don't run it.
  The .tex still presents it as the lead candidate with its own PE and boundary rules.
- **Heavy Edge is not a method** — it is S4's fallback variant above `max_spectral_nodes=5000`.
- **Condensation (GCond/DosCond) is excluded**, with four written reasons. Not mentioned in the
  .tex at all, and a reviewer will ask.
- **FRAIG is a leakage negative control**, not a summarizer. Not in the .tex.
- **MFFC** exists only on `claude/aig-graph-summarization-coarsening-8a9ea6` as an appended
  S6 (`METHODS=(... lsh mffc)`). That branch is **not** a descendant of `origin/summarization`
  — it forked at `b75ae89` and misses the last 4 commits, including the scope decision. Decide
  whether S6 is in before writing, and rebase it if so.
- **Random within-type merging** is a proposed but unbuilt 4th arm (the naive floor). The scope
  box states the gap plainly: with all three arms sophisticated, "all three are good" cannot be
  distinguished from "this task does not care how you merge."

### B2. `cone` (S1) is described incorrectly — factual error, not just staleness

Methodology line 171 says: *"Restricts merging to gates on tightly bounded topological depths
(e.g. $\pm 1$) sharing a common immediate dominator node."* Three things are wrong:

1. **Dominator → post-dominator.** Measured: ~99% of AND gates (2996/3000) have the virtual
   source as immediate dominator, which collapses S1 to "merge everything on a level". The
   implementation uses the immediate **post**-dominator. The same degeneracy reappeared on the
   output side (73% of AND gates, 5821/8000, post-dominate to the virtual sink) and had to be
   closed separately.
2. **$\pm 1$ → fixed window.** "Within ±1 level" is not an equivalence relation, so it cannot
   define a partition. Implementation uses `level // (band+1)`; production runs `level_band=0`.
   Consequence worth stating: widening the band does **not** monotonically increase compression.
3. **The depth axis is missing entirely.** S1 has two axes; the .tex describes only width.
   The depth axis contracts maximal fanout-free chains (`max_chain_length=4`), which is what
   reduces circuit depth and connects to the over-squashing argument.

Also unreported and worth a sentence: at `level_band=0` **S1 is provably DAG-preserving** and
the two axes compose safely.

### B3. Baselines: a one-line TODO covering three implemented models

`sec:method:architecture:baselines` is `% TODO: some optimizability baselines from pre-existing?`,
and `sec:results:rq1:baselines` names only trivial predictors (mean/median, size-only regressor).

`origin/baselines/openabc-synthnet-hoga` has **SynthNet** (OpenABC-D, Chowdhury et al.),
**HOGA** (Deng et al., DAC'24), and **DeepGate4**, wired through `src/train_baseline.py`
(917 lines) with three SLURM scripts and ~2,250 lines of tests. `train_baseline.py`'s docstring
is already thesis-grade methodology prose:

- which hyperparameters are the papers' published values and which had to be assumed;
- why HOGA's node-budget batching is *closer* to upstream than fixed graph batching, since
  upstream minibatches over nodes for a per-node task;
- why budget and `accumulate_grad_batches` must be retuned together (their product is the
  effective batch), and that Lightning's constant divisor makes effective sample size the
  harmonic mean;
- why DeepGate4's virtual edge set (7.36M virtual edges on an average 40k-node graph, ~45 GB
  across 12 layers) forces gradient checkpointing, and that the paper's own "w/o Partition"
  row is OOM.

`DIAGNOSIS.md` documents SynthNet **collapsing to a constant** (flat `val_loss` 4.4e-4,
`val_r2` -0.167), that OpenABC-D's own Table 6 reports negative R² on the same split variant
(unseen IP = your `split_by=design`), and — importantly — that upstream z-scores per design,
so **only the sign transfers**, not the numbers. That caveat has to survive into the thesis or
the comparison is invalid.

Also: the notes recommend **GCNConv / GraphSAGE / GINConv** as naive-model baselines alongside
mean/median/size-only. Three tiers, none currently in the .tex.

### B4. The exact-compression model track is absent ★ theoretically the strongest gap

`origin/summarization` adds a **second model track**: `src/data/exact_graph.py`,
`src/models/base_model_exact.py`, `src/models/layers/gcn_exact.py`, with tests.

`fold_inversions_into_x` drops `edge_attr` and encodes polarity as a per-node inverted-fanin
count; `GCNConvExact` then applies multiplicity via `edge_weight` **after** the message
nonlinearity. That ordering is the whole point: it is the only way "k identical incoming
messages" can be represented exactly by one aggregated computation, and it is what makes
`wl` at `count_cap=None` **provably lossless for the trained GNN**. Folding multiplicity into
`edge_attr` instead scales a value fed *into* the nonlinearity, which cannot decompose over sums.

`sec:method:architecture` describes exactly one architecture and states it is "used for
unreduced graphs, sparsified graphs, and summarized graphs" — which is now false for the exact
track. This needs its own subsection, plus the equitable-partition/orbit theory from the notes
(colour refinement computes the coarsest equitable partition; permutation-equivariant GNNs give
identical representations to nodes in the same orbit → the quotient is lossless).

### B5. Sparsification naming and parameters

- `sec:method:sparse:mst` is titled **"Random Minimum Spanning Tree (MST) Sparsity"**. The code
  is `spanning_forest_sparsification`, and the results table in the same repo already calls it
  `spanning_forest`. Pick one name.
- The .tex does not say it runs on the **undirected projection** (`to_networkx(to_undirected=True)`),
  which discards edge direction on a DAG. That is a substantive property.
- The docstring records it **replaced a spanner algorithm** that was ineffective on AIGs "due to
  their lack of dense cyclic redundancy" — a genuine negative finding worth one sentence.
  `config.SPARSIFICATION_SPANNER_STRETCH = 3.0` is still there, now unused.
- **No parameter values anywhere.** Fill in: dropout rate 0.3, PageRank keep_ratio 0.8 / α 0.85,
  seed 42, `and_gate_only` parameter-free.

### B6. Experiment section vs. what is actually configured

| .tex says | code says |
|---|---|
| "max X,000 nodes per AIG" | `config.MAX_NUM_GATES = 366040` |
| "Dynamic batching … maximum number of nodes per batch" | **two** budgets: train 3M, eval 8M (`EVAL_MAX_TOTAL_NODES_PER_BATCH`), deliberately separate |
| "80/10/10 volume split and a functional design level split" | three strategies via `--split_by` (`design`/`recipe`/`random`); design is the untagged default |
| — | eval sweep is **9 configs** (`none` + 4 sparsification + 4 partition); summarization is a reserved commented line in `test.sh`, not yet run |

The eval-budget comment in `config.py` carries a warning that belongs in the thesis verbatim in
spirit: raising it **invalidates cross-config hardware comparisons**, so every config must be
re-run at the same value. Every RQ2 VRAM/throughput number depends on that.

Also for `sec:method:experiment:hardware`: Snellius, `gpu_h100`/`genoa`, H100 80GB, measured
~6.8 GB per 1M nodes, kernels latency-bound on message-passing gather/scatter (SM Active ~100%
at 82% occupancy, DRAM ~37%).

For `sec:method:experiment:reproducibility`: seeds are fixed at 42 across sparsification,
partition, summarization and training — but there is **no multi-seed protocol**. As it stands
the answer to "how many seeds per configuration?" is one, which directly weakens RQ4's
matched-compression gap. Either add seeds or state it in Limitations.

### B7. RQ numbering has drifted in three places

RQ4 was split out (domain-informed adaptation), pushing generalization to RQ5. Not everywhere:

- `src/results_to_latex.py` captions cross-state generalization **"(RQ4)"**.
- `thesis-overview.tex` still carries the **old 4-RQ list** where RQ4 = generalization.
- `sec:method:experiment:phases` has a TODO acknowledging phases are no longer 1:1 with RQs.

### B8. Results chapter ignores the table generator that already exists

`src/results_to_latex.py` on `main` writes booktabs tables **explicitly "ready to \input{} into
sections/4-results.tex"**:

| file | label | RQ |
|---|---|---|
| `baseline_accuracy.tex` | `tab:baseline_accuracy` | RQ1 |
| `reduction_efficiency.tex` | `tab:reduction_efficiency` | RQ2 |
| `predictive_retention.tex` | `tab:predictive_retention` | RQ3 |
| `cross_state_generalization.tex` | `tab:cross_state_generalization` | RQ5 (captioned RQ4) |
| `vram_scaling.tex` | `tab:vram_scaling` | RQ2 |
| `pareto_front.csv` | — | RQ3 |

Ch. 4 describes these tables in prose instead of inputting them. Two follow-ons: **`msc_thesis.tex`
does not load `booktabs`**, so nothing here compiles today; and the generator knows only
`sparsification_stats` / `partition_stats` — no summarization, no baselines.

### B9. Related Work is missing its nearest competitor

**CTS-Bench** (arXiv 2602.19330, Feb 2026) — graph coarsening trade-offs for GNNs in clock tree
synthesis. The notes contain a full, careful positioning: it costs you only the unqualified
phrase "first coarsening study in EDA", and it *strengthens* the motivation (generic clustering
gave 17.2× memory / 3× speed but **negative R² zero-shot**, and it explicitly calls for
domain-aware coarsening). Its method consumes physical XY coordinates and so is **not applicable
to an AIG at all**.

Its methodological lesson belongs in `sec:method:experiment:metrics`: their MAE barely moved
(0.16 → 0.17) while R² fell below zero. **Never judge a reduction on RMSE/MAE alone.** That is
the citation justifying reporting RMSE + R² + Spearman together.

### B10. Hypotheses that exist only in the notes

The .tex states no hypotheses (`sec:method:experiment:design` is empty). These are written and
argued in the notes:

1. **Over-squashing / receptive field** — sparsification *removes edges* and can worsen
   propagation; summarization *contracts paths* and can improve it. So coarsening may **raise**
   accuracy rather than trade it, and beat sparsification at matched compression. Caveat to state
   honestly: `measure_summarization.py` and the receptive-field metric are **not built**, so this
   is currently asserted, not evidenced.
2. **AIG symmetry sources** — interchangeable AND fan-ins, replicated datapath bit-slices,
   repeated sub-logic create real orbits a random graph lacks, predicting AIGs compress well
   under `wl`. Measurable as a dataset statistic.
3. **Residual redundancy after strash** — ABC strashes every graph, so `d=1` should find almost
   nothing. Reportable either way; probe not yet run.

### B11. Findings from building the methods that belong in Ch. 3/4

Beyond B2, four more measured results are sitting in notes only:

- **ConvMatch's Eq. 7 objective favours low-degree nodes regardless of similarity** — verified on
  a hand-built graph (twins 1.876, mismatched same-degree 1.967, degree-2 PI pair 1.257).
  Explains why its behaviour on a deep polarity-carrying DAG differs from the node-classification
  setting it was published in.
- **S4's spectral cap is part of the method's definition, not an implementation detail** —
  local variation costs 880 ms at n=4k vs 9 ms for heavy-edge, hence `max_spectral_nodes=5000`
  with heavy-edge fallback. Which graphs took the fallback is **not recorded**; if most of the
  corpus is above 5k nodes, S4 must be described as "heavy-edge with a spectral rule on small
  graphs."
- **S5's compression is bounded at both ends, and neither bound is `bin_width`.** Ceiling =
  distinct-descriptor count (0.2246 at n=5k, 0.3644 at 50k, 0.3738 at 200k), so S5 cannot compress
  *less* than ~63%. At the production `reduction_ratio = 0.5` the ceiling binds on every AIG
  measured, so **S5 cannot participate in a matched-compression comparison** without a decision
  (three options are written up; option 1 — report S5 on its own curve — costs nothing).
  The 37%-distinct-descriptor figure is itself direct evidence for the redundancy the whole
  summarization argument rests on.
- **Per-graph cost at the corpus's largest size (370,801 nodes)**: `identity` 0 s, `lsh` 0.6 s,
  `wl` 2.7 s, `cone` 8.9 s, `spectral` 11.4 s, `convmatch` ~25 s / ~1.6 GB. That is the RQ2
  offline-cost column, already measured.

**Not yet measured, and it matters:** `cone`'s compression on the real corpus is unknown.
Synthetic numbers must not be quoted (finding 7). If tier0/tier1 AIGs have few fanout-free cones,
the contribution has to lean on retention at low compression rather than on compression itself.

---

## C. Build and structure issues

1. **`booktabs` is not loaded** in `msc_thesis.tex` — the commented sparsification table and
   every generated table need it.
2. **`thesis-overview.tex` (538 lines) is a stale full duplicate** carrying the old 4-RQ
   numbering. Superseded by commit `8c64898`; not `\include`d, so it does not compile — but it is
   a live drift risk. Delete or move to an `archive/` folder.
3. **Page limit is 35, two-column.** The outline currently has ~120 headings across 6 chapters.
   Ch. 2 alone has 30. A per-chapter page budget is needed before drafting, or Preliminaries will
   eat the thesis.
4. `\usepackage{mwe}` is example-only; `sections/0-list-symbols.tex` is fully commented out
   (the thesis has real notation to list: count-cap $c$, depth $d$, level band, compression ratio).
5. `references.bib` needs ~20 entries; all are characterised in the notes but none are BibTeX yet.

---

## B12. Three more drifts found on second pass

- **PE pooling is stated backwards.** Methodology line 150 (SNAP bullet) says *"Mean-pool 32D
  PEs of merged nodes"*. `apply_merge_map` **min**-pools (`_pool_min` over `_MIN_POOLED_ATTRS =
  ("level", "pos_enc")`). Min is the right choice and defensible — a super-node enters the
  circuit at its earliest member's level — but the .tex says the opposite, and line 209 still
  asks *"How to deal with the positional encoding??"* as an open question that the code answered.
- **The node/edge feature schema changes under merging, and the Architecture section denies it.**
  `sec:method:architecture` says node features are a *"4-dimensional one-hot vector"* and edge
  attributes a *"2-dimensional one-hot vector"*. After `apply_merge_map` both are **count
  vectors**: `x` = member type counts `[#const, #PI, #AND, #PO]`, `edge_attr` = super-edge
  polarity counts `[#normal, #inverted]`. A size-1 super-node reproduces the one-hot exactly,
  which is *why* R1 holds and why cross-state inference needs no re-processing — that is a
  load-bearing design property (enriched superset schema) and it is currently invisible.
- **The super-node features the design rationale calls critical are computed and then discarded.**
  C3 in the notes cites Generale's ablation that keeping node content was critical, and lists
  type frequency, size, and level `[min, max, mean, var]`. What is actually implemented:
  type counts ✓, level **min only** (no max/mean/var), and `internal_edges` / `num_edges` /
  `num_pis` / `num_pos` are attached to the graph but **never read by the encoder**. `node_size`
  is not even attached (recoverable as `x.sum(1)`). So the C3 ablation is not currently
  runnable, and one of the cheapest possible accuracy wins is sitting unconsumed. Either wire
  them into `base_model.py` or drop the C3 claim from the writeup — do not write it as if done.

---

## E. Missing definitions, by section

Terms used in existing prose whose definition site is empty or absent. Ordered by how much
breaks without them.

### E1. Blocking — a result cannot be interpreted without these

| Term | Used in | Defined where | Status |
|---|---|---|---|
| **Compression ratio** (node vs edge retention) | RQ2, RQ3, RQ4 | `sec:prelim:reduction:measuring` | empty — and **RQ4 is unevaluable without it** |
| **Matched compression** | RQ4, `sec:results:rq4:pairings` | nowhere | undefined, and partly unachievable (see F3) |
| **Optimizability** | Title, abstract, every RQ | `sec:method:data:label:definition` | one line, buried in Ch. 3 — needs a clause in Ch. 1 |
| **Cross-state / structural state** | RQ5, Ch. 4, Ch. 5 | nowhere | undefined term used ~12 times |
| **Level / topological depth** | PE, S1, level slicing, span weighting | `sec:prelim:aig:properties` | empty — four methods depend on it |
| **(Post-)dominator, reconvergence** | S1, S6 | `sec:prelim:aig:properties` | empty, and the .tex currently uses "dominator" *incorrectly* (B2) |

### E2. Needed for the summarization chapter to make sense

**strash / structural hashing** (CA1 — the residual-redundancy argument and S6's premise both
rest on every corpus graph being strashed); **FRAIG / functional vs structural equivalence**
(the optimization-overlap boundary, and the negative control); **MFFC** (S6's whole definition);
**colour refinement / 1-WL / equitable partition / bisimulation / count-cap $c$** (S2);
**quotient graph, super-node, multigraph, edge multiplicity** (used throughout Ch. 3, never
introduced); **over-squashing / effective receptive field** (hypothesis 1).

### E3. Needed for the baselines and the efficiency numbers

**VRAM allocated vs reserved** (RQ2 reports "peak VRAM" without saying which — the config
comments distinguish them and warn about NVML sampling gaps); **mixed precision** (bf16-mixed on
H100 changes every memory number); **gradient checkpointing** (what makes DeepGate4 runnable at
all); **`torch.compile`** (on by default — it changes every timing number and must be disclosed);
**gradient accumulation / effective batch** (central to the HOGA and DeepGate4 comparability
argument).

### E4. Metrics — defined nowhere, all four used

Smooth L1 (**with `beta = 0.01`**, a non-default that must be stated), RMSE, $R^2$
(and that it is logged for val/test only, since per-batch $R^2$ is meaningless), Spearman.
`sec:method:experiment:metrics` and its four subsubsections are all empty.

### E5. Points missing entirely from Methodology (not just unwritten — unplanned)

1. **No formal problem statement.** `sec:prelim:formalization` is empty. Needs: graph $G$,
   target $y \in [0,1]$, reduction operator $R$, and the claim under test
   $\hat{y}(R(G)) \approx \hat{y}(G)$. Every RQ is an instance of it; without it RQ4 and RQ5
   read as separate topics rather than two probes of one operator.
2. **No description of `apply_merge_map`** — the single shared rewrite every summarization
   method routes through. This is the actual method contribution surface, and it is absent.
3. **No global statement of the boundary constraint (C4).** PI/PO preservation is currently
   asserted per-method in three places; it is a property of the whole family.
4. **Two different pipeline designs, neither stated.** Sparsification and partitioning apply
   **masks at `get()` time**; summarization **materializes coarsened graphs to disk**. That
   difference drives the offline-cost column in RQ2 and the storage discussion, and explains
   why summarization needed a precompute job at all.
5. **No test protocol.** `test.py` runs **two passes per config** — `full_graph` and
   `matched_reduction` — and that pairing *is* the RQ5 mechanism plus RQ3's matched-state
   number. Ch. 3 never says so; Ch. 4 §`sec:results:rq5:dropoff` describes the output without
   the procedure.
6. **No determinism/seeding statement**, and no disclosure of the **summarization cache being
   keyed by method only, not parameters** — a real reproducibility hazard already hit once
   (the `num_probes` 2→8 change), and a Limitations item.

---

## F. Decisions — my reading, with reasons

You asked me to settle these rather than hand them back.

### F1. S6 `mffc`: include it, and run it. ✅

It is the strongest domain method in the set, for four reasons that hold independently:

- **It contracts exactly the unit the optimizer operates on.** `refactor` collapses and
  re-expresses one MFFC at a time. No other method has that tight a coupling to the
  label-generating process — S1's level band is a proxy for the same intuition.
- **Parameter-free.** S1, S3, S4 and S5 all carry uncalibrated knobs (open item: "set to
  plausible defaults, not calibrated"). S6 has none, so it cannot be accused of being the one
  method you tuned. That is a real defence against the obvious reviewer attack.
- **Verified against an independent oracle** — 600/600 random DAGs against a post-dominator
  implementation, now a unit test. None of the other five have that.
- **It resolves the objection your own .tex raises against it.** The wider-pool bullet says
  *"MFFCs are prime targets for synthesis rewriting — merging them might obscure the exact
  optimizability features"*. The S6 docstring answers it: intra-cone wiring is precisely what
  local rewriting is *free to change*, so it is noise w.r.t. the label, while the sharing
  structure that *constrains* optimization survives untouched. **Do not delete that objection —
  promote it.** It is the sharpest testable hypothesis in the summarization chapter, and S6 vs
  the full-graph baseline settles it empirically.

**But make S1 and S6 an ablation pair, not two rival contributions.** S6 is a strict
generalization of S1's depth axis (uncapped, and it catches internally-reconvergent cones S1
provably cannot). Running both answers "does the capped, level-banded version lose anything
against the exact decomposition?" — a much better story than one arbitrarily-parameterised
domain method. Frame the domain contribution as *one family with two settings*.

**Cost:** the branch forked at `b75ae89` and is missing the last four commits of
`origin/summarization`, including the scope decision. Rebase it before writing.

### F2. The random within-type merge floor: build it — and it is not optional. ✅

It is currently written up as a nice-to-have naive control. It is actually **load-bearing for
RQ4**, because it is the only method in the entire summarization family that can be dialled to
*any* compression ratio exactly (merge random same-type nodes until the node count matches).

That makes it the **matching instrument**, not just the floor: for each real method, run
random-merge at that method's *achieved* ratio and compare pairwise. This converts RQ4 from
"find methods that happen to compress equally" (which mostly fails — see F3) into a paired
design that works for every arm including the parameter-free ones.

Cost is ~10 lines plus a `METHODS` append, and one precompute + training run per matched point.

### F3. RQ4's "matched compression ratios" premise is broken as written. ⚠ Fix required

Sort the methods by whether compression is dialable:

| dialable | fixed by construction |
|---|---|
| `convmatch`, `spectral` (`reduction_ratio`) | `mffc`, `wl`, `identity` |
| `random_edge_dropout` (rate), `pagerank` (`keep_ratio`) | `and_gate_only`, `spanning_forest` |
| the proposed random-merge floor (any target, exactly) | `cone` only coarsely — `level_band` is an integer, and `band>0` gives up both DAG-preservation and the exact level PE |
| | `lsh` — **cannot reach 0.5 at all**; the distinct-descriptor ceiling binds on every AIG measured |

So RQ4 as phrased ("compared at matched compression ratios") is **not achievable for most of
the family**. Three concrete consequences:

1. **The `and_gate_only` vs `pagerank` pairing is already matched and you should say so.**
   Measured node retention 82.1% vs `keep_ratio = 0.8` → 80.0%. That is a genuinely tight pair
   and the single cleanest RQ4 comparison you have. Claim it deliberately rather than leaving it
   to look accidental.
2. **The `spanning_forest` pairing is not matched, and is fixable today.** `spanning_forest`
   holds 58.1% of edges (parameter-free); `random_edge_dropout` at the configured rate 0.3 holds
   69.7%. **Set the dropout rate to ≈0.419** to match 58.1% and the pair becomes valid.
   This is a config change, not a redesign — but it invalidates the existing
   `random_edge_dropout` runs, so decide before the next sweep.
3. **Reword RQ4** so matched compression is the *method*, not a precondition: something like
   *"…retain more predictive accuracy than generic techniques at equivalent compression,
   established by pairwise matching against a compression-matched random control where a
   direct pairing is not available?"* Then F2's random floor is what makes the question
   answerable, and the parameter-free methods stop being excluded.

### F4. `lsh` in the matched-compression table: report it on its own curve. ✅

Option 1 of the three you wrote. It costs nothing, it is honest, and *"hashing cannot be dialled
to a target compression on AIGs"* is itself a finding about the method — supported by the
measured distinct-descriptor ceiling (0.2246 / 0.3644 / 0.3738 at n = 5k / 50k / 200k). Option 3
(AH-UGC consistent hashing) is the stronger result but is a second implementation, and `lsh` is
already demoted to cite-only, so the payoff does not justify the cost.

### F5. `wl` at `count_cap=None` is a positive control for RQ5 — use it as one. ✅

RQ5 currently has no positive control, which means a poor cross-state result is uninterpretable:
you cannot tell "reduced-trained models do not generalize" from "the eval plumbing is wrong".

Exact colour refinement on the exact track is **provably lossless for the trained GNN**. So a
model trained on `wl`-coarsened graphs and queried on full graphs *must* score essentially
identically to the full-graph baseline. If it does not, the finding is a bug, not a result.
That is the single most valuable thing the exact track buys you and it should be stated as a
validity check in `sec:method:experiment:design`.

**Caveat that must be in the text:** the exact track changes the input schema
(`fold_inversions_into_x` drops `edge_attr` and appends an inverted-fanin count), so R1 does
**not** hold trivially there — a full graph must be run through `fold_inversions_into_x` before
an exact-track model can be queried on it. For the five standard-track methods R1 *is* trivial
(size-1 super-nodes reproduce the one-hot schema) and full graphs need no preprocessing at all.
Two different answers for two tracks; say both.

---

## G. Changes to the experiments and the RQs

### G1. RQ-level changes

| RQ | Verdict | Change |
|---|---|---|
| RQ1 | **Reword** | "Can a GNN *accurately* predict…" has no referent, and the question is now stronger than "can it at all": you have three published baselines implemented, one of which (**SynthNet**) collapses to a constant on this split. Reframe as "how does a GNN compare against published circuit-representation baselines and trivial predictors". Also drop **"theoretical"** — the label is empirical, produced by running ABC. |
| RQ2 | **Keep**, scope-check | It promises all three families. Summarization has **no runs yet** and is still a commented-out line in `test.sh`'s config array. Either the runs happen or RQ2 narrows. |
| RQ3 | **Add the hypothesis** | The over-squashing argument (coarsening contracts paths → summarization may *beat* sparsification, and may *raise* accuracy rather than trade it) belongs here as an explicit stated hypothesis. Do **not** make it a sixth RQ — 35 pages, two columns. |
| RQ4 | **Reword — required** | The matched-compression premise is unachievable for most methods (F3). |
| RQ5 | **Keep**, add the control | Add the `wl`-lossless positive control (F5) and the exact-track schema caveat. |

### G2. Experimental changes, ranked

1. **Set `SPARSIFICATION_RANDOM_DROPOUT_RATE ≈ 0.419`** so the `random_edge_dropout` vs
   `spanning_forest` pair is matched at 58.1% edge retention (F3.2). Invalidates existing
   dropout runs — decide before the next sweep, not after.
2. **Build and run the random within-type merge arm** as the compression-matching instrument
   (F2). Without it, RQ4 has exactly one valid pairing on the summarization side: none.
3. **Rebase the S6 branch onto `origin/summarization`** and run `cone` and `mffc` as an
   ablation pair (F1).
4. **Run one shard each of `cone` and `mffc` before committing to the full sweep.**
   `sbatch --array=32` and `--array=192`. Both compressions are unknown on the real corpus, and
   S1's synthetic numbers must not be quoted. If `cone` compresses poorly, the contribution has
   to lean on retention at low compression — that changes how Ch. 1 is written, so find out early.
5. **Add summarization to `test.sh`'s `CONFIGS` array** (the reserved commented line) and to
   `results_to_latex.py`, which currently loads only `sparsification_stats` and
   `partition_stats`. No RQ2/RQ3 summarization row can be produced until both are done.

### G3. Two things to decide that affect what you can claim

- **Seeds.** Everything is seeded at 42, but there is **one run per configuration**. RQ4 asks
  whether domain-informed methods retain *more* accuracy — a claim about a gap that is likely to
  be small. With no run-to-run variance you cannot say a gap is real.
  `build_paired_savings` already does bootstrap CIs and a Wilcoxon test **for the efficiency
  numbers**; the accuracy numbers have nothing. Either run ≥3 seeds on the RQ4 pairs
  specifically (cheapest useful version) or state in Limitations that accuracy gaps are
  reported without variance.
- **The motivation-to-metric gap.** The motivation is *script/pipeline selection* — choosing
  between synthesis algorithms for a circuit. The model is trained on **Orchestrate only**
  (`config.VALID_ALGORITHMS = {"Orchestrate"}`), and Spearman ranks *circuits by optimizability*,
  not *algorithms for one circuit*. Those are different tasks. Deepsyn/Syn4/C2RS graphs already
  exist on disk. Either name this in `sec:intro:rqs:scope` as a deliberate delimitation, or the
  Discussion has to absorb the objection unprepared. Naming it is much cheaper.

### G4. Also worth fixing while in there

- **`build_paired_savings` already computes bootstrap CIs and Wilcoxon p-values.** Ch. 4 does not
  mention statistical treatment anywhere. Say what is tested and how.
- **`EVALUATION.md` is a fourth place with the old RQ numbering** ("RQ1–RQ4 tables",
  "RQ2/RQ3/RQ4 charts"), alongside `results_to_latex.py`, `thesis-overview.tex` and the phases
  TODO. Fix all four together.
- **§`sec:results:summary`** promises one consolidated table across every method; nothing in
  `results_to_latex.py` builds it. Either add a builder or drop the section.

---

## D. Suggested order

Ranked. Steps 1–4 need no cluster runs.

0. **Reword RQ4 and add the RQ3 hypothesis** (G1, F3.3). Fifteen minutes, and everything below
   is written against the new phrasing. Do this before drafting anything else.
1. **Rewrite `sec:method:reduction:summarization` against the locked set** (B1, B2, B11, B12) and
   add the exact-compression subsection to `sec:method:architecture` (B4). This is the largest
   block of wrong-and-confident text and the strongest contribution. S6 and the random floor are
   both **in** (F1, F2), so the method table is seven rows: `identity` (fixture), `cone`, `mffc`,
   `wl`, `convmatch`, `spectral` (cite-only), `lsh` (cite-only), plus the random floor.
   **~half a day.**
2. **Fill Preliminaries + Related Work from the notes** (A1, B9, B10) and build `references.bib`.
   Unblocks 13 dangling `sec:prelim:*` references from the Introduction. **~1–2 days**; the
   `/academic-research-skills:ars-outline` skill fits here, since the raw material is already
   written and needs structuring, not researching.
3. **Write the Baselines subsection and the three-tier RQ1 baseline story** (B3), including the
   SynthNet non-comparability caveat. Most prose can be adapted from `train_baseline.py`'s
   docstring and `DIAGNOSIS.md`. **~2–3 hours.**
4. **Reconcile the Experiment section with `config.py`** (B5, B6), fix the RQ numbering in all
   three places (B7), and add `booktabs` + the `\input{}` hooks with placeholder tables (B8, C1).
   **~2–3 hours.**
5. **Fill front matter** (A4) and set the per-chapter page budget (C3). **~1 hour.**

Blocked until runs land: `sec:intro:contributions:results`, label distribution, dataset
statistics, all of Ch. 4, and Discussion §Interpretation.

Writable today despite no results: Discussion §Limitations (B6's single-seed issue, B11's
unmeasured `cone` compression, the bounded-scale honesty point already drafted in comments) and
§Ethical Considerations.

---

## H. Still open after the pass

Everything below is either blocked on a cluster run or needs a decision only you can make.
Each has an inline marker in the `.tex` at the point it is needed.

### H1. Blocked on runs (cannot be written yet)

| Section | Needs |
|---|---|
| `sec:intro:contributions:results` | RQ2/RQ3 headline numbers |
| `sec:method:data:label:distribution` | histogram + per-tier stats |
| `sec:method:data:statistics` | node/edge/depth distributions |
| All of Ch. 4 | every result |
| `sec:discussion:interpretation`, all of Ch. 6 | Ch. 4 |

### H2. Runs to launch, in order

1. **One shard each of `cone` and `mffc`** — `sbatch --array=32` and `--array=192`.
   Both compressions are unknown on the real corpus and synthetic numbers must not be quoted.
   If `cone` compresses poorly the framing of Ch. 1 changes, so find out first.
2. **Set the dropout rate to ≈0.419** and re-run `random_edge_dropout`, so its pairing with
   `spanning_forest` is matched at 58.1% edge retention.
3. **Build + run the random within-type merge arm** — the matching instrument RQ4 depends on.
4. **≥3 seeds on the RQ4 pairings only** — otherwise accuracy gaps are point estimates with
   no variance, which is stated as a limitation but weakens the headline claim.
5. **Add summarization to `test.sh`'s `CONFIGS`** and to `results_to_latex.py`, which loads
   only sparsification and partition offline stats today.

### H3. Code changes the text now assumes or flags

- **RQ numbering** still says RQ4 for cross-state in `results_to_latex.py`, `EVALUATION.md`
  and `thesis-overview.tex`. Ch. 4 carries a note; fix all three together.
- **Summary table** (`sec:results:summary`) has no builder — add one or drop the section.
- **Receptive-field metric** is specified in `sec:method:experiment:metrics:reduction` but not
  implemented. Until it exists, H1 is asserted, not evidenced.
- **Super-node content** (`internal_edges`, `num_pis`, `num_pos`, level max/mean/var) is
  computed but never read by the encoder. Cheapest untested improvement available.
- **Summarization cache is keyed by method, not parameters** — disclosed in
  `sec:method:experiment:reproducibility` as a limitation; fixing it would remove the caveat.
- **Rebase the S6 `mffc` branch** onto `origin/summarization`; it forked before the scope
  decision.

### H4. Decisions and facts only you have

- **Front matter**: student number, dates, examiner, supervisors (`msc_thesis.tex`).
- **Source circuits** (`sec:method:data:sources`): which two sources, how many designs each,
  licensing. Needed for the ethics section and any dataset release.
- **`sec:intro:context` vs `sec:intro:motivation`** overlap almost entirely. Recommended:
  delete the former and let Motivation open the chapter.
- **`thesis-overview.tex`** is a stale 538-line duplicate with the old 4-RQ numbering. Delete
  or archive.
- **Bibliography**: ~32 TODO markers. Entries were reconstructed from the working notes;
  authors/venues need checking and five have `TODO` in an author field.
- **Page budget**: builds at 42 pages single-column, so roughly half that two-column, before
  results. Ch. 2 has 20 preliminaries subsubsections and will need merging.
