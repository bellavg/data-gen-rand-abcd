# Summarization: write-up versus implementation

Working note, 2026-08-03. Produced by comparing the thesis prose against the implementation.
Not part of the build.

## Read this first: where the implementation actually is

There is no single implementation branch. Verify before trusting anything here.

| Location | Branch | State |
|---|---|---|
| `.claude/worktrees/summarization` | `summarization` | **20 uncommitted files. Newest code. Has the exact track, no MFFC.** |
| `.claude/worktrees/open-abcd-baseline-setup-4ae523` | `claude/aig-graph-summarization-coarsening-8a9ea6` | clean, older. **Has MFFC, no exact track.** |
| `.claude/worktrees/exact-gcn-summarization-7a2753` | `claude/exact-gcn-summarization-7a2753` | dirty, superseded |

The two diverge at `b75ae89`. **Neither is a superset.** Everything below treats the
`summarization` *working tree* as the implementation, because it is newest and holds
`wl_exact`.

**A `git checkout` in that worktree destroys the uncommitted work. Commit it before anything
else.**

Shipped method set is `cone wl convmatch wl_exact` (`src/shell/summarization_methods.sh:25`,
matching `SUMMARIZATION_REGISTRY` in `src/data/summarization.py:815-821`). `identity`,
`spectral` and `lsh` were deleted from the code on 2026-08-02, which the thesis already
reflects.

## Already fixed in the thesis (commit `cb80f9e`), do not redo

1. **The exactness coefficient was false as written.** `eq:prelim:exact-aggregation` in
   `2.5-prelim-reduction.tex` carried the raw multiplicity `w` as the message coefficient, but
   its left side aggregates at a single node while `w` counts member edges across all members
   of the receiving cluster. Verified by hand: a two-member target cluster with four member
   edges needs coefficient 2, not 4. The code was right all along
   (`exact_graph.py:137-138`, `edge_weight = counts / target_size`). Corrected at the
   definition in `2.5` and the claim in `3.1:275`, with `m_{v'}` now defined once alongside the
   multiplicity. The readout condition was already correct.
2. **The refinement-depth justification had both directions reversed**, in the prose at
   `3.5:204` and in two code comments (`summarize_graphs.py:226-235` and its docstring at
   `:180-185`). Refinement grows finer with depth, so a larger depth stays exact but forfeits
   compression, while a smaller depth breaks the guarantee. Exactness needs `L <= d`. The
   conclusion `d = L` was right, only the reason was inverted. **The code comments still say it
   backwards and should be corrected too.**

## Open: prose is wrong, the implementation is right

Ranked. Each needs a prose change, not a code change.

1. **Two rewrites exist; the thesis describes only the lossy one.** `3.5:47` and `3.5:59-60`
   say intra-cluster edges become self-loops and are dropped. `summarize_graph`
   (`summarization.py:962-989`) dispatches on `EXACT_METHODS` to either `apply_merge_map` or
   `apply_exact_merge_map`. The exact rewrite **keeps** intra-cluster edges as weighted
   self-loops, and `exact_graph.py:111-112` names dropping them as the source of lossiness. So
   the losslessness claim currently rests on a rewrite the thesis never describes.
2. **The exact schema's node vector is a class representative, not member counts.**
   `2.5:130-132` and `3.1:303-306` say the member counts the summarized schema carries make the
   weighting available. `exact_graph.py:121-126` computes `x = x_sum / node_size`, the average,
   and its docstring says the member sum would be wrong. The count lives in a separate
   attribute `out.node_size`, which `base_model_exact.py:77-93` reads for size-weighted
   pooling. A super-node of fifty AND gates is `[0,0,1,0,k]`, not `[0,0,50,0,.]`. So
   `eq:prelim:exact-readout` is satisfied by `node_size`, not by `x`.
3. **The exact track's node input is five-dimensional.** `3.1:21-22` and the config table row
   at `3.1:51` say four, unqualified. `src/config.py:45`: `EXACT_NODE_INPUT_DIM =
   NODE_INPUT_DIM + 1`, asserted in `test_gcn_exact.py:429`. `3.1:279-282` explains why the
   extra column exists but never says it widens the input.
4. **The exact encoder drops normalisation and the positional encoding.** `3.1` lists two
   encoder requirements for exactness; the code enforces at least four.
   `gcn_exact.py:43-62`: no normalisation anywhere, because graph-scoped statistics cannot be
   reproduced on a coarsened super-node. `base_model_exact.py:14,20,29-31`: no positional
   encoding, `pe_type="none"` only. Both are enforced, not advisory:
   `test_gcn_exact.py:224-235` asserts the module *raises* on `pe_type="level"` and on
   non-mean pooling. `test_summarization.py:587-598` is a `strict=True` xfail proving exact
   refinement is **not** lossless for the production encoder, precisely because of
   normalisation and unweighted mean pooling.
5. **`wl_exact`'s initial colour is not the type and level pair.** `3.5:173` gives
   `(tau(v), level(v))` and `3.5:181-183` derives pooled-level exactness from it. On the exact
   path `fold_inversions_into_x` runs before clustering and drops `level` and `pos_enc`
   entirely (`exact_graph.py:61-69`). The initial colour is (type, inverted-fanin count).
   `summarization_notes.md:87-94` says this was measured, same 26 classes either way on
   `adder`, and warns "Do not write it up the other way round." The prose currently states it
   the forbidden way round.
6. **The cone parameters have values.** `3.5:128-129` says neither has been given a value, plus
   a `% TODO`. `config.py:73`: `{"max_chain_length": 4, "level_band": 0}`, exercised by
   `test_summarization.py:815-905`. The TODO is honest about *calibration* not being done, but
   "neither has been given a value" is false.
7. **The residual-redundancy probe has been run and its premise refuted.** `3.5:226-229` says
   how much exists is unknown. `summarization_notes.md:291-298`: structural hashing dedupes
   identical fanin *pointers* while colour refinement groups by fanin *colours*, so hashing
   removes essentially none of the refinement redundancy. Drop the false premise and the
   "unknown" hedge. **Do not paste the measured retention figures in**: the author's own guard
   at `summarization_notes.md:267-275` says they come from 50 unrandomised seed designs, not
   the training corpus, and every one must be re-measured before entering the thesis.
8. **ConvMatch is adapted in more than one way.** `3.5:238-241` says "adapted only by imposing
   the boundary constraint, leaving the convolution objective itself unmodified."
   `summarization.py:524-573` documents exact kNN replaced by random-projection neighbour
   pairs, mutual-best matching instead of the sequential greedy scan, inherited self-loops left
   at pre-merge values, and a substituted convolved signal at `:588`. Commit `d521d4d` adds
   three more deviations found against the reference implementation. The boundary filter clause
   is right; "only" is not.
9. **The ConvMatch target ratio is already set, by a different rule.** The `% TODO` at
   `3.5:242` says to set it from the domain candidate's achieved compression.
   `config.py:62-69` fixes `SUMMARIZATION_REDUCTION_RATIO = 0.5` to match the sparsification
   sweep's midpoint so the Pareto fronts are comparable. Two different rules; the code has
   committed to one.
10. **The count-cap sweep.** `3.5` promised two endpoints, `c = 1` and `c = infinity`. The code
    registers `count_cap: None` on both `wl` and `wl_exact` and nothing at `c = 1`.
    `summarization_notes.md:299-306` records the cap as inert on AIGs, because fan-in is fixed
    by node type, with identical class counts in 50 of 50 seed designs. A `% TODO` now marks
    this in `3.5`; the arm's fate is an experimental decision.

## Open: implementation is behind the prose

1. **`3.7`'s positive control cannot pass as specified. High value.**
   `3.7-experiment-setup.tex:35-40` says a model trained on the exact reduction must score
   identically to the full-graph baseline on both passes, and any deviation is a defect rather
   than a finding. `summarization_notes.md:178-193` contradicts this: the exact model has no
   normalisation, no positional encoding and no edge attributes, while `cone` and `convmatch`
   train on the production encoder with all three, so "their accuracies cannot go on the same
   Pareto front, they are different models." Two separate problems. The RQ1 full-graph baseline
   is the production encoder and an exact-track model cannot match it. And the actual guarantee
   is that *one set of weights* scores identically on reduced and full inputs, which is a
   different statement. The notes put it correctly at `:210-216`: for the exact method RQ5 is a
   verification, not an experiment. The baseline the exact arm needs, an exact model on
   uncoarsened graphs, is an unbudgeted extra training run (`:879-882`).
2. **MFFC contraction is not on the working branch.** `3.5:37`, `:110-111`, `:113-122`,
   `:141-145` present two AIG-native candidates. `mffc` appears nowhere in the `summarization`
   worktree. It exists only on `claude/aig-graph-summarization-coarsening-8a9ea6:543-628`.
   `summarization_notes.md:159-170` says deciding between them means porting MFFC across and
   running one shard of each. The existing `% TODO` covers unmeasured compression, not the
   missing port. Where the prose and that branch's code do overlap they agree closely, so the
   prose is not wrong, just ahead of any single tree.
3. **Random within-type merging is not built, and is correctly marked.** Not a defect. One
   knock-on: `3.5:16-17` counts it among "Four candidates", so the thesis's four and the
   code's four are different sets.

## Open: genuinely ambiguous, author's decision

1. **Does the exact arm share an axis with the lossy ones?** The thesis reads it as a
   zero-accuracy-cost reference point every lossy method is measured against (`3.5:217-219`,
   `3.7:35-40`). `summarization_notes.md:190-193` says the opposite: it answers a different
   question, does not sit on their Pareto front, and the write-up has to say so rather than
   imply a shared axis. `:195-200` adds that exactness and improvement are mutually exclusive.
   This changes `3.5`, `3.7` and the RQ5 story.
2. **Boundary preservation is one constraint in the prose and three strengths in the code.**
   `3.5:77-85` states one rule imposed uniformly. In the code, `cone` merges AND gates only;
   `convmatch` filters candidates to same-type pairs; `wl` under backward refinement collapses
   **every** primary input into a single super-node, because sourceless nodes are
   indistinguishable (`test_summarization.py:526-534` asserts exactly this). The last satisfies
   the stated rule but arguably violates the justification at `3.5:82-83` that the interface is
   the circuit's contract. Either strengthen the rule or weaken the justification.
3. **Is the lossy `wl` arm viable at all?** `config.py` records measured node retention of
   99.0% on `sqrt`, 95.2% on `div`, 41.5% on `c6288`. A method retaining 99% of nodes on the
   deep datapath designs is not a reduction. `3.5:38` calls its compression "graph-dependent",
   which is true but does not convey that the arm may be inoperative on exactly the graphs the
   memory problem lives on.
4. **`wl_exact` has no row in the thesis table.** The code ships it as a distinct arm with its
   own model, schema, node width, rewrite and precompute range. The thesis folds it into the
   single graded-refinement row and treats the difference as an encoder variant. Given the
   findings above, the merge differs too.
5. **Should the inversion fold count as part of the reduction's cost?** It is lossy in its own
   right, since `AND(a', b)` and `AND(a, b')` become identical rows. `3.1:279-282` discloses
   this. It is applied to both the reduct and the "full graph", so the exactness comparison is
   internally valid while the exact arm reads strictly less than the production model.

## Bug-shaped, not documentation

1. **The precompute cache is keyed on method, not parameters.** `ARCHIVE_DIR=.../${METHOD}`
   with `.shardNNN.done` sentinels short-circuiting resubmission, and `summarize_graphs.py`
   skipping existing outputs. Changing `num_probes`, `reduction_ratio` or `sgc_depth` and
   resubmitting recomputes nothing and silently produces a mixed-vintage corpus. `params` is
   written to `_summary_stats_*.json` but never read back. **This will silently corrupt any
   parameter sweep.** Author-flagged at `summarization_notes.md:840-851` and hit for real.
2. **`apply_exact_merge_map`'s docstring over-claims uniformity**, saying the per-target-member
   count is guaranteed uniform by the equitable-partition property. That holds only after
   refinement converges. The notes give the correct weaker argument at `:128-140`. Behaviour is
   right; the justification is not.
3. **`summarization_notes.md` has internal staleness.** Line 857 says only `cone` guarantees an
   acyclic quotient, contradicting the MFFC acyclicity proof and the notes' own 300-graph
   measurement at `:166-168`; here the thesis agrees with the code and the note is wrong. Line
   877 says `wl_exact` is `METHODS` index 6 when it is index 3. Lines 862-864 and 883-904 still
   discuss deleted arms as live.
4. **`src/verify_exact_rq4.py` is a standalone entry point**, staged but uncommitted, plus an
   untracked `src/verify_exact_sanity.py`. `src/test.py` is 0 bytes on this branch because
   main's eval harness postdates the fork, so the RQ5 verification lives outside it and needs
   folding in at merge. This is the code path `3.7`'s positive control depends on.
5. **`_immediate_postdominators` materialises a NetworkX DiGraph**, roughly 0.35 GB and 2.6 s
   on a 366k-node graph, which at 96 CPUs is about 100 GB if many workers hit large graphs at
   once. Affects the `cone` precompute only.

## Where prose and code agree

Stated so they are not re-litigated: multiplicity applied outside the nonlinearity; refinement
depth tied to encoder depth at 4; minimum pooling of positional encodings on the lossy track;
inverted and non-inverted edges kept distinct on the lossy track; level bands as fixed windows
because "within k levels" is not transitive; ConvMatch's degree bias and direction-blindness;
the MFFC prose against `mffc_clustering`; and the removal of spectral and LSH from the study.
