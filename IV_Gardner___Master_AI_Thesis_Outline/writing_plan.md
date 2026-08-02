# Writing plan (archived source comments)

Working notes lifted out of the `.tex` sources so the submitted source carries no
self-addressed commentary. Each entry records the file and the sectioning command the
note sat under. This file is a scratchpad, not part of the document.

## `sections/1-introduction.tex`

### chapter: Introduction}\label{sec:introduction

```
% TERM CHECKLIST -- working note, not thesis content, produces no output.
% Every term below is used by the Motivation and must be named there in at least a clause
% before it is leaned on. The full definition belongs in Preliminaries; the trailing label
% on each line records where, so no term can silently end up without a home. Keep the
% mapping updated if headings move, and delete this block once the chapter is finished.
%
%   Electronic Design Automation (EDA)                  -> sec:prelim:synthesis:flow
%   Logic synthesis                                     -> sec:prelim:synthesis:flow
%   Optimization objectives: area, power, delay         -> sec:prelim:synthesis:objectives
%   And-Inverter Graph (AIG); Directed Acyclic Graph    -> sec:prelim:aig:definition
%   Synthesis algorithm, script, and pipeline           -> sec:prelim:synthesis:scripts
%   Optimizability (the prediction target)              -> sec:prelim:formalization
%   Graph Neural Network (GNN)                          -> sec:prelim:gnn:mp
%   Graph reduction: partitioning / sparsification /
%     summarization / coarsening                        -> sec:prelim:reduction
%   Compression ratio                                   -> sec:prelim:reduction:measuring
%   Training vs. inference                              -> sec:prelim:gnn:cost
%   VRAM; memory bottleneck; distributed training       -> sec:prelim:gnn:cost
%   Causal cone, logical depth, longest path,
%     topological level, gate role                      -> sec:prelim:aig:properties
%   RMSE, R^2, Spearman correlation                     -> sec:method:experiment:metrics
%     (defined where first used, not in Preliminaries -- they are standard)
%
% Added after auditing Chapter 3 for terms it uses but never introduces. Not all are
% needed by the Motivation, but every one is load-bearing later:
%   Reconvergence; immediate dominator / post-dominator -> sec:prelim:aig:properties
%   Maximum Fanout-Free Cone (MFFC)                     -> sec:prelim:aig:properties
%   Structural hashing (strash); functional vs.
%     structural equivalence                            -> sec:prelim:algorithms:primitives
%   Quotient graph, super-node, edge multiplicity       -> sec:prelim:reduction:summarization
%   Colour refinement (1-WL), equitable partition,
%     bisimulation                                      -> sec:prelim:gnn:expressivity
%   Over-squashing; effective receptive field           -> sec:prelim:gnn:cost
```

## `sections/introduction/contributions.tex`

### section: Contributions  `sec:intro:contributions`

```
% To enable further ML applications in EDA and solve the memory bottleneck, this thesis proposes a four-part methodology:
% \begin{itemize}
%     \item \textbf{Domain-Aware AIG Reduction Framework:} First, this thesis contrasts generic graph reduction techniques with targeted, AIG-specific heuristics (such as level-aware span weighting and structural coarsening). The objective is to effectively compress massive logic graphs (reducing memory footprint and computation time) while preserving the causal logic and structural information required for predictive accuracy. This yields a domain-aware reduction methodology and a dataset of compressed graphs that can be readily applied to unblock other ML avenues in EDA.
%     \item \textbf{Deep-Dive Optimizability Prediction:} Second, these reduction techniques are applied and rigorously validated on a high-value regression task: predicting circuit optimizability for a complex target synthesis algorithm. By focusing deeply on a single target algorithm, this approach provides a robust benchmark to measure exactly how well different reduction methods (generic vs. AIG-specific) retain critical structural features without the confounding variables of cross-algorithm comparisons.
%     \item \textbf{A Provably Lossless Coarsening for AIGs:} Third, one member of the
%     summarization family is exact rather than approximate. Merging nodes that share a
%     $d$-step colour-refinement class, where $d$ is the encoder depth, removes only structure
%     the network provably cannot distinguish \citep{bollen2023exact,grohe2014colourrefinement},
%     so it compresses the input without changing the model's output at all. Realising that
%     guarantee on an AIG requires a schema in which inverter polarity and edge multiplicity
%     survive the merge exactly (\ref{sec:method:architecture:exact}). This gives the study a
%     fixed anchor, a compression whose accuracy cost is known to be zero by construction,
%     against which every lossy method can be read, and a positive control for RQ5.
%     \item \textbf{Cross-Structural Generalization:} Finally, to establish practical
%     applicability, this thesis evaluates cross-state inference. The objective is to determine
%     whether a model trained exclusively on memory-friendly compressed AIGs can generalize and
%     accurately predict optimizability when queried with normal, full-sized AIGs at inference
%     time.
%     \item \textbf{Protocol-Sensitivity Measurement (RQ1a):} A controlled comparison of
%     three train/test protocols (random, recipe-disjoint, design-disjoint) under otherwise
%     identical conditions, quantifying how much leaky evaluation inflates reported
%     performance, with per-design error distributions. The surveyed literature motivates on
%     the unseen-design gap but rarely measures it (matched seen/unseen results in only 4 of
%     63 surveyed papers); all reported results here use the strictest
%     protocol.
% \end{itemize}
```

### subsection: Summary of Results  `sec:intro:contributions:results`

```
% Supervisor note: state the main numbers here (e.g. "X\% memory reduction at
% Y\% accuracy retention over the full-graph baseline"). Fill once RQ2/RQ3 land.
```

## `sections/introduction/motivation.tex`

### subsection: Problem Statement \& Research Gap  `sec:intro:gap`

```
% NOTE: this claim is scoped deliberately to logic synthesis / AIGs / optimizability.
% CTS-Bench \citep{cts_bench2026} benchmarks coarsening for GNNs one EDA stage later
% (post-placement netlists, clock skew) using a spatial clustering heuristic that cannot
% apply to a pre-physical AIG. It costs us the unqualified phrase "first coarsening study
% in EDA" and nothing more specific -- see \ref{sec:relwork:gap}.
```

## `sections/introduction/research-questions.tex`

### subsection: RQ1: Task Feasibility \& Baseline  `sec:intro:rq1`

```
% Reworded on two counts. (1) "accurately" had no referent, so the answer could not be
% wrong; it is now defined by a three-tier comparison (see \ref{sec:results:rq1:baselines}).
% (2) "theoretical optimizability" is dropped -- the label is empirical, obtained by
% actually running the synthesis script, not derived from theory.
```

### paragraph: RQ1a (protocol sensitivity).

```
% Added as a subquestion rather than a sixth RQ: it bounds the validity of RQ1's answer
% (and of every downstream comparison) but opens no new line of inquiry. Most published
% results in this literature report a single protocol; matched seen/unseen comparisons are
% the exception (4 of the 63 papers surveyed report both sides, and OpenABC-D's unseen-IP
% variant confounds design novelty with size extrapolation), so reporting all three cells
% under matched conditions is itself a contribution worth naming in
% \ref{sec:intro:contributions}.
```

### paragraph: Hypothesis H1 (path contraction).

```
% This is the strongest claim in the thesis and currently the least evidenced: the
% receptive-field metric (mean k-hop fanin-cone size, k = number of encoder layers) is
% specified in \ref{sec:method:experiment:metrics:reduction} but NOT yet implemented.
% Either build it or downgrade H1 to a discussion point -- do not report it as tested.
```

### subsection: RQ4: Value of Domain-Informed Adaptation  `sec:intro:rq4`

```
% Reworded because the original premise ("compared at matched compression ratios") is not
% satisfiable. Only convmatch, pagerank and random_edge_dropout take a target ratio; wl,
% mffc, and_gate_only and spanning_forest are parameter-free, and cone's band knob is
% integer-valued and gives up its DAG guarantee above 0. The random control
% (\ref{sec:method:summary:random}) is the only method that hits an arbitrary ratio exactly,
% which is what makes it the matching instrument rather than merely a naive floor.
```

### subsection: RQ5: Reduced to Unreduced? Generalization  `sec:intro:rq5`

```
% Positive control, and the reason the exact-compression track exists: colour refinement
% at count-cap infinity is provably lossless for this encoder (\ref{sec:method:architecture:exact}),
% so a model trained on wl-coarsened graphs MUST score essentially identically to the
% full-graph baseline when queried on full graphs. Without it, a poor cross-state result
% cannot be distinguished from a broken evaluation path.
```

## `sections/2-related-work.tex`

### section: Preliminaries  `sec:prelim`

```
% What a reader needs to know to follow the rest of the thesis. Definitions and
% notation only -- no argumentation, no comparison to other work.
%
% PAGE BUDGET: at 35 pages two-column, this section gets ~4 pages. It currently has
% 20 subsubsections. Several will have to become a sentence inside a neighbouring
% subsection rather than a heading of their own -- the ones flagged "merge candidate"
% below are the first to go.
```

### subsubsection: Position of Logic Synthesis in the Design Flow  `sec:prelim:synthesis:flow`

```
% Define: Electronic Design Automation (EDA); the flow from RTL through logic synthesis
% and technology mapping to placement and routing. One paragraph. The only load-bearing
% point for this thesis: logic synthesis is PRE-PHYSICAL -- an AIG carries no coordinates.
% That is what makes CTS-Bench's spatial coarsening inapplicable here
% (\ref{sec:relwork:domain:circuits}) and it needs to be established before that claim.
```

### subsubsection: Optimization Objectives: Area, Power, Delay  `sec:prelim:synthesis:objectives`

```
% Define the three classical objectives. State that this work uses node count as an area
% proxy and models nothing else -- forward-reference \ref{sec:intro:scope:target}, and note
% that the construct-validity question is taken up in \ref{sec:discussion:limitations:validity}.
```

### subsubsection: Synthesis Scripts and Pipelines  `sec:prelim:synthesis:scripts`

```
% Define: a script/recipe is an ordered sequence of transformation commands; the result
% depends on the order, and the best sequence is design-dependent. The motivating statistic
% is that the best recipes across designs overlap by under 30\%, which is why script choice
% cannot be fixed once and reused -- source it before citing.
% TODO(source): the <30\% overlap figure is recorded in summarization_notes.md without a
% citation. Find the paper or drop the number.
```

### subsubsection: Rewriting, Refactoring, and Balancing  `sec:prelim:algorithms:primitives`

```
% Define the primitives the target scripts are built from \citep{mishchenko2006rewriting}:
% rewriting replaces a cut with a smaller equivalent; refactoring collapses and
% re-expresses one MFFC at a time; balancing restructures for depth.
%
% Two definitions that must land here because later arguments depend on them:
%   - STRUCTURAL HASHING (strash): merges nodes with identical (fanin, polarity) pairs on
%     construction. Every graph in this corpus is strashed, so the trivial one-hop
%     structural redundancy is ALREADY GONE before any summarization runs -- which is why
%     colour refinement at depth 1 is expected to find almost nothing
%     (\ref{sec:method:summary:wl}).
%   - FUNCTIONAL vs STRUCTURAL equivalence, and FRAIG \citep{mishchenko2005fraig}, which
%     merges functionally equivalent nodes via SAT sweeping. This defines the boundary the
%     summarization methods deliberately stay on the near side of: merging functionally
%     equivalent logic performs part of the optimization being predicted, and would leak
%     the label. That boundary is argued, not measured -- see
%     \ref{sec:discussion:limitations:methods}.
```

### subsubsection: The Orchestrate Script  `sec:prelim:algorithms:orchestrate`

```
% What Orchestrate does and why it is the target. Command template and per-algorithm
% defaults are recorded in AIG_DATASET_README.md; the abc invocation is
%   strash; orchestrate
% with deterministic=true, max_passes=20, score_metric=area_delay.
% Say enough that "optimizability under Orchestrate" is a well-defined quantity.
```

### subsubsection: The Non-Target Scripts: Deepsyn, Syn4, and C2RS  `sec:prelim:algorithms:others`

```
% One sentence each on what \texttt{Deepsyn}, \texttt{Syn4} and \texttt{C2RS} actually do,
% sourced from AIG_DATASET_README.md and abc.rc. C2RS is an alias expanding to a long
% balance/resub/rewrite chain -- give the expansion, not the alias name.
%
% These belong here rather than in Chapter 3
% (\ref{sec:method:data:generation:algorithms}) for two reasons:
%   - They are not merely "also on disk". Three quarters of the corpus is their output:
%     every tier-1 graph was produced by one of them (\ref{sec:method:data:tier1}), and how
%     much Orchestrate can still remove from a graph depends sharply on which of the three
%     ran first (\ref{fig:dataset_label_by_source}). The reader needs to know how they
%     differ before that figure can be read.
%   - They are the named extension in \ref{sec:conclusion:futurework:algorithms}, which
%     only means something if the reader knows how each differs from Orchestrate.
% Keep it to one sentence each; command templates and parameters stay in
% \ref{sec:apx:first_appendix}.
```

### subsubsection: Message Passing Framework  `sec:prelim:gnn:mp`

```
% Standard neighbourhood aggregation \citep{kipf2017gcn,hamilton2017graphsage}. Establish
% the notation Chapter 3 reuses, and make the point that after L layers a node's
% representation depends on exactly its L-hop neighbourhood -- the fact every depth-coupling
% argument in this thesis rests on (colour-refinement depth d = L, SGC depth = L, receptive
% field = L hops).
```

### subsubsection: Graph-Level Readout and Pooling  `sec:prelim:gnn:readout`

```
% Pooling node representations to one graph vector; jumping knowledge \citep{xu2018jk}.
% Note the constraint that shapes the whole experimental setup: a graph-level target means
% every node of a graph must be present in the same forward pass, so a graph cannot be
% split across batches. That is why batching is by node budget and why the baselines need
% gradient accumulation (\ref{sec:method:architecture:baselines}).
```

### subsubsection: Positional and Structural Encodings  `sec:prelim:gnn:pe`

```
% Why message passing alone does not see position, and what a PE adds. This thesis uses a
% level-based PE, which is a DAG-native choice: on an AIG the topological level is a
% meaningful absolute coordinate, unlike on an undirected graph.
% Merge candidate: could fold into \ref{sec:prelim:gnn:mp} if space is short.
```

### subsubsection: Expressivity and the Weisfeiler--Leman Hierarchy  `sec:prelim:gnn:expressivity`

```
% The theoretical core of the exact-compression contribution. Define, in this order:
%   - COLOUR REFINEMENT / 1-WL: iteratively refine node colours by the multiset of
%     neighbour colours.
%   - EQUITABLE PARTITION, and the fact that colour refinement computes the coarsest one
%     \citep{grohe2014colourrefinement}.
%   - The expressivity bound: message-passing GNNs are at most as discriminative as 1-WL
%     \citep{xu2019gin}.
%   - BISIMULATION as the set-valued (rather than multiset-valued) variant, and the
%     count-cap c that interpolates between them: c = infinity is colour refinement,
%     c = 1 is bisimulation \citep{bollen2023exact}.
%   - ORBITS: automorphism orbits form an equitable partition, and a permutation-equivariant
%     network gives identical representations to nodes in the same orbit -- so quotienting
%     by an equitable partition is lossless.
% The AIG-specific consequence, worth stating here and measuring later: interchangeable AND
% fanins, replicated datapath bit-slices (adders, multipliers, registers are many isomorphic
% cones) and repeated standard sub-logic create real orbits that a random graph does not
% have. That is the structural reason to expect AIGs to compress well under this method.
```

### subsubsection: Training Cost, Memory, and Inference  `sec:prelim:gnn:cost`

```
% Defines the vocabulary the whole motivation leans on but never introduces:
% VRAM, the memory bottleneck and out-of-memory failure, distributed/multi-GPU training,
% and the training-vs-inference distinction (activations and gradients are stored for the
% former, not the latter -- which is the mechanism RQ5 depends on).
%
% Also define here, because Chapter 3's numbers are meaningless without them:
%   - ALLOCATED vs RESERVED memory (the thesis reports allocated).
%   - MIXED PRECISION (bf16), which changes every activation-memory figure.
%   - GRADIENT CHECKPOINTING -- trading recomputation for memory. This is what makes the
%     DeepGate4 baseline runnable at all (\ref{sec:method:architecture:baselines}).
%   - GRADIENT ACCUMULATION and effective batch size.
%   - OVER-SQUASHING \citep{alon2021oversquashing}: a fixed-width representation cannot
%     carry the information of an exponentially growing receptive field, so long-range
%     signal is crushed. Needed by hypothesis H1 (\ref{sec:intro:rq3}); the connection to
%     coarsening is that contracting a path shortens the distance signal must travel
%     \citep{dwivedi2022lrgb}.
```

### subsection: Graph Reduction: Definitions  `sec:prelim:reduction`

```
% Introduce the three-family taxonomy used throughout, following \citet{hashemi2024survey}:
% sparsification, coarsening, condensation. Note that this thesis substitutes PARTITIONING
% for condensation as the third family, because partitioning is what a distributed training
% setup forces on you and condensation is excluded for the reasons in
% \ref{sec:relwork:reduction:summarization}. Say so explicitly -- a reader who knows the
% survey will otherwise read the substitution as an error.
```

### subsubsection: Partitioning  `sec:prelim:reduction:partitioning`

```
% Splitting a graph into k parts, cutting edges that span parts. Distinguish from the other
% two families on the property that matters here: partitioning KEEPS EVERY NODE and removes
% only edges, so its node compression ratio is always 1.
```

### subsubsection: Sparsification  `sec:prelim:reduction:sparsification`

```
% Removing edges (or nodes, and their incident edges) while preserving some structural
% property. Note that node-removing and edge-removing methods differ in a way that matters
% for measurement -- see \ref{sec:prelim:reduction:measuring}.
```

### subsubsection: Summarization and Coarsening  `sec:prelim:reduction:summarization`

```
% Merging nodes into SUPER-NODES. Define, because Chapter 3 uses all of them:
% super-node, QUOTIENT GRAPH, the merge map (a surjection from nodes to clusters), EDGE
% MULTIPLICITY and the MULTIGRAPH that results when parallel super-edges are kept rather
% than collapsed. State that a coarsening is exactly a partition of the node set, which is
% why "within +/- 1 level" cannot define one (it is not transitive) -- a point Chapter 3
% depends on.
```

### subsubsection: Measuring Reduction: Compression Ratio  `sec:prelim:reduction:measuring`

```
% Node vs edge retention, and why the distinction matters here: edge-mask methods
% preserve node counts while node-mask methods do not, so a single "compression"
% number is ambiguous. This is the definition RQ4's matched-compression comparison
% rests on, so it has to be pinned down before the Results use it.
%
% Fix the convention explicitly and use it everywhere afterwards:
%   retention = kept / original; reduction = 1 - retention;
%   reported per node and per edge, never as one number.
% State the consequence for RQ4 that motivated its rewording: methods differ in whether
% compression is a KNOB or a fixed outcome of the graph, so "compare at matched
% compression" is a design problem, not just a reporting convention. See
% \ref{sec:method:summary:random}.
```

### subsection: Problem Formalization  `sec:prelim:formalization`

```
% Formal statement, referenced throughout the Methodology. Sketch:
%   G = (V, E, X, A) an AIG with node features X and edge attributes A;
%   an algorithm S (here Orchestrate) maps G to S(G);
%   the target y(G) = 1 - |V(S(G))| / |V(G)| in [0, 1], the optimizability;
%   a reduction operator R maps G to a smaller graph R(G);
%   f_theta is the encoder, trained to approximate y.
% Then every research question is one statement about the same objects:
%   RQ1: how well does f_theta(G) approximate y(G)?
%   RQ2: what do |R(G)|, memory and time cost, and what does computing R cost offline?
%   RQ3: how much worse is f_theta trained and tested on R(G)?
%   RQ4: does R chosen with AIG knowledge beat generic R at equal |R(G)|?
%   RQ5: does f_theta trained on R(G) still approximate y(G) when given G?
% Writing this out is what makes the five RQs read as one study rather than five topics,
% and it is where the feature-schema compatibility requirement (a super-node graph and a
% full graph must live in the same input space, so one set of weights ingests both)
% belongs.
```

### section: Related Work  `sec:relwork`

```
% Template guidance: announce the research gap at the START of this section and
% return to it at the END. Not a laundry list -- every citation should support the
% gap or motivate a design decision. Consider stating hypotheses.
%
% OPENING PARAGRAPH (to write): state the gap up front -- graph reduction is a mature
% literature, circuit representation learning is a mature literature, and no work applies
% the first to whole AIGs as input reduction for the second. The four subsections then
% establish each half and the two attempts to bridge them.
```

### subsubsection: Quality-of-Result and Optimizability Prediction  `sec:relwork:ml4eda:qor`

```
% The direct precedents for the task. OpenABC-D \citep{chowdhury2021openabc} is the
% reference large-scale ML4EDA dataset and publishes graph-level labels including
% "percentage of nodes optimized" -- essentially the label used here, which is what makes
% its SynthNet a fair baseline rather than a repurposed model. Joint recipe+circuit models
% \citep{qor_transformer2022} and LOSTIN \citep{wu2022lostin} predict QoR for a given
% recipe; note LOSTIN's use of a super-node to encode the synthesis SEQUENCE, which is a
% temporal use of the same device this thesis uses structurally -- a terminological
% collision worth heading off explicitly.
```

### subsubsection: Evaluation Protocols and the Unseen-Design Gap  `sec:relwork:ml4eda:protocols`

```
% Grounds RQ1a. Keep to one tight paragraph + optional small table. Bullets to cover:
%   - "unseen" means three different things in this literature: unseen recipe, unseen
%     design/IP, unseen design-recipe PAIR (both entities seen). The third is much easier
%     than the second and is easily mistaken for it.
%   - the gap is the stated motivation of much of the field, but matched seen/unseen
%     results are rare: of the papers surveyed for this thesis, only OpenABC-D
%     \citep{chowdhury2021openabc}, LOSTIN \citep{wu2022lostin}, LSOformer, and
%     Jiang et al.'s XGBoost AIG-timing predictor report both sides. HOGA
%     \citep{deng2024hoga} motivates on the gap but publishes no seen-design baseline,
%     so its degradation cannot be computed.
%   - the canonical unseen-IP split (OpenABC-D V2) trains on 16 small IPs and tests on 8
%     large ones -- design novelty confounded with size extrapolation.
%   - benchmark diversity dominates architecture: the same task family yields ~1-3\% MAPE
%     on 11 homogeneous EPFL designs (LOSTIN) but 22\%+ on OpenABC-D (LSOformer).
%   - error is design-concentrated, not uniform (OpenABC-D: aes_xcrypt/wb_conmax carry
%     it); pooled means hide this, motivating the per-design reporting used here.
```

### subsubsection: Learned Synthesis Script Generation  `sec:relwork:ml4eda:scripts`

```
% The downstream application the motivation invokes but this thesis does not attempt
% (\ref{sec:intro:scope:prediction}). Keep short -- it exists to show the prediction task
% is worth solving, not to be surveyed.
```

### subsubsection: Circuit Representation Learning  `sec:relwork:ml4eda:representation`

```
% The DeepGate line \citep{li2022deepgate,shi2024deepgate3,deepgate4_2025} and PolarGate
% \citep{polargate2024}. Two things to extract, both of which shape decisions in Chapter 3:
%   1. PolarGate identifies polarity/functionality as the representational bottleneck for
%      AIG GNNs, which is the external support for treating inverter polarity as a
%      first-class relation that summarization must preserve rather than average away.
%   2. DeepGate3/4 scale the MODEL where this thesis reduces the INPUT. That contrast is
%      the cleanest one-sentence statement of the contribution, and DeepGate4 doubles as a
%      baseline (\ref{sec:method:architecture:baselines}), so the comparison is not merely
%      rhetorical.
% Also note what these papers do for scale in practice: they train on extracted
% subcircuits of tens to a few thousand gates. That IS a reduction, it is simply never
% named or evaluated as one -- which is half of the gap.
```

### subsubsection: Benchmarks and Datasets for ML in EDA  `sec:relwork:ml4eda:datasets`

```
% OpenABC-D \citep{chowdhury2021openabc} and OpenLS-DGF; benchmark framing from "Towards
% the Imagenets of ML4EDA". Position the dataset built here against them: custom, larger,
% and tiered by optimization state (\ref{sec:method:data:tiers}) rather than by recipe.
% TODO(citation): OpenLS-DGF and the "Imagenets of ML4EDA" position paper are named in the
% working notes without full references. Add them to references.bib or drop them.
```

### subsubsection: Sampling-Based Approaches  `sec:relwork:scaling:sampling`

```
% Neighbour and subgraph sampling \citep{hamilton2017graphsage}. Why it is not the
% approach taken here: sampling is a per-epoch stochastic operation for node-level tasks,
% whereas a graph-level target needs the whole graph in one forward pass, and the reduction
% has to be a deterministic, cacheable, offline artifact if it is to be reused across runs.
```

### subsubsection: Partitioning and Distributed Training  `sec:relwork:scaling:distributed`

```
% The standard answer to a graph that does not fit: partition it across devices. Note that
% this is the setting in which cutting edges is forced rather than chosen, which is why
% partitioning is included as a reduction family here even though it removes no nodes.
```

### subsubsection: Memory-Efficient Training Techniques  `sec:relwork:scaling:memory`

```
% Gradient checkpointing, mixed precision, accumulation. These are orthogonal to input
% reduction and are used here rather than studied -- but they must be named, because the
% baselines depend on them (DeepGate4 is only runnable with checkpointing) and because a
% reader will otherwise ask why reduction is needed when checkpointing exists. The answer
% to state: they trade compute for memory at a fixed input size; reduction changes the
% input size, and the two compose.
```

### subsubsection: Graph Partitioning  `sec:relwork:reduction:partitioning`

```
% METIS \citep{karypis1997metis} as the classical balanced min-cut partitioner and the one
% used here. Note that its objective -- minimise cut edges subject to balance -- is
% topology-blind, which is exactly what the span-weighted variant in
% \ref{sec:method:partition:spanmetis} modifies.
```

### subsubsection: Graph Sparsification  `sec:relwork:reduction:sparsification`

```
% Spectral sparsification and spanners as the principled end of the family. Worth one
% sentence of the negative result found here: spanner-based sparsification was implemented
% and abandoned because AIGs lack the dense cyclic redundancy spanners exploit
% (\ref{sec:method:sparse:forest}). That is a small domain finding and belongs in the
% record.
```

### subsubsection: Graph Summarization and Coarsening  `sec:relwork:reduction:summarization`

```
% The largest related-work subsection, because it is where the methods come from. Group as:
%   - EXACT / equivalence-based: colour refinement and bisimulation
%     \citep{bollen2023exact}, the equitable-partition foundation
%     \citep{grohe2014colourrefinement}, attribute/IO summaries and k-SNAP
%     \citep{tian2008ksnap} -- noting k-SNAP is the depth-1 case of graded refinement and
%     is therefore cited rather than run.
%   - SPECTRAL / classical: Local Variation and heavy-edge matching
%     \citep{loukas2019localvariation}, the scientific-computing lineage
%     \citep{chen2022coarsening}, Kron reduction and its directed extension
%     \citep{sugiyama2023kron}.
%   - GNN-AWARE: ConvMatch \citep{dickens2024convmatch}, which merges nodes equivalent with
%     respect to the graph convolution itself rather than to a graph property, reporting
%     ~95\% of performance at 1\% of size on node classification.
%   - HASHING / cheap: UGC and AH-UGC \citep{ugc2024}, linear-time LSH-based coarsening.
%   - SUMMARIZATION-WITH-GNNS survey framing \citep{shabani2023summarization}.
%
% CONDENSATION, and why it is excluded (\ref{sec:intro:scope:reductions}) -- state all four
% reasons here since this is the section a reader will look for them in:
% GCond/DosCond \citep{jin2022gcond} synthesise a new small graph rather than merging the
% real one, so (1) there is no correspondence between synthetic and real nodes, (2) the
% method is label-dependent by construction, (3) gradient matching ties the result to one
% architecture, and (4) with no real nodes there is nothing to run cross-state inference
% ON, so it cannot answer RQ5 even in principle.
```

### subsubsection: Reduction as Preprocessing for GNN Training  `sec:relwork:reduction:forgnn`

```
% The specific precedent for this thesis's setup. SCAL \citep{huang2021scal} trains on a
% coarsened graph and infers directly with the same weights -- the shared-weights mechanism
% RQ5 tests. \citet{generale2022scaling} instead transfers weights back through a
% node-to-super-node mapping for a node-level task, and its ablation finding that
% super-node CONTENT was critical is the direct justification for carrying member type
% counts and level statistics on super-nodes (\ref{sec:method:summary:mergemap}).
% \citet{buffelli2022sizeshift} frames the same problem as size shift.
% State the distinction that makes RQ5 well-posed: for a NODE-level task a mapping back is
% required, but for a graph-level regression the weights are already size- and
% identity-agnostic, so shared weights need no mapping at all -- only a compatible input
% feature space.
```

### subsubsection: Reduction on Directed Acyclic Graphs  `sec:relwork:domain:dag`

```
% What changes when the graph is directed and acyclic: DAG-native architectures
% \citep{thost2021dagnn,zhang2019dvae}, directed Kron reduction \citep{sugiyama2023kron},
% forward vs backward refinement direction. Most coarsening theory is stated for undirected
% graphs, and the guarantees do not automatically carry -- which is one reason the
% coarsening methods run here are compared against a random control at matched compression
% rather than trusted on their stated guarantees.
```

### subsubsection: Reduction in Circuit and Netlist Domains  `sec:relwork:domain:circuits`

```
% Two distinct things happen in this space and conflating them would be an error.
%   1. AIG-NATIVE reduction already exists but is not framed as graph summarization:
%      structural hashing merges structurally identical nodes, FRAIG
%      \citep{mishchenko2005fraig} merges functionally equivalent ones, and MFFC-based
%      decomposition \citep{mishchenko2006rewriting} is how refactoring already carves the
%      graph. These are equivalence-based coarsenings by another name, and connecting them
%      to the generic exactness framework is part of the contribution.
%   2. CTS-Bench \citep{cts_bench2026} is the nearest published competitor and must be
%      handled carefully and fairly. Facts: it benchmarks coarsening trade-offs for GNNs on
%      POST-PLACEMENT gate-level netlists for clock-skew prediction; its coarsening is one
%      bespoke three-step heuristic (BFS from flip-flops, spatial-variance filtering,
%      gravity-vector-aligned merging) that consumes physical XY coordinates and therefore
%      cannot be applied to a pre-physical AIG at all; it surveys none of the coarsening
%      literature. Its results SUPPORT the premise here: up to 17.2x memory reduction and
%      3x training speedup, but accuracy degraded to NEGATIVE R^2 under zero-shot
%      evaluation, and it explicitly calls for domain-aware coarsening.
%      Two consequences to carry forward: it is a warning for RQ5 (expect cross-state
%      transfer to be hard), and its MAE-vs-R^2 behaviour -- MAE barely moved, 0.16 to
%      0.17, while R^2 fell below zero -- is the citation justifying why this thesis
%      reports RMSE, R^2 AND Spearman together rather than a single error number
%      (\ref{sec:method:experiment:metrics}).
```

### subsubsection: Size and Distribution Shift in GNNs  `sec:relwork:generalization:shift`

```
% \citet{buffelli2022sizeshift} on size generalization; the inductive-GNN line
% \citep{hamilton2017graphsage} establishing that learned weights are size-agnostic in
% principle, which is why train-reduced/infer-full is plausible rather than obviously
% doomed.
```

### subsubsection: Train--Test Structural Mismatch  `sec:relwork:generalization:mismatch`

```
% SCAL's direct-inference result \citep{huang2021scal} as the positive precedent, and
% CTS-Bench's negative zero-shot R^2 \citep{cts_bench2026} as the negative one. The two
% together are why RQ5 is an open question rather than a formality, and why it needs a
% positive control (\ref{sec:intro:rq5}).
```

### subsection: Research Gap  `sec:relwork:gap`

```
% Close the chapter by restating the gap, now grounded in the work above.
% Name the key papers the eventual results will be compared against.
%
% Scope the claim precisely -- this is the paragraph a reviewer will attack, so it should
% concede what it must and hold what it can. What CANNOT be claimed after CTS-Bench: to be
% the first coarsening study in EDA, unqualified. What CAN be claimed:
%   - first to benchmark MULTIPLE coarsening families (exact colour
%     refinement/bisimulation, GNN-aware convolution matching, and domain-specific) against
%     a matched random control for GNN training in EDA -- CTS-Bench benchmarks one bespoke
%     spatial heuristic. Do NOT list spectral or hashing here: they are surveyed in
%     \ref{sec:relwork:reduction:summarization}, not run.
%   - first on AIGs, logic synthesis, and optimizability regression specifically;
%   - first to bring PROVABLY LOSSLESS compression into an EDA GNN setting;
%   - first to connect AIG-native equivalence (strash, FRAIG, MFFC) to the generic graph
%     summarization framework;
%   - first controlled three-protocol split comparison (random / recipe-disjoint /
%     design-disjoint) under matched conditions for a node-level AIG prediction task,
%     with per-design error distributions (\ref{sec:relwork:ml4eda:protocols} shows the
%     field reports one protocol at a time, pooled).
% Comparison targets for the eventual results: OpenABC-D/SynthNet
% \citep{chowdhury2021openabc}, HOGA \citep{deng2024hoga} and DeepGate4
% \citep{deepgate4_2025} on the prediction task; CTS-Bench \citep{cts_bench2026} on the
% memory/speed/accuracy trade-off shape.
```

### subsection: Research Gap  `sec:relwork:gap`

```
% Use \cite{} for a reference that is part of the sentence, and \citep{} for
% references in parentheses.
```

## `sections/related-work/prelim-aig.tex`

### subsubsection: Structural Properties: Levels, Logic Cones, Fanout  `sec:prelim:aig:properties`

```
% TODO(measure): the degeneracy of idom is asserted qualitatively, which is deliberate. An
% earlier draft quoted ~99% of AND gates; that figure was never computed on this corpus.
% Measure it before any number goes in this paragraph.
```

## `sections/related-work/prelim-algorithms.tex`

### subsubsection: Rewriting, Refactoring, and Balancing  `sec:prelim:algorithms:primitives`

```
% Define the primitives the target scripts are built from \citep{mishchenko2006rewriting}:
% rewriting replaces a cut with a smaller equivalent; refactoring collapses and
% re-expresses one MFFC at a time; balancing restructures for depth.
%
% Two definitions that must land here because later arguments depend on them:
%   - STRUCTURAL HASHING (strash): merges nodes with identical (fanin, polarity) pairs on
%     construction. Every graph in this corpus is strashed, so the trivial one-hop
%     structural redundancy is ALREADY GONE before any summarization runs -- which is why
%     colour refinement at depth 1 is expected to find almost nothing
%     (\ref{sec:method:summary:wl}).
%   - FUNCTIONAL vs STRUCTURAL equivalence, and FRAIG \citep{mishchenko2005fraig}, which
%     merges functionally equivalent nodes via SAT sweeping. This defines the boundary the
%     summarization methods deliberately stay on the near side of: merging functionally
%     equivalent logic performs part of the optimization being predicted, and would leak
%     the label. That boundary is argued, not measured -- see
%     \ref{sec:discussion:limitations:methods}.
```

### subsubsection: The Orchestrate Script  `sec:prelim:algorithms:orchestrate`

```
% What Orchestrate does and why it is the target. Command template and per-algorithm
% defaults are recorded in AIG_DATASET_README.md; the abc invocation is
%   strash; orchestrate
% with deterministic=true, max_passes=20, score_metric=area_delay.
% Say enough that "optimizability under Orchestrate" is a well-defined quantity.
```

### subsubsection: The Non-Target Scripts: Deepsyn, Syn4, and C2RS  `sec:prelim:algorithms:others`

```
% TODO: confirm the exact &syn4 expansion against the ABC source before this sentence is
% final; AIG_DATASET_README.md records only max_passes=4 and the flow name, not the passes.
%
% These belong here rather than in Chapter 3
% (\ref{sec:method:data:generation:algorithms}) for two reasons:
%   - They are not merely "also on disk". Three quarters of the corpus is their output, and
%     how much Orchestrate can still remove from a graph depends sharply on which of the
%     three ran first (\ref{fig:dataset_label_by_source}). The reader needs to know how they
%     differ before that figure can be read.
%   - They are the named extension in \ref{sec:conclusion:futurework:algorithms}, which only
%     means something if the reader knows how each differs from Orchestrate.
% Keep it to one sentence each; command templates and parameters stay in
% \ref{sec:apx:first_appendix}.
```

## `sections/related-work/prelim-synthesis.tex`

### section: Preliminaries  `sec:prelim`

```
% What a reader needs to know to follow the rest of the thesis. Definitions and
% notation only -- no argumentation, no comparison to other work.
%
% PAGE BUDGET: at 35 pages two-column, this section gets ~4 pages. It currently has
% 20 subsubsections. Several will have to become a sentence inside a neighbouring
% subsection rather than a heading of their own -- the ones flagged "merge candidate"
% below are the first to go.
```

### subsubsection: Position of Logic Synthesis in the Design Flow  `sec:prelim:synthesis:flow`

```
% Define: Electronic Design Automation (EDA); the flow from RTL through logic synthesis
% and technology mapping to placement and routing. One paragraph. The only load-bearing
% point for this thesis: logic synthesis is PRE-PHYSICAL -- an AIG carries no coordinates.
% That is what makes CTS-Bench's spatial coarsening inapplicable here
% (\ref{sec:relwork:domain:circuits}) and it needs to be established before that claim.
```

### subsubsection: Optimization Objectives: Area, Power, Delay  `sec:prelim:synthesis:objectives`

```
% Define the three classical objectives. State that this work uses node count as an area
% proxy and models nothing else -- forward-reference \ref{sec:intro:scope:target}, and note
% that the construct-validity question is taken up in \ref{sec:discussion:limitations:validity}.
```

### subsubsection: Synthesis Scripts and Pipelines  `sec:prelim:synthesis:scripts`

```
% Define: a script/recipe is an ordered sequence of transformation commands; the result
% depends on the order, and the best sequence is design-dependent. The motivating statistic
% is that the best recipes across designs overlap by under 30\%, which is why script choice
% cannot be fixed once and reused -- source it before citing.
% TODO(source): the <30\% overlap figure is recorded in summarization_notes.md without a
% citation. Find the paper or drop the number.
```

## `sections/3-methodology.tex`

### chapter: Methodology  `sec:methodology`

```
% Focus on what you add to the existing method. Explain what you will do and why (and how). Do not forget to characterize your research design. There should be a sub-section on the evaluation.
%For DS students, this normally means using manually labelled or ground truth data. For IS students, it is not always needed to have a separate methodology section. You can also integrate the approach with the results in one section. It depends on your type of research what is best fitting.
% Write about your methodology here. Focus on your own contribution. Indicate exactly how you will assess your work in terms of evaluation.
% % It is possible to use a separate section for the Experimental Setup, which then focuses on all settings used in your experiments. It also possible to address the settings in a sub-section under Methodology.
% \section{Equations}
%         We estimate the deformable template parameters~$\theta_t$ and the deformation fields for every data point using maximum likelihood. Letting~$\mathcal{V} = \{\boldsymbol{v}_i\}$ and~$\mathcal{A} = \{a_i\}$,
%         %
%         \begin{align}
%                 \hat{\theta_t}, \hat{\mathcal{V}} &= \arg \max_{\theta_t, \mathcal{V}} \log p_{\theta_t}(\mathcal{V} | \mathcal{X},  \mathcal{A}) \nonumber \\
%                 &= \arg \max_{\theta_t, \mathcal{V}} \log p_{\theta_t}(\mathcal{X} | \mathcal{V}; \mathcal{A}) + \log p(\mathcal{V}),
%                 \label{eq:logpost}
%         \end{align}
%         %
%         where the first term captures the likelihood of the data and deformations, and the second term controls a prior over the deformation fields.
```

### chapter: Methodology  `sec:methodology`

```
%         \begin{proof}
%                 Awesome proof.
%         \end{proof}
```

### chapter: Methodology  `sec:methodology`

```
% \section{Long equations in two columns}
%         Note that equations can fill the width pretty quickly when there are two columns.
%         Different mathemtical environments, beyond the basic \verb|\begin{equation}| may be of use. Taking as an example the binomial formula which overflows in Eq. \eqref{binom:eq}:
%         \begin{equation}
%                 \label{binom:eq}
%                 (1+x)^n = \underbrace{(1+x)\times...\times(1+x)}_{n \text{ times}} = \sum_{k=0}^n \binom{n}{k}x^k = \sum_{k=0}^n\frac{n!}{k!(n-k)!}x^k .
%         \end{equation}
```

### chapter: Methodology  `sec:methodology`

```
%         The \verb|\begin{multline}| environment is an easy although crude fix:
%         \begin{multline}
%                 (1+x)^n = \underbrace{(1+x)\times...\times(1+x)}_{n \text{ times}} \\
%                 = \sum_{k=0}^n \binom{n}{k}x^k \\
%                 = \sum_{k=0}^n\frac{n!}{k!(n-k)!}x^k .
%         \end{multline}
%         \verb|\begin{align}| tends to give the most control over the final result:
%         \begin{align}
%                 (1+x)^n &= \underbrace{(1+x)\times...\times(1+x)}_{n \text{ times}} \nonumber\\
%                         &= \sum_{k=0}^n \binom{n}{k}x^k \nonumber\\
%                         &= \sum_{k=0}^n\frac{n!}{k!(n-k)!}x^k .
%         \end{align}
```

### chapter: Methodology  `sec:methodology`

```
% \section{Math styles}
%         Different font styles can be used for equations:
%         \begin{itemize}
%                 \item \verb|$a b c A B C 1 2 3$|: $ a b c A B C 1 2 3 $
%                 \item \verb|$\mathbf{a b c A B C 1 2 3}$|: $ \mathbf{a b c A B C 1 2 3} $
%                 \item \verb|$\mathfrak{a b c A B C 1 2 3}$|: $ \mathfrak{a b c A B C 1 2 3} $
%                 \item \verb|$\mathcal{ABC}$|: $ \mathcal{ABC} $
%                 \item \verb|$\mathbb{ABC}$|: $ \mathbb{ABC} $
%                 \item \verb|$\mathsf{ABC}$|: $ \mathsf{ABC} $
%                 \item \verb|$\mathtt{ABC}$|: $ \mathtt{ABC} $
%         \end{itemize}
```

### chapter: Methodology  `sec:methodology`

```
%         Text and names in equations should be dealt with the \verb|\text| command, for instance:\\
%         \verb|$\mathcal L_{\text{SuperLoss}}$|: $\mathcal L_{\text{SuperLoss}}$ and not $\mathcal L_{SuperLoss}$.
```

## `sections/methodology/architecture.tex`

### section: Architecture  `tab:method:architecture`

```
% Degree normalisation was disabled because it performed better here; GNN+ reports the same
% finding. Add the pointer to their ablation table before this is defended.
```

### paragraph: Edge features enter the message (GNN+ Eq.~6).

```
% The paper typesets Eq. 6 with sigma applied once to the completed sum; the
% implementation applies it per message, which is the OGB edge-aware GCN convention GNN+
% cites (its Sec. 3.1) as the source of edge integration. Worth one check against
% github.com/LUOyk1999/GNNPlus before this sentence is defended in a viva.
```

### paragraph: What is set here rather than inherited.

```
    % TODO: one sentence on why level over RWSE. The mechanical reason is cost (RWSE is
    % a per-graph precompute on graphs of this size); if there is a structural argument
    % as well, it belongs here.
```

### paragraph: Batching deviations, and why they are deviations in the papers' direction.

```
% Full derivations, measured virtual-edge counts, activation-memory estimates and the exact
% list of which hyperparameters are published vs assumed are in src/train_baseline.py's
% module docstring and src/baselines/*/regressor.py. Condense, do not re-derive.
```

### paragraph: A baseline that fails is a result, and needs care.

```
% Detail and the diagnostic script live in src/baselines/openabc_synthnet/DIAGNOSIS.md.
% TODO: run src/diagnose_synthnet_baseline.py against the checkpoint and quote the measured
% per-split target mean and std rather than the values inferred from the metrics.
```

## `sections/methodology/data.tex`

### subsection: Source Circuits  `tab:source_circuits`

```
% SETTLED (was TODO 1 and 2):
%   - Provenance. 29 of the 47 real designs are OpenABC-D's own, listed in the dataset
%     README at the NYU record; the other 18 are EPFL (8) / ISCAS-85 (4) / MCNC (6). Both
%     the 29-design list and the BSD-3-Clause license are confirmed at
%     https://ultraviolet.library.nyu.edu/records/mw6q2-a8p15 (DOI 10.58153/mw6q2-a8p15)
%     and https://github.com/NYU-MLDA/OpenABC. EPFL suite: MIT, https://github.com/lsils/benchmarks.
%   - Acquisition. All 47 came from the OpenABC-D distribution's bench/ directory, reached
%     via the NYU link on that GitHub README. Both URLs are now in the text.
% STILL TODO:
%   1. FIX THE README. data/DATA_README.md and AIG_DATASET_README.md both say the corpus
%      "combines two AIG sources" and attribute all 47 non-synthetic designs to OpenABC-D.
%      Correct both files so the thesis and the repo agree.
%   2. The license column is the upstream suite's. The 18 classical benchmarks were
%      redistributed inside a BSD-3-Clause dataset, so state which term governs the copy
%      actually used before any release (\ref{sec:discussion:ethics}).
%   3. The SEED and the exact tool path for the eight synthetic designs. What IS settled:
%      the generator is mockturtle's random_aig_generator (its parameters and its
%      constant-node-as-first-PO signature match all eight files exactly), and the
%      parameters are num_pis = the design name, num_gates = 8x that.
%      What is NOT settled, and should not be guessed in the text:
%        - The seed. aigverse's default (3405688830 = 0xcafeaffe) does NOT reproduce these
%          files, and neither does any seed in 0..4999 under aigverse 0.1.1. Writing a seed
%          in would be a false methods claim, so the text says the seed is unrecoverable.
%        - The route. It was NOT aigverse: random_aig landed in aigverse on 2026-03-09
%          (v0.1.0), five weeks AFTER these AIGs were committed as random_aigs.zip on
%          2026-02-03, and neither 0.0.26 nor 0.0.27 exposes any generator at all. So it was
%          mockturtle directly, or some other wrapper around it. Recover the actual script
%          if it still exists anywhere outside the repo.
%      This matters more than it looks: these designs behave unlike every real design in the
%      label distribution (16384 and 8192 reach a maximum y of 0.0003 over 32,000 evaluation
%      graphs, i.e. Orchestrate is at a fixed point on them).
```

### subsubsection: Random Structural Transformation  `sec:method:data:generation:transform`

```
% One claim here still wants a measurement before the results lean on it:
%   dch (step 21) builds a choice network, so step-21 graphs are not plain AIGs and abc's
%   own node count understates them; see the comment at src/data/dataset.py:65, which
%   measured 43/200 mismatches. That is 1/21 of tier 0. Decide whether it belongs here or
%   in \ref{sec:method:data:statistics}.
% The distinctness count that used to sit here is now a TODO in
% \ref{sec:method:data:statistics}, where the corpus is described numerically.
```

### subsubsection: Target Synthesis Algorithms  `sec:method:data:generation:algorithms`

```
% What each script does is defined once in the preliminaries --
% \ref{sec:prelim:algorithms:orchestrate} for the target,
% \ref{sec:prelim:algorithms:others} for the other three. Do not restate them here; this
% subsubsection records only which scripts were run and which one is the prediction target.
```

### subsection: Tiered Dataset Structure  `sec:method:data:tiers`

```
% COUNT: 231,055 (tier 0) + 3 x 231,055 (tier 1) = 924,220 graphs carrying an
% Orchestrate label, of which ~874,220 remain after the 50,000-graph
% hyperparameter holdout (\ref{sec:method:data:hpsubset}).
%
% THREE tier-1 scripts, not four. Measured on the evaluation splits: the tier-1
% to tier-0 ratio is 3.00 in every one of the 11 val+test designs, and no
% Orchestrate-sourced tier-1 graph appears anywhere. That is right by
% construction -- an Orchestrate-produced graph relabelled by Orchestrate IS the
% discarded sequential step of \ref{sec:method:data:tiers:labelling}, so it
% cannot be a sample. The observed 16,087 graphs per design matches the 15,895
% this predicts to within 1.2%.
%
% An earlier draft said "over 3.9 million", which counted the sequential-
% optimization outputs as a third tier. Those runs happened but their graphs
% were never saved, so they are label sources rather than samples.
%
% Confirm the exact figure before it goes in the text:
%   wc -l data/designs/design_metadata/algo_Orchestrate_ml.csv
%   awk -F, 'NR>1 {print $TIER_COLUMN}' ...algo_Orchestrate_ml.csv | sort | uniq -c
```

### subsubsection: Label Distribution  `sec:method:data:label:distribution`

```
% BLOCKED: needs a histogram plus per-tier summary statistics (mean, std, quantiles).
% This is not cosmetic -- three later claims depend on the numbers:
%   1. R^2 is a ratio against the target's own variance, so a narrow label distribution
%      makes a respectable RMSE compatible with a near-zero or negative R^2. The baseline
%      diagnosis in \ref{sec:method:architecture:baselines} turns on exactly this.
%   2. If the distribution is concentrated, a constant predictor is a strong baseline and
%      the tier-1 comparison in \ref{sec:results:rq1:baselines} carries most of the weight.
%   3. Tiers differ by construction: a tier-1 graph has already been through one full
%      synthesis script, so Orchestrate has less left to remove than on a tier-0 base
%      graph. MEASURED: the difference is real but smaller than the difference between
%      SOURCE SCRIPTS -- a Syn4-derived tier-1 graph is far more optimizable than a
%      C2RS- or Deepsyn-derived one (fig:dataset_label_by_source). Report the source
%      script alongside the tier; on its own the tier is the weaker cut of the data.
% Report per tier AND pooled; the pooled distribution is what the model actually sees.
```

### subsection: Dataset Statistics  `sec:method:data:statistics`

```
% BLOCKED on a stats pass over the corpus. Report, per tier and pooled: node and edge count
% distributions (min/median/max and a histogram), depth distribution, and the ratio of AND
% gates to interface nodes.
% Three things downstream need specific numbers from here:
%   - the graph-scale bound in \ref{sec:method:experiment:scale} is justified by the node
%     count distribution, and currently states the bound without showing the distribution
%     it was drawn from;
%   - the node-budget batch sizes in \ref{sec:method:experiment:batching} only make sense
%     against the mean and maximum graph size;
%   - the AND-gate fraction is what determines and_gate_only's compression
%     (\ref{sec:method:sparse:andgate}), so it explains a measured retention figure rather
%     than merely accompanying it.
% Worth adding as a structural statistic, since the summarization argument rests on it:
% if the probe is run, the residual redundancy after strash at refinement depths 1-4
% (\ref{sec:method:summary:wl}).
%
% TODO(distinctness): how many of the 231,055 tier-0 graphs are actually distinct.
% \ref{sec:method:data:generation:transform} deliberately does NOT claim "structurally
% distinct": a pass that finds nothing to do leaves the graph unchanged, so consecutive
% steps of one recipe can coincide. Hash each AIG after strash, count the distinct hashes,
% and report the number here -- the corpus size is only meaningful next to it. Then the
% generation text can state the figure instead of avoiding the adjective.
```

### subsection: Dataset Statistics  `tab:corpus_stats`

```
% tab:corpus_stats is generated by data/creation/corpus_tier_stats.py from the per-graph ML
% CSV, which lives on the cluster. Regenerate with:
%   python data/creation/corpus_tier_stats.py --latex > media/tables/corpus_tiers.tex
% The \IfFileExists guard in the .tex prints a visible placeholder until that file exists,
% rather than silently dropping the table. The old fabricated version has been deleted.
```

### subsection: Tiered Dataset Structure  `sec:method:data:tiers` (counts, lifted 2026-08-01)

```
% Counts: 231,055 tier-0 graphs plus three times that at tier 1 gives 924,220 carrying an
% Orchestrate label, of which about 874,220 remain after the 50,000-graph hyperparameter
% holdout. Three tier-1 scripts rather than four, since an Orchestrate-produced graph
% relabelled by Orchestrate is the discarded sequential step of
% \ref{sec:method:data:tiers:labelling}. The measured tier-1 to tier-0 ratio is 3.00 in
% every evaluation design.
```

## `sections/methodology/experiment-metrics.tex`

### paragraph: Four memory quantities, not one.

```
% MISMATCH: the savings pipeline currently divides raw peak allocated memory, not
% \eqref{eq:method:marginal-memory}, even though the benchmark records the floor and the
% increment. Either the savings are recomputed on Delta m or this paragraph overstates what
% is reported. Do not defend the paragraph until the numbers behind it use the floor.
```

### subsubsection: Reduction Quality Metrics  `sec:method:experiment:metrics:reduction`

```
% \ref{sec:prelim:reduction:measuring} has to fix eta_V and eta_E as THE retention symbols
% when that stub is written. They are used here and in the Results, and belong in one place.
```

### paragraph: Effective receptive field.

```
% NOT YET IMPLEMENTED. Until it is, H1 is asserted and not evidenced -- see
% \ref{sec:intro:rq3}. Either build it or downgrade H1 to a discussion point; do not report
% it as tested.
```

## `sections/methodology/experiment-reproducibility.tex`

### subsubsection: Numerical Determinism  `sec:method:reproducibility:determinism`

```
% OUTLINE -- this is the gap between "same seeds, same versions" and "same numbers".
% Versions are pinned in \ref{sec:method:experiment:hardware}; do NOT restate them here.
%   - State plainly whether GPU execution is bit-deterministic. It is not by default, and
%     saying so matters because \ref{sec:method:reproducibility:seeding} would otherwise
%     imply more than it delivers.
%   - Two independent sources: PyG's scatter/gather aggregation is nondeterministic on GPU
%     (atomics, order-dependent float addition), and mixed precision
%     (\ref{sec:method:experiment:training}) adds numerical drift on top.
%   - Say what this does and does not threaten: it perturbs the last digits, not the
%     ranking of reduction methods, which is what the thesis actually claims. Quantify it
%     if cheap -- rerunning one configuration twice at the same seed bounds the effect and
%     costs one run.
%   - Snellius/SLURM allocation needed to reproduce the environment: module set
%     (\texttt{module load 2025}), partition (\texttt{gpu\_h100} / \texttt{genoa}), node and
%     GPU count per job.
```

### subsubsection: Code and Data Availability  `sec:method:reproducibility:availability`

```
% OUTLINE -- code will be released. Decide and state:
%   - Hosting and license for the code itself (e.g. GitHub, MIT/Apache/etc.).
%   - What ships alongside the code: trained checkpoints, the reduction precompute artifacts
%     of \ref{sec:method:reproducibility:caching}, and/or the generated AIG corpus itself
%     versus only the generation scripts (\texttt{data/creation/}), given its size.
%   - Licensing of the underlying source circuits (OpenABC-D and the second source,
%     \ref{sec:method:data:sources}) -- this can restrict redistribution independently of
%     this thesis's own code license, and is also a dependency of the ethics section
%     (\ref{sec:discussion:ethics}).
%   - Release timing: at submission, or held until after examination/publication.
```

## `sections/methodology/experiment-setup.tex`

### subsubsection: Data Splitting  `tab:split_protocols`

```
% Results to gather for RQ1a:
%   - IN FLIGHT: the two additional training runs at the headline config (split_by=random
%     and split_by=recipe; the design-disjoint run already exists). Identical encoder,
%     budget and seed. Until they land, the random and recipe rows of tab:rq1a_protocol
%     are placeholders carrying [TODO/FAKE] -- see src/analysis/fake_data.py
%     (SPLIT_PROTOCOL). Delete that block and rerun `python -m analysis.make_all` once
%     src/test.py has been run for each.
%   - per-design metrics on the design-disjoint test set, computed post-hoc from the
%     persisted per-graph predictions CSVs (design key recoverable from graph path);
%   - matched seen/unseen comparison: for each design in the design-disjoint test set,
%     its error here (unseen) vs its error in the random-split run (seen) -- same designs,
%     both conditions, no extra training.
```

### subsubsection: Hardware \& Software Environment  `sec:method:experiment:hardware`

```
% TODO: library versions (torch, PyG, Lightning, Optuna, CUDA/cuDNN) from pyproject.toml or
% the lockfile; the abc version used for dataset generation, which is not currently recorded
% anywhere and which results depend on; and the total wall-clock/GPU-hour budget consumed --
% the last is also needed for \ref{sec:discussion:ethics}.
% SCOPE: every version number lives in this subsection. \ref{sec:method:reproducibility:determinism}
% covers only what pinning those versions does NOT fix (GPU nondeterminism, allocation), and
% must cross-reference here rather than restate.
```

## `sections/methodology/reduction-sparsification.tex`

### subsubsection: Random Edge Dropout  `sec:method:sparse:dropout`

```
% Configured rate: 0.3 (config.SPARSIFICATION_RANDOM_DROPOUT_RATE), giving a measured 69.7\%
% edge retention.
% ACTION REQUIRED (see \ref{sec:results:rq4:pairings}): to pair this against spanning forest
% at matched compression the rate must be raised to approximately 0.419, matching spanning
% forest's measured 58.1\% edge retention. At 0.3 the two are not comparable. Changing it
% invalidates the existing random_edge_dropout runs, so decide before the next sweep.
```

### subsubsection: Random Spanning Forest  `sec:method:sparse:forest` (lifted 2026-08-01)

```
% The unused config.SPARSIFICATION_SPANNER_STRETCH constant is a leftover of the abandoned
% spanner attempt.
```

## `sections/methodology/reduction-summarization.tex`

### paragraph: Boundary preservation.

```
% The extra graph attributes (internal_edges, num_pis, num_pos) are attached but the encoder
% never reads them, and the level PE is min-pooled only. Do NOT write as if super-node content
% is being exploited until that is wired into the model.
```

### subsubsection: Domain-Specific Coarsening (Method TBD)  `sec:method:summary:domain`

```
% One implementation finding worth reporting if this method is written up: the width axis
% must use post-dominators, not dominators. Grouping by common immediate dominator is the
% textbook phrasing and degenerates on an AIG -- most AND gates are dominated only by the
% virtual source, so the axis collapses to "merge every gate on a level". The mirror
% degeneracy at the virtual sink means gates reaching two POs without a common gate must be
% excluded from merging. The percentages quoted in an earlier draft were NOT measured on the
% real corpus; measure them before putting any number in the text.
% Also: level bands are fixed windows floor(level / (b+1)), not +-k neighbourhoods (which are
% not transitive and do not define a partition). Widening the band therefore does not
% monotonically increase compression.
% Production settings: cone -- max_chain_length = 4, level_band = 0; MFFC -- absorption capped
% at 64 rounds (real netlists measure 2-3; any prefix of the iteration is a valid partition,
% so the cap is safe rather than approximate).
% NEITHER IS MEASURED ON THE REAL CORPUS. Run one shard
% (sbatch --array=32 src/shell/precompute_summarization.sh) and read the summary stats before
% quoting compression for either.
```

### subsubsection: Graded Colour Refinement / Bisimulation (Adapted, Exact)  `sec:method:summary:wl`

```
% Production setting: depth = NUM_LAYERS (4), count_cap = None (exact), direction = backward.
% OPEN: the residual-redundancy probe (d = 1..4) has not been run. Depth 1 should find almost
% nothing precisely because of strash; that is a reportable result either way. This subsumes
% k-SNAP/IO-summary as its d=1 case -- cite \citep{tian2008ksnap}, do not run it separately.
```

### subsubsection: Convolution Matching (General SOTA Bar)  `sec:method:summary:convmatch`

```
% Production setting: reduction_ratio = 0.5, sgc_depth = NUM_LAYERS, num_probes = 8, seed 42.
% num_probes replaces the reference's exact kNN over the SGC embedding. Checked against the
% authors' reference implementation. This is the most expensive method in the family, so
% precompute must be sharded by graph size -- measure the per-graph cost before quoting it.
```

### subsubsection: Random Within-Type Merging (Control and Matching Instrument)  `sec:method:summary:random`

```
% NOT YET BUILT -- ~10 lines on top of the shared merge rewrite, plus a METHODS registry
% entry, its own precompute, and one training run per matched point. The runs, not the code,
% are the cost.
```

## `sections/4-results.tex`

### chapter: Results  `sec:results`

```
% Give the outcomes for each research question in the form of a table or graphic (with caption).
% Sometimes, especially if you have quite different experiments or research questions, it makes sense to interleave the experimental setup and the results sections, so the reader does not get lost. It is then helpful to structure clearly in (sub)subsections.
```

### chapter: Results  `sec:results`

```
% TABLE SOURCES -- read before editing any table in this chapter.
% Every numbered table below is GENERATED, not hand-written. src/results_to_latex.py reads
% the inference and benchmark CSVs and writes booktabs tables into results/tables/. Copy
% that directory next to msc_thesis.tex (\resultstables points at it) and the \input lines
% below pick them up. Do not paste numbers in by hand: they will drift from the run that
% produced them.
%
% Generated so far:            baseline_accuracy, reduction_efficiency,
%                              predictive_retention, cross_state_generalization,
%                              vram_scaling, and pareto_front.csv (plotted, not tabulated).
% NOT generated yet, needed:   (a) the summarization rows -- the generator loads
%                              sparsification and partition offline stats only;
%                              (b) the baseline-model rows for RQ1 tier 2/3;
%                              (c) the consolidated summary table of \ref{sec:results:summary}.
%
% NOTE: the generator's caption for cross_state_generalization still says "(RQ4)". Cross-state
% generalization is RQ5 in this document; RQ4 is the domain-informed comparison. Fix the
% caption in src/results_to_latex.py, EVALUATION.md, and thesis-overview.tex together.
```

### section: Baseline Predictive Performance on Full AIGs (RQ1)  `sec:results:rq1`

```
% Answers \ref{sec:intro:rq1}: how well does a GNN predict optimizability from the full,
% unreduced AIG, relative to trivial predictors, standard encoders and published circuit
% models? Everything downstream is measured against this.
```

### subsection: Comparison Against Naive Baselines  `sec:results:rq1:baselines`

```
% The three-tier comparison defined in \ref{sec:method:architecture:baselines}. Report all
% three in one table so "accurately" has a referent:
%   Tier 1 -- mean, median, graph-size-only regressor. Establishes what R^2 = 0 looks like.
%   Tier 2 -- GCN, GraphSAGE, GIN under identical pooling/head/training. Isolates the
%             contribution of the edge-aware, positionally-encoded architecture.
%   Tier 3 -- SynthNet, HOGA, DeepGate4. Positions the result against published work.
%
% Read the tiers in order when writing the prose: beating tier 1 shows the task is learnable
% at all; beating tier 2 shows the architecture earns its complexity; beating tier 3 is the
% claim that matters externally, and is the weakest of the three because the published models
% were tuned for other tasks (see the comparability caveats in
% \ref{sec:method:architecture:baselines}).
```

### subsection: Baseline Model Outcomes  `sec:results:rq1:baselinemodels`

```
% Where the published baselines are reported and, where they fail, diagnosed. SynthNet
% collapsed to a constant prediction on this split; \ref{sec:method:architecture:baselines}
% gives the mechanism and the reason it is not simply a porting error (OpenABC-D reports
% negative R^2 on the same split variant).
%
% CRITICAL, do not get this wrong in the table: upstream z-scores its targets per design,
% which removes between-design variance from the R^2 denominator. Their R^2 and ours are not
% the same quantity and must NOT appear in the same column. Report ours, and cite theirs in
% prose with the normalization stated. Only the sign transfers.
```

### subsection: Error Analysis  `sec:results:rq1:erroranalysis`

```
% Where the baseline model fails: residuals broken down by graph size, by tier
% (0/1/2), and by source design. Feeds the Discussion and sets up which reduction
% methods are expected to hurt most.
```

### subsection: Protocol Sensitivity: How Much of the Score Is Design Recognition? (RQ1a)  `sec:results:rq1:protocol`

```
% Answers RQ1a under \ref{sec:intro:rq1}. One table, three rows: random, recipe-disjoint,
% design-disjoint -- identical encoder, budget and data; only the split changes
% (\ref{sec:method:experiment:splitting}). Same metric columns as
% \ref{sec:results:rq1:accuracy}, plus an inflation column: each leakier protocol's score
% relative to the design-disjoint row.
%
% Second exhibit: per-design metrics on the design-disjoint test set, computed from the
% persisted per-graph predictions (design recovered from the graph path). Report the
% distribution, not just the mean -- OpenABC-D finds unseen-design error concentrated in a
% minority of designs, and a pooled mean cannot show which regime this corpus is in. This
% breakdown also decides whether a multi-fold design split is needed: uniform degradation
% supports the single grouped holdout; concentrated degradation means the headline number
% depends on which designs drew the test block.
%
% Results to gather (see also \ref{sec:method:experiment:splitting}):
%   - runs: split_by=random and split_by=recipe at the headline config (design run exists);
%   - per-design groupby over the persisted predictions CSVs (no retraining);
%   - matched seen/unseen per-design pairs from the random-split vs design-split runs.
% Presentation:
%   - Exhibit 1: 3-row protocol table (same columns as \ref{sec:results:rq1:accuracy}
%     + inflation factor vs the design-disjoint row);
%   - Exhibit 2: per-design bar/box plot of RMSE and Spearman on the design-disjoint test
%     set, one bar per held-out design -- distribution, not mean;
%   - Exhibit 3 (small table): seen-vs-unseen error per test design (random-split vs
%     design-split condition), the LOSTIN-style two-sided comparison at design granularity.
%
% Framing for the prose: this is the experiment \ref{sec:discussion:limitations:validity}
% flags for verification -- once written, that limitation upgrades to a settled number and
% its text must be flipped accordingly. Do not frame the design-split scores as a weakness
% of the model: the finding is that leakier protocols overstate performance, which
% strengthens every design-split number in the thesis.
```

### subsection: Graph Reduction Offline Profile  `sec:results:rq2:offline`

```
% One method-level finding belongs in this section rather than in the trade-off analysis,
% because it is a property of the reduction itself and holds regardless of any training run:
% which summarization methods have a compression knob at all. ConvMatch takes a target ratio
% directly; the domain-specific candidate has at most a coarse integer band; colour
% refinement has none. That is what forces the matched-random design
% (\ref{sec:method:summary:random}) and it should be stated here before any matched table.
```

### subsection: Graph Reduction Offline Profile  `sec:results:rq2:offline`

```
% Measured sparsification retention statistics (already collected offline). Superseded by
% the generated reduction_efficiency table above once that includes all three families;
% kept meanwhile because these numbers exist and the generated table does not yet cover
% summarization.
% \begin{table}[htbp]
% \centering
% \caption{Sparsification Retention Statistics over 10,000 Graphs}
% \label{tab:sparsification_stats}
% \small
% \begin{tabular}{llccccr}
% \toprule
% \textbf{Method} & \textbf{Metric} & \textbf{Mean (\%)} & \textbf{Std (\%)} & \textbf{Min (\%)} & \textbf{Max (\%)} & \textbf{Avg Time (ms)} \\
% \midrule
% and\_gate\_only & Edge Retention & 73.3 & 13.0 & 48.9 & 99.1 & 91.85 \\
%                 & Node Retention & 82.1 & 11.4 & 63.3 & 99.9 & \\
%                 & Edge Reduction & 26.7 & --   & --   & --   & \\
% \midrule
% pagerank        & Edge Retention & 62.7 & 6.3  & 41.5 & 76.8 & 1,832.14 \\
%                 & Node Retention & 80.0 & 0.0  & 79.8 & 80.0 & \\
%                 & Edge Reduction & 37.3 & --   & --   & --   & \\
% \midrule
% random\_edge\_dropout & Edge Retention & 69.7 & 0.6  & 66.7 & 70.3 & 26.44 \\
%                       & Node Retention & 100.0& 0.0  & 100.0& 100.0& \\
%                       & Edge Reduction & 30.3 & --   & --   & --   & \\
% \midrule
% spanning\_forest       & Edge Retention & 58.1 & 5.3  & 50.0 & 70.0 & 3,519.05 \\
%                       & Node Retention & 100.0& 0.0  & 100.0& 100.0& \\
%                       & Edge Reduction & 41.9 & --   & --   & --   & \\
% \bottomrule
% \end{tabular}
% \end{table}
```

### subsection: Training Memory Dynamics and Compute Efficiency  `sec:results:rq2:memory`

```
% Host-side memory (process resident set, system utilisation) is logged as well. Report it
% only if it turns out to constrain a configuration; otherwise it is noise in this section.
```

### subsection: Training Memory Dynamics and Compute Efficiency  `sec:results:rq2:memory`

```
% Report peak memory as a function of graph size, not only as a per-configuration scalar:
% memory scales close to linearly with nodes and edges per batch, so a single peak figure
% conflates the reduction's effect with which graphs happened to land in the peak batch.
% \input{\resultstables/vram_scaling}   % tab:vram_scaling
%
% One result to state explicitly because it bounds every memory claim in this chapter: a
% graph larger than the batch budget forms a batch of its own, so peak memory for any method
% that does not reduce NODE counts is floored by the largest circuit in the corpus. That
% predicts the edge-only methods (random edge dropout, spanning forest, and all four
% partitioners) show smaller peak-memory gains than their edge compression suggests -- check
% whether the measurements bear it out, because if they do it is the cleanest illustration in
% the thesis of why node and edge compression must be reported separately.
```

### subsection: Throughput and Training Speedup  `sec:results:rq2:throughput`

```
% Per-graph savings are paired against the baseline by construction, so report bootstrap
% confidence intervals and the Wilcoxon signed-rank test that src/results_to_latex.py already
% computes -- not bare means. Say so in the text; an unqualified speedup figure invites the
% question of whether it is within noise.
%
% Expect speedups to under-deliver relative to compression, and explain why rather than
% treating it as a disappointment: the kernels are latency-bound on message-passing
% gather/scatter (\ref{sec:method:experiment:hardware}), so removing work does not translate
% proportionally into wall-clock. Edge-removing methods should do relatively better here than
% their node compression predicts, which is the mirror image of the memory result above.
```

### subsection: Efficiency Ranking Across Methods  `sec:results:rq2:ranking`

```
% The subsections above are organised by metric; this one cuts the same data the
% other way -- one row per reduction method, ranked on what it buys you.
% How much smaller, how much faster, at what offline cost.
% Tag each method as generic or domain-informed in a column rather than splitting
% the section by it: it is one attribute of a method, not the organising axis.
```

### section: Predictive Retention and Performance Trade-offs (RQ3)  `sec:results:rq3`

```
% Answers \ref{sec:intro:rq3}: how much accuracy survives reduction, and which
% methods sit on the best accuracy-to-efficiency trade-off.
% The domain-informed comparison is its own question -- see \ref{sec:results:rq4}.
```

### subsection: Accuracy Degradation  `sec:results:rq3:degradation`

```
% \input{\resultstables/predictive_retention}   % tab:predictive_retention
%
% Report RMSE, R^2 AND Spearman together throughout, per
% \ref{sec:method:experiment:metrics}: a reduction can hold mean error steady while
% destroying explained variance, which is exactly what \citet{cts_bench2026} observed. A
% table showing only RMSE would miss the failure this chapter exists to detect.
%
% VALIDITY CHECK, report first and separately: the exact colour-refinement configuration must
% land on the full-graph baseline to within numerical tolerance
% (\ref{sec:method:experiment:design}). It is the positive control, not a result. If it does
% not match, everything else in this chapter is suspect and the deviation must be resolved
% before the rest is interpreted -- say so in the text rather than reporting it as one row
% among many.
```

### subsection: Ranking Preservation  `sec:results:rq3:ranking`

```
% Spearman under reduction. Ranking can survive even when absolute error degrades,
% which matters for the script-selection use case in the motivation.
```

### subsection: Accuracy Loss Attributable to Reduction  `sec:results:rq3:attribution`

```
% How much accuracy each method costs, and what structural damage explains it
% (severed causal cones, lost depth, broken longest paths).
% IMPORTANT: compare at matched compression ratio, or "which method is best"
% confounds a genuinely better method with one that simply reduced less.
% The measured retention spread makes this concrete -- and\_gate\_only keeps 73.3\%
% of edges while spanning\_forest keeps 58.1\%, so their raw accuracies are not
% comparable as-is. The formal paired version of this lives in \ref{sec:results:rq4}.
% Present this as one row per reduction method (accuracy retained vs the RQ1
% baseline) so it lines up directly against the efficiency ranking in
% \ref{sec:results:rq2:ranking}. The two tables together feed the Pareto front below.
```

### section: Value of Domain-Informed Adaptation (RQ4)  `sec:results:rq4`

```
% Answers \ref{sec:intro:rq4}. RQ2 and RQ3 rank every method on efficiency and on
% accuracy; this section asks the narrower question of whether AIG-specific
% knowledge bought anything at all.
```

### subsection: Matched-Compression Pairings  `sec:results:rq4:pairings`

```
% Define the comparisons BEFORE reporting them, and state how close each pairing actually is
% -- an unstated mismatch is what turns "method A is better" into "method A reduced less".
%
% Compression is a free parameter for only some methods, which is why RQ4 is phrased as
% "at equivalent compression" with a random control rather than "at matched compression
% ratios" (\ref{sec:intro:rq4}). Three kinds of pairing result:
%
% 1. MATCHED BY CONSTRUCTION -- partitioning. All four partitioners keep every node and
%    differ only in which edges they cut, and k is set by the same size heuristic for all
%    four. So span-weighted METIS vs plain METIS, and level slicing vs random hashing, are
%    matched without any tuning. These are the cleanest pairs in the thesis.
%
% 2. MATCHED BY CALIBRATION -- sparsification.
%    - and_gate_only (82.1\% node retention, parameter-free) vs PageRank at keep_ratio 0.8
%      (80.0\%). Close enough to compare directly; say so explicitly rather than leaving the
%      match to look accidental.
%    - spanning_forest (58.1\% edge retention, parameter-free) vs random_edge_dropout. NOT
%      currently matched: the configured drop rate of 0.3 gives 69.7\% retention. Raising the
%      rate to ~0.419 matches it. Until that run exists, do not report this pair as matched.
%
% 3. MATCHED AGAINST A RANDOM CONTROL -- summarization. No two real summarization methods can
%    be made to meet at the same ratio (colour refinement has no compression knob at all, and
%    the domain-specific candidate has at most a coarse integer band). Each method is
%    therefore paired against random within-type merging run at that method's own achieved
%    compression (\ref{sec:method:summary:random}). State the achieved ratio for each arm.
%
% The domain-vs-generic pair for summarization is the domain-specific method
% (\ref{sec:method:summary:domain}) against ConvMatch -- GNN-aware but domain-blind -- with
% each read against its own random control, since the two cannot be made to meet at one ratio.
```

### subsection: Retention Gap at Matched Compression  `sec:results:rq4:gap`

```
% The paired result: RMSE / R^2 / Spearman difference within each pairing.
%
% HONESTY REQUIREMENT. Each configuration is trained once
% (\ref{sec:method:experiment:metrics:stats}), so there is no run-to-run variance to judge
% these gaps against, and RQ4's expected effect is small. Either re-run the pairings above
% across several seeds -- the cheapest useful version of this is seeds on the RQ4 pairs only,
% not on every configuration -- or report the differences as point estimates and say plainly
% that a small gap cannot be distinguished from noise. Do not report a gap of a few percent
% as a finding without one or the other.
```

### subsection: Structural Interpretation  `sec:results:rq4:interpretation`

```
% What the outcome implies either way. If domain-informed methods do not win, the
% predictive signal is likely distributed rather than concentrated in the causal
% cones and long paths the heuristics were built to protect -- which is a finding
% about AIG representation learning, not just about these four heuristics.
```

### subsection: Adequacy of the Heuristics Tested  `sec:results:rq4:adequacy`

```
% Honest scoping: these are relatively simple adaptations (edge span weighting,
% level slicing, gate-role filtering). A null result bounds these heuristics, not
% the idea of domain-aware reduction in general. Carry this into the Discussion.
```

### subsection: Reduced-to-Full Inference Accuracy  `sec:results:rq5:accuracy`

```
% \input{\resultstables/cross_state_generalization}   % tab:cross_state_generalization
%
% Two caveats that belong in the text next to this table:
%   1. POSITIVE CONTROL FIRST. The exact colour-refinement configuration must transfer
%      essentially perfectly -- it is lossless by construction. Report it before the others
%      so that a poor transfer elsewhere reads as a finding rather than as a possible bug in
%      the evaluation path (\ref{sec:method:experiment:design}).
%   2. SCHEMA ASYMMETRY. For the standard-track methods a full graph needs no conversion: it
%      already is a valid graph of size-1 super-nodes. The exact track is the exception -- it
%      uses a different input schema, so full graphs must be converted before an exact-track
%      model can be queried on them (\ref{sec:method:architecture:exact}). State this;
%      otherwise the two tracks look like the same experiment and they are not.
%
% Prior expectation to set before reporting: \citet{cts_bench2026} found negative R^2 under
% zero-shot evaluation after generic coarsening. If transfer works here at all, that is the
% result -- and there is now a citation showing the generic case fails.
```

### subsection: Generalization by Reduction Family  `sec:results:rq5:byfamily`

```
% Which family survives the structural shift best? Node-count-preserving methods
% (edge dropout, spanning forest) may transfer differently to node-removing ones
% (PageRank, and-gate-only) or to partitioning, which changes the pooling path.
```

### subsection: Inference Cost  `sec:results:rq5:cost`

```
% The practical pay-off: inference needs no gradients or activations, so a model
% trained cheaply on reduced graphs may serve full graphs on modest hardware.
% Report full-graph inference memory and latency, CPU included.
```

### section: Summary of Findings  `sec:results:summary`

```
% One consolidated table: every reduction method as a row, with columns for
%   family (partition / sparsify / summarize), generic or domain-informed,
%   node retention, edge retention, offline cost,
%   peak VRAM vs baseline, speedup vs baseline,
%   matched-state accuracy (RMSE, R^2, Spearman),
%   cross-state accuracy.
% Tag generic vs domain-informed as a COLUMN, not by splitting the table -- it is one
% attribute of a method, not the organising axis.
%
% This is the single artifact the Discussion and Conclusion refer back to, so it is worth
% the page it costs. src/results_to_latex.py does NOT build it today; it writes five
% per-RQ tables and a Pareto CSV. Either add a builder that joins them on method name, or
% drop this section rather than leaving a promise the chapter does not keep.
```

### section: Summary of Findings  `sec:results:summary`

```
% Template examples for tables and figures, kept for reference.
% \section{Table and Figures}
%         \begin{table}[h] % h for Here, t for Top, b for Bottom
%                 \centering
%                 \caption{By convention, Table caption goes on top.}
%                 \begin{tabular}{lcr}
%                         \textbf{Left} & \textbf{center} & \textbf{right} \\
%                         \hline
%                         111 & 222 & 333 \\
%                         444 & 555 & 666
%                 \end{tabular}
%         \end{table}
%
%         \begin{figure}[h]
%                 \centering
%                 \includegraphics[width=0.5\linewidth]{example-image}
%                 \caption[Example figure]{Example figure. Notice that the caption goes below and that there is a way to have a shorter caption for the list of Figures.}
%         \end{figure}
%
%         \begin{figure*}[b]
%                 \centering
%                 \includegraphics[width=0.8\linewidth, height=.2\textheight]{example-image}
%                 \caption{Example figure spanning the two columns.}
%         \end{figure*}
```

### section: Summary of Findings  `sec:results:summary`

```
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  UNSORTED FIGURE AND TABLE GALLERY
%%
%%  Everything below is a holding pen, not a chapter structure. Move each float
%%  up into the section it belongs to, push it to the appendix, or delete it.
%%  Nothing here is referenced from the prose yet.
%%
%%  ALL of it is GENERATED. Rebuild after any new run with:
%%
%%      PYTHONPATH=src python -m analysis.make_all
%%
%%  which writes media/results/figures/*.pdf and media/results/tables/*.tex.
%%  Do not edit a figure or a table by hand: the next rebuild overwrites it, and
%%  a number typed in here drifts from the run that produced it.
%%
%%  FABRICATED DATA. Several floats below are placeholders for runs that do not
%%  exist. They carry a red frame, cross-hatched bars, a diagonal watermark and
%%  a [TODO/FAKE] tag on every invented row. src/analysis/fake_data.py holds
%%  every one of those numbers and names the run that would replace it:
%%
%%      grep TODO_ src/analysis/fake_data.py
%%
%%  Outstanding as of this build:
%%    - the ENTIRE summarization family (no method trained or measured);
%%    - RQ1 baseline tiers 2 and 3 (only SynthNet ran, and it collapsed);
%%    - RQ1a protocol sensitivity (needs --split_by random and --split_by recipe);
%%    - the H1 receptive-field metric (specified, not implemented);
%%    - RQ4 seed variance (every configuration is trained exactly ONCE);
%%    - CPU inference (every surviving inference CSV is device=cuda);
%%    - whole-corpus statistics beyond the evaluation splits.
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
```

### subsection: Corpus and Label  `sec:results:gallery:corpus`

```
% Belongs in \ref{sec:method:data:statistics} and \ref{sec:method:data:label:distribution},
% both of which are currently marked BLOCKED on exactly these numbers.
```

### subsection: Summarization and H1 (no data yet)  `sec:results:gallery:summarization`

```
% Every float in this subsection is a placeholder. If the summarization runs do
% not happen, this is the subsection to delete -- and the summarization family
% then has to come out of the methodology and the research questions too, rather
% than being left as a promise the results do not keep.
```

### subsection: Consolidated Summary  `sec:results:gallery:summary`

```
% This is the table \ref{sec:results:summary} promises. It is now generated
% (analysis/tables.py::summary), so that section can \input it rather than being
% dropped.
```

## `sections/5-discussion.tex`

### chapter: Discussion  `sec:discussion`

```
% Compare your results with the state-of-the-art and reflect upon the results and limitations of the study. You can already hint at future work to which you come back in the conclusion section.
% Template guidance: start by comparing to other studies as precisely as possible.
% Reflect on limitations in terms of reproducibility, scalability, generalizability,
% reliability and validity. Ethical concerns must be mentioned.
```

### section: Comparison with Related Work  `sec:discussion:relatedwork`

```
% Goes first per template guidance. Compare as precisely as the literature allows,
% and say so explicitly where a direct numerical comparison is not possible because
% no prior work reduces AIGs for this task.
```

### subsection: Positioning Against Optimizability Prediction Work  `sec:discussion:relatedwork:prediction`

```
% Bullets to cover:
%   - protocol-match before comparing numbers: most published figures sit in leakier
%     protocol cells (random or recipe-level splits, \ref{sec:relwork:ml4eda:protocols});
%     comparing their scores to this thesis's design-disjoint scores without stating the
%     protocol is apples-to-oranges. Use the RQ1a inflation factor
%     (\ref{sec:results:rq1:protocol}) to translate.
%   - SynthNet's collapse on this split (\ref{sec:results:rq1:baselinemodels}) is
%     consistent with the protocol audit, not an implementation artifact: OpenABC-D
%     itself reports negative R^2 on its unseen-IP variant.
```

### section: Interpretation of Findings  `sec:discussion:interpretation`

```
% Explain the results rather than restating them -- Results reports what happened,
% this section argues why. Grouped by theme rather than strictly per-RQ so RQ2 and
% RQ3 can be discussed as the single trade-off they describe.
```

### subsection: What Reduction Buys and What It Costs  `sec:discussion:interpretation:tradeoff`

```
% \ref{sec:intro:rq2} and \ref{sec:intro:rq3} together. Which point on the Pareto
% front is actually worth taking, and is the offline reduction cost recovered?
```

### subsection: Why Domain-Informed Adaptation Did or Did Not Help  `sec:discussion:interpretation:domain`

```
% \ref{sec:intro:rq4}. If the AIG-aware heuristics do not win, argue what that says
% about where the predictive signal lives -- distributed across the graph rather
% than concentrated in causal cones and long paths. Distinguish clearly between
% "domain knowledge does not help here" and "these particular heuristics were weak".
```

### subsection: Unexpected Results  `sec:discussion:interpretation:unexpected`

```
% Anything that contradicted the hypotheses, including negative results. Better
% surfaced deliberately here than left for a reader to notice.
```

### section: Practical Implications  `sec:discussion:implications`

```
% Concrete recommendation: given a memory budget and an accuracy tolerance, which
% reduction should a practitioner reach for? This is the payoff of the Pareto front.
```

### section: Limitations  `sec:discussion:limitations`

```
% These are limitations discovered or imposed -- distinct from the deliberate
% scoping choices declared in \ref{sec:intro:rqs:scope}. Do not simply repeat those.
```

### subsection: Reliability and Validity  `sec:discussion:limitations:validity`

```
% Once RQ1a numbers land, replace the sentence above's hedge with the settled figure:
%   "the random-split score exceeds the design-disjoint score by [X]x; that factor is
%    design recognition, not structural inference." Also state whether the per-design
% breakdown showed uniform or concentrated degradation, and (if concentrated) note the
% single-holdout caveat: the headline number then depends on which designs drew the test
% block, and a second design fold is the mitigation (cost: one training run).
```

### section: Outlook  `sec:discussion:outlook`

```
% Short. Per the template, the Discussion hints at future directions and the
% Conclusion develops them -- so a few sentences pointing forward, not a second
% Future Work section. Full treatment lives in \ref{sec:conclusion:futurework}.
```

### paragraph: Compute and energy cost.

```
% TODO: quote the actual GPU-hours and node-hours consumed rather than describing them
% qualitatively. The figures make this paragraph honest instead of gestural.
```

## `sections/6-conclusion.tex`

### chapter: Conclusion  `sec:conclusion`

```
% Answer each research question and address how the limitations of the study qualify the conclusion.
% Template guidance: make the relation between the research gap and the contribution
% clear, and be honest about how limitations qualify each answer.
```

### section: Answers to the Research Questions  `sec:conclusion:answers`

```
% One direct answer each -- a sentence or two, not a summary of the Results chapter.
% Where a limitation qualifies the answer, say so in the same breath.
%
% BLOCKED on Chapter 4. When writing, pair each answer with the limitation that qualifies it,
% rather than deferring all caveats to the Discussion:
%   RQ1 -> \ref{sec:discussion:limitations:validity} (node reduction is an area proxy)
%   RQ2 -> \ref{sec:discussion:limitations:scalability} (bounded graph scale)
%   RQ3 -> \ref{sec:discussion:limitations:generalizability} (one algorithm, one encoder)
%   RQ4 -> \ref{sec:discussion:limitations:reproducibility} (single seed; small gaps)
%          and \ref{sec:results:rq4:adequacy} (weak heuristics bound the null result)
%   RQ5 -> \ref{sec:discussion:limitations:methods} (which methods were actually run)
```

### section: Contributions Revisited  `sec:conclusion:contributions`

```
% Close the loop with \ref{sec:intro:contributions} and the gap stated in
% \ref{sec:relwork:gap}: what was claimed, and what the evidence actually supports.
```

### subsection: Extending Beyond a Single Synthesis Algorithm  `sec:conclusion:futurework:algorithms`

```
% The Deepsyn/Syn4/C2RS graphs already exist on disk; algorithm-ranking was the
% original motivation and is the most natural next step.
```

### subsection: Stronger Domain-Aware Reduction  `sec:conclusion:futurework:domain`

```
% Especially if RQ4 came out null: what a genuinely strong AIG-aware method would
% need to do differently. Three concrete directions, in increasing ambition:
%   1. SUPER-NODE CONTENT. Member statistics (size, level distribution, discarded internal
%      edge count, structural rewrite-potential features such as MFFC size and reconvergence
%      count) are already computed but not consumed by the encoder. The literature finds
%      content matters critically \citep{generale2022scaling}; wiring it in is the cheapest
%      untested improvement available and is an ablation, not a new method.
%   2. FUNCTION-AWARE COARSENING. Annotating each merged cone with its NPN function class --
%      the synthesis tool's own rewrite-library index -- would give super-nodes a semantic
%      identity rather than a purely structural one, while stopping short of the functional
%      merging that would leak the label. Designed but not built; it requires a model-schema
%      change.
%   3. LEARNED REDUCTION. Every method here is a fixed heuristic. A reduction trained to
%      preserve the label is the obvious next step and the obvious risk: it reintroduces
%      label dependence, which is one of the reasons condensation was excluded
%      (\ref{sec:relwork:reduction:summarization}).
```

### subsection: Scaling to Industrial AIGs  `sec:conclusion:futurework:scale`

```
% Removing the bounded-scale constraint, which is the limitation closest to the
% original motivation.
```

### subsection: From Prediction to Script Generation  `sec:conclusion:futurework:generation`

```
% Closing the loop the motivation opens with: using predicted optimizability to
% drive data-driven script construction.
```

## `sections/7-appendix.tex`

### (file preamble)

```
% You can choose whether you prefer a single or double column appendix.
% Whatever you choose, you will need to stick to it throughout the appendix.
% \onecolumn
```

### chapter: First Appendix  `sec:apx:first_appendix`

```
% Everything here is reference material the main text points at but does not need inline.
% The thesis must remain readable without it.
```

### section: Synthesis Algorithm Configurations  `sec:apx:algorithms`

```
% Referenced from \ref{sec:method:data:generation:algorithms}. Reproduce, per algorithm
% (Orchestrate, Deepsyn, Syn4, C2RS): the exact abc command template, the parameters that
% differ from abc's defaults, and the alias definitions -- C2RS in particular expands via
% abc.rc into a long balance/resubstitute/rewrite chain that is worth showing in full.
% Source: AIG_DATASET_README.md and abc.rc. Pin the abc version here too; the labels depend
% on it and \ref{sec:discussion:limitations:reproducibility} flags that it is unrecorded.
```

### section: Reduction Method Parameters  `sec:apx:params`

```
% One table: every reduction method, its configured parameters, and whether each was
% calibrated against measured retention or set to a plausible default. The distinction
% matters for \ref{sec:results:rq4:gap} -- an uncalibrated knob is a confound in a
% matched-compression comparison -- and putting it in one place keeps the methodology
% chapter from having to caveat each method individually.
```

### section: Baseline Hyperparameters  `sec:apx:baselines`

```
% For SynthNet, HOGA and DeepGate4: which hyperparameters are the papers' published values
% and which had to be assumed or adapted, with the reason for each deviation. The two
% substantive ones (node-budget batching plus gradient accumulation) are argued in
% \ref{sec:method:architecture:baselines}; this is where the full table belongs.
% Source: src/train_baseline.py module docstring and src/baselines/*/regressor.py.
```

### section: Experimental Configuration  `sec:apx:config`

```
% Referenced from \ref{sec:method:reproducibility:seeding}, which no longer enumerates the
% stochastic components inline. Reproduce the full run configuration in one table: seed
% values, split settings, batch budget, optimizer settings, and precision. Keep it as
% configuration values, not as source listings or key names.
```

### section: Extended Results  `sec:apx:results`

```
% Per-configuration tables too detailed for Chapter \ref{sec:results}: full metric sets per
% reduction method, per-tier error breakdowns, and the offline retention statistics per
% method and shard.
```

## `sections/0-list-symbols.tex`

### (file preamble)

```
% \chapter{List of Symbols}
%         This list of notation can help the reader decipher the thesis:
```

### (file preamble)

```
%         \begin{align*}
%                 \mathbb R_+&, \mathbb R_+^*, \mathbb R_-^* & \text{set of real numbers $\geq 0, >0, <0$, respectively;} \\
%                 a & \in \mathbb R & \text{scalar containing the value of the Answer.}
%         \end{align*}
```

### (file preamble)

```
% \chapter{List of Abbreviations}
%         A list of acronyms could similarly be defined
%         \begin{table}[h]
%                 \centering
%                 \begin{tabular}{cl}
%                                 CNN & Convolutional neural network \\
%                                 CPU & Central processing unit \\
%                                 SBU & System Billing Unit \\
%                                 SGD & Student Gradient Descent
%                 \end{tabular}
%         \end{table}
```

