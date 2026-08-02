# Terms to introduce before the preliminaries

Working file. Not `\input` anywhere, not part of the build, delete before handing in.

Scope: only the terms a reader must already hold to follow the motivation
(`1.1-motivation.tex`) and the research questions (`1.2-research-questions.tex`). Everything
else waits for Chapter 2. Tick a box once the term is defined at or before its first use in
the introduction.

Columns per entry: where it is first used, whether it is defined at that point, and where
its full treatment lives.

## Defined at first use (verify wording, then tick)

- [ ] **Electronic Design Automation (EDA)**. First use: chapter opener,
  `1-introduction.tex`. Defined there in one clause (compiles a behavioural description into
  a layout through a sequence of stages). Full: `sec:prelim:synthesis:flow`.
- [ ] **Logic synthesis**. First use: chapter opener. Defined there as the stage that fixes
  the gate count. No abbreviation: `sec:prelim:synthesis` introduces LS, but the body spells
  the term out everywhere, so do not abbreviate it in the introduction.
- [ ] **And-Inverter Graph (AIG)**. First use: chapter opener. Defined there as a DAG of
  two-input conjunctions with optionally inverted edges. Full: `sec:prelim:aig:definition`.
- [ ] **Synthesis script**. First use: `1.1-motivation.tex`, section 1.1.1. Defined there as
  a sequence of rewriting passes. Full: `sec:prelim:synthesis:scripts`.
- [ ] **Optimizability**. First use: `1.1-motivation.tex`, section 1.1.2, as the fraction of
  nodes a script removes. Defined a second time at the head of
  `1.2-research-questions.tex`. One of the two should go, and the equation stays in
  `sec:prelim:formalization`.
- [ ] **Graph reduction**, and its three families (**partitioning**, **sparsification**,
  **summarization**, also called **coarsening**). First use: `1.1-motivation.tex`, section
  1.1.4, where the three are named as the methods that shrink a graph before a model reads
  it. RQ2 is the first place the summarization/coarsening synonym pair appears. Individual
  definitions: `sec:prelim:reduction:partitioning`, `sec:prelim:reduction:sparsification`,
  `sec:prelim:reduction:summarization`.
- [ ] **Compression ratio**, as **node retention** and **edge retention**. First use: head of
  `1.2-research-questions.tex`, which distinguishes the two and forwards to
  `sec:prelim:reduction:measuring`. Check that the forward reference is enough for RQ2 and
  RQ4 to be readable in place.

## Used before any definition (decide: gloss in the introduction, or leave to Chapter 2)

- [ ] **Directed Acyclic Graph (DAG)**. First use: chapter opener, inside the AIG
  definition. Not defined anywhere in the introduction. Standard enough to assume for this
  audience.
- [ ] **Graph Neural Network (GNN)**. First use: `1.1-motivation.tex`, section 1.1.2. The
  abbreviation is introduced but the mechanism is not; message passing is only defined at
  `sec:prelim:gnn:mp`. Assume the reader has it, or add half a clause.
- [ ] **Causal cone** (also fanin cone, logic cone). First use: `1.1-motivation.tex`, section
  1.1.4, where severing one is the stated risk. Undefined until `eq:prelim:cone` in
  `sec:prelim:aig:properties`. This one carries the argument, so it needs a gloss.
- [ ] **Logic depth** and **topological level**. Two names for one quantity: depth in
  `1.1-motivation.tex` section 1.1.4, level in RQ4 and in H1. Defined at
  `eq:prelim:level`. Pick one name for the introduction.
- [ ] **Recipe**. First use: RQ1a, in "recipe-disjoint split", and again in
  `1.3-contributions.tex`. Never defined in the introduction, and
  `2.1-prelim-synthesis.tex` already carries a `\todo` about defining it. Needed to read
  RQ1a at all, since the whole point is what the split holds out.
- [ ] **Design**, in the sense of a source circuit. First use: RQ1a, in "design-disjoint
  split". Undefined until `sec:method:data:sources`. Same problem as recipe, and the two
  should be glossed together in one sentence.
- [ ] **Reconvergence** and **gate role**. First use: RQ4, as the structure the domain
  heuristics exploit. Defined at `sec:prelim:aig:properties` and `eq:prelim:role`. RQ4 reads
  as a list of unexplained features without at least the gist of them.
- [ ] **Structural hashing**. First use: `1.1-motivation.tex`, section 1.1.5, in the claim
  that generic summarization work does not know about it. Defined at
  `sec:prelim:algorithms:primitives`. Possibly cuttable from the introduction instead of
  glossed.
- [ ] **Over-squashing** and **receptive field**. First use: H1 in
  `1.2-research-questions.tex`, which is where the whole mechanism of the hypothesis sits.
  Cited to the over-squashing literature but not explained; `sec:prelim:gnn:mp` covers the
  receptive field.

## No pre-definition needed

- [ ] **Smooth L1 loss**, **RMSE**, **$R^2$**, **Spearman correlation**. Named in RQ3 as the
  accuracy and ranking metrics. Standard, and defined in the metrics section
  (`sec:method:experiment:metrics`). Confirm the reader can take them on trust in RQ3 and
  tick.
- [ ] **Graph condensation**. Named once in the scope statement, only to be excluded, with
  the reason given in place. No definition owed.
