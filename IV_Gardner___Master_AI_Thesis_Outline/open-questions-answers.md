# Open questions from the review pass

Working note. Not part of the thesis build.

## 1. Confidence intervals, alpha, p-values, and single-run training

**Current venue practice.** Reporting an interval alongside a point estimate is close to
mandatory at top ML venues now, not a nicety. NeurIPS's paper checklist asks directly whether
results carry error bars, confidence intervals, or a significance test, and requires stating
both what source of variability the interval captures (a train and test split, initialization,
a random draw, or an entire run under fixed conditions) and how it was computed (closed form,
a library call, bootstrap). Answering "no" is allowed with justification, but the checklist
itself institutionalizes the expectation. Agarwal et al., "Deep Reinforcement Learning at the
Edge of the Statistical Precipice" (NeurIPS 2021, outstanding paper award), argue directly for
interval estimates via stratified bootstrap in place of bare point estimates, precisely
because a small number of runs makes a point estimate unreliable on its own. That is the same
tool, a percentile bootstrap, the methodology chapter already uses.

**Are p-values out?** Not out, but demoted. The 2016 American Statistical Association
statement on p-values, and Dror et al.'s 2018 ACL survey ("The Hitchhiker's Guide to Testing
Statistical Significance in NLP"), converge on the same complaint: significance testing in
published work is routinely misused or run without the assumptions it depends on being
checked. The shift since is toward reporting an effect size with an interval as the primary
claim, and running a single well-chosen test only where a natural hypothesis exists, such as
a paired comparison, rather than scattering many uncorrected tests across a table and
reducing each to a star. The methodology chapter's own line, "intervals are the primary
report and p-values are not," is exactly this consensus position stated in one sentence,
not an unusual stance.

**Training each configuration exactly once.** This is the sharpest part of the question, and
current guidance is unambiguous: a bootstrap interval computed by resampling the test set
measures sampling uncertainty over which circuits happened to land in the test split. It says
nothing about run-to-run variance from initialization, data order, or nondeterministic
kernels, a distinct source of uncertainty that requires multiple seeds to estimate. Reimers
and Gurevych (2017) and Bouthillier et al. (2021), both already cited in the bibliography,
show comparisons of this kind reversing under reseeding when that second source is ignored.
Because training every configuration at multiple seeds is often not affordable at the scale
this thesis runs at, the field's answer is not "withhold the interval until seeds exist," it
is "report the interval you can compute, name exactly what it covers, and say plainly what it
does not." That is the current design: a percentile bootstrap over the test set, alpha fixed
at 0.05, paired with an explicit statement that seed variance is unmeasured, flagged hardest
at RQ4 where the predicted effect is smallest and most at risk of being swamped by exactly
that missing variance.

**Assessment.** The three choices, a 95 percent percentile bootstrap over test graphs with a
single fixed alpha, a Wilcoxon signed-rank test reserved for the one paired comparison
(efficiency against the full-graph baseline, matched per graph by construction), and an
explicit statement of what the interval does not cover, match current best practice closely
enough that nothing here needs to change. Two points worth naming for her own judgment, not
corrections: a 95 percent interval and alpha of 0.05 remain the standard default, nothing
unusual in that choice. Separately, 2000 bootstrap resamples is comfortably inside the usual
guidance of at least 1000 to 2000 for a stable percentile interval.

## 2. Bold and underline in tables

Confirmed, and already implemented correctly: bold for the best value in a column, underline
for second best, restricted to measured rows. This is a de facto convention, not a rule from
any formal style body. It shows up across current ML papers in near-identical wording, for example "the best
performance is set in bold, and the second best is set in underline," spread by repeated
imitation the same way an up or down arrow for "higher is better" spread. It is usually paired
with exactly that arrow, or a stated direction per column, which her table captions already
carry ("lower is better for error, memory, time and offline cost; higher is better for $R^2$,
correlation and savings"). The "over measured rows only" qualifier is not a deviation from the
convention. It is the same convention applied to a filtered row set, necessary while fabricated
placeholder rows sit in the same tables, and worth keeping once those rows are replaced with
real numbers only if new methods keep entering mid-comparison.

## 3. "Generalization" versus "Cross-State Inference"

**What "generalization" means in ML.** The gap between performance on the training sample and
performance on unseen data drawn from the same, or a deliberately shifted, distribution. Every
standard variant, domain generalization, compositional generalization, out-of-distribution
generalization, and "size generalization" specifically (Buffelli et al. 2022, already cited,
is exactly this: accuracy under test graphs larger than anything seen in training) describes a
model's behavior under a shift in the data distribution, holding the model and its input
representation fixed.

**Why it is the wrong word for RQ5.** RQ5 is not a question about unseen data. The same
graphs, or the same distribution of graphs, are evaluated in two different structural states,
reduced and full, fed to one already-trained, fixed model. It is a question about whether a
fixed function is robust to a change in its input's representation, not about whether a model
trained on a sample covers a wider population. Calling both of those "generalization" would
also collide with a sense the thesis already uses correctly: the discussion chapter's
Limitations section has its own subsection titled "Generalizability," covering exactly the
conventional sense, whether the reduction rankings hold across the three unused synthesis
algorithms, across encoder depth, and across the one modeled target. Two different technical
senses of the same word inside one document is confusing regardless of which one arrived
first. Here it is avoidable.

**Is "Cross-State Inference" a good replacement?** Yes. It is already formally grounded, not
just relabeled: the compatibility requirement in the problem formalization
(`sec:prelim:formalization`) states precisely what "state" means, whether the encoder receives
$G$ or $R(G)$, the same underlying circuit in two representations that must share one input
space. It is used consistently, in some form of "cross-state" on close to thirty separate
lines across the introduction, related work, methodology, results, discussion, and
conclusion, so this is not a one-off rename sitting next to older language. It avoids two collisions, not one: "generalization" (already claimed, see
above), and the more natural-sounding alternative, "transfer." "Structural Transfer" or
"Representation Transfer" would be the next-best names, but "transfer" carries a strong
connotation of an adaptation step, fine-tuning on a new task or domain, which is precisely what
does not happen here: RQ5 queries frozen weights on a different input form, zero adaptation.
"Cross-Representation Inference" was also considered and rejected for the same reason
"generalization" was: "representation" collides even harder, with representation learning,
the standard name for what a GNN's embeddings already are. One accurate caveat: her related
work already redescribes CTS-Bench's finding using "cross-state" language ("negative $R^2$
under zero-shot cross-state evaluation"), but CTS-Bench's own abstract calls this "zero-shot
evaluation," not "cross-state." The term is her thesis's own vocabulary applied to their
result, not one borrowed from them. Worth being precise about that rather than claiming the
field already converged on it. One small gap, not about the name: "state" itself never gets
the one-line first-use definition her own writing rules require for a technical term (compare
"inverted," which gets an equation reference on first use). A single defining clause at its
first appearance would close the last bit of ambiguity a reader unfamiliar with the chapter
could have.

## 4. Emulating Bollen and Generale on AIGs

This is the most concrete open item, and the one worth reading in full before touching either
experiment. Both papers were read directly, not summarized secondhand.

### Bollen, Steegmans, Van den Bussche, and Vansummeren, "Learning Graph Neural Networks using
Exact Compression," GRADES-NDA 2023 (`bollen2023exact`, DBLP `conf/grades/BollenSBV23`)

**What the model must satisfy.** An aggregate-combine GNN: every layer is a pair of an
aggregation function, mapping a multiset of neighbour vectors to one vector, and a combine
function, folding that into the node's own state. This covers GCN and effectively every
mainstream message-passing architecture. It excludes anything that reads more than the
immediate neighbour multiset. The guarantee is stated relative to a fixed, stated maximum
depth $d$, the number of layers in every model that will ever be trained on the compressed
graph. A deeper model used afterward falls outside what was proved. A bounded aggregation
width $c$ (a cap on how many same-colored neighbours the aggregator can distinguish, which
GraphSAGE-style neighbour sampling already imposes implicitly) allows coarser compression that
is still exact, but only for hypothesis spaces that actually respect that cap. After merging,
parallel edges collapse into one edge carrying a multiplicity, so the message function has to
treat that multiplicity correctly, that is, decompose linearly over repeated edges, or be
modified so it does.

**What the graph must satisfy.** Any directed, node-colored graph works in principle. The
condition that actually bites in practice is that the initial node coloring must not already
be node-unique, or there is nothing to merge. Bollen's own experiment hit this directly:
ogbn-arxiv's continuous 128-dimensional word-embedding features gave every node a distinct
color, so compression was zero until they inserted an extra step, pretraining a small
multilayer perceptron without any graph structure to turn each node's embedding into a
discrete estimated class label, and only then running colour refinement on that coarser
coloring. Exact compression is only as good as the redundancy already present in the input
features. Rich continuous per-node features destroy that redundancy by construction.

**What they evaluate.** Two separate experiments, worth keeping apart. First, a purely
structural, label-free compression study across eight real graphs (three variants of
ogbn-arxiv differing only in edge direction, ogbn-products, three SNAP road networks, one
SNAP social network): every node given one shared placeholder color, then node and edge
retention measured as a function of refinement depth (0 to 6) and, separately, of a graded
width cap (1 to 5). This isolates topological redundancy from any specific task. It is the
direct analogue of the depth-probe figure already scoped as a placeholder in the results
chapter (`fig:summ_wl_depth_probe`, currently marked "ENTIRELY FABRICATED. NO SUMMARIZATION
DATA EXISTS"). Second, a single learning experiment: node classification on ogbn-arxiv-inv
(one of 40 subject areas per paper), a 3-layer mean-aggregation network, 256 epochs, one fixed
learning rate, comparing the uncompressed problem against three compressed variants on test
accuracy, training time, and training memory. One run per configuration, no repeated seeds, no
interval, no significance test. Test accuracy runs 1 to 5 points below the uncompressed
problem across the three compressed variants, worst at the coarsest setting (bisimulation,
$c=1$), and the gap is read as "comparable" and attributed to "the stochastic nature of
learning" rather than tested. The authors call this "a single learning problem," a
"preliminary insight that requires further evaluation," not a validated benchmark result.
Worth knowing when calibrating how much rigor the source paper itself claims for its own
empirical half.

**What running a comparable experiment on AIGs requires, and what is already done.** The
model condition is already satisfied by construction: the exact-compression encoder variant
(`sec:method:architecture:exact`) already uses sum aggregation, repurposes the edge attribute
as the multiplicity, and ties colour-refinement depth to the encoder's layer count. This is
not outstanding work, it is argued through in the methodology already. The graph condition is
also already satisfied, and better than in the source paper: AIG node features are a small,
discrete alphabet by construction, constant, primary input, AND gate, or primary output, which
sidesteps the exact failure mode that forced Bollen's own two-stage fix. That is worth stating
as a design strength rather than an unexamined borrowing: the architecture avoids a documented
pitfall in the paper it depends on. What remains outstanding matches Bollen's own two-part
structure directly: (a) the structural compression-ratio measurement, colour refinement at
depth 1 through the encoder's layer count over the AIG corpus, node and edge retention
reported, a purely offline pass with no label needed, exactly what the depth-probe figure is a
placeholder for, and (b) training the exact configuration and comparing test accuracy, RMSE,
$R^2$, Spearman correlation, training time, and memory against the full-graph baseline, which
the thesis already scopes as the RQ3 and RQ5 positive control, designed but not yet run. One
respect in which the planned RQ5 experiment already exceeds Bollen's own validation: Bollen
never evaluates a model on the original, uncompressed graph object after training on the
compressed one. His comparison is compressed-problem against uncompressed-problem, each scored
on its own graph. RQ5 explicitly queries the exact-trained model on the full, unreduced graph
it never trained on, a more direct empirical test of the "provably equivalent" claim than the
source paper itself performs.

### Generale, Blume, and Cochez, "Scaling R-GCN Training with Graph Summarization," WWW 2022
Companion (`generale2022scaling`, DBLP `conf/www/GeneraleBC22`)

**What the model must satisfy.** Essentially nothing architecture-specific. The mechanism does
not depend on exactness or on any particular aggregation function, since the guarantee here is
empirical, not proved. What it does require is a node-level target, because "transfer the
weights back" means literally copying each summary node's learned representation onto every
original node that summary node stands for, through the explicit merge map the summarization
computed, then continuing to train or evaluate per original node.

**What the graph must satisfy.** A relational, multi-typed graph, in their case an RDF-style
knowledge graph with typed edges and a distinct class of leaf "literal" nodes (strings,
numbers, dates), that a structural, schema-level summarizer can partition. They test two
approximate, heuristic summarizers, neither carrying a formal equivalence guarantee the way
Bollen's colour refinement does: an Attributes Summary (nodes sharing the same set of outgoing
edge labels merge) and forward $k$-bisimulation (nodes with equivalent local neighbourhood
schema up to $k$ hops merge, $k = 3$). Literal nodes are stripped out or grouped into one
placeholder node before summarizing, since they inflate node count without carrying structure.

**What they evaluate.** Multi-label node classification, predicting each entity's type, on
three standard benchmark knowledge graphs of increasing size (roughly 8 thousand entities up
to roughly 1.7 million). Three models are compared. The first is an R-GCN trained only on the
graph summary. The second is the same architecture, initialized by copying summary-node
weights onto original nodes through the merge map, then optionally trained further on the
full original graph. The third is a baseline R-GCN trained from a random initialization
directly on the full graph, at the same epoch budget. Two readings of the middle model
matter: performance with no further training at all (the
transferred weights alone) against performance after 50 further epochs on the full graph. That
no-further-training-versus-more-training contrast is the closest published precedent to a
matched-state versus cross-state comparison, though at the node level rather than the graph
level. Statistics are thin: 5-fold cross-validation, mean plus standard deviation, for the two
smaller graphs. Only two runs, no cross-validation, for the largest, "due to computational
cost." No intervals in the modern sense. Result: the summary-pretrained model starts far above
the random baseline in almost every case and often stays ahead after further training. The one
exception ends about four points below baseline. An ablation removing literal-node information
before summarizing makes the transferred model "less consistent and often fall below the
baseline," read as evidence that what a super-node carries, not merely that it exists, decides
downstream performance.

**What running a comparable experiment on AIGs requires, and what is already done.** The
reason Generale's transfer machinery itself does not port over is already correctly stated in
the thesis's own related work (`sec:relwork:reduction:forgnn`): their target is per node, so a
mapping back from summary node to original member is required, along with a rule for what a
summary node's single prediction means when it represents several original nodes, which they
solve with a weighted multi-label target by type frequency inside the partition. The AIG
target is graph-level, one scalar per whole circuit, so no such mapping exists to fail: a full
graph is already a valid input to a summary-trained model, being the special case where every
super-node has exactly one member, already argued in `sec:method:architecture:summarization`.
Their transfer step is not something that needs reproducing. The shared input schema makes it
unnecessary by design. What does transfer from their paper, concretely, is narrower and
already identified in the thesis's own limitations section as missing: the super-node content
ablation. The merge-map schema already computes member counts and level statistics per
super-node (`sec:method:summary:mergemap`), but the encoder does not yet consume them, per
`sec:discussion:limitations:methods`. Running the encoder with and without that content,
matching Generale's literal-nodes-in versus literal-nodes-out ablation, is a direct,
already-scoped, not-yet-run experiment. It is the one piece of this paper that would let the
thesis cite the super-node-content finding as tested rather than only cited. Their
no-further-training-versus-more-training framing is also a useful precedent for how to present
the RQ5 numbers, the closest published example of treating a "before any adaptation" score as
informative on its own, even though it sits at the node level and hers sits at the graph
level. Their exclusion of literal nodes before summarizing is the same kind of design decision
as choosing which AIG nodes are eligible to seed a merge, not a new requirement, just a
precedent worth citing if that choice needs one.

## 5. Where does the exactness proof belong: discussion or results?

Neither, in the sense the question implies, and the thesis already has this right without
needing a decision. A proof or a formal argument, deriving conditions, stating a theorem,
adapting someone else's theorem to a new setting, is not a measurement. It belongs wherever
the object it is a property of gets defined, which in a methods-first thesis is the
methodology chapter. The discussion chapter interprets and compares findings. It is not where
an argument is first stated. The results chapter reports what was measured. An empirical
confirmation of a proof, does this configuration's number actually match what the theorem
predicts, is a measurement and belongs there, typically reported ahead of the rest of that
research question's results specifically because a deviation there flags a broken evaluation
path rather than a genuine finding about what is being studied.

The thesis's own discussion section already states this division explicitly, in
`sec:discussion:relatedwork:reduction`: "The exact-compression result this work depends on is
not proved here... What this thesis adds is the set of conditions under which that theorem
applies to an AIG encoder... Establishing them is an argument, not a measurement, so it
belongs where the encoder is defined rather than in the results." Bollen's theorem itself
stays a cited result, attributed to Bollen et al., not reproved. What belongs to this thesis,
and correctly sits in the methodology as exposition rather than in results as a finding, is
the derivation of the conditions under which that theorem transfers to this specific encoder
and AIG input schema (`sec:method:architecture:exact`). The empirical half, whether the exact
configuration actually reproduces the full-graph baseline under both matched-state and
cross-state evaluation, is the positive control reported in results
(`sec:results:rq3:degradation`, `sec:results:rq5:accuracy`). The discussion's remaining job,
once that result exists, is only to say what the positive control showed and what that implies
for how far the rest of the thesis can lean on the theorem holding. Nothing here needs to
move.

## 6. Already settled, recorded for completeness

**The "-1 clipped but I see -4" figure.** Not a bug. The $R^2$ axis in the RQ3 figures is
deliberately floored at $-1$ (`clip_bar_axis` in `src/analysis/style.py`). Some
configurations, METIS in particular, reach $R^2$ around $-3$ to $-750$ depending on design,
which would compress every other bar to a single pixel column if drawn to scale. A value
below the floor is drawn at the floor with a distinct marker and labelled with its real value
as text, so a number like $-4.01$ that looks like it sits on a bar running off the chart is a
label, not the bar's true plotted extent. Confirmed directly in the plotting code, both the
`clip_bar_axis` docstring and its use in `src/analysis/fig_rq3.py` (floor fixed at $-1.0$
throughout, off-scale points drawn with a separate marker and their value written out).

**The grey bars on the memory figure.** Mean, p95, and max order statistics, not error bars.
`sections/4-results/4.2-rq2.tex` states this directly: "Peak allocated device memory per
configuration at three order statistics. The mean, p95 and maximum disagree by an order of
magnitude: a graph larger than the batch budget forms a batch of its own, so the maximum is
floored by the largest circuit in the corpus no matter what the reduction did to the average
one." Three separate summary statistics of one distribution shown side by side, not a mean
with an uncertainty band drawn around it.

**Colour scheme centralization.** Confirmed. `src/analysis/style.py` is the single source
every figure script imports from: `FAMILY_COLORS` maps the four reduction families, baseline,
partition, sparsification, summarization, to four fixed hues from a validated categorical
palette, chosen once rather than per figure, and `DOMAIN_HATCH` marks domain-informed methods
with a hatch pattern regardless of family. The module's own docstring states the rule
directly: "Colour encodes the reduction family, never the individual method... Domain-informed
methods are hatched rather than given their own colour." Hue carries family, hatching carries
domain-informed status, and both are fixed in one file rather than re-decided per figure.
