# Gamora vendored source — provenance

Upstream: <https://github.com/Yu-Maryland/Gamora>
Branch: `master` (the repository has no `main`)
Commit: `344d5e4530072cd95a07f0557640ad641fd6cfcb`
Paper: Wu, Li, Hao, Dai, Yu, Xie, *GAMORA: Graph Learning based Symbolic
Reasoning for Large-Scale Boolean Networks*, DAC 2023.
<https://arxiv.org/abs/2303.08256> (v2, 12 Jun 2023 — the version cited
throughout this directory)

## Licence — record the ambiguity, do not resolve it

GitHub reports **NOASSERTION** for this repository (`GET /repos/Yu-Maryland/Gamora`
returns `{"key": "other", "name": "Other", "spdx_id": "NOASSERTION"}`). That is
not a badge glitch; `LICENSE.txt` genuinely does not say who is licensing what.

The complete file is vendored here as `LICENSE_UPSTREAM` and reproduced
verbatim below — all 27 non-blank-padded lines of it:

```
The MIT License

Portions of this code base were orginally forked from ABC: , which is under the following License:

ABC license

Portions of this code base were orginally forked from GraphSAGE: , which is under the following License:

Copyright (c) 2017 William L. Hamilton, Rex Ying

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

Read it closely, because it does not say what a glance suggests:

1. The heading is "The MIT License", but the only copyright line in the file is
   `Copyright (c) 2017 William L. Hamilton, Rex Ying` — that is **GraphSAGE's**
   copyright, introduced by the sentence "Portions of this code base were
   orginally forked from GraphSAGE". The MIT grant that follows attaches to
   that copyright line, so on its face the file licenses the GraphSAGE-derived
   portions and says nothing about the rest.
2. **Gamora's own code carries no stated copyright holder and no explicit
   grant.** No author, institution, or year appears for the Gamora work itself,
   in `LICENSE.txt` or as a header in `abc2pyg/gnn_multitask.py`.
3. The ABC clause is **empty**. It reads "Portions of this code base were
   orginally forked from ABC: , which is under the following License:" followed
   by the bare words "ABC license" and nothing else — no URL after the colon,
   no licence text. The repository does vendor Berkeley ABC (the whole `abc/`
   subtree, including `src/proof/acec/acecXor.c`, which this port cites for the
   node featurisation), so that clause is pointing at real code whose terms the
   file never states. ABC's own distribution carries a UC Berkeley
   permissive licence; this file does not reproduce or reference it.

**What this means here.** "MIT" is not a safe summary and must not be written
in the thesis, a table caption, or a release note. What can be said accurately
is: the upstream file is titled MIT, grants MIT over GraphSAGE's 2017
copyright, states no copyright holder or grant for Gamora's own contribution,
and leaves the terms of the vendored ABC blank.

Consequences, same position as `../deepgate4/PROVENANCE.md` and for a
comparable reason (no unambiguous grant covering the authors' own work):
vendoring for private research use is one question and redistribution is
another, and only the first has happened. Before this code is published — a
thesis appendix, an artifact release, a public fork of this repository — either
obtain clarification from the authors or replace `model.py` with a clean-room
reimplementation. That is a small job: the vendored class is ~40 lines of
constructor plus a 10-line forward, and the architecture it expresses
(a GraphSAGE stack) is itself published under MIT by Hamilton and Ying.

## Files taken

| File | Upstream path | Change |
|---|---|---|
| `model.py` | `abc2pyg/gnn_multitask.py:39-105` (class `SAGE_MULT`) | see below |
| `LICENSE_UPSTREAM` | `LICENSE.txt` | none, byte-for-byte |

`model.py` takes `__init__`, `reset_parameters` and `forward_nosampler`
with one behaviour-visible change and two omissions. "Verbatim" is used loosely
below: four commented-out debug lines (`# print(x[0])`, a commented-out
`F.dropout`, `# print(self.linear[0].weight)`, `# print(x1[0])`) are dropped
and PEP8 comment spacing is applied. Neither is behaviour-visible; the code
bodies are otherwise byte-identical modulo trailing whitespace.

- **Change.** `forward_nosampler` upstream calls `F.dropout(x, p=0.5, ...)`
  with the rate hardcoded, while `__init__` stores `self.dropout = dropout` and
  never reads it — so upstream's `--dropout` flag is dead. This copy uses
  `p=self.dropout`. Numerically identical at the default (`dropout=0.5` is
  upstream's own argparse default, `gnn_multitask.py:530`); the change makes
  the constructor argument real rather than a trap.
- **Omission 1.** `forward(self, x, adjs)` (`:68-84`), which consumes the
  `(edge_index, e_id, size)` triples a `NeighborSampler` yields.
- **Omission 2.** `inference(self, x_all, subgraph_loader, device)`
  (`:107-139`), which iterates a `NeighborSampler` directly.

Both omissions are the point of the port rather than incidental — see the next
section and `regressor.py`'s module docstring.

The two no-op statements at the head of `forward_nosampler` (`x.to(device)` and
`adj_t.to(device)`, whose return values are discarded) are kept verbatim rather
than fixed, and are the reason the method takes a `device` argument it does not
use.

## Files NOT taken, and why

Everything else in `abc2pyg/gnn_multitask.py` is training-loop or
task-specific: `train()` (`:141-197`, the sampled training loop),
`post_processing()` (`:199-255`, the xor/maj reconciliation that turns per-node
predictions into adder boundaries), `test()` / `test_nosampler()` /
`confusion_matrix_plot()` / `write_txt()`, and `main()`. None of it applies to
supervised graph-level regression, and `train()` in particular is the file's
only optimizer loop and is inseparable from the `NeighborSampler` it iterates.

`abc2pyg/dataset_prep/` and `abc2pyg/ABC_dataset_generation.py` are not taken
either. They generate CSA/Booth multipliers with ABC and label them with the
adder-tree extraction command; this project's dataset already exists and its
target is not Gamora's. `regressor.py`'s `gamora_node_features` reproduces the
one part that does transfer — the node featurisation written by
`Gia_edgelist` in `abc/src/proof/acec/acecXor.c:382-421` — and cites the
upstream lines it mirrors. The vendored ABC subtree is not taken in any form.

The released checkpoint `abc2pyg/SAGE_mult8` (a 4-layer/32-channel `SAGE_MULT`
trained on an 8-bit multiplier) is not loaded. It is a per-node classifier for
multiplier structure; its three heads are exactly what this port deletes, and
its trunk was trained under a different task, a different input distribution
and a sampled objective. This baseline trains from scratch, matching the
SynthNet and HOGA ports.

## Upstream trains with sampling; this port does not

Recorded here because it is the one deviation from Gamora's published procedure
that a reviewer is most likely to ask about, and because it is easy to overstate
in the flattering direction.

- The architecture is sampling-free, and upstream wrote the full-graph forward
  themselves: `forward_nosampler` (`:86-105`) loops over the entire adjacency.
- Upstream's **released trainer samples**, with no alternative in the
  repository: `:570-572` builds
  `NeighborSampler(data.adj_t, node_idx=train_idx, sizes=[8, 5, 5, 5], batch_size=20, shuffle=True)`
  and `train()` iterates it. `forward_nosampler` is reached from exactly one
  caller, `test_nosampler` (`:342`), which is evaluation — and `main()` reaches
  it only from the evaluation block at `:662`, while training at `:597`/`:600`
  goes through the sampled `train()`/`test()`. There is no `train_nosampler`
  anywhere in the repository.
- The two sibling entrypoints do not change this. `gnn_multitask_inference.py`
  matches, and `gnn_multitask_v2.py` is if anything stronger evidence: its
  non-`use_old` branch swaps `NeighborSampler` for `NeighborLoader`
  (`:619-621`) and still samples `[8, 5, 5, 5]` at `batch_size=20`. Every
  released trainer samples.
- This port trains full-graph. It is a **deviation from the published training
  procedure**, not "an option upstream provides". Upstream provides the forward
  computation; it does not provide a trainer that optimizes through it.

The deviation is available because this codebase supplies its own Lightning
loop, so upstream's `train()` never enters the repository. No `NeighborSampler`,
`ClusterLoader`, or subgraph call is vendored here or reachable from
`train_baseline.py`'s Gamora path.
`src/unittests/baselines/test_gamora.py::TestNoSamplingInPort` pins that with
an AST-level assertion over both this package and every module the Gamora
training path executes (`train_baseline.py`, `train_utils.py`,
`data/sampler.py`, `baselines/common/lightning_wrapper.py`, and the SynthNet
regressor this port imports one helper from).

## What the baseline measures

Stated in `regressor.py`'s module docstring and repeated here because it
belongs with the provenance: once the three per-node classification heads are
removed, what remains is a GraphSAGE encoder. Gamora's contribution is the
multi-task formulation and the adder-tree post-processing, both of which a
graph-level regression task discards. This baseline measures **Gamora's encoder
adapted to graph-level regression**, not Gamora's published task, and its score
is not comparable to any number in the DAC'23 paper. Label the thesis row so a
reader cannot mistake it for a claim about Gamora's results.
