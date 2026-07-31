# DeepGate4 vendored source — provenance

Upstream: <https://github.com/zyzheng17/DeepGate4-ICLR-25>
Commit: `85e20742a2d4702426f94e86666cdcdb408fba8a` (2026-06-25)
Paper: Zheng et al., *DeepGate4: Efficient and Effective Representation
Learning for Circuit Design at Scale*, ICLR 2025. <https://arxiv.org/abs/2502.01681>

## Licence

**Upstream declares no licence.** The DeepGate4-ICLR-25 repository contains no
`LICENSE` file, no `COPYING`, and no licence header in any source file, so
there is no `LICENSE_UPSTREAM` here to sit alongside the ones in
`../hoga/` (BSD 3-Clause) and `../openabc_synthnet/` (BSD 3-Clause).

That is a statement of fact about the upstream repository, not a grant. Absent
an explicit licence, default copyright applies and no redistribution right is
conveyed. Before this code is published — thesis appendix, an artifact
release, or a public fork of this repository — either obtain permission from
the authors or replace these four files with a clean-room reimplementation.
Vendoring it for private research use is a separate question from
redistributing it, and only the first has happened so far.

## Files taken unmodified (bar the noted deletions)

| File | Upstream path | Change |
|---|---|---|
| `mlp.py` | `src/models/mlp.py` | deleted unused `import torch` |
| `tfmlp.py` | `src/models/tfmlp.py` | deleted unused `Adj`, `MLP` imports |
| `dg2.py` | `src/models/dg2.py` | deleted unused `import deepgate as dg`, `import copy` |
| `plain_tf_linear.py` | `src/models/plain_tf_linear.py` | deleted unused `import torch.nn.functional as F` |

Every change is a deleted import of a name the file never references, so all
four are behaviour-neutral by inspection. Verified with `diff -w -B` against
the upstream originals: apart from the added docstring header, the deletions
above, and trailing-whitespace normalisation, the code bodies are identical —
no reordered statements, no changed literals, no logic changes. Two also matter practically: `import
deepgate as dg` pulls in the external `python-deepgate` package, which is not a
dependency of this project, so leaving it would make the module unimportable;
and `ruff check src` is expected to stay clean repo-wide, which unused imports
would break.

Nothing else was touched. Two Ruff findings inside `dg2.py`'s own *code* were
deliberately left in place and silenced through `per-file-ignores` in
`pyproject.toml` instead — one of them (`G.batch == None`) is elementwise on a
tensor, so "fixing" it to `is None` would be a behaviour change rather than a
style one.

## Files NOT taken, and why

`src/models/dg4.py`'s `DeepGate4` class is not vendored. Its `forward()` is
inseparable from the self-supervised pretraining objectives (probability,
truth-table similarity, connectivity, GED, hop-level tasks) and from the
`History` embedding tables that implement the partitioned updating strategy.
Neither applies to a supervised graph-level regression baseline. `regressor.py`
in this directory reproduces the *encoder* path of that `forward()` —
structural encoding, tokenizer, sparse transformer, residual sum — line for
line, and cites the upstream line numbers it mirrors.

`src/dg_datasets/data_preparation.py` is not vendored either; see
`aig_features.py` for what replaces it and why the original cannot run at this
project's scale.

## Released checkpoints

`trained/model_last.pth` and `trained/model_last_workload.pth` in the upstream
repository are **DeepGate2 tokenizer weights only** — 64 tensors, all under
`aggr_{and,not}_{strc,func}`, `update_{and,not}_{strc,func}` and
`readout_prob`, at `dim_hidden=128`. They contain no sparse-transformer
weights, so there is no pretrained DeepGate4 to transfer from; upstream loads
them through `DeepGate2.load_pretrained()` purely as tokenizer initialisation.
`--deepgate4_pretrained_tokenizer` in `train_baseline.py` exposes that same
initialisation. It is off by default, matching how the SynthNet and HOGA
baselines train from scratch.
