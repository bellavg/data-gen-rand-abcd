## Commands

### Environment

The `src/` package is **not** run as an installed CLI — only the `data` and `models` subpackages are declared under `[tool.setuptools.packages.find]`. Top-level modules (`train.py`, `config.py`, etc.) are imported via `PYTHONPATH`, not package imports:

```bash
export PYTHONPATH=src   # from repo root
```

Rebuild the HPC venv from `pyproject.toml` (also documents the exact torch/PyG wheel pins):
```bash
bash src/shell/rebuild_venv.sh          # override with VENV_PATH=...
```

### Tests

`src/unittests/` is the **only** test location (mirrors the `src/` package layout, imports as `from data.xxx import ...` / `from models.xxx import ...`). A root-level `tests/` directory and root-level `test_*.py` scratch scripts used to exist alongside it and were removed — they duplicated/shadowed `src/unittests/` coverage and one pair (`test_sparsification*.py`) had silently gone stale (imported a function renamed years ago from `spanner_sparsification` to `spanning_forest_sparsification`). If you ever see test files outside `src/unittests/`, treat that as regression, not a second valid location — port the coverage in and remove the duplicate rather than maintaining both.

A local `.venv` with all pipeline dependencies (torch/PyG/Lightning/wandb, etc.)
already exists at the main repo checkout root (`/Users/bellavg/data-gen-rand-abcd/.venv`)
— it's what makes running the test suite off the HPC cluster possible at all.
`git worktree` checkouts (e.g. under `.claude/worktrees/`) do **not** get their
own `.venv`; point `PYTHONPATH` at the worktree's own `src/` but invoke the
interpreter from the main checkout's `.venv`:

```bash
PYTHONPATH=src pytest src/unittests
PYTHONPATH=src pytest src/unittests/data/test_dataset.py::TestClassName::test_name   # single test
# From a worktree, use the main checkout's interpreter instead of a bare `pytest`:
PYTHONPATH=src /Users/bellavg/data-gen-rand-abcd/.venv/bin/python -m pytest src/unittests
```

The suite is fully green (`256 passed, 5 skipped`) and `ruff check src` is clean. Two things worth knowing if drift happens again:
- `datamodule._loader_kwargs` applies `persistent_workers` uniformly to train/val/test loaders (not train-only) as of commit `fb022ae` ("maybe speed up?") — val/test loaders are recreated often under fractional `val_check_interval`, so keeping their workers alive too avoids repeated spawn overhead. Tests are named `test_persistent_workers_applies_to_all_*` to match.
- `train.main()` reads `args.torch_compile`; any test mocking `args` via `SimpleNamespace` must include it.

### Lint

```bash
ruff check src
```
No `[tool.ruff]` section exists in `pyproject.toml`, so this runs on Ruff defaults.

### Training

`src/train.py` is a single-algorithm-run entrypoint driven by argparse (defaults sourced from `src/config.py`). It's normally launched via the SLURM scripts, which show the full flag set and expected directory layout (checkpoint/log/cache dirs, tiered caches, HP-tuning split file):
```bash
cat src/shell/train.sh                  # sparsification sweep (array job)
cat src/shell/train_no_sparsification.sh
```
Minimal direct invocation shape:
```bash
PYTHONPATH=src python -m train --algorithm Orchestrate --csv_paths <path.csv> --sparsification none
```

Other `src/shell/*.sh` scripts worth knowing about: `warmup_train_cache.sh`/`warmup_cache.sh` (pre-populate the graph cache on CPU before a GPU job so the GPU doesn't idle during loading — chain with `--dependency=afterok`), `precompute_sparsification_masks.sh` (precompute and cache sparsification masks so they aren't recomputed every epoch), `measure_sparsity.sh` (report per-method edge/node retention for calibrating sparsification parameters), `big_hp_tuning.sh` (Optuna sweep).

## Architecture

### `config.py` is the single source of truth for defaults/constants

There used to be a second `src/constants.py` with structural constants (`NODE_INPUT_DIM`, `EDGE_ATTR_DIM`, the lazy `ENCODER_REGISTRY`) that duplicated several names already in `config.py` (`MAX_DEPTH`, `MAX_NUM_GATES`, `TASK_OUT_DIM`, `get_output_dim_for_encoder`, and a `VALID_ALGORITHMS` that had drifted to a *different value* than `config.py`'s). It's been merged into `config.py`; `constants.py` no longer exists. If a memory, doc, or old branch references `from constants import ...`, that import is stale — the same names now live in `config.py`.

`config.py` keeps two intentionally distinct algorithm-name sets, because they answer different questions and merging their *values* (not just the files) would be wrong:
- `VALID_ALGORITHMS = {"Orchestrate"}` — algorithms `train.py` will currently accept via `--algorithm`. This project only trains on Orchestrate.
- `KNOWN_ALGORITHMS = {"Orchestrate", "Deepsyn", "Syn4", "C2RS"}` — every algorithm name that appears in tier0/tier1/tier2 filenames already on disk from the data-generation pipeline (used by `data/dataset_utils.py` to parse those filenames via regex). This has to stay all four regardless of training scope, since old filenames don't change.

`data/preprocess_data.py` used to have its own separate hardcoded `VALID_ALGORITHMS = {"Orchestrate", "Deepsyn", "Syn4", "C2RS"}` — a third, independent copy not wired to `config.py` at all. It's now `from config import KNOWN_ALGORITHMS as VALID_ALGORITHMS`, so there is exactly one place (`config.KNOWN_ALGORITHMS`) that knows the full algorithm set.

### Dataset caching pipeline (`src/data/dataset.py`)

`AIGGraphRegressionDataset` layers several independent caches, each keyed by a signature hash so Optuna sweeps instantiating many `Dataset` objects per process don't recompute or leak memory:
- **CSV sample cache** (`_CSV_SAMPLE_CACHE`) and **splits cache** (`_SPLITS_CACHE`) — module-level dicts, cleared via `clear_dataset_global_caches()`.
- **Graph cache manifest** (JSON, per cache-signature) — maps each raw AIG `.pt` path to a stable cache path + node count, so repeated runs skip `torch.load`/`stat()` via a fast set-lookup path.
- **Tiered cache directories** — `tier0_cache_dir`/`tier1_cache_dir` are shared across runs (keyed by path containing `/tier0/` or `/tier1/`); a per-run `cache_dir` is the fallback.
- **Sparsification mask index** (`src/data/sparsification.py`, chunked `_sparse_<algo>*.pt` files) — precomputed masks looked up by `(cache_dir, algo_name)`, not recomputed per epoch.

Sparsification is applied at `get()` time (edge masks for `random_edge_dropout`/`spanning_forest`, node masks for `pagerank`/`and_gate_only`); `get_num_nodes_list()` has to special-case node-based sparsification because it changes node counts post-mask, which matters for dynamic batching.

### Dynamic batching (`src/data/sampler.py`, `AIGDataModule`)

When `dynamic_batching=True`, batches are built to a total-node budget (`MAX_TOTAL_NODES_PER_BATCH`) rather than a fixed graph count, via `BalancedDynamicBatchSampler`. The batch plan itself is cached to disk (`load_or_build_batch_plan`) since AIGs vary wildly in size and replanning is expensive.

### Model (`src/models/`)

`base_model.UnifiedGraphBaseModel` is encoder-agnostic: it projects raw node/edge features, optionally concatenates a positional encoding (`models/layers/positional_encodings.py`), runs the encoder, pools to graph level, and applies a regression head (`Linear → ReLU → Dropout → Linear → Sigmoid`, since targets are in `[0, 1]`). Encoders are resolved through `config.ENCODER_REGISTRY`, a lazy-import dict-like registry (only `"gcn"` → `models/layers/gcn.py:GCNEncoder` is currently registered). `models/lightning_model.AIGRegressionLightningModule` wraps it with linear LR warmup + `ReduceLROnPlateau`, and logs RMSE at every stage but R² only for val/test (per-batch R² is statistically meaningless).

### Positional encodings

`pe_type="level"` reads a precomputed graph attribute rather than computing anything at runtime (`ExtractPrecomputedPE` in `positional_encodings.py`) — it deletes the source attribute (`level`/`pi_paths`/`local_sp_sum`) after extraction, and `dataset.py` mops up the unused siblings so they aren't persisted in cache files.
