#!/bin/bash
#SBATCH --job-name=test_cache_warmup
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=genoa
#SBATCH --output=logs/warmup_test_%j.out

# ---------------------------------------------------------------------------
# Pre-warm the test-split graph cache for the Orchestrate workspace.
#
# warmup_train_cache.sh deliberately only warms train+val (test does not need
# to be preloaded for training). test.sh / test_cpu.sh / results_to_latex.py
# need the test split's graph cache (and, downstream, its sparsification /
# partition masks) actually built, so run this once before those jobs.
#
# CHAIN WITH TEST + MASK PRECOMPUTE JOBS
# ---------------------------------------
# This job is step 1 of the chain in EVALUATION.md — follow it there rather
# than copying the submission commands into this header, which is how earlier
# copies drifted out of sync with it.
#
# If the sentinel already exists the warmup skips it, so re-running or
# re-chaining is always safe.
# ---------------------------------------------------------------------------

set -euo pipefail

echo "=========================================="
echo "TEST CACHE WARMUP JOB"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "CPUs available: $(nproc)"
echo "=========================================="

# =========================================================
# 1. Environment
# =========================================================

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load SciPy-bundle/2025.06-gfbf-2025a

VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
source "$VENV_PATH/bin/activate"

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BASE_DIR/src"

cd "$BASE_DIR"

# =========================================================
# 2. EVAL CACHE PATHS — a SEPARATE workspace from training, so test-split
#    graphs are never written into the train cache. The design-level split
#    makes train/test graphs disjoint, so nothing is shared anyway. These must
#    match test.sh / test_cpu.sh exactly (same EVAL_ROOT) so the warmed cache
#    is found at eval time. Override with EVAL_ROOT=...
#    The HP-tuning splits file is the SAME as training's: with the same seed +
#    CSVs it reproduces the identical test split regardless of cache location.
# =========================================================

ALGO="Orchestrate"
CSV_PATH="$BASE_DIR/data/designs/design_metadata/algo_${ALGO}_ml.csv"

HP_TUNING_WORKSPACE="/scratch-shared/$USER/big_optuna_run"
HP_TUNING_SPLITS="$HP_TUNING_WORKSPACE/shared_dataset_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_50000_splits.json"

EVAL_ROOT="${EVAL_ROOT:-/scratch-shared/$USER/aig_eval_run}"
TIER0_CACHE_DIR="$EVAL_ROOT/shared_tier0_cache"
TIER1_CACHE_DIR="$EVAL_ROOT/shared_tier1_cache"
mkdir -p "$TIER0_CACHE_DIR" "$TIER1_CACHE_DIR"

CACHE_DIR="$EVAL_ROOT/${ALGO}/cache"
mkdir -p "$CACHE_DIR"

N_IO_WORKERS="${N_IO_WORKERS:-$(nproc)}"

# Split strategy — must match the SPLIT_BY passed to test.sh/test_cpu.sh (and
# the one the checkpoint was trained with), because a different strategy puts
# a different set of graphs in the test split. Unset means config.SPLIT_BY.
# The sentinel is keyed on it so warming "design" does not make a later
# "recipe" warmup skip itself with a cache that lacks its graphs.
SPLIT_BY="${SPLIT_BY:-}"
DEFAULT_SPLIT_BY=$(python -c 'import config; print(config.SPLIT_BY)')
if [[ -n "$SPLIT_BY" ]]; then
    SPLIT_BY_PY="\"$SPLIT_BY\""
else
    SPLIT_BY_PY="config.SPLIT_BY"
fi
# Only non-default strategies get a tag, matching
# data.dataset.splits_cache_filename and train.py's run_label. So an explicit
# SPLIT_BY=design reuses the untagged sentinel instead of forking a second one.
if [[ -n "$SPLIT_BY" && "$SPLIT_BY" != "$DEFAULT_SPLIT_BY" ]]; then
    SENTINEL="$CACHE_DIR/test_cache_ready_${SPLIT_BY}.sentinel"
else
    SENTINEL="$CACHE_DIR/test_cache_ready.sentinel"
fi

if [[ -f "$SENTINEL" ]]; then
    echo "[warmup:test] Cache already warm (sentinel exists). Skipping."
    exit 0
fi

if [[ ! -f "$CSV_PATH" ]]; then
    echo "[warmup:test] ERROR: CSV not found at $CSV_PATH" >&2
    exit 1
fi

splits_arg="None"
if [[ -f "$HP_TUNING_SPLITS" ]]; then
    splits_arg="\"$HP_TUNING_SPLITS\""
else
    echo "[warmup:test] WARNING: splits file not found at $HP_TUNING_SPLITS — using auto-generated splits."
fi

echo "[warmup:test] Building test-split cache in $CACHE_DIR (split_by=${SPLIT_BY:-<config default>}) ..."

python -u - <<PYEOF
import sys, time
sys.path.insert(0, "$BASE_DIR/src")
import config
from data.datamodule import AIGDataModule

t0 = time.monotonic()

dm = AIGDataModule(
    csv_paths=["$CSV_PATH"],
    positional_encoding=config.PE_TYPE if config.PE_TYPE != "none" else None,
    batch_size=config.BATCH_SIZE,
    split_ratios=(0.8, 0.1, 0.1),
    split_by=$SPLIT_BY_PY,
    seed=42,
    cache_dir="$CACHE_DIR",
    tier0_cache_dir="$TIER0_CACHE_DIR",
    tier1_cache_dir="$TIER1_CACHE_DIR",
    num_workers=$N_IO_WORKERS,
    hp_tuning_splits_path=$splits_arg,
)

dm.setup("test")
n_test = len(dm.test_ds)

elapsed = time.monotonic() - t0
print(f"[warmup:test] done in {elapsed:.1f}s — test={n_test}", flush=True)
PYEOF

touch "$SENTINEL"
echo "[warmup:test] Sentinel written: $SENTINEL"

echo "=========================================="
echo "Test cache warmup complete."
echo "End time: $(date)"
echo "=========================================="
