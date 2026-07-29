#!/bin/bash
#SBATCH --job-name=hoga_hop_cache_warmup
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=genoa
#SBATCH --output=logs/warmup_hoga_hop_cache_%j.out

# ---------------------------------------------------------------------------
# Pre-compute HOGA's hop-stacked node features (see
# src/baselines/hoga/hop_features.py) for train/val/test, on CPU, before
# submitting train_baseline_hoga.sh -- so the GPU job never stalls computing
# them. This is the same idea as warmup_train_cache.sh, applied to the
# baseline's own on-disk hop-feature cache instead of the primary model's
# graph cache.
#
# CHAIN WITH THE HOGA BASELINE TRAIN JOB
# ---------------------------------------
#   WID=$(sbatch --parsable src/shell/warmup_hoga_hop_cache.sh)
#   sbatch --dependency=afterok:$WID src/shell/train_baseline_hoga.sh
#
# Already-cached samples are skipped (HopFeatureCache checks the on-disk file
# before recomputing), so re-running this script is always safe.
#
# HOGA_NUM_HOPS / HOGA_DIRECTED must match the values passed to
# train_baseline_hoga.sh, or the cache built here won't be reused (cache
# filenames are keyed by both).
# ---------------------------------------------------------------------------

set -euo pipefail

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

ALGORITHM="Orchestrate"
CSV_PATH="$BASE_DIR/data/designs/design_metadata/algo_${ALGORITHM}_ml.csv"

WORKSPACE="/scratch-shared/$USER/aig_baseline_run/hoga_${ALGORITHM}"
CACHE_DIR="$WORKSPACE/cache"
TIER0_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier0_cache"
TIER1_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier1_cache"
HP_TUNING_SPLITS="/scratch-shared/$USER/big_optuna_run/shared_dataset_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_50000_splits.json"

HOGA_HOP_CACHE_DIR="/scratch-shared/$USER/aig_baseline_run/hoga_hop_cache"
HOGA_NUM_HOPS="${HOGA_NUM_HOPS:-5}"
HOGA_DIRECTED="${HOGA_DIRECTED:-true}"
N_IO_WORKERS="${N_IO_WORKERS:-$(nproc)}"

mkdir -p "$CACHE_DIR" "$HOGA_HOP_CACHE_DIR"

echo "=========================================="
echo "HOGA HOP-FEATURE CACHE WARMUP"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "HOGA_NUM_HOPS=$HOGA_NUM_HOPS  HOGA_DIRECTED=$HOGA_DIRECTED"
echo "=========================================="

python -u - <<PYEOF
import sys, time
sys.path.insert(0, "$BASE_DIR/src")

import config
from data.datamodule import AIGDataModule
from baselines.hoga.hop_features import HopFeatureCache

dm = AIGDataModule(
    csv_paths=["$CSV_PATH"],
    positional_encoding=None,
    batch_size=config.BATCH_SIZE,
    split_ratios=(0.8, 0.1, 0.1),
    seed=42,
    cache_dir="$CACHE_DIR",
    tier0_cache_dir="$TIER0_CACHE_DIR",
    tier1_cache_dir="$TIER1_CACHE_DIR",
    num_workers=$N_IO_WORKERS,
    hp_tuning_splits_path="$HP_TUNING_SPLITS" if __import__("os").path.isfile("$HP_TUNING_SPLITS") else None,
    dynamic_batching=False,
)
dm.setup("fit")
dm.setup("test")

directed = "$HOGA_DIRECTED".lower() in ("true", "1", "yes")
num_hops = int("$HOGA_NUM_HOPS")

for split_name, ds in (("train", dm.train_ds), ("val", dm.val_ds), ("test", dm.test_ds)):
    t0 = time.monotonic()
    cache = HopFeatureCache(ds, num_hops=num_hops, cache_dir="$HOGA_HOP_CACHE_DIR", directed=directed)
    print(f"[warmup:hoga] {split_name}: {len(cache)} samples", flush=True)
    cache.precompute_all(log_every=1000)
    print(f"[warmup:hoga] {split_name} done in {time.monotonic() - t0:.1f}s", flush=True)

print("[warmup:hoga] All splits cached.", flush=True)
PYEOF

echo "=========================================="
echo "Warmup complete."
echo "End time: $(date)"
echo "=========================================="
