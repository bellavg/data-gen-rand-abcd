#!/bin/bash
#SBATCH --job-name=train_cache_warmup
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=genoa
#SBATCH --output=logs/warmup_train_%j.out

# ---------------------------------------------------------------------------
# Pre-warm the per-algorithm dataset cache for final training.
#
# Run this BEFORE submitting train.sh so the GPU node does not waste time on
# disk I/O.  This job runs on a cheap CPU partition and builds the graph
# cache (and optional node-sizes JSON) for each algorithm.
#
# ALGORITHM SELECTION
# -------------------
# By default all four algorithms are processed.  To run a subset, pass a
# space-separated list via the TRAIN_ALGORITHMS environment variable:
#
#   TRAIN_ALGORITHMS="Orchestrate Syn4" sbatch src/shell/warmup_train_cache.sh
#
# CHAIN WITH TRAIN JOB
# --------------------
#   WID=$(sbatch --parsable src/shell/warmup_train_cache.sh)
#   sbatch --dependency=afterok:$WID src/shell/train.sh
#
# If the sentinel already exists for an algorithm the warmup skips it, so
# re-running or re-chaining is always safe.
# ---------------------------------------------------------------------------

set -euo pipefail

echo "=========================================="
echo "TRAINING CACHE WARMUP JOB"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "CPUs available: $(nproc)"
echo "Memory available: $(free -h | awk '/^Mem:/{print $2}')"
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
# 2. ALGORITHM SELECTION
# =========================================================
# Edit the default list here, or override at submission time:
#   TRAIN_ALGORITHMS="Orchestrate" sbatch ...

ALL_ALGORITHMS=("Orchestrate" "Deepsyn" "Syn4" "C2RS")

if [[ -n "${TRAIN_ALGORITHMS:-}" ]]; then
    read -r -a SELECTED_ALGORITHMS <<< "$TRAIN_ALGORITHMS"
else
    SELECTED_ALGORITHMS=("${ALL_ALGORITHMS[@]}")
fi

echo "Algorithms to warm: ${SELECTED_ALGORITHMS[*]}"

# =========================================================
# 3. SHARED PATHS
# =========================================================

# Must match the HP_TUNING_SPLITS path used in train.sh.
HP_TUNING_WORKSPACE="/scratch-shared/$USER/aig_optuna_run"
HP_TUNING_SPLITS="$HP_TUNING_WORKSPACE/hp_tuning/shared_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_20000_splits.json"

# Number of parallel I/O workers.  Default: all SLURM-allocated CPUs.
N_IO_WORKERS="${N_IO_WORKERS:-$(nproc)}"

# =========================================================
# 4. WARMUP FUNCTION (one call per algorithm)
# =========================================================

warm_algorithm() {
    local algo="$1"

    local csv_path="$BASE_DIR/data/designs/design_metadata/algo_${algo}_ml.csv"
    local workspace="/scratch-shared/$USER/aig_train_run/${algo}"
    local cache_dir="$workspace/cache"
    local sentinel="$cache_dir/train_cache_ready.sentinel"

    mkdir -p "$cache_dir"

    if [[ -f "$sentinel" ]]; then
        echo "[warmup:${algo}] Cache already warm (sentinel exists). Skipping."
        return 0
    fi

    if [[ ! -f "$csv_path" ]]; then
        echo "[warmup:${algo}] ERROR: CSV not found at $csv_path — skipping."
        return 1
    fi

    echo "[warmup:${algo}] Building cache in $cache_dir ..."

    # Resolve optional splits file (pass None if not found).
    local splits_arg="None"
    if [[ -f "$HP_TUNING_SPLITS" ]]; then
        splits_arg="\"$HP_TUNING_SPLITS\""
    else
        echo "[warmup:${algo}] WARNING: splits file not found at $HP_TUNING_SPLITS — using auto-generated splits."
    fi

    python -u - <<PYEOF
import sys, time
sys.path.insert(0, "$BASE_DIR/src")
from data.datamodule import AIGDataModule

t0 = time.monotonic()

dm = AIGDataModule(
    csv_paths=["$csv_path"],
    batch_size=4,
    split_ratios=(0.8, 0.1, 0.1),
    seed=42,
    cache_dir="$cache_dir",
    num_workers=$N_IO_WORKERS,
    use_full_test_set=True,
    hp_tuning_splits_path=$splits_arg,
    # Precompute node-sizes so dynamic_batching=True is instant at training time.
    dynamic_batching=True,
)

# Warm train + val
dm.setup("fit")
n_train = len(dm.train_ds)
n_val   = len(dm.val_ds)
n_sizes = len(getattr(dm, "_train_sizes", []))

# Warm test
dm.setup("test")
n_test = len(dm.test_ds)

elapsed = time.monotonic() - t0
print(
    f"[warmup:${algo}] done in {elapsed:.1f}s — "
    f"train={n_train}  val={n_val}  test={n_test}  node-sizes={n_sizes}",
    flush=True,
)
PYEOF

    touch "$sentinel"
    echo "[warmup:${algo}] Sentinel written: $sentinel"
}

# =========================================================
# 5. MAIN LOOP — warm each algorithm sequentially
#    (parallel warmups would contend on GPFS; sequential is safer)
# =========================================================

for ALGO in "${SELECTED_ALGORITHMS[@]}"; do
    echo ""
    echo "------------------------------------------"
    echo "Processing algorithm: $ALGO"
    echo "------------------------------------------"
    warm_algorithm "$ALGO"
done

echo ""
echo "=========================================="
echo "All warmups complete."
echo "End time: $(date)"
echo "=========================================="
