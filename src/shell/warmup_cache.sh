#!/bin/bash
#SBATCH --job-name=cache_warmup
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=genoa
#SBATCH --output=logs/cache_warmup_15.out

# ---------------------------------------------------------------------------
# Dedicated cache pre-warm job.
#
# Run this before submitting big_hp_tuning.sh. Builds the shared dataset
# cache on a CPU node so HP tuning workers skip GPFS I/O at trial start.
#
# Usage:
#   sbatch src/shell/warmup_cache.sh
#
# Or chain automatically:
#   WID=$(sbatch --parsable src/shell/warmup_cache.sh)
#   STAGE=1 sbatch --dependency=afterok:$WID src/shell/big_hp_tuning.sh
# ---------------------------------------------------------------------------

set -euo pipefail

echo "=========================================="
echo "CACHE WARMUP JOB"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "CPUs available: $(nproc)"
echo "Memory available: $(free -h | awk '/^Mem:/{print $2}')"
echo "=========================================="

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

SHARED_CACHE="/scratch-shared/$USER/big_optuna_run/shared_dataset_cache"
mkdir -p "$SHARED_CACHE"

CSV_1="$BASE_DIR/data/designs/design_metadata/algo_Orchestrate_ml.csv"
CSV_2="$BASE_DIR/data/designs/design_metadata/algo_Deepsyn_ml.csv"
CSV_3="$BASE_DIR/data/designs/design_metadata/algo_Syn4_ml.csv"
CSV_4="$BASE_DIR/data/designs/design_metadata/algo_C2RS_ml.csv"

# Use all CPUs allocated by SLURM for parallel I/O during cache build.
N_IO_WORKERS="${N_IO_WORKERS:-$(nproc)}"

warm_cache() {
    local n_samples=$1
    local sentinel="$SHARED_CACHE/cache_ready_n${n_samples}.sentinel"

    if [[ -f "$sentinel" ]]; then
        echo "[warmup] Cache already warm for n_samples=${n_samples} (sentinel exists). Skipping."
        return 0
    fi

    echo "[warmup] Building cache for n_samples=${n_samples} ..."
    python -u - <<PYEOF
import sys
sys.path.insert(0, "$BASE_DIR/src")
from data.datamodule import AIGDataModule

dm = AIGDataModule(
    csv_paths=["$CSV_1", "$CSV_2", "$CSV_3", "$CSV_4"],
    batch_size=4,
    split_ratios=(0.8, 0.2, 0.0),
    seed=42,
    cache_dir="$SHARED_CACHE",
    train_num_samples=${n_samples},
    num_workers=${N_IO_WORKERS},
    # Enable dynamic_batching so setup() also calls get_num_nodes_list() and
    # writes the node-sizes JSON cache.  Subsequent HP trials read this in < 1 s
    # instead of computing it from scratch.
    dynamic_batching=True,
)
dm.setup("fit")
print(
    f"[warmup] n_samples=${n_samples}: "
    f"{len(dm.train_ds)} train / {len(dm.val_ds)} val graphs cached, "
    f"{len(dm._train_sizes)} node-sizes written.",
    flush=True,
)
PYEOF

    touch "$sentinel"
    echo "[warmup] Sentinel written: $sentinel"
}

# Warm 15K for Stage 1. Re-run with n_samples=35000 before Stage 2.
warm_cache 15000

echo "=========================================="
echo "Cache warmup complete."
echo "End time: $(date)"
