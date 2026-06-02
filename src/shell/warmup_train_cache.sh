#!/bin/bash
#SBATCH --job-name=train_cache_warmup
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=genoa
#SBATCH --array=0-3
#SBATCH --output=logs/warmup_train_%A_%a.out

# ---------------------------------------------------------------------------
# Pre-warm the per-algorithm dataset cache for final training.
#
# Run this BEFORE submitting train.sh so the GPU node does not waste time on
# disk I/O.  This job runs on a cheap CPU partition and builds the graph
# cache (and optional node-sizes JSON) for the algorithm assigned to this
# array task (index 0-3 → Orchestrate/Deepsyn/Syn4/C2RS).
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

# Array-task id, or 0 when run outside an array for local debugging.
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "=========================================="
echo "TRAINING CACHE WARMUP JOB (array_task=${TASK_ID})"
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
# 2. ARRAY MAPPING (Algorithm)
# =========================================================

ALGORITHMS=("Orchestrate" "Deepsyn" "Syn4" "C2RS")
ALGO=${ALGORITHMS[$TASK_ID]}
echo "Task ${TASK_ID} assigned to ALGORITHM: ${ALGO}"

# =========================================================
# 3. SHARED PATHS
# =========================================================

# Must match the HP_TUNING_SPLITS path used in train.sh.
HP_TUNING_WORKSPACE="/scratch-shared/$USER/big_optuna_run"

# Shared cache for tier-0 (base) graphs — all 4 algorithms read from here
# so we store exactly one copy of each base graph instead of four.
TIER0_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier0_cache"
mkdir -p "$TIER0_CACHE_DIR"
# All 50K graphs used across both HP tuning stages (15K Stage-1 + 35K Stage-2).
# Using this file ensures zero HP tuning leakage into final train/val/test splits.
HP_TUNING_SPLITS="$HP_TUNING_WORKSPACE/shared_dataset_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_50000_splits.json"

# Number of parallel I/O workers.  Default: all SLURM-allocated CPUs.
N_IO_WORKERS="${N_IO_WORKERS:-$(nproc)}"
SPLIT_CACHE_VERSION="${SPLIT_CACHE_VERSION:-2}"

S1_SPLITS="$HP_TUNING_WORKSPACE/shared_dataset_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_15000_splits.json"
S2_SPLITS="$HP_TUNING_WORKSPACE/shared_dataset_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_35000_splits.json"

# =========================================================
# 3b. BUILD 50K EXCLUSION FILE (combine Stage-1 + Stage-2)
# =========================================================
# Merge the 15K and 35K HP-tuning split files into a single exclusion file so
# that no graph seen during either HP tuning stage leaks into final training.
# Idempotent: skips if the file already exists.
if [[ -f "$HP_TUNING_SPLITS" ]]; then
    echo "[exclusion] 50K exclusion file already exists. Skipping rebuild."
elif [[ ! -f "$S1_SPLITS" || ! -f "$S2_SPLITS" ]]; then
    echo "ERROR: Stage-1 or Stage-2 split file not found — cannot build exclusion file." >&2
    echo "  Expected:" >&2
    echo "    $S1_SPLITS" >&2
    echo "    $S2_SPLITS" >&2
    exit 1
else
    echo "[exclusion] Building 50K exclusion file from Stage-1 + Stage-2 splits..."
    python -u - <<PYEOF
import json
from pathlib import Path

s1 = json.loads(Path("$S1_SPLITS").read_text(encoding="utf-8"))
s2 = json.loads(Path("$S2_SPLITS").read_text(encoding="utf-8"))

all_keys = list(dict.fromkeys(
    s1.get("train", []) + s1.get("val", []) + s1.get("test", []) +
    s2.get("train", []) + s2.get("val", []) + s2.get("test", [])
))

out = Path("$HP_TUNING_SPLITS")
out.write_text(
    json.dumps({"train": all_keys, "val": [], "test": []}, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(f"[exclusion] Written {len(all_keys)} paths → {out.name}", flush=True)
PYEOF
    echo "[exclusion] Done."
fi

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
        if python - "$cache_dir" "$SPLIT_CACHE_VERSION" <<'PYEOF'
import json
import sys
from pathlib import Path

cache_dir = Path(sys.argv[1])
expected_version = int(sys.argv[2])
split_files = sorted(cache_dir.glob("*_splits.json"))
if not split_files:
    raise SystemExit(1)

payload = json.loads(split_files[0].read_text(encoding="utf-8"))
meta = payload.get("__meta__")
if not isinstance(meta, dict):
    raise SystemExit(1)

raise SystemExit(
    0
    if meta.get("version") == expected_version and meta.get("split_by") == "design"
    else 1
)
PYEOF
        then
            echo "[warmup:${algo}] Cache already warm with current split metadata. Skipping."
            return 0
        fi

        echo "[warmup:${algo}] Sentinel exists, but split cache metadata is stale. Rebuilding."
        rm -f "$sentinel"
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
import sys, time, json
from pathlib import Path
sys.path.insert(0, "$BASE_DIR/src")
import config
from data.datamodule import AIGDataModule

t0 = time.monotonic()

# Parse dynamic bucket rules from config
parsed_rules = []
if getattr(config, "DYNAMIC_BUCKET_RULES", None):
    parsed_rules = [
        tuple(map(int, s.split(':')))
        for s in config.DYNAMIC_BUCKET_RULES.split(',')
        if s.strip()
    ]

dm = AIGDataModule(
    csv_paths=["$csv_path"],
    batch_size=config.BATCH_SIZE,
    split_ratios=(0.8, 0.1, 0.1),
    seed=42,
    cache_dir="$cache_dir",
    tier0_cache_dir="$TIER0_CACHE_DIR",
    num_workers=$N_IO_WORKERS,
    hp_tuning_splits_path=$splits_arg,
    # Precompute node-sizes so dynamic_batching=True is instant at training time.
    dynamic_batching=config.DYNAMIC_BATCHING,
    dynamic_bucket_rules=parsed_rules,
)

# Warm train + val only; test does not need to be preloaded.
dm.setup("fit")
n_train = len(dm.train_ds)
n_val   = len(dm.val_ds)
n_sizes = len(dm.train_ds.get_num_nodes_list()) if getattr(dm, "train_ds", None) is not None else 0

# Log test split membership without building the graph cache for it.
n_test = 0
split_files = sorted(Path("$cache_dir").glob("*_splits.json"))
if split_files:
    test_splits = json.loads(split_files[0].read_text(encoding="utf-8"))
    n_test = len(test_splits.get("test", []))
    print(f"[warmup:${algo}] test split: {n_test} samples logged (not warmed up)", flush=True)

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
# 5. RUN WARMUP FOR THIS ARRAY TASK'S ALGORITHM
# =========================================================

echo ""
echo "------------------------------------------"
echo "Processing algorithm: $ALGO"
echo "------------------------------------------"
warm_algorithm "$ALGO"

echo ""
echo "=========================================="
echo "Warmup complete for ${ALGO}."
echo "End time: $(date)"
echo "=========================================="
