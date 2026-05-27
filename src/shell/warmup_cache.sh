#!/bin/bash
#SBATCH --job-name=cache_warmup
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=genoa
#SBATCH --output=logs/cache_warmup_35.out

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
USE_CONDA_ENV="${USE_CONDA_ENV:-false}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3/2025.06-1}"
if [[ "$USE_CONDA_ENV" == "true" ]]; then
    module load "$CONDA_MODULE"
else
    module load Python/3.13.1-GCCcore-14.2.0
    module load SciPy-bundle/2025.06-gfbf-2025a
fi

if [[ "$USE_CONDA_ENV" == "true" ]]; then
    CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/scratch-shared/$USER/.conda/envs/data-gen-py312}"
    echo "Activating conda environment at: $CONDA_ENV_PREFIX"
    # set +u: Anaconda's activate.d/qt-main_activate.sh uses $QT_XCB_GL_INTEGRATION
    # without a default, fatal under set -u.
    set +u
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    elif [[ -n "${EBROOTANACONDA3:-}" && -f "${EBROOTANACONDA3}/etc/profile.d/conda.sh" ]]; then
        source "${EBROOTANACONDA3}/etc/profile.d/conda.sh"
    else
        set -u
        echo "ERROR: conda command not found after loading $CONDA_MODULE" >&2
        exit 1
    fi
    conda activate "$CONDA_ENV_PREFIX"
    set -u
else
    VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
    source "$VENV_PATH/bin/activate"
fi

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
DYNAMIC_BUCKET_RULES="${DYNAMIC_BUCKET_RULES:-240000:1,160000:1,100000:2}"

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


def parse_rules(text: str):
    out = []
    for chunk in (c.strip() for c in text.split(",") if c.strip()):
        min_nodes, batch_size = chunk.split(":", 1)
        out.append((int(min_nodes), int(batch_size)))
    return out


dynamic_rules = parse_rules("$DYNAMIC_BUCKET_RULES")

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
    dynamic_bucket_rules=dynamic_rules,
)
dm.setup("fit")
n_plan_batches = len(getattr(dm, "_train_batch_plan", []) or [])
print(
    f"[warmup] n_samples=${n_samples}: "
    f"{len(dm.train_ds)} train / {len(dm.val_ds)} val graphs cached, "
    f"dynamic_batch_plan_batches={n_plan_batches} "
    f"(rules=$DYNAMIC_BUCKET_RULES).",
    flush=True,
)
PYEOF

    touch "$sentinel"
    echo "[warmup] Sentinel written: $sentinel"
}

# ---------------------------------------------------------------------------
# Step 1: pre-compute disjoint splits for both stages.
#
# Shuffle all unique graph paths with seed=42, then partition:
#   Stage 1: first 15K unique paths  (80/20 train/val)
#   Stage 2: next  35K unique paths  (80/20 train/val)
# Writes the splits JSONs to the exact paths _load_or_create_split_keys
# expects, so warm_cache and HP workers both pick them up without rebuilding.
# ---------------------------------------------------------------------------
echo "[splits] Generating disjoint 15K / 35K splits from 50K pool..."
python -u - <<PYEOF
import json
import random
from pathlib import Path

import pandas as pd

csv_paths = ["$CSV_1", "$CSV_2", "$CSV_3", "$CSV_4"]
shared_cache = Path("$SHARED_CACHE")
shared_cache.mkdir(parents=True, exist_ok=True)

df = pd.concat(
    [pd.read_csv(p, dtype=str).fillna("") for p in csv_paths],
    ignore_index=True,
)

df["unoptimized_graph_path"] = df["unoptimized_graph_path"].str.replace(
    "/gpfs/scratch1/shared", "/scratch-shared"
)

unique_keys = list(dict.fromkeys(df["unoptimized_graph_path"].tolist()))
print(f"[splits] Total unique graph paths: {len(unique_keys)}", flush=True)

rng = random.Random(42)
rng.shuffle(unique_keys)

stage1_keys = unique_keys[:15000]
stage2_keys = unique_keys[15000:50000]

def make_split(keys):
    n = len(keys)
    n_train = int(n * 0.8)
    n_val = int(n * 0.2)
    return {
        "train": keys[:n_train],
        "val": keys[n_train : n_train + n_val],
        "test": keys[n_train + n_val :],
    }

algo_tag = "_".join(Path(p).stem for p in csv_paths)
for n, stage_keys in [(15000, stage1_keys), (35000, stage2_keys)]:
    split = make_split(stage_keys)
    out = shared_cache / f"{algo_tag}_{n}_splits.json"
    out.write_text(json.dumps(split, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[splits] {n}: {len(split['train'])} train / {len(split['val'])} val "
        f"from {len(stage_keys)} unique graphs → {out.name}",
        flush=True,
    )
PYEOF

# ---------------------------------------------------------------------------
# Step 2: cache the graphs for Stage 1 now.
# Before Stage 2, resubmit this script — warm_cache 15000 will skip (sentinel
# already present) and warm_cache 35000 will run on the disjoint graph set.
# ---------------------------------------------------------------------------
# warm_cache 15000
warm_cache 35000  # uncomment (or resubmit) before Stage 2

echo "=========================================="
echo "Cache warmup complete."
echo "End time: $(date)"
