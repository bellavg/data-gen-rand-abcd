#!/bin/bash
#SBATCH --job-name=train_cache_warmup
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=genoa
#SBATCH --array=0-2%1
#SBATCH --output=logs/warmup_train_%A_%a.out

# ---------------------------------------------------------------------------
# Pre-warm the train+val dataset cache for final training.
#
# Run this BEFORE submitting the train job so the GPU node does not waste time
# on disk I/O.  This job runs on a cheap CPU partition and writes the splits
# JSON plus the dataset manifest (graph paths + node counts) for the split
# strategy assigned to this array task (index 0-2 → design/recipe/random),
# matching train_no_sparsification.sh's array convention so the two line up
# task-for-task.
#
# It does NOT build a preprocessed graph cache -- it runs with
# use_graph_cache=False and sizes graphs from the CSV's abc stats instead.
# train.sh (sparsification sweep) and test.sh still default to
# use_graph_cache=True and will build their own cache in $WORKSPACE/cache on
# first run; this job no longer warms it for them.
#
# The %1 throttle runs the tasks one at a time on purpose. Unlike the previous
# per-algorithm array, every strategy writes the SAME cache_dir and shared tier
# dirs, and their graph sets overlap heavily -- run concurrently they would
# re-read the same graphs from GPFS and clobber each other's last-writer-wins
# _num_nodes_global.json updates.
#
# Only Orchestrate is warmed: it is the sole algorithm training accepts
# (config.VALID_ALGORITHMS). The other three exist on disk from the data
# generation pipeline but nothing reads their train caches.
#
# CHAIN WITH TRAIN JOB
# --------------------
#   WID=$(sbatch --parsable src/shell/warmup_train_cache.sh)
#   sbatch --dependency=afterok:$WID --array=0-2 src/shell/train_no_sparsification.sh
#
# If the sentinel already exists for a strategy the warmup skips it, so
# re-running or re-chaining is always safe.
# ---------------------------------------------------------------------------

set -euo pipefail

# Array-task id, or 0 (design) when run outside an array for local debugging.
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
# 2. ARRAY MAPPING (split strategy)
# =========================================================

# Training only ever runs on Orchestrate (config.VALID_ALGORITHMS), so the
# array slot selects the split strategy instead of the algorithm.
ALGO="Orchestrate"

# Same mapping as train_no_sparsification.sh, so `--array=0-2` on both means
# the same three runs. SLURM_ARRAY_TASK_ID is injected by the scheduler
# directly (unlike an exported env var, its delivery does not depend on
# --export policy), so the strategy is selected with no environment passing.
# Falls back to the SPLIT_BY env var for a plain, non-array submission.
declare -A SPLIT_BY_MODES=([0]="design" [1]="recipe" [2]="random")
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SPLIT_BY="${SPLIT_BY_MODES[$SLURM_ARRAY_TASK_ID]:?Unknown SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID, expected one of: ${!SPLIT_BY_MODES[*]}}"
else
    SPLIT_BY="${SPLIT_BY:-design}"
fi
echo "Task ${TASK_ID} assigned to SPLIT_BY: ${SPLIT_BY} (ALGORITHM: ${ALGO})"

# =========================================================
# 3. SHARED PATHS
# =========================================================

# Must match the HP_TUNING_SPLITS path used in train.sh.
HP_TUNING_WORKSPACE="/scratch-shared/$USER/big_optuna_run"

# Shared cache for tier-0 (base) graphs — all 4 algorithms read from here
# so we store exactly one copy of each base graph instead of four.
TIER0_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier0_cache"
mkdir -p "$TIER0_CACHE_DIR"
# Shared cache for tier-1 graphs — tier-2 training rows across target
# algorithms reuse the same tier-1 input graphs, so cache them once.
TIER1_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier1_cache"
mkdir -p "$TIER1_CACHE_DIR"
# All 50K graphs used across both HP tuning stages (15K Stage-1 + 35K Stage-2).
# Using this file ensures zero HP tuning leakage into final train/val/test splits.
HP_TUNING_SPLITS="$HP_TUNING_WORKSPACE/shared_dataset_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_50000_splits.json"

# Number of parallel I/O workers.  Default: all SLURM-allocated CPUs.
N_IO_WORKERS="${N_IO_WORKERS:-$(nproc)}"
SPLIT_CACHE_VERSION="${SPLIT_CACHE_VERSION:-2}"
# Bumped 3 -> 4 with use_graph_cache=False: manifests now hold raw .pt paths
# instead of sha1 cache filenames, so a sentinel from a layout-3 run must not
# let this job skip.
CACHE_LAYOUT_VERSION="${CACHE_LAYOUT_VERSION:-4}"

# The default strategy stays untagged and the others get a tag, so the caches
# and sentinels written before split_by was configurable remain valid. Same
# rule as data.dataset.splits_cache_filename and train.py's run_label. Both the
# sentinel and the staleness check are keyed on it, so warming "design" cannot
# make a later "recipe" warmup skip itself with a cache that lacks its graphs.
DEFAULT_SPLIT_BY=$(python -c 'import config; print(config.SPLIT_BY)')
if [[ "$SPLIT_BY" != "$DEFAULT_SPLIT_BY" ]]; then
    SENTINEL_SUFFIX="_$SPLIT_BY"
else
    SENTINEL_SUFFIX=""
fi

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
    tmp_exclusion="${HP_TUNING_SPLITS}.tmp.${SLURM_JOB_ID:-$$}.${SLURM_ARRAY_TASK_ID:-0}.$$"
    cleanup_exclusion_tmp() {
        rm -f "$tmp_exclusion"
    }
    trap cleanup_exclusion_tmp EXIT
    python -u - <<PYEOF
import json
from pathlib import Path

s1 = json.loads(Path("$S1_SPLITS").read_text(encoding="utf-8"))
s2 = json.loads(Path("$S2_SPLITS").read_text(encoding="utf-8"))

all_keys = list(dict.fromkeys(
    s1.get("train", []) + s1.get("val", []) + s1.get("test", []) +
    s2.get("train", []) + s2.get("val", []) + s2.get("test", [])
))

tmp_out = Path("$tmp_exclusion")
tmp_out.write_text(
    json.dumps({"train": all_keys, "val": [], "test": []}, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(f"[exclusion] Prepared {len(all_keys)} paths → {tmp_out.name}", flush=True)
PYEOF
    if mv -n "$tmp_exclusion" "$HP_TUNING_SPLITS" 2>/dev/null; then
        echo "[exclusion] Published 50K exclusion file."
    else
        echo "[exclusion] Another worker published the exclusion file first. Keeping existing file."
        rm -f "$tmp_exclusion"
    fi
    trap - EXIT
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
    local sentinel="$cache_dir/train_cache_ready${SENTINEL_SUFFIX}.sentinel"
    # Suffixed like the sentinel: one shared layout file per cache_dir would be
    # written by whichever strategy rebuilt first, letting the remaining array
    # tasks read the NEW version next to their own OLD sentinel and skip.
    local layout_version_file="$cache_dir/cache_layout_version${SENTINEL_SUFFIX}.txt"

    mkdir -p "$cache_dir"

    if [[ -f "$sentinel" ]]; then
        if python - "$cache_dir" "$SPLIT_CACHE_VERSION" "$CACHE_LAYOUT_VERSION" "$csv_path" "$SPLIT_BY" "$layout_version_file" <<'PYEOF'
import json
import sys
from pathlib import Path

from data.dataset import splits_cache_filename

cache_dir = Path(sys.argv[1])
expected_version = int(sys.argv[2])
expected_layout_version = int(sys.argv[3])
csv_path = sys.argv[4]
split_by = sys.argv[5]
layout_file = Path(sys.argv[6])

# The exact file this warmup would write, NOT the alphabetically-first
# "*_splits.json": a cache_dir legitimately holds one per (num_samples,
# split_by) combination, so globbing checks a subset's or another strategy's
# metadata and reports a cache warm that was never built.
splits_file = cache_dir / splits_cache_filename([csv_path], None, split_by)
if not splits_file.is_file():
    raise SystemExit(1)

payload = json.loads(splits_file.read_text(encoding="utf-8"))
meta = payload.get("__meta__")
if not isinstance(meta, dict):
    raise SystemExit(1)

if not layout_file.is_file():
    raise SystemExit(1)

try:
    layout_version = int(layout_file.read_text(encoding="utf-8").strip())
except ValueError:
    raise SystemExit(1)

raise SystemExit(
    0
    if meta.get("version") == expected_version
    and meta.get("split_by") == split_by
    and layout_version == expected_layout_version
    else 1
)
PYEOF
        then
            echo "[warmup:${algo}] Cache already warm with current split metadata. Skipping."
            return 0
        fi

        echo "[warmup:${algo}] Sentinel exists, but cache metadata/layout is stale. Rebuilding."
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
from data.dataset import splits_cache_filename

t0 = time.monotonic()

dm = AIGDataModule(
    csv_paths=["$csv_path"],
    positional_encoding=config.PE_TYPE if config.PE_TYPE != "none" else None,
    batch_size=config.BATCH_SIZE,
    split_ratios=(0.8, 0.1, 0.1),
    split_by="$SPLIT_BY",
    seed=42,
    cache_dir="$cache_dir",
    tier0_cache_dir="$TIER0_CACHE_DIR",
    tier1_cache_dir="$TIER1_CACHE_DIR",
    num_workers=$N_IO_WORKERS,
    hp_tuning_splits_path=$splits_arg,
    # No graph cache: with NORMALIZE_EDGES=False the "cached" copy is just the
    # raw graph minus pi_paths/local_sp_sum (which get() drops anyway), so
    # rewriting 700k+ files bought ~10% file size for >30h of wall time. The
    # manifest is built from the CSV's abc stats instead.
    use_graph_cache=False,
    # Precompute node-sizes so dynamic_batching=True is instant at training time.
    dynamic_batching=config.DYNAMIC_BATCHING,
    max_total_nodes=config.MAX_TOTAL_NODES_PER_BATCH,
)

# Warm train + val only; test does not need to be preloaded.
dm.setup("fit")
n_train = len(dm.train_ds)
n_val   = len(dm.val_ds)
n_sizes = len(dm.train_ds.get_num_nodes_list()) if getattr(dm, "train_ds", None) is not None else 0

# Log test split membership without building the graph cache for it. Read the
# file this run just wrote, not the first "*_splits.json" in the directory --
# that picks up another num_samples/split_by tag and logs its test count here.
n_test = 0
splits_file = Path("$cache_dir") / splits_cache_filename(
    ["$csv_path"], None, dm.split_by
)
if splits_file.is_file():
    test_splits = json.loads(splits_file.read_text(encoding="utf-8"))
    n_test = len(test_splits.get("test", []))
    print(f"[warmup:${algo}] test split: {n_test} samples logged (not warmed up)", flush=True)

elapsed = time.monotonic() - t0
print(
    f"[warmup:${algo}] done in {elapsed:.1f}s — "
    f"train={n_train}  val={n_val}  test={n_test}  node-sizes={n_sizes}",
    flush=True,
)
PYEOF

    printf '%s\n' "$CACHE_LAYOUT_VERSION" > "$layout_version_file"

    touch "$sentinel"
    echo "[warmup:${algo}] Sentinel written: $sentinel"
}

# =========================================================
# 5. RUN WARMUP FOR THIS ARRAY TASK'S ALGORITHM
# =========================================================

echo ""
echo "------------------------------------------"
echo "Processing $ALGO with split_by=$SPLIT_BY"
echo "------------------------------------------"
warm_algorithm "$ALGO"

echo ""
echo "=========================================="
echo "Warmup complete for ${ALGO} (split_by=${SPLIT_BY})."
echo "End time: $(date)"
echo "=========================================="
