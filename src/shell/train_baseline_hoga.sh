#!/bin/bash
#SBATCH --job-name=aig_train_baseline_hoga
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --constraint=scratch-node
#SBATCH --output=logs/train_baseline_hoga_%j.out

set -euo pipefail

export TEMP="$TMPDIR"
export TMP="$TMPDIR"

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load SciPy-bundle/2025.06-gfbf-2025a
module load gperftools/2.16-GCCcore-14.2.0
export LD_PRELOAD="${EBROOTGPERFTOOLS}/lib/libtcmalloc.so${LD_PRELOAD:+:${LD_PRELOAD}}"

VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
source "$VENV_PATH/bin/activate"

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BASE_DIR/src"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export AIG_REQUIRE_GPU=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$BASE_DIR"

ALGORITHM="Orchestrate"
CSV_PATH="$BASE_DIR/data/designs/design_metadata/algo_${ALGORITHM}_ml.csv"

WORKSPACE="/scratch-shared/$USER/aig_baseline_run/hoga_${ALGORITHM}"
CHECKPOINT_DIR="$WORKSPACE/checkpoints"
LOG_DIR="$WORKSPACE/logs"
CACHE_DIR="$WORKSPACE/cache"
TIER0_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier0_cache"
TIER1_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier1_cache"

# Same HP-tuning exclusion file as the primary model's train.sh, so the
# baseline's train/val/test rows never overlap with graphs used for HP
# tuning, and both models are compared on identical held-out splits.
HP_TUNING_SPLITS="/scratch-shared/$USER/big_optuna_run/shared_dataset_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_50000_splits.json"
[ ! -f "$HP_TUNING_SPLITS" ] && echo "WARNING: HP Tuning split file not found at $HP_TUNING_SPLITS"

# Hop-feature cache -- run src/shell/warmup_hoga_hop_cache.sh first (CPU-only
# partition) so this GPU job never stalls computing hop features; see
# src/baselines/hoga/hop_features.py. Must match warmup's HOGA_NUM_HOPS /
# HOGA_DIRECTED for the cache to be reused rather than recomputed.
HOGA_HOP_CACHE_DIR="/scratch-shared/$USER/aig_baseline_run/hoga_hop_cache"
HOGA_NUM_HOPS="${HOGA_NUM_HOPS:-5}"
HOGA_DIRECTED="${HOGA_DIRECTED:-true}"

NUM_WORKERS="${NUM_WORKERS:-12}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PIN_MEMORY="${PIN_MEMORY:-true}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
SPLIT_BY="${SPLIT_BY:-design}"

echo "Using NUM_WORKERS=$NUM_WORKERS for data loading."
echo "Using SPLIT_BY=$SPLIT_BY."
echo "Using HOGA_NUM_HOPS=$HOGA_NUM_HOPS, HOGA_DIRECTED=$HOGA_DIRECTED."
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# =========================================================
# EXECUTE TRAINING (HOGA baseline. hidden_dim/num_layers/num_hops/lr are
# published QoR-task values from Deng et al. DAC'24 Sec 3.3/4.1; --hoga_heads
# has no QoR-task source and carries over from HOGA's Gamora-task run.sh --
# see src/baselines/hoga/regressor.py for the full breakdown.)
# =========================================================

echo "Starting HOGA baseline training for $ALGORITHM on GPU 0..."

srun python -u -m train_baseline \
    --baseline           "hoga" \
    --algorithm          "$ALGORITHM" \
    --csv_paths          "$CSV_PATH" \
    --checkpoint_dir     "$CHECKPOINT_DIR" \
    --log_dir            "$LOG_DIR" \
    --cache_dir          "$CACHE_DIR" \
    --tier0_cache_dir    "$TIER0_CACHE_DIR" \
    --tier1_cache_dir    "$TIER1_CACHE_DIR" \
    --hp_tuning_splits_path "$HP_TUNING_SPLITS" \
    --split_by           "$SPLIT_BY" \
    --hoga_hop_cache_dir "$HOGA_HOP_CACHE_DIR" \
    --hoga_num_hops      "$HOGA_NUM_HOPS" \
    --hoga_directed      "$HOGA_DIRECTED" \
    --prefetch_factor    "$PREFETCH_FACTOR" \
    --num_workers        "$NUM_WORKERS" \
    --pin_memory         "$PIN_MEMORY" \
    --persistent_workers "$PERSISTENT_WORKERS" \
    --patience           4
