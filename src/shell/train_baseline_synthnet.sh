#!/bin/bash
#SBATCH --job-name=aig_train_baseline_synthnet
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --constraint=scratch-node
#SBATCH --output=logs/train_baseline_synthnet_%j.out

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
# CSV_PATH intentionally does NOT derive from BASE_DIR -- see
# train_baseline_hoga.sh's matching comment: _build_cache_signature() hashes
# this path's literal absolute string, so it must stay pinned to the same
# checkout train.sh runs from even when BASE_DIR is overridden to a different
# worktree for PYTHONPATH.
DATA_BASE_DIR="${DATA_BASE_DIR:-$HOME/data-gen-rand-abcd}"
CSV_PATH="$DATA_BASE_DIR/data/designs/design_metadata/algo_${ALGORITHM}_ml.csv"

WORKSPACE="/scratch-shared/$USER/aig_baseline_run/synthnet_${ALGORITHM}"
CHECKPOINT_DIR="$WORKSPACE/checkpoints"
LOG_DIR="$WORKSPACE/logs"
# Reuse the primary model's own per-algorithm cache_dir (not a separate
# aig_baseline_run/.../cache) -- see train_baseline_hoga.sh's matching
# comment: the graph-cache manifest is keyed by cache_dir (not the cache
# signature), so pointing here at train.sh's own CACHE_DIR lets this job
# find the manifest the primary run already built instead of rebuilding it
# from scratch for all ~700k samples.
CACHE_DIR="/scratch-shared/$USER/aig_train_run/${ALGORITHM}/cache"
TIER0_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier0_cache"
TIER1_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier1_cache"

# Same HP-tuning exclusion file as the primary model's train.sh, so the
# baseline's train/val/test rows never overlap with graphs used for HP
# tuning, and both models are compared on identical held-out splits.
HP_TUNING_SPLITS="/scratch-shared/$USER/big_optuna_run/shared_dataset_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_50000_splits.json"
[ ! -f "$HP_TUNING_SPLITS" ] && echo "WARNING: HP Tuning split file not found at $HP_TUNING_SPLITS"

NUM_WORKERS="${NUM_WORKERS:-16}"  # of the 18 cores auto-assigned per GPU on gpu_h100; 2 left for the main process + pin_memory thread
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PIN_MEMORY="${PIN_MEMORY:-true}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
# Must match the strategy warmup_train_cache.sh warmed (its array slot 0 is
# "design"), and --use_graph_cache must match that job's use_graph_cache=False:
# both are hashed into the dataset's cache signature, so a mismatch renames the
# manifest, misses the warm one, and re-derives all ~700k train samples on the
# GPU node. Same default as train_no_sparsification.sh's array slot 0.
#
# NOTE these baseline scripts are single-submission only: unlike
# train_no_sparsification.sh and warmup_train_cache.sh, they do NOT map
# SLURM_ARRAY_TASK_ID to a strategy. Submitting one with --array=0-2 would run
# three identical "design" jobs into one checkpoint dir. To run another
# strategy, submit separately with
#   sbatch --export=ALL,SPLIT_BY=recipe <script>
# (bare VAR=value sbatch does not propagate on Snellius).
SPLIT_BY="${SPLIT_BY:-design}"

# Edge direction fed to the GCN trunk. true is upstream's own direction:
# andAIG2Graphml.py:56 writes edges node -> fanin and pygDataFromNetworkx
# passes list(G.edges) through unreversed, so under PyG's default
# flow="source_to_target" each node summarises its fanout cone. That is the
# direction every number in the OpenABC-D paper's Table 6 was produced with,
# so it is what this job runs. Pinned explicitly rather than left to
# train_baseline.py's default so the log records which variant produced the
# result.
#
# false restores this project's native fanin -> node direction. It is not a
# neutral flip: model.py:78 computes deg on edge_index[0], so upstream's
# direction makes deg a fanin count (2, 3 or 4 after self-loops) and the GCN
# normalisation nearly degree-blind, while native makes it a fanout count
# spanning orders of magnitude. src/baselines/openabc_synthnet/DIAGNOSIS.md
# asks for both to be reported; run the pair with
#   sbatch --export=ALL,SYNTHNET_UPSTREAM_EDGE_DIRECTION=false src/shell/train_baseline_synthnet.sh
# (bare VAR=value sbatch does not propagate on Snellius).
SYNTHNET_UPSTREAM_EDGE_DIRECTION="${SYNTHNET_UPSTREAM_EDGE_DIRECTION:-true}"

echo "Using NUM_WORKERS=$NUM_WORKERS for data loading."
echo "Using SPLIT_BY=$SPLIT_BY."
echo "Using SYNTHNET_UPSTREAM_EDGE_DIRECTION=$SYNTHNET_UPSTREAM_EDGE_DIRECTION."
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# =========================================================
# EXECUTE TRAINING (SynthNet baseline, GNN hyperparameters and optimizer
# default to models/qor/SynthNetV3/train.py's own published config -- see
# src/baselines/openabc_synthnet/regressor.py and src/train_baseline.py).
# =========================================================

echo "Starting SynthNet baseline training for $ALGORITHM on GPU 0..."

srun python -u -m train_baseline \
    --baseline           "synthnet" \
    --algorithm          "$ALGORITHM" \
    --csv_paths          "$CSV_PATH" \
    --checkpoint_dir     "$CHECKPOINT_DIR" \
    --log_dir            "$LOG_DIR" \
    --cache_dir          "$CACHE_DIR" \
    --tier0_cache_dir    "$TIER0_CACHE_DIR" \
    --tier1_cache_dir    "$TIER1_CACHE_DIR" \
    --hp_tuning_splits_path "$HP_TUNING_SPLITS" \
    --split_by           "$SPLIT_BY" \
    --use_graph_cache    "false" \
    --synthnet_upstream_edge_direction "$SYNTHNET_UPSTREAM_EDGE_DIRECTION" \
    --prefetch_factor    "$PREFETCH_FACTOR" \
    --num_workers        "$NUM_WORKERS" \
    --pin_memory         "$PIN_MEMORY" \
    --persistent_workers "$PERSISTENT_WORKERS" \
    --patience           4
