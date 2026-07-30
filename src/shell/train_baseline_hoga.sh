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
# CSV_PATH intentionally does NOT derive from BASE_DIR: dataset.py's
# _build_cache_signature() hashes this path's literal absolute string, so
# running from a different worktree than train.sh (e.g. BASE_DIR overridden so
# PYTHONPATH resolves the baselines/ package) would change the signature and
# miss train.sh's already-built manifest even though the CSV content is
# identical. Pin it to the checkout train.sh runs from.
DATA_BASE_DIR="${DATA_BASE_DIR:-$HOME/data-gen-rand-abcd}"
CSV_PATH="$DATA_BASE_DIR/data/designs/design_metadata/algo_${ALGORITHM}_ml.csv"

WORKSPACE="/scratch-shared/$USER/aig_baseline_run/hoga_${ALGORITHM}"
CHECKPOINT_DIR="$WORKSPACE/checkpoints"
LOG_DIR="$WORKSPACE/logs"
# Reuse the primary model's own per-algorithm cache_dir (not a separate
# aig_baseline_run/.../cache). The graph-cache manifest lives at
# cache_dir/metadata/dataset_<sig>_manifest.json and is keyed by cache_dir
# path, not by the cache signature -- so pointing here at train.sh's own
# CACHE_DIR lets this job load the manifest the primary run already built
# (sub-second) instead of re-walking all ~700k samples to rebuild it. The
# per-graph .pt files themselves live in the shared tier0/tier1 dirs below
# regardless of this setting.
CACHE_DIR="/scratch-shared/$USER/aig_train_run/${ALGORITHM}/cache"
TIER0_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier0_cache"
TIER1_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier1_cache"

# Same HP-tuning exclusion file as the primary model's train.sh, so the
# baseline's train/val/test rows never overlap with graphs used for HP
# tuning, and both models are compared on identical held-out splits.
HP_TUNING_SPLITS="/scratch-shared/$USER/big_optuna_run/shared_dataset_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_50000_splits.json"
[ ! -f "$HP_TUNING_SPLITS" ] && echo "WARNING: HP Tuning split file not found at $HP_TUNING_SPLITS"

# Hop features are computed in the dataloader workers, NOT cached to disk.
# An on-disk cache is not viable at this dataset's scale: [num_nodes, 11, 4]
# float32 = 176 B/node, and Orchestrate train+val total ~32.4e9 nodes over
# ~788k graphs -> ~5.7 TB in ~788k files, against an 8 TiB / 3M-inode scratch
# quota already ~90% full. Recomputing is a handful of O(nnz) sparse ops per
# graph and parallelises over NUM_WORKERS, overlapped with GPU compute. Pass
# --hoga_hop_cache_dir only for subset runs small enough to fit. See
# src/baselines/hoga/hop_features.py.
HOGA_NUM_HOPS="${HOGA_NUM_HOPS:-5}"
HOGA_DIRECTED="${HOGA_DIRECTED:-true}"
# Node-budget batching replaces a fixed graph count. Graphs here average ~40k
# nodes (max 366k) and HOGA's trunk holds [N, 11, 256] activations -- ~5.6
# KB/node under the bf16-mixed AMP this script gets on H100 -- so a fixed
# batch_size=32 (~1.29M nodes) is ~7.2 GB for a single activation tensor and
# OOMs once the attention layer and backward are counted. 150k nodes is
# ~845 MB/activation for a typical batch.
#
# Caveat: a graph bigger than the budget cannot be split (graph-level pooling
# needs all its nodes in one pass), so it forms a singleton batch, and the
# largest graph sets an irreducible peak of ~2.1 GB/activation at 366k nodes.
# Lowering this budget does NOT reduce that peak. If it still OOMs, cap graph
# SIZE (a dataset choice) rather than touching hidden_dim/num_layers/heads --
# those are published DAC'24 QoR values and changing them would stop this
# being a faithful HOGA baseline. Note a size cap must be applied to the
# primary model too, or the comparison is no longer like-for-like.
# See src/train_baseline.py's module docstring for the full rationale.
HOGA_MAX_NODES_PER_BATCH="${HOGA_MAX_NODES_PER_BATCH:-150000}"

# The node budget above yields only ~4 graphs per micro-batch (150k / ~40k avg
# nodes) -- one graph is one label, so node count buys no variance reduction.
# The primary model averages ~75 graphs per step (its 3M budget / ~40k), so
# without accumulation HOGA trains on far fewer loss terms per update than the
# model it is compared against (gradient-noise std scales ~1/sqrt(N), so
# roughly 4x noisier, not 20x). Accumulating 20 steps closes most of that gap
# while leaving the published lr=0.0001 untouched.
#
# Two honest caveats on "parity". Lightning divides each micro-batch loss by
# the CONSTANT accumulate_grad_batches, so micro-batches are weighted equally
# regardless of how many graphs they hold; since node-budget packing puts
# fewer graphs in batches holding big graphs, large graphs end up carrying
# more per-graph gradient weight. And the effective sample size over a 20-step
# window is the harmonic mean, ~60-65 graphs, not 20 x 4 = 80. So this
# approximates the primary model's regime rather than matching it exactly.
# TrainingStartupCallback prints the real avg_graphs_per_batch each epoch --
# calibrate this value from an actual log rather than trusting the estimate.
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-20}"

NUM_WORKERS="${NUM_WORKERS:-12}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PIN_MEMORY="${PIN_MEMORY:-true}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
# No SPLIT_BY: this branch's dataset hardcodes design-level splitting (see
# data/dataset.py) -- there's no --split_by flag to pass on train_baseline.py.

echo "Using NUM_WORKERS=$NUM_WORKERS for data loading."
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
    --hoga_num_hops      "$HOGA_NUM_HOPS" \
    --hoga_max_nodes_per_batch "$HOGA_MAX_NODES_PER_BATCH" \
    --accumulate_grad_batches  "$ACCUMULATE_GRAD_BATCHES" \
    --hoga_directed      "$HOGA_DIRECTED" \
    --prefetch_factor    "$PREFETCH_FACTOR" \
    --num_workers        "$NUM_WORKERS" \
    --pin_memory         "$PIN_MEMORY" \
    --persistent_workers "$PERSISTENT_WORKERS" \
    --patience           4
