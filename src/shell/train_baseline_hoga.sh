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
#
# Note this job trains on UNREDUCED graphs (train_baseline.py hardcodes
# sparsification=None -- sparsification is this project's contribution, not
# part of either baseline paper). Compare its results against
# train_no_sparsification.sh, not against the train.sh sparsification array.
HP_TUNING_SPLITS="/scratch-shared/$USER/big_optuna_run/shared_dataset_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_50000_splits.json"
[ ! -f "$HP_TUNING_SPLITS" ] && echo "WARNING: HP Tuning split file not found at $HP_TUNING_SPLITS"

# Hop features are computed in the dataloader workers, NOT cached to disk.
# An on-disk cache is not viable at this dataset's scale: [num_nodes, 6, 4]
# float32 = 96 B/node, and Orchestrate train+val total ~32.4e9 nodes over
# ~788k graphs -> ~3.1 TB in ~788k files, against an 8 TiB / 3M-inode scratch
# quota already ~90% full. Recomputing is a handful of O(nnz) sparse ops per
# graph and parallelises over NUM_WORKERS, overlapped with GPU compute. Pass
# --hoga_hop_cache_dir only for subset runs small enough to fit. See
# src/baselines/hoga/hop_features.py.
HOGA_NUM_HOPS="${HOGA_NUM_HOPS:-5}"
# Undirected, matching the paper and upstream rather than deviating from both.
# Paper Section 3.1 defines a single symmetric-normalized adjacency
# (A_hat = D^-1/2 A D^-1/2, X^(k) = A_hat X^(k-1)) and stacks K+1 = 6 slots;
# upstream's --directed is action='store_true' (default False) and its
# published run.sh never passes it, so the released experiment is undirected
# too. The previous `true` here was this project's own extension, motivated by
# AIG causal cones -- defensible on its own terms, but it was the single
# largest cost in the run: 11 slots instead of 6 means ~1.85x the work
# throughout. (Cost is dominated by the four Linear(256,256) projections,
# which are LINEAR in slot count; only the score/PV matmuls scale with
# slots^2, and at hidden=256/heads=32 those are ~2% of the attention module.)
# Reverting it makes the baseline both faster and more faithful. Set to `true` to restore
# the fanin/fanout variant (hop_features.py's directed branch also fixes an
# upstream copy-paste bug in the reverse direction; that fix still stands).
HOGA_DIRECTED="${HOGA_DIRECTED:-false}"
# Node-budget batching replaces a fixed graph count. Graphs here average ~40k
# nodes (max 366k) and HOGA's trunk holds [N, 6, 256] activations -- ~3.0
# KB/node under the bf16-mixed AMP this script gets on H100 -- so a fixed
# batch_size=32 (~1.29M nodes) is ~3.9 GB for a single activation tensor and
# OOMs once the attention layer and backward are counted. 150k nodes is
# ~460 MB/activation for a typical batch.
#
# NOTE 150000 was calibrated at 11 slots (~5.5 KB/node) and has NOT been
# retuned since HOGA_DIRECTED flipped to false and the attention stopped
# materializing its score tensor. Both cut peak memory substantially -- the
# run this was set for sat at ~91 GB allocated on the H100 -- so the budget is now
# expected to be roughly 1.8x conservative. Read the real figure off
# `nvidia-smi` (or wandb's GPU Memory Allocated) on the first epoch and raise
# it; a bigger budget means fewer, better-utilised steps, which is the same
# problem LIMIT_TRAIN_BATCHES below works around from the other end.
#
# Caveat: a graph bigger than the budget cannot be split (graph-level pooling
# needs all its nodes in one pass), so it forms a singleton batch, and the
# largest graph sets an irreducible peak of ~1.1 GB/activation at 366k nodes.
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

# Epoch subsampling. A full epoch under the node budget above is 149,485
# micro-batches plus 32,211 val batches. MEASURED AT ~12h TOTAL PER EPOCH, but
# on the PRE-CHANGE configuration (11 hop slots, and an attention that
# materialized its [N, 11, 11, heads] score tensor); GPU utilisation was ~100%
# with tensor-core activity at ~2%, i.e. latency/bandwidth-bound, not underfed.
# The same commit that added these caps also flipped HOGA_DIRECTED to false
# and switched to a fused attention kernel, which together cut a CPU-side
# fwd+bwd microbenchmark of the trunk by ~16x. The GPU figure will be smaller,
# but the 12h number is stale in the conservative direction and 25000 below is
# a placeholder sized off it.
#
# RECALIBRATE from the first "[train] Epoch summary" line (train_utils.py
# prints avg_step_s and epoch_s) rather than trusting these values -- if the
# epoch now runs in 2-3h, raise them and stop discarding data for no
# scheduling benefit.
#
# The point of capping at all is cadence: at ~12h/epoch the 72h walltime
# allows ~6 epochs; --patience 4 needs 5 non-improving epochs to fire, so it
# would at best trigger right as the walltime expires. The
# train sampler reshuffles per epoch, so each epoch draws a different subset;
# the val plan is shuffled once off a fixed seed in train_baseline.py (see
# _hoga_loader) so a capped val pass is representative AND identical epoch to
# epoch. With the defaults below 25000 % 20 == 0, so no partial accumulation
# window is left at the epoch boundary (Lightning does step on a trailing
# partial window, giving one under-scaled update); re-check that if you
# override either value.
#
# NOTE this makes "epoch" mean something different here than in train.py,
# and it also makes val_loss mean something different: the baseline's is
# computed on a fixed ~8% of the val split while train.py's uses 100%. That
# number drives ModelCheckpoint, PreciseEarlyStopping and ReduceLROnPlateau,
# so quote the FULL val/test split from a separate eval pass when reporting
# results -- do not compare these two val_loss curves directly. Compare the
# models on graphs seen, not on epoch index.
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-25000}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-2500}"

NUM_WORKERS="${NUM_WORKERS:-16}"  # of the 18 cores auto-assigned per GPU on gpu_h100; 2 left for the main process + pin_memory thread
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
    --limit_train_batches "$LIMIT_TRAIN_BATCHES" \
    --limit_val_batches   "$LIMIT_VAL_BATCHES" \
    --hoga_directed      "$HOGA_DIRECTED" \
    --prefetch_factor    "$PREFETCH_FACTOR" \
    --num_workers        "$NUM_WORKERS" \
    --pin_memory         "$PIN_MEMORY" \
    --persistent_workers "$PERSISTENT_WORKERS" \
    --patience           4
