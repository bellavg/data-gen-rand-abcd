#!/bin/bash
#SBATCH --job-name=aig_train_baseline_gamora
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --constraint=scratch-node
#SBATCH --output=logs/train_baseline_gamora_%j.out

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

WORKSPACE="/scratch-shared/$USER/aig_baseline_run/gamora_${ALGORITHM}"
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
# part of any baseline paper). Compare its results against
# train_no_sparsification.sh, not against the train.sh sparsification array.
HP_TUNING_SPLITS="/scratch-shared/$USER/big_optuna_run/shared_dataset_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_50000_splits.json"
[ ! -f "$HP_TUNING_SPLITS" ] && echo "WARNING: HP Tuning split file not found at $HP_TUNING_SPLITS"

# Unlike the HOGA and DeepGate4 scripts, this baseline needs no feature
# precomputation and no on-disk feature cache. Gamora's node features are the
# 4 columns its own ABC exporter writes (acecXor.c:392-415) and they are
# derived inside the model's forward from .x / .edge_index / .edge_attr, which
# every cached graph already carries. Nothing extra is written to scratch.

# Published architecture. 4 layers / 32 channels is BOTH the paper's shallow
# configuration (Sec IV.A: "a shallow 4-layer model with the hidden channel of
# 32") and upstream's argparse default (gnn_multitask.py:528-529), so it is the
# hardcoded default below. The paper's OTHER published configuration is 8
# layers / 80 channels, used there "for Booth multipliers and after complex
# technology mapping" -- i.e. for the harder, more irregular netlists. This
# dataset's graphs are neither multipliers nor small, so 8/80 is a defensible
# second run; report which was used either way. Do not invent a third size.
#
# TO RUN 8/80 VIA SBATCH: DO NOT rely on
# `sbatch --export=ALL,GAMORA_NUM_LAYERS=8,GAMORA_HIDDEN_DIM=80 <script>` or a
# bare `GAMORA_NUM_LAYERS=8 sbatch ...` prefix. Neither reaches a job SUBMITTED
# via sbatch on this cluster -- confirmed 2026-08-05 (memory: snellius-sbatch-
# env-propagation) -- and both exit 0 while silently training 4/32 anyway,
# with nothing in the log to say so except this script's own
# "Using GAMORA_NUM_LAYERS=..." echo further down: CHECK IT before trusting
# any run submitted this way. The `${VAR:-default}` form below still works for
# a direct (non-sbatch) invocation on an already-allocated interactive node,
# where normal env-var inheritance applies -- it is only the sbatch submission
# path that silently drops the override. For an sbatch run, the only
# confirmed-working mechanism is editing the two literals below in place,
# submitting, and reverting before the next 4/32 run -- the same pattern
# test.sh's SKIP_FULL_GRAPH block uses. If you do this, ALSO edit
# GAMORA_MAX_NODES_PER_BATCH below (10.10 KB/node at 8/80 vs 2.08 at 4/32, see
# that comment) and re-derive ACCUMULATE_GRAD_BATCHES so the effective batch
# still targets ~75 graphs/step -- neither follows from this edit automatically.
GAMORA_NUM_LAYERS="${GAMORA_NUM_LAYERS:-4}"
GAMORA_HIDDEN_DIM="${GAMORA_HIDDEN_DIM:-32}"

# Node-budget batching replaces a fixed graph count, as for the other two
# large-graph baselines -- but for the opposite reason. HOGA and DeepGate4 need
# a budget to avoid OOM; Gamora needs one only to keep the effective batch
# comparable. So this is set to config.MAX_TOTAL_NODES_PER_BATCH, the primary
# model's own budget (3,000,000), giving ~75 graphs per step -- the same
# effective batch train_no_sparsification.sh runs at, with
# ACCUMULATE_GRAD_BATCHES left at 1.
#
# That means NEITHER of the accumulation caveats in the HOGA and DeepGate4
# scripts applies here: there is no window over which Lightning's constant
# divisor mis-weights micro-batches, because there is no window.
#
# MEASURED on the largest single graph (config.MAX_NUM_GATES = 366,040 nodes,
# 677,172 edges, fp32) at these defaults: 743 MB retained for backward, i.e.
# 2.08 KB/node, for a model of 8,193 parameters. That is the sum of the tensors
# autograd packs for backward, deduplicated by storage pointer -- NOT an RSS
# delta, which reads only 561 MB on the same forward because the CPU allocator
# reuses resident freed pages. It is still a lower bound on peak: transient
# per-op buffers freed inside the forward are not counted.
#
# So 3M nodes is ~6.2 GB fp32, roughly half that under the bf16-mixed AMP this
# script gets on H100, and the irreducible singleton-batch peak that constrains
# the other two baselines is not a constraint here. VERIFY the steady-state
# figure on the first epoch anyway (nvidia-smi, or wandb GPU Memory Allocated)
# -- if it lands far below the card there is nothing to gain by raising this,
# since the point of the value is parity with the primary model, not
# utilisation.
#
# The 8-layer/80-channel configuration is a different story: the same
# measurement gives 10.10 KB/node, ~30 GB fp32 at this budget. If you edit
# GAMORA_NUM_LAYERS/GAMORA_HIDDEN_DIM to 8/80 (see that block above for how --
# an env override does not work here), lower this budget too.
#
# DO NOT lower this below config.MAX_NUM_GATES (366,040) without reading
# src/baselines/gamora/regressor.py first: a graph bigger than the budget forms
# a singleton batch, and in train mode a singleton batch's graph embedding is
# exactly bn0's bias -- upstream's BatchNorm normalises over the same node set
# the pooling then averages, so the encoder contributes nothing to that step.
#
# The literal below MUST equal config.MAX_TOTAL_NODES_PER_BATCH, which is where
# train_baseline.py's own default comes from and what the parity argument above
# rests on. test_train_baseline_cli.py asserts the two agree, so retuning the
# config constant without touching this line fails the suite rather than
# silently breaking the comparison.
GAMORA_MAX_NODES_PER_BATCH="${GAMORA_MAX_NODES_PER_BATCH:-3000000}"

# Left at 1 deliberately. The budget above already delivers the primary model's
# ~75 graphs per update, so accumulating would OVERSHOOT it rather than close a
# gap. If GAMORA_MAX_NODES_PER_BATCH is ever lowered, raise this to keep the
# product near 3,000,000 nodes' worth of graphs -- the two must move together.
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"

# NOT capped, unlike train_baseline_hoga.sh (12500 / 1250). At a 3M-node budget
# an epoch is roughly 700k train graphs / ~75 per batch = ~9.3k steps, and a
# Gamora step is a small fraction of a HOGA step, so a full epoch is expected
# to fit the checkpoint/early-stop cadence without subsampling. Leaving these
# at 1.0 keeps val_loss computed on the FULL val split, which makes this
# baseline's val_loss directly comparable to train.py's -- something HOGA's
# capped run cannot offer.
#
# This is an ESTIMATE, not a measurement: no Gamora epoch has been timed on the
# cluster. RECALIBRATE from the first "[train] Epoch summary" line
# (train_utils.py prints avg_step_s and epoch_s). If an epoch turns out to
# exceed ~6h, set LIMIT_TRAIN_BATCHES/LIMIT_VAL_BATCHES here the way the HOGA
# script does and say so when reporting, because capping changes what val_loss
# means.
#
# Either way, compare models on GRAPHS SEEN, not on epoch index:
# TrainingStartupCallback logs `graphs_seen` (cumulative, wandb + the epoch
# summary line) precisely because --patience 4 buys a different amount of
# training in each of these four jobs.
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"

# Gamora publishes NO regression loss -- its task is per-node classification
# trained with F.nll_loss (gnn_multitask.py:183) -- so there is nothing to be
# faithful to and this defaults to the PRIMARY model's loss, SmoothL1 at
# beta=0.01 (train.py:151). Scoring the baseline under a different loss than
# the model it is compared against would confound architecture with loss
# choice, and the two are not interchangeable here: an out-of-repo analysis of
# the label distribution puts ~49% of it at exactly zero, which MSE and
# SmoothL1(beta=0.01) weight very differently. Set "mse" for a like-for-like
# check against the other three baselines, which do keep MSE.
LOSS="${LOSS:-smooth_l1}"

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
# three identical "design" jobs into one checkpoint dir.
#
# To run another strategy VIA SBATCH: neither
# `sbatch --export=ALL,SPLIT_BY=recipe <script>` nor a bare
# `SPLIT_BY=recipe sbatch <script>` prefix reaches a job submitted this way --
# confirmed 2026-08-05 (memory: snellius-sbatch-env-propagation) -- and both
# exit 0 while silently training "design" anyway, with nothing in the log to
# say so except the "Using SPLIT_BY=..." echo below. For an sbatch run, edit
# the literal here, submit separately, and revert it afterward. The
# `${VAR:-default}` form still works for a direct (non-sbatch) invocation on an
# already-allocated interactive node.
SPLIT_BY="${SPLIT_BY:-design}"

echo "Using NUM_WORKERS=$NUM_WORKERS for data loading."
echo "Using SPLIT_BY=$SPLIT_BY."
echo "Using GAMORA_NUM_LAYERS=$GAMORA_NUM_LAYERS, GAMORA_HIDDEN_DIM=$GAMORA_HIDDEN_DIM."
echo "Using LOSS=$LOSS."
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# =========================================================
# EXECUTE TRAINING
#
# Gamora baseline. num_layers/hidden_dim are published (Wu et al. DAC'23 Sec
# IV.A) and also upstream's argparse defaults; lr=0.008, weight_decay=5e-5 and
# max_epochs=100 come from the released code only (gnn_multitask.py:531,595,532)
# with no value in the paper.
#
# TWO THINGS TO CARRY INTO THE THESIS, both detailed in
# src/baselines/gamora/regressor.py:
#
#   1. This trains FULL-GRAPH. Gamora's architecture is sampling-free and
#      upstream wrote the full-graph forward themselves (forward_nosampler,
#      gnn_multitask.py:86-105), but their RELEASED TRAINER samples --
#      NeighborSampler(sizes=[8,5,5,5], batch_size=20) at :570-572 -- and there
#      is no train_nosampler. Training full-graph is a deviation from their
#      published procedure and must be reported as one.
#
#   2. This measures GAMORA'S ENCODER, not Gamora. Removing the three per-node
#      xor/maj/root heads removes the multi-task formulation that is the
#      paper's actual contribution; what trains here is a GraphSAGE trunk with
#      Gamora's node features and hyperparameters. Label the row so no reader
#      takes it as a claim about the DAC'23 results.
# =========================================================

echo "Starting Gamora baseline training for $ALGORITHM on GPU 0..."

srun python -u -m train_baseline \
    --baseline           "gamora" \
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
    --loss               "$LOSS" \
    --gamora_num_layers  "$GAMORA_NUM_LAYERS" \
    --gamora_hidden_dim  "$GAMORA_HIDDEN_DIM" \
    --gamora_max_nodes_per_batch "$GAMORA_MAX_NODES_PER_BATCH" \
    --accumulate_grad_batches  "$ACCUMULATE_GRAD_BATCHES" \
    --limit_train_batches "$LIMIT_TRAIN_BATCHES" \
    --limit_val_batches   "$LIMIT_VAL_BATCHES" \
    --prefetch_factor    "$PREFETCH_FACTOR" \
    --num_workers        "$NUM_WORKERS" \
    --pin_memory         "$PIN_MEMORY" \
    --persistent_workers "$PERSISTENT_WORKERS" \
    --patience           4
