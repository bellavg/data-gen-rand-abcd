#!/bin/bash
#SBATCH --job-name=aig_train_baseline_gamora
#SBATCH --time=72:00:00
#SBATCH --nodes=1
# gpu_h100, not gpu_a100. Tried gpu_a100 on 2026-08-06 (twice: once with
# torch.compile on, once off) and both runs showed the same broken pattern --
# a step that reports finishing in ~50ms, immediately followed by ~25-30s of
# data_wait_s on the NEXT batch. A ~3M-node batch cannot really finish
# forward+backward in 50ms on any current GPU, so that isn't real GPU speed --
# the time is landing on the wrong side of the batch boundary. Confirmed NOT
# caused by torch.compile (identical pattern with it off), so most likely
# NUM_WORKERS=16 below (tuned for gpu_h100's documented 18 cores/GPU) is
# oversubscribed on gpu_a100's actual core-per-GPU allocation, which this
# project has never verified -- or some other A100-node-specific difference.
# Real wall-clock for 3 batches was ~10s on gpu_h100 (original run) vs. ~108s
# on gpu_a100 even with compile off -- roughly 10x worse, well beyond the
# ~2-3x a slower card alone would explain. Verify the actual cause (check
# `sinfo -p gpu_a100 -o "%c %G"` for cores/GPU, retune NUM_WORKERS to match)
# before trying gpu_a100 again.
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
# that comment -- at the current 15M-node budget, 8/80 is ~145 GB fp32, likely
# over 80 GB even under bf16-mixed, so lower the budget for 8/80 rather than
# leaving it) and re-derive ACCUMULATE_GRAD_BATCHES if you also change the
# budget back toward primary-model parity -- neither follows from this edit
# automatically.
GAMORA_NUM_LAYERS="${GAMORA_NUM_LAYERS:-4}"
GAMORA_HIDDEN_DIM="${GAMORA_HIDDEN_DIM:-32}"

# Node-budget batching replaces a fixed graph count, as for the other two
# large-graph baselines -- but originally for the opposite reason. HOGA and
# DeepGate4 need a budget to avoid OOM; this budget was ORIGINALLY set to
# config.MAX_TOTAL_NODES_PER_BATCH (3,000,000), the primary model's own
# budget, purely to keep the effective batch comparable (~75 graphs/step, the
# same effective batch train_no_sparsification.sh runs at).
#
# RAISED to 15,000,000 on 2026-08-06 as a deliberate, documented DEVIATION
# from that parity, for wall-clock speed rather than memory: a real H100 run
# at the 3M budget used only 4.6 GB of the card's 80 GB (5.8%), and GPU
# utilization (wandb system metrics, DCGM sampling -- not the app-level
# step_s/data_wait_s split, which is not trustworthy without a
# torch.cuda.synchronize() the callback doesn't have) sat at a moderate 60%
# mean with near-idle CPU (4%) and near-idle GPU memory traffic (0.9%) --
# a pattern more consistent with many small, sequential kernel launches
# (4 SAGEConv layers' gather/scatter + relu/dropout/batchnorm, an
# 8,193-parameter model) paying fixed per-launch overhead than with either a
# data-loading or a compute-bound workload. Fewer, larger batches per epoch
# amortizes that fixed overhead over more nodes, at the cost of no longer
# matching the primary model's per-step gradient noise.
#
# THIS BREAKS THE PARITY ARGUMENT ABOVE. ~75 graphs/step no longer holds --
# see the recalculated estimate below -- and test_train_baseline_cli.py's
# test_gamora_node_budget_reflects_the_documented_speed_deviation pins THIS
# value rather than asserting equality with config.MAX_TOTAL_NODES_PER_BATCH,
# specifically because the two are now expected to differ. Compare this run
# against the primary model on GRAPHS SEEN (TrainingStartupCallback's
# graphs_seen), not on step or epoch index -- see the LIMIT_TRAIN_BATCHES
# comment below, which already establishes that pattern for a different
# reason (HOGA's own capped run).
#
# That means NEITHER of the accumulation caveats in the HOGA and DeepGate4
# scripts applies here: there is no window over which Lightning's constant
# divisor mis-weights micro-batches, because there is no window.
#
# MEASURED on the largest single graph (config.MAX_NUM_GATES = 366,040 nodes,
# 677,172 edges, fp32) at published hyperparameters: 743 MB retained for
# backward, i.e. 2.08 KB/node, for a model of 8,193 parameters. That is the
# sum of the tensors autograd packs for backward, deduplicated by storage
# pointer -- NOT an RSS delta, which reads only 561 MB on the same forward
# because the CPU allocator reuses resident freed pages. It is still a lower
# bound on peak: transient per-op buffers freed inside the forward are not
# counted.
#
# Extrapolating: 15M nodes is ~31.2 GB fp32, roughly half that (~15.6 GB)
# under the bf16-mixed AMP this script gets on H100 -- comfortably under 80 GB
# even allowing for the real 3M-node run's higher-than-extrapolated actual
# usage (4.6 GB observed vs. ~3.1 GB extrapolated bf16-mixed, i.e. real
# overhead beyond this backward-only estimate), which scaled the same way
# would be ~23 GB at 15M. VERIFY the steady-state figure on the first batch
# anyway (nvidia-smi, or wandb GPU Memory Allocated) before trusting either
# estimate -- if it's near the card, back off; if it's still far below,
# raising further is not unreasonable but was not tried here.
#
# The 8-layer/80-channel configuration is a different story: the same
# measurement gives 10.10 KB/node, ~30 GB fp32 at the ORIGINAL 3M budget --
# scale accordingly (or lower the budget back down) if editing
# GAMORA_NUM_LAYERS/GAMORA_HIDDEN_DIM to 8/80 (see that block above for how --
# an env override does not work here).
#
# DO NOT lower this below config.MAX_NUM_GATES (366,040) without reading
# src/baselines/gamora/regressor.py first: a graph bigger than the budget forms
# a singleton batch, and in train mode a singleton batch's graph embedding is
# exactly bn0's bias -- upstream's BatchNorm normalises over the same node set
# the pooling then averages, so the encoder contributes nothing to that step.
GAMORA_MAX_NODES_PER_BATCH="${GAMORA_MAX_NODES_PER_BATCH:-15000000}"

# Left at 1 deliberately. Even at the raised 15M-node budget this is a single
# batch per step with no window, so Lightning's constant accumulation divisor
# never mis-weights a micro-batch -- the accumulation caveats in the HOGA and
# DeepGate4 scripts still do not apply. If GAMORA_MAX_NODES_PER_BATCH is ever
# lowered back toward 3,000,000 to restore primary-model parity, this can stay
# at 1 for that too; accumulation was never what made the two comparable, the
# node budget alone was.
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"

# NOT capped, unlike train_baseline_hoga.sh (12500 / 1250). At the 15M-node
# budget an epoch is roughly 700k train graphs / ~375 per batch (15M / ~40k
# avg graph size) = ~1.9k steps -- down from the ~9.3k estimated at the
# original 3M budget -- and a Gamora step is already a small fraction of a
# HOGA step, so a full epoch is expected to fit the checkpoint/early-stop
# cadence without subsampling. Leaving these at 1.0 keeps val_loss computed on
# the FULL val split, which makes this baseline's val_loss directly comparable
# to train.py's -- something HOGA's capped run cannot offer.
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

# DEFAULT ON, 2026-08-06 -- a deliberate re-test, not a return to the
# original default-on state. History:
#   1. First tried on gpu_a100: 3 batches that take ~10s uncompiled took
#      ~202s compiled (~20x), so it was turned off.
#   2. But a SEPARATE gpu_a100 run with compile OFF also showed ~108s for the
#      same 3 batches (~10x vs. gpu_h100's ~10-15s), with the identical
#      fast-step/slow-next-wait log pattern the compiled run showed --
#      proving that pattern (and likely most of the 20x) is gpu_a100-specific
#      and NOT caused by torch.compile. The one real compiled run this
#      project has ever done was on the one node type independently confirmed
#      to be pathologically slow for this workload for an unrelated reason.
#      Compile has never actually been tested clean, on gpu_h100.
# THIS RUN is that test. Also relevant: the graph break at global_mean_pool
# this comment used to warn about (PyG's scatter() calling int(index.max())
# when dim_size isn't given) is fixed in the same change that raised
# GAMORA_MAX_NODES_PER_BATCH above -- regressor.py now passes
# size=batch.num_graphs, so Dynamo may fuse more of the forward pass than it
# did during the original A100 test, not just the SAGEConv stack.
#
# torch.compile is invoked with dynamic=True regardless (real batches vary in
# node count, edge count AND graph count every step, so this is required, not
# optional -- see baselines/common/lightning_wrapper.py's module docstring for
# what dynamic=True actually buys on this torch version). Checkpoints are
# unaffected by this setting either way -- the wrapper strips torch.compile's
# key prefix on save.
#
# WATCH: batch 0 will be slow (one-time JIT compile, expect tens of seconds).
# Judge steady state from batch 1 onward's step_s against the gpu_h100 eager
# baseline (1.463s, 1.717s on the same two batch shapes) -- if it's not
# meaningfully faster than that by a few batches in, this isn't paying off and
# should go back to false, same revert procedure as before (env override does
# not reach an sbatch job on this cluster -- edit the literal below).
TORCH_COMPILE="${TORCH_COMPILE:-true}"

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
echo "Using TORCH_COMPILE=$TORCH_COMPILE."
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
    --torch_compile      "$TORCH_COMPILE" \
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
