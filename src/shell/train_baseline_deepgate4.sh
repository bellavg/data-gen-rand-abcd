#!/bin/bash
#SBATCH --job-name=aig_train_baseline_deepgate4
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --constraint=scratch-node
#SBATCH --output=logs/train_baseline_deepgate4_%j.out

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
# CSV_PATH intentionally does NOT derive from BASE_DIR -- same reason as
# train_baseline_hoga.sh: dataset.py's _build_cache_signature() hashes this
# path's literal absolute string, so running from a different worktree would
# change the signature and miss the manifest train.sh already built.
DATA_BASE_DIR="${DATA_BASE_DIR:-$HOME/data-gen-rand-abcd}"
CSV_PATH="$DATA_BASE_DIR/data/designs/design_metadata/algo_${ALGORITHM}_ml.csv"

WORKSPACE="/scratch-shared/$USER/aig_baseline_run/deepgate4_${ALGORITHM}"
CHECKPOINT_DIR="$WORKSPACE/checkpoints"
LOG_DIR="$WORKSPACE/logs"
# Reuse the primary model's per-algorithm cache_dir and shared tier caches, so
# this job loads the already-built manifest instead of re-walking ~700k
# samples. See train_baseline_hoga.sh for the full explanation.
CACHE_DIR="/scratch-shared/$USER/aig_train_run/${ALGORITHM}/cache"
TIER0_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier0_cache"
TIER1_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier1_cache"

# Same HP-tuning exclusion file as train.sh and the other baselines, so all
# models are compared on identical held-out splits.
#
# This job trains on UNREDUCED graphs (train_baseline.py hardcodes
# sparsification=None). Compare against train_no_sparsification.sh, not the
# train.sh sparsification array.
HP_TUNING_SPLITS="/scratch-shared/$USER/big_optuna_run/shared_dataset_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_50000_splits.json"
[ ! -f "$HP_TUNING_SPLITS" ] && echo "WARNING: HP Tuning split file not found at $HP_TUNING_SPLITS"

# =========================================================
# DeepGate4-specific configuration
# =========================================================
# Virtual-edge radius k, paper Sec 4.1 ("we set k to 8"). This is the dominant
# cost in the whole baseline. MEASURED on synthetic AIGs matching this
# dataset's shape (40k nodes, depth 100), virtual edges per expanded node:
#     k=2 -> 4.2   k=3 -> 8.3   k=4 -> 15.0   k=6 -> 42.6   k=8 -> 111.8
# At k=8 an average graph carries ~7.36M virtual edges.
#
# If memory forces this down, k=6 is the right step: Appendix A.3 (Table 8)
# ablates it directly, reporting the BEST functional loss of any setting
# (L_func 0.4629 vs 0.4863 at k=8) at roughly half the training memory. It is a
# published configuration, not an improvisation. Report whichever k was used.
DEEPGATE4_NUM_HOPS="${DEEPGATE4_NUM_HOPS:-8}"

# false (default) matches BOTH the paper and the released code. This is easy to
# get backwards: get_fanin_fanout_cone looks symmetric but marks the fanin cone
# with 1 and the fanout cone with 2 (data_preparation.py:222 and :233), and the
# consumer selects `argwhere(ff_cone.T == 1)` (line 523), so fanout pairs are
# structurally excluded and the released edge set is one-way,
# ancestor -> descendant -- exactly Section 3.5's `E_bar = {(u,v) : u <=_k v}`.
# Setting this true departs from paper AND code and doubles the edge count.
DEEPGATE4_SYMMETRIC="${DEEPGATE4_SYMMETRIC:-false}"

# Upstream applies NO gradient clipping -- dg4_trainer.py never calls any
# clipping function. train_baseline.py's shared --gradient_clip_val default is
# 1.0 (this project's choice, inherited by all three baselines), so it is set
# explicitly here to match upstream instead. 0 disables it in Lightning.
# If training diverges, 1.0 is the fallback -- report it if you change it,
# since it is then a training-setup difference from the published method.
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-0}"

# Recompute sparse-transformer activations in backward instead of storing them.
# Verified numerically transparent -- forward values to 1e-6 and gradients to
# 1e-5 (src/unittests/baselines/test_deepgate4.py). Not a bitwise claim, and
# the test covers 4 layers on CPU in fp32, not the shipped 12 layers under
# bf16-mixed. The mechanism is sound at the shipped config regardless:
# use_reentrant=False replays dropout RNG on recompute, and the checkpointed
# block has LayerNorm only, no BatchNorm running stats to double-update.
# Do NOT turn this off: 12 GATConv
# layers over ~7.36M edges retain ~45 GB for a single average graph, over half
# an 80 GB H100. With it, peak is one layer at a time (~3.8 GB).
DEEPGATE4_GRADIENT_CHECKPOINTING="${DEEPGATE4_GRADIENT_CHECKPOINTING:-true}"

# Node budget per batch, counted in PRE-expansion nodes (NOT-node expansion
# then adds ~60% more). At the measured ~182 virtual edges per original node,
# 200k nodes is ~36M edges, ~18.6 GB for one checkpointed GAT layer, and about
# 5 average graphs.
#
# As with HOGA, a graph larger than the budget cannot be split (graph-level
# pooling needs all its nodes at once), so it forms a singleton batch and sets
# an irreducible peak that lowering this does NOT reduce: at
# config.MAX_NUM_GATES = 366,040 that is ~67M virtual edges, ~34 GB for one
# checkpointed layer. If those graphs OOM, lower DEEPGATE4_NUM_HOPS to 6 --
# do NOT reach for DEEPGATE4_SYMMETRIC, which is already at its
# paper-and-code-faithful setting.
#
# That singleton peak is why 200k is the right budget on an 80 GB H100 rather
# than the 100k a memory-only reading would pick. ~34 GB is already the
# binding peak whatever this is set to, so raising the budget to ~18.6 GB
# cannot become the constraint -- it buys ~2x fewer micro-batches for the same
# worst case. On a smaller card the budget is not the lever either, for the
# same reason: on the L40 (48 GB) the paper used for its own OpenABC-D
# experiments (Appendix A.4) the ~34 GB singleton eats most of the card
# whatever this is set to, so lower DEEPGATE4_NUM_HOPS to 6 there rather than
# lowering this.
DEEPGATE4_MAX_NODES_PER_BATCH="${DEEPGATE4_MAX_NODES_PER_BATCH:-200000}"

# The budget above yields ~5 graphs per micro-batch, against the primary
# model's ~75 (its 3M budget / ~40k avg nodes). One graph is one label is one
# loss term, so node count buys no gradient-variance reduction; without
# accumulation this baseline would train on ~15x fewer loss terms per update
# than the model it is compared against. 15 x 5 brings the effective batch
# back to ~75, matching the primary model, while leaving the published lr=1e-4
# untouched.
#
# Same two caveats as the HOGA script: Lightning divides each micro-batch loss
# by the CONSTANT accumulate_grad_batches, so micro-batches weigh equally
# however many graphs they hold, and the effective sample size over the window
# is the harmonic mean, below 15 x 5. This approximates the primary model's
# regime; it does not match it. TrainingStartupCallback prints the real
# avg_graphs_per_batch each epoch -- calibrate from an actual log.
#
# MUST be retuned together with DEEPGATE4_MAX_NODES_PER_BATCH -- their product
# is the effective batch, and that product is the whole point of the pairing.
# 200k x 15 is the same product as the 100k x 30 this pairing started at, so
# the H100 budget bump above changes throughput, not the optimization regime.
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-15}"

# Epoch subsampling -- mandatory here, not a nicety. At ~5 graphs per
# micro-batch a full epoch over ~707k train graphs is ~141k micro-batches, so a
# single epoch would far outrun the 72h walltime, let alone the --patience 4
# early-stop cadence. 7200 of those is ~5% of an epoch.
#
# THESE TWO NUMBERS ARE PLACEHOLDERS. They are sized off an ESTIMATED ~5 s
# per micro-batch at the 200k budget (12 GATConv layers over ~36M edges, plus
# the tokenizer's level loop, plus checkpoint recompute in backward). That
# estimate has NOT been measured on an H100 -- only the structural quantities
# behind it have (edge counts, expanded depth, adapter time). RECALIBRATE from
# the first "[train] Epoch summary" line (train_utils.py prints avg_step_s and
# epoch_s) before trusting any result.
#
# Halved alongside the 100k -> 200k budget bump, so an epoch still covers the
# same ~36k graphs and still performs 480 optimizer steps. Only the micro-batch
# count changed.
#
# Wall-clock is NOT expected to halve, and may not improve at all. The GATConv
# term is edge-linear and so is flat under this trade, while the tokenizer term
# is level-synchronous over the whole batch (dg2.py:114, :236 loop to
# max(forward_level)), so doubling graphs per batch doubles the per-level work
# and the batch's max depth can only grow -- the packer anchors the largest
# graph and backfills with the smallest. Treat ~5 s/micro-batch as a floor.
#
# The estimate is WEAKEST ON DEEP CIRCUITS, and per-step cost varies far more
# than the node budget suggests. DeepGate2's tokenizer is level-synchronous:
# get_slices and the update loop each iterate once per logic level, doing full
# edge-list and per-node scans every iteration, and that is inherently
# sequential -- no batching or GPU parallelism removes it. NOT-node expansion
# roughly doubles the level count (measured: a depth-500 AIG expands to 988
# levels). So a typical depth-100 graph costs ~200 iterations, while
# config.MAX_DEPTH = 24,972 means the worst case is ~50,000 -- a ~250x spread
# in the tokenizer term alone, on top of the memory spread the node budget
# already handles. Batches are packed by node count, which does NOT correlate
# with depth, so a deep-but-small circuit can dominate a step. If avg_step_s
# comes in far above 5 s, check the depth distribution before assuming the
# transformer is at fault.
#
# 7200 % 15 == 0, so no partial accumulation window is left at the epoch
# boundary (Lightning does step on a trailing partial window, giving one
# under-scaled update). Preserve that property if you override either value.
#
# The train sampler reshuffles per epoch, so each epoch draws a different
# subset; the val plan is shuffled once off a fixed seed in train_baseline.py
# (_node_budget_loader) so a capped val pass is representative AND identical
# epoch to epoch.
#
# NOTE this makes "epoch" mean something different here than in train.py, and
# makes val_loss mean something different too: it is computed on a fixed slice
# of the val split. That number drives ModelCheckpoint, PreciseEarlyStopping
# and ReduceLROnPlateau, so quote the FULL val/test split from a separate eval
# pass when reporting. Compare models on graphs seen, not on epoch index.
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-7200}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-720}"

# Set explicitly rather than left at config.LOG_EVERY_N_STEPS = 1000.
# Lightning counts log_every_n_steps in OPTIMIZER steps, and 7200
# micro-batches / 15 accumulation = 480 steps per epoch -- under the 1000
# default, per-step train_loss/train_rmse would reach W&B only about once
# every 2.1 epochs. Epoch-level val_loss is logged unconditionally, so
# checkpointing, early stopping and ReduceLROnPlateau were never affected.
#
# MUST be retuned whenever LIMIT_TRAIN_BATCHES or ACCUMULATE_GRAD_BATCHES is
# overridden, since steps/epoch is their quotient: keep it well under
# LIMIT_TRAIN_BATCHES / ACCUMULATE_GRAD_BATCHES. Env-overridable for exactly
# that reason.
#
# Not a DeepGate4-only problem, just worst here: at 12500/10 = 1250 steps
# HOGA clears the 1000 default by so little that it logs one step-point per
# epoch. SynthNet is fine (it caps neither value, so it runs tens of thousands
# of steps per epoch). Fixing HOGA is out of scope for this script.
LOG_STEPS="${LOG_STEPS:-100}"

# DELIBERATELY LOWER than the other baselines' 16/4. The dataloader payload
# here is the virtual edge list: ~7.36M edges x 2 rows x 8 B = ~118 MB PER
# GRAPH, shipped from worker to main process through shared memory. A batch
# holds ~5 graphs, so ~590 MB per batch; at 16 workers x prefetch 4 that
# queue would hold ~38 GB and exhaust /dev/shm, while 6 x 1 holds ~3.5 GB. The
# adapter itself costs only ~0.2 s per graph (measured), well under the
# per-step GPU time, so few workers still keep the GPU fed. Raise only
# alongside a lower DEEPGATE4_NUM_HOPS.
#
# PREFETCH_FACTOR dropped 2 -> 1 when the node budget doubled, so the in-flight
# shm total stays at ~3.5 GB. Six workers each holding one ready batch is still
# six batches of queue depth against a ~5 s step, and each worker only has to
# produce 5 x 0.2 s = ~1 s of adapter work per batch.
NUM_WORKERS="${NUM_WORKERS:-6}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
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

echo "Using NUM_WORKERS=$NUM_WORKERS for data loading."
echo "Using SPLIT_BY=$SPLIT_BY."
echo "Using DEEPGATE4_NUM_HOPS=$DEEPGATE4_NUM_HOPS, SYMMETRIC=$DEEPGATE4_SYMMETRIC."
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# =========================================================
# EXECUTE TRAINING
#
# DeepGate4 baseline. hidden(128), sparse-transformer depth(12), k(8) and
# lr(1e-4) are published values from Zheng et al. ICLR'25 Sec 4.1; --deepgate4_heads
# and --deepgate4_tf_dropout are upstream's constructor defaults with no
# published source.
#
# This port does NOT implement the cone partitioning / History updating
# strategy of paper Sec 3.2-3.4. The authors do define a "w/o Partition"
# setting, but report it as OOM on both their benchmarks (Table 4), concluding
# partitioning is necessary for memory -- so that ablation names this setting
# rather than endorsing it. What makes it run here is different machinery:
# gradient checkpointing plus the node budget above. Report accordingly: this
# measures DeepGate4's representation, not its scalability contribution, and
# it does so via memory handling the authors did not use.
# See src/baselines/deepgate4/regressor.py.
# =========================================================

echo "Starting DeepGate4 baseline training for $ALGORITHM on GPU 0..."

srun python -u -m train_baseline \
    --baseline           "deepgate4" \
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
    --deepgate4_num_hops "$DEEPGATE4_NUM_HOPS" \
    --deepgate4_symmetric_virtual_edges "$DEEPGATE4_SYMMETRIC" \
    --deepgate4_gradient_checkpointing  "$DEEPGATE4_GRADIENT_CHECKPOINTING" \
    --deepgate4_max_nodes_per_batch     "$DEEPGATE4_MAX_NODES_PER_BATCH" \
    --gradient_clip_val  "$GRADIENT_CLIP_VAL" \
    --accumulate_grad_batches  "$ACCUMULATE_GRAD_BATCHES" \
    --limit_train_batches "$LIMIT_TRAIN_BATCHES" \
    --limit_val_batches   "$LIMIT_VAL_BATCHES" \
    --prefetch_factor    "$PREFETCH_FACTOR" \
    --num_workers        "$NUM_WORKERS" \
    --pin_memory         "$PIN_MEMORY" \
    --persistent_workers "$PERSISTENT_WORKERS" \
    --log_steps          "$LOG_STEPS" \
    --patience           4
