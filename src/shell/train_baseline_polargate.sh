#!/bin/bash
#SBATCH --job-name=aig_train_baseline_polargate
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --constraint=scratch-node
#SBATCH --output=logs/train_baseline_polargate_%j.out

# =========================================================
# TODO before this job's results go in the thesis -- none block launching a
# run, all block trusting or reporting its output. Full detail in
# src/baselines/polargate/PROVENANCE.md.
#
# 1. VERIFY THE OLS CLAIM BEFORE SPENDING GPU HOURS. This baseline's loss
#    default and its size covariates (see #2) are both justified by "a
#    two-parameter OLS on log|V|, log|E| outranks the primary encoder on
#    Spearman" -- supplied by the thesis author, computed nowhere in this
#    repo. src/test.py already has spearmanr wired up. If the claim holds,
#    it says something about every baseline here, not just PolarGate.
#
# 2. RUN THE size_covariates PAIRED ABLATION, not just the default. PolarGate
#    is the ONLY model in this suite that sees |V|/|E| explicitly -- HOGA,
#    DeepGate4, SynthNet and the primary encoder all pool size-blind. A
#    PolarGate WIN over them is not attributable to its architecture unless
#    the size-blind arm is also reported:
#      sbatch --export=ALL,POLARGATE_SIZE_COVARIATES=false <this script>
#
# 3. NO BASELINE EVAL PATH EXISTS YET. src/test.py:555 hardcodes
#    AIGRegressionLightningModule.load_from_checkpoint, which cannot load a
#    BaselineRegressionLightningModule checkpoint -- true for all four
#    baselines, not new here. This job will finish and leave an unusable
#    .ckpt until that loader (or a baseline-specific one) exists.
#
# 4. RECALIBRATE FROM THE FIRST EPOCH, don't trust the defaults below as
#    final. POLARGATE_MAX_NODES_PER_BATCH / ACCUMULATE_GRAD_BATCHES are sized
#    to hit upstream's published 256-graph effective batch from TWO
#    disagreeing estimates (see the comment above their definitions) --  read
#    the real avg_graphs_per_batch off "[train] Epoch summary" and retune if
#    it's off target. The GPU memory figures were extrapolated from a CPU
#    measurement (never run on an H100) -- read the real peak off nvidia-smi
#    or wandb and record it in PROVENANCE.md.
# =========================================================

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

WORKSPACE="/scratch-shared/$USER/aig_baseline_run/polargate_${ALGORITHM}"
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

# ---------------------------------------------------------------------------
# WHICH PolarGate THIS IS
#
# The ICCAD'24 conference version, which is what BUPT-GAMMA/PolarGate releases
# and implements completely. The TODAES'25 journal extension adds a
# structure-aware preprocessing module (SAP) and an optimal global attention
# module (OGA); neither is in the released code (last push 2024-10-08) and
# neither is reimplemented here. Cite the ICCAD'24 bibtex from upstream's
# README. See src/baselines/polargate/PROVENANCE.md.
#
# SCALE CAVEAT, for the results caption and not only for this file: the paper's
# evaluated dataset "includes circuits with up to 3214 nodes". This project's
# average graph is ~40,000 nodes and the largest is 366,040, roughly 114x their
# largest evaluated circuit.
# ---------------------------------------------------------------------------

# Published: upstream train.sh passes --layer_num 9. out_dim is train.py's
# argparse default (256), which is the value train.sh actually runs under --
# NOT the 64 in model.py's class signature (argparse always overrides it) and
# NOT the 128 the TODAES paper normalises all baselines to. Report which was
# used. See src/baselines/polargate/regressor.py.
POLARGATE_LAYER_NUM="${POLARGATE_LAYER_NUM:-9}"
POLARGATE_OUT_DIM="${POLARGATE_OUT_DIM:-256}"

# Mean pooling plus explicit log-size covariates, NOT upstream's readout.
# Upstream predicts one value per NODE, so it never pools; a graph-level mean
# is invariant to |V| and |E|, and on this dataset a two-parameter OLS on log
# node and edge count alone already outranks the primary encoder on Spearman.
# A size-blind baseline would therefore be beaten by a trivial predictor before
# its architecture was tested at all. Set POLARGATE_POOLING=sum for the
# alternative encoding of the same information; leaving covariates on with sum
# pooling is redundant but harmless.
#
# CONFOUND WARNING, read before building any cross-model table: PolarGate is
# the ONLY model in this suite that sees graph size. HOGA, DeepGate4, SynthNet
# and the primary encoder all pool without any size covariate. So a PolarGate
# WIN cannot be credited to ambipolar message passing -- it may be the size
# head-start alone. (A PolarGate LOSS is unaffected, and is if anything
# stronger evidence for the out-of-regime reading.) Run the paired ablation:
#   sbatch --export=ALL,POLARGATE_SIZE_COVARIATES=false src/shell/train_baseline_polargate.sh
# and report both arms. See src/baselines/polargate/PROVENANCE.md.
POLARGATE_POOLING="${POLARGATE_POOLING:-mean}"
POLARGATE_SIZE_COVARIATES="${POLARGATE_SIZE_COVARIATES:-true}"

# Loss. Defaults to smooth_l1 (beta=0.01) for this baseline, matching
# train.py:151 and therefore the primary model, because the label is 48.8%
# exactly zero (mean 0.020, SD 0.053) and MSE through a terminal sigmoid on
# that distribution collapses toward the mean. The other three baselines still
# default to MSE, so this is the one baseline scored on the primary model's own
# objective. Run the MSE arm too and report both:
#   sbatch --export=ALL,POLARGATE_LOSS=mse src/shell/train_baseline_polargate.sh
# train_baseline.py suffixes the run label with a non-default loss, so the two
# runs get separate checkpoint dirs.
POLARGATE_LOSS="${POLARGATE_LOSS:-smooth_l1}"

# Node budget replaces a fixed graph count, as for HOGA and DeepGate4.
#
# MEASURED cost, at the published out_dim=256 / layer_num=9 in float32:
# autograd retains 84,066 bytes (82.1 KiB) per node for the backward pass --
# constant to three significant figures across 25k, 50k and 100k nodes, since
# the trunk is nine fixed-width convs with no attention and no virtual-edge
# expansion. So a 500k budget is ~39 GiB fp32, roughly half that under the
# bf16-mixed AMP this script gets on H100. Unlike HOGA and DeepGate4, the
# largest single graph is NOT what sets the peak: a 366,040-node singleton
# batch retains 28.7 GiB fp32, below a full 500k budget.
#
# The GPU figure itself was NOT measured -- the port was written on a machine
# with no access to the cluster. Take it from nvidia-smi or wandb "GPU Memory
# Allocated" on the first epoch and record it in PROVENANCE.md.
#
# Unlike HOGA and DeepGate4, PolarGate publishes an effective batch:
# train.sh's --batch_size 256 is gradient accumulation over 256 one-graph
# forwards (train.py:339 steps the optimizer every batch_size iterations of a
# single-graph loop). Sizing the pair needs the graphs-per-micro-batch figure,
# and the two available estimates disagree, so treat it as a RANGE:
#   - From the ~40k mean node count quoted throughout this repo: 500k/40k =
#     12.5, and that is an upper bound, since packing is imperfect and any
#     graph over the budget forms a singleton.
#   - From HOGA's own MEASURED 149,485 train micro-batches at its 150k budget:
#     ~45k micro-batches here, i.e. ~707k/45k = ~16 graphs each.
# Those cannot both be right. The second implies a train-split mean nearer 32k
# nodes than 40k, which is plausible (the ~40k figure is quoted for train+val
# and is itself approximate) but unverified. So 20 accumulation steps gives
# somewhere around 250-310 graphs per update against upstream's 256, and the
# 12.5-graph reading is the one that lands on target.
# DO NOT treat either end as known -- read avg_graphs_per_batch off the first
# epoch summary and retune from that.
#
# THE TWO MUST BE RETUNED TOGETHER -- their product is the effective batch, and
# the whole point of the pairing is to hit the paper's number. Note this makes
# PolarGate optimize under a ~3.4x larger effective batch than the primary
# model's ~75 graphs/update, which HOGA and DeepGate4 deliberately match
# instead. That is not an oversight: for those two no published effective batch
# exists, so parity with the primary model is the best available choice, while
# here the paper's own number exists and fidelity to it wins. Say so when
# reporting.
#
# VERIFY on the first epoch: train_utils.py prints the real
# avg_graphs_per_batch in "[train] Epoch summary", and nvidia-smi / wandb "GPU
# Memory Allocated" gives the real peak. Recalibrate from those, not from the
# ~40k mean.
POLARGATE_MAX_NODES_PER_BATCH="${POLARGATE_MAX_NODES_PER_BATCH:-500000}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-20}"

# FULL EPOCHS, deliberately, unlike train_baseline_hoga.sh.
#
# That script caps LIMIT_TRAIN_BATCHES at ~13-17% of the corpus so its "epoch"
# fits a useful checkpoint cadence, which makes its epoch index incomparable
# with train.py's and its val_loss incomparable with train.py's -- while both
# still run --patience 4, so the two models get patience over very different
# amounts of data. Not truncating is the better default when the epoch fits.
#
# WHETHER IT FITS IS NOT MEASURED. The honest scaling: HOGA measured 149,485
# train micro-batches at a 150k node budget, so at 500k this is ~45k
# micro-batches -- fewer than HOGA's ~75k at its current 300k, on a model with
# no attention and no virtual-edge expansion. That argues a full pass is
# affordable, but HOGA's ~12h epoch was diagnosed as bandwidth-bound rather
# than compute-bound (~100% GPU utilisation at ~2% tensor-core activity), and
# this job streams the same corpus through the same loader, so the dataloader
# floor may dominate and the saving may be much smaller than the model's
# cheapness suggests.
#
# The risk if it does not fit: ModelCheckpoint runs with
# save_on_train_epoch_end=True, so nothing is written until an epoch ends. A
# walltime kill mid-epoch loses everything since the last boundary, and
# --patience 4 needs four completed epochs to fire at all. That is the other
# half of HOGA's rationale for capping, and it applies here too.
#
# So: RECALIBRATE from the first "[train] Epoch summary" line (avg_step_s,
# epoch_s) before trusting this default. If a full epoch does not comfortably
# fit several times over inside 72h, cap these -- and when you do, report
# GRAPHS SEEN, not epoch index: train_utils.py logs `train_graphs_seen` to
# WandB and prints `graphs_seen_total=` on every epoch summary precisely so a
# truncated run stays comparable. Also quote the FULL val/test split from a
# separate eval pass, since a capped val pass changes what val_loss means.
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"

# Upstream's published patience is 50 epochs (train.py argparse, and TODAES
# Sec 6.2: "500 epochs ... early stopping strategy using a patience of 50").
# Unreachable inside a 72h walltime here, and it would also make this the one
# baseline allowed to run far longer than the others. 4 matches
# train_baseline_hoga.sh and train_baseline_deepgate4.sh, so early stopping is
# not what distinguishes the models. Report the deviation.
PATIENCE="${PATIENCE:-4}"

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

echo "Using NUM_WORKERS=$NUM_WORKERS for data loading."
echo "Using SPLIT_BY=$SPLIT_BY."
echo "Using POLARGATE_LAYER_NUM=$POLARGATE_LAYER_NUM, POLARGATE_OUT_DIM=$POLARGATE_OUT_DIM."
echo "Using POLARGATE_LOSS=$POLARGATE_LOSS, POLARGATE_POOLING=$POLARGATE_POOLING."
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# =========================================================
# EXECUTE TRAINING (PolarGate baseline, ICCAD'24 version. layer_num is
# published in upstream's train.sh; out_dim/lr/weight_decay/max_epochs are
# train.py's argparse defaults, which train.sh runs under. The readout,
# batching and loss are this port's -- see
# src/baselines/polargate/regressor.py and PROVENANCE.md.)
# =========================================================

echo "Starting PolarGate baseline training for $ALGORITHM on GPU 0..."

srun python -u -m train_baseline \
    --baseline           "polargate" \
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
    --loss               "$POLARGATE_LOSS" \
    --polargate_layer_num "$POLARGATE_LAYER_NUM" \
    --polargate_out_dim  "$POLARGATE_OUT_DIM" \
    --polargate_pooling  "$POLARGATE_POOLING" \
    --polargate_size_covariates "$POLARGATE_SIZE_COVARIATES" \
    --polargate_max_nodes_per_batch "$POLARGATE_MAX_NODES_PER_BATCH" \
    --accumulate_grad_batches  "$ACCUMULATE_GRAD_BATCHES" \
    --limit_train_batches "$LIMIT_TRAIN_BATCHES" \
    --limit_val_batches   "$LIMIT_VAL_BATCHES" \
    --prefetch_factor    "$PREFETCH_FACTOR" \
    --num_workers        "$NUM_WORKERS" \
    --pin_memory         "$PIN_MEMORY" \
    --persistent_workers "$PERSISTENT_WORKERS" \
    --patience           "$PATIENCE"
