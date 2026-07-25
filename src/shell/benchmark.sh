#!/bin/bash
#SBATCH --job-name=aig_train_benchmark_array
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --constraint=scratch-node
#SBATCH --array=0-8
#SBATCH --output=logs/benchmark_train_%A_%a.out

# ---------------------------------------------------------------------------
# Controlled training-hardware benchmark — step time, throughput, peak VRAM,
# GPU utilization, host memory — on a small seeded sample of graphs, for each
# of the 9 configs. Batch size and worker count are FIXED identically across
# every array task below (not read from the actual training runs, which
# drifted in these settings between runs) so the comparison is controlled,
# per the reproducibility notes in the thesis experiments plan.
#
# No cache warmup dependency needed — this samples from the train split,
# which warmup_train_cache.sh + the existing mask precompute scripts already
# cover for real training.
# ---------------------------------------------------------------------------

set -euo pipefail

export TEMP="$TMPDIR"
export TMP="$TMPDIR"

echo "=========================================="
echo "JOB ARRAY ID: $SLURM_ARRAY_JOB_ID, TASK ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$BASE_DIR"

# Same "type:method" array convention as test.sh / test_cpu.sh.
ALGORITHM="Orchestrate"
CONFIGS=(
    "none:none"
    "sparsification:and_gate_only" "sparsification:random_edge_dropout"
    "sparsification:spanning_forest" "sparsification:pagerank"
    "partition:random" "partition:metis"
    "partition:level_slicing" "partition:span_weighted_metis"
    # "summarization:<method>"   # appended here once implemented
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
REDUCTION_TYPE="${CONFIG%%:*}"
REDUCTION_METHOD="${CONFIG##*:}"

echo "Task $SLURM_ARRAY_TASK_ID assigned to reduction_type=$REDUCTION_TYPE reduction_method=$REDUCTION_METHOD"

CSV_PATH="$BASE_DIR/data/designs/design_metadata/algo_${ALGORITHM}_ml.csv"

WORKSPACE="/scratch-shared/$USER/aig_train_run/${ALGORITHM}"
CACHE_DIR="$WORKSPACE/cache"
TIER0_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier0_cache"
TIER1_CACHE_DIR="/scratch-shared/$USER/aig_train_run/shared_tier1_cache"

HP_TUNING_WORKSPACE="/scratch-shared/$USER/big_optuna_run"
HP_TUNING_SPLITS="$HP_TUNING_WORKSPACE/shared_dataset_cache/algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_50000_splits.json"

RESULTS_DIR="$BASE_DIR/results"
mkdir -p "$RESULTS_DIR"

# Fixed across every array task — this is what makes the comparison
# controlled. The benchmark measures ONE graph per batch (not real training's
# node-budget dynamic batching, which holds per-batch VRAM ~constant across
# methods and hides reduction's memory benefit). See benchmark.py's docstring.
NUM_WORKERS=8
NUM_WARMUP_GRAPHS=5
NUM_MEASURE_GRAPHS=100

nvidia-smi -L

REDUCTION_ARGS=()
if [[ "$REDUCTION_TYPE" != "none" ]]; then
    REDUCTION_ARGS=(--reduction_method "$REDUCTION_METHOD")
fi

srun python -u -m benchmark \
    --algorithm          "$ALGORITHM" \
    --reduction_type     "$REDUCTION_TYPE" \
    ${REDUCTION_ARGS[@]+"${REDUCTION_ARGS[@]}"} \
    --csv_paths          "$CSV_PATH" \
    --cache_dir          "$CACHE_DIR" \
    --tier0_cache_dir    "$TIER0_CACHE_DIR" \
    --tier1_cache_dir    "$TIER1_CACHE_DIR" \
    --hp_tuning_splits_path "$HP_TUNING_SPLITS" \
    --num_workers        "$NUM_WORKERS" \
    --num_warmup_graphs  "$NUM_WARMUP_GRAPHS" \
    --num_measure_graphs "$NUM_MEASURE_GRAPHS" \
    --results_dir        "$RESULTS_DIR/training_benchmark" \
    --per_graph_dir      "$RESULTS_DIR/benchmark_per_graph"

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID training benchmark ($REDUCTION_TYPE/$REDUCTION_METHOD) complete."
echo "End time: $(date)"
