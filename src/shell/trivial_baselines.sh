#!/bin/bash
#SBATCH --job-name=rq1_trivial_baselines
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=genoa
#SBATCH --output=logs/rq1_trivial_baselines_%j.out

# ---------------------------------------------------------------------------
# RQ1 non-learned baselines. Reads the generation CSV and the splits JSON the
# training run already wrote, fits on train, scores on test, and writes two
# result files. No graph is loaded and no model is run, so this needs no GPU
# and no warm cache -- only the splits file, which train.sh writes.
#
# CPU-only, but not a login-node job: the generation CSV is on the order of a
# million rows of long scratch paths, and the splits JSON holds every one of
# those paths again as a Python string, so peak memory runs to several GB.
# Eight genoa cores carry that comfortably. The work itself is single
# threaded; the cores are asked for their share of node memory, not for
# parallelism.
#
# Submit with:
#   sbatch src/shell/trivial_baselines.sh
#
# No environment variable placed on the sbatch line reaches the job on this
# cluster, --export included. To score a different algorithm or split
# protocol, edit ALGORITHM or SPLIT_BY below and resubmit.
# ---------------------------------------------------------------------------

set -euo pipefail

export TEMP="$TMPDIR"
export TMP="$TMPDIR"

echo "=========================================="
echo "RQ1 NON-LEARNED BASELINES"
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

cd "$BASE_DIR"

ALGORITHM="Orchestrate"
SPLIT_BY="design"
WANDB="true"

# Matches train.sh and test.sh: without it a compute node that cannot reach the
# W&B backend hangs in wandb.init() until the wall clock kills the job.
export WANDB_INIT_TIMEOUT=120

CSV_PATH="$BASE_DIR/data/designs/design_metadata/algo_${ALGORITHM}_ml.csv"
CACHE_DIR="/scratch-shared/$USER/aig_train_run/${ALGORITHM}/cache"

# Same rule as data.dataset.splits_cache_filename: the default strategy stays
# untagged and every other one gets a tag, so splits files written before
# split_by was configurable are still found under their original names.
DEFAULT_SPLIT_BY=$(python -c 'import config; print(config.SPLIT_BY)')
if [[ "$SPLIT_BY" != "$DEFAULT_SPLIT_BY" ]]; then
    SPLIT_TAG="_$SPLIT_BY"
else
    SPLIT_TAG=""
fi
SPLITS_PATH="$CACHE_DIR/algo_${ALGORITHM}_ml_all${SPLIT_TAG}_splits.json"

OUT_PATH="$BASE_DIR/results/rq1_trivial_baselines_${ALGORITHM}${SPLIT_TAG}.csv"

echo "CSV_PATH=$CSV_PATH"
echo "SPLITS_PATH=$SPLITS_PATH"
echo "OUT_PATH=$OUT_PATH"
echo ""

# The splits file is read, never regenerated: a freshly sampled split would not
# be the split the model was trained on. trivial_baselines.py refuses to start
# without it rather than creating one.
# W&B run named baseline_trivial_<algorithm>, mirroring train.py's train_<config>
# and test.py's test_<config> so all three sit together in the project.
python -u "$BASE_DIR/src/trivial_baselines.py" \
    --csv_paths    "$CSV_PATH" \
    --splits_path  "$SPLITS_PATH" \
    --out          "$OUT_PATH" \
    --algorithm    "$ALGORITHM" \
    --split_by     "$SPLIT_BY" \
    --wandb        "$WANDB"

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "Fair competitors: $OUT_PATH"
echo "Oracles (never rank these against the model):"
echo "  ${OUT_PATH%.csv}_oracles.csv"
echo "=========================================="
