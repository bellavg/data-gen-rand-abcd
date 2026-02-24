#!/bin/bash
#SBATCH --job-name=populate_openabc_metadata
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=genoa
#SBATCH --output=logs/populate_openabc_metadata_%j.out

set -euo pipefail

echo "=========================================="
echo "JOB 6B: Populate FULL Metadata"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-manual_run}"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0
module load SciPy-bundle/2025.06-gfbf-2025a

echo "Loaded modules: 2025, foss/2025a, Python/3.13.1, SciPy-bundle/2025.06"
echo ""

BASE_DIR="$HOME/data-gen-rand-abcd"
DATASET_TOOLS_DIR="${BASE_DIR}/dataset_tools"
RANDOM_DATASET="${RANDOM_DATASET:-${BASE_DIR}/OPENABC_DATASET}"
OPENABC_DATASET="${OPENABC_DATASET:-/scratch-shared/igardner1/openabc_full/OPENABC_DATASET}"
OUTPUT_DATASET="${OUTPUT_DATASET:-/scratch-shared/$USER/FULL_DATASET}"

echo "Base directory: $BASE_DIR"
echo "Random dataset source: $RANDOM_DATASET"
echo "OpenABC source: $OPENABC_DATASET"
echo "Target FULL_DATASET: $OUTPUT_DATASET"
echo ""

for required_dir in "$DATASET_TOOLS_DIR" "$RANDOM_DATASET" "$OPENABC_DATASET" "$OUTPUT_DATASET"; do
    if [ ! -d "$required_dir" ]; then
        echo "ERROR: Required directory missing: $required_dir"
        exit 1
    fi
done

for cmd in python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: $cmd"
        exit 1
    fi
done

python3 -c "import pandas, numpy; print('✓ pandas and numpy available')" || {
    echo "ERROR: Required Python packages not available"
    exit 1
}

WORKERS="${SLURM_CPUS_PER_TASK:-24}"
if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [ "$WORKERS" -lt 1 ]; then
    WORKERS=4
fi

echo "Populating metadata for ALL designs (random + OpenABC)..."
echo "Workers: $WORKERS"
echo ""

cd "$DATASET_TOOLS_DIR"

python3 generate_metadata.py \
    "$OUTPUT_DATASET" \
    --workers "$WORKERS" \
    --validate \
    --summary \
    --random-source "$RANDOM_DATASET" \
    --openabc-source "$OPENABC_DATASET"

echo ""
echo "JOB 6B completed successfully."
echo "End time: $(date)"
