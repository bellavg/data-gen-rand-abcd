#!/bin/bash
#SBATCH --job-name=synth_2048
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/synthesis_2048_%j.out

# Step 4e: Run Synthesis for Design 2048

set -e
echo "STEP 4e: Running Synthesis for Design 2048"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

BASE_DIR="$HOME/data-gen-rand-abcd"
DATASET_DIR="${BASE_DIR}/OPENABC_DATASET"

if ! command -v yosys-abc &> /dev/null; then
    echo "✗ Error: yosys-abc not found"
    exit 1
fi

cd "${DATASET_DIR}/bench"
./synthesisBulk_2048.sh

echo "✓ Synthesis complete for design 2048"
echo "End time: $(date)"
