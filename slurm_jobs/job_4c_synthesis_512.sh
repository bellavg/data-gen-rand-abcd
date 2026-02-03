#!/bin/bash
#SBATCH --job-name=synth_512
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=thin
#SBATCH --output=logs/synthesis_512_%j.out

# Step 4c: Run Synthesis for Design 512

set -e
echo "STEP 4c: Running Synthesis for Design 512"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

BASE_DIR="$HOME/data-gen-rand-abcd"
DATASET_DIR="${BASE_DIR}/OPENABC_DATASET"

if ! command -v yosys-abc &> /dev/null; then
    echo "✗ Error: yosys-abc not found"
    exit 1
fi

cd "${DATASET_DIR}/bench"
./synthesisBulk_512.sh

echo "✓ Synthesis complete for design 512"
echo "End time: $(date)"
