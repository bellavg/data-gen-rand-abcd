#!/bin/bash
#SBATCH --job-name=synth_4096
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/synthesis_4096_%j.out

# Step 4f: Run Synthesis for Design 4096

set -e
echo "STEP 4f: Running Synthesis for Design 4096"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

BASE_DIR="$HOME/data-gen-rand-abcd"
DATASET_DIR="${BASE_DIR}/OPENABC_DATASET"

if ! command -v yosys-abc &> /dev/null; then
    echo "✗ Error: yosys-abc not found"
    exit 1
fi

cd "${DATASET_DIR}/bench"
./synthesisBulk_4096.sh

echo "✓ Synthesis complete for design 4096"
echo "End time: $(date)"
