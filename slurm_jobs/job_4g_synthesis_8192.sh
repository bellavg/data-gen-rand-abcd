#!/bin/bash
#SBATCH --job-name=synth_8192
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/synthesis_8192_%j.out

# Step 4g: Run Synthesis for Design 8192

set -e
echo "STEP 4g: Running Synthesis for Design 8192"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

BASE_DIR="$HOME/data-gen-rand-abcd"
DATASET_DIR="${BASE_DIR}/OPENABC_DATASET"

if ! command -v yosys-abc &> /dev/null; then
    echo "✗ Error: yosys-abc not found"
    exit 1
fi

cd "${DATASET_DIR}/bench"
./synthesisBulk_8192.sh

echo "✓ Synthesis complete for design 8192"
echo "End time: $(date)"
