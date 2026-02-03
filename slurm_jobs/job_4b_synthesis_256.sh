#!/bin/bash
#SBATCH --job-name=synth_256
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/synthesis_256_%j.out

# Step 4b: Run Synthesis for Design 256

set -e
echo "=========================================="
echo "STEP 4b: Running Synthesis for Design 256"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

BASE_DIR="$HOME/data-gen-rand-abcd"
DATASET_DIR="${BASE_DIR}/OPENABC_DATASET"

# Try to load yosys module
module load 2025 2>/dev/null || true
module load yosys 2>/dev/null || module load Yosys 2>/dev/null || true

if ! command -v yosys-abc &> /dev/null; then
    echo "✗ Error: yosys-abc not found. Run check_yosys.sh for help."
    exit 1
fi

cd "${DATASET_DIR}/bench"
echo "Running synthesis for design 256..."
./synthesisBulk_256.sh

echo "✓ Synthesis complete for design 256"
echo "End time: $(date)"
