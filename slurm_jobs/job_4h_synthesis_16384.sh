#!/bin/bash
#SBATCH --job-name=synth_16384
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/synthesis_16384_%j.out

# Step 4h: Run Synthesis for Design 16384

set -e
echo "=========================================="
echo "STEP 4h: Running Synthesis for Design 16384"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

# Load required modules for Snellius
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

echo "Loaded modules:"
module list
echo ""

# Add ABC to PATH (adjust path if ABC is installed elsewhere)
export PATH="$HOME/abc:$PATH"

BASE_DIR="$HOME/data-gen-rand-abcd"
DATASET_DIR="${BASE_DIR}/OPENABC_DATASET"

if ! command -v abc &> /dev/null; then
    echo "✗ Error: abc not found"
    echo "Current PATH: $PATH"
    exit 1
fi

echo "✓ Using abc: $(which abc)"
echo ""

cd "${DATASET_DIR}/bench"
./synthesisBulk_16384.sh

echo "✓ Synthesis complete for design 16384"
echo "End time: $(date)"
