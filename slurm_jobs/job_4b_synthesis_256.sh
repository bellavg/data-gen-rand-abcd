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

# Load required modules for Snellius
module purge
module load 2025
module load foss/2025a

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

# Verify library and design files
if [ ! -f "${DATASET_DIR}/lib/nangate45.lib" ] || [ ! -f "${DATASET_DIR}/bench/256/256_orig.aig" ]; then
    echo "✗ Error: Required files missing"
    exit 1
fi

cd "${DATASET_DIR}/bench"
echo "Running synthesis for design 256..."
./synthesisBulk_256.sh

echo "✓ Synthesis complete for design 256"
echo "End time: $(date)"
