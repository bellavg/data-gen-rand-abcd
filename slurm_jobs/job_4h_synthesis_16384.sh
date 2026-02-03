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
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

echo "Loaded modules:"
module list
echo ""

# Add ABC to PATH (adjust path if ABC is installed elsewhere)
export PATH="$HOME/abc:$PATH"

BASE_DIR="$HOME/data-gen-rand-abcd"
DATASET_DIR="${BASE_DIR}/OPENABC_DATASET"

# Copy abc.rc for ABC aliases
if [ -f "${BASE_DIR}/abc.rc" ]; then
    cp "${BASE_DIR}/abc.rc" "$HOME/.abc.rc"
    cp "${BASE_DIR}/abc.rc" "${DATASET_DIR}/bench/abc.rc"
fi

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
