#!/bin/bash
#SBATCH --job-name=synth_4096
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/synthesis_4096_%j.out

# Step 4f: Run Synthesis for Design 4096

set -e
echo "=========================================="
echo "STEP 4f: Running Synthesis for Design 4096"
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
./synthesisBulk_4096.sh

echo "✓ Synthesis complete for design 4096"
echo "End time: $(date)"
