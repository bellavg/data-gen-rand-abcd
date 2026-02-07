#!/bin/bash
#SBATCH --job-name=synth_4b
#SBATCH --time=01:00:00
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
echo "✓ Modules loaded: 2025, foss/2025a"
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
echo "Running synthesis for design 256..."
./synthesisBulk_256.sh

echo ""
echo "=========================================="
echo "Synthesis Complete for Design 256"
echo "=========================================="
echo ""

# Count generated files
zip_count=$(ls -1 256/syn*.zip 2>/dev/null | wc -l | tr -d ' ')
log_count=$(ls -1 256/log_256/*.log 2>/dev/null | wc -l | tr -d ' ')

echo "Generated:"
echo "  - ${zip_count} synthesis result zip files"
echo "  - ${log_count} log files"
echo ""
echo "End time: $(date)"
