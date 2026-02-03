#!/bin/bash
#SBATCH --job-name=synth_128
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/synthesis_128_%j.out

# Step 4a: Run Synthesis for Design 128
# Executes 1500 synthesis recipes for the 128 design
# Each recipe produces 21 AIG files (step0 through step20)

set -e  # Exit on error

echo "=========================================="
echo "STEP 4a: Running Synthesis for Design 128"
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

# Define paths
BASE_DIR="$HOME/data-gen-rand-abcd"
DATASET_DIR="${BASE_DIR}/OPENABC_DATASET"

# Check if abc is available
if ! command -v abc &> /dev/null; then
    echo "✗ Error: abc not found in PATH"
    echo "Current PATH: $PATH"
    echo "Please ensure abc is installed at $HOME/abc or update the PATH export above"
    exit 1
fi

echo "✓ Using abc: $(which abc)"
abc -h 2>&1 | head -3 || echo "(ABC help unavailable)"
echo ""

# Change to bench directory
cd "${DATASET_DIR}/bench"

echo "Running 1500 synthesis recipes for design 128..."
echo "This will generate 31,500 AIG files (1500 × 21)"
echo ""

# Execute the synthesis
./synthesisBulk_128.sh

echo ""
echo "=========================================="
echo "Synthesis Complete for Design 128"
echo "=========================================="
echo ""

# Count generated files
zip_count=$(ls -1 128/syn*.zip 2>/dev/null | wc -l | tr -d ' ')
log_count=$(ls -1 128/log_128/*.log 2>/dev/null | wc -l | tr -d ' ')

echo "Generated:"
echo "  - ${zip_count} synthesis result zip files"
echo "  - ${log_count} log files"
echo ""
echo "End time: $(date)"
