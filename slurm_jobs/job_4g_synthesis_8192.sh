#!/bin/bash
#SBATCH --job-name=syn_4g
#SBATCH --time=06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/synthesis_8192_%j.out

# Step 4g: Run Synthesis for Design 8192
# Executes 1500 synthesis recipes for the 8192 design
# Each recipe produces 21 AIG files (step0 through step20)

set -e
echo "=========================================="
echo "STEP 4g: Running Synthesis for Design 8192"
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
echo "Running 1500 synthesis recipes for design 8192..."
echo "This will generate 31,500 AIG files (1500 × 21)"
echo ""

# Execute the synthesis
./synthesisBulk_8192.sh

echo ""
echo "=========================================="
echo "Synthesis Complete for Design 8192"
echo "=========================================="
echo ""

# Count generated files
zip_count=$(ls -1 8192/syn*.zip 2>/dev/null | wc -l | tr -d ' ')
log_count=$(ls -1 8192/log_8192/*.log 2>/dev/null | wc -l | tr -d ' ')

echo "Generated:"
echo "  - ${zip_count} synthesis result zip files"
echo "  - ${log_count} log files"
echo ""
echo "End time: $(date)"
