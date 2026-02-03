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

# Define paths
BASE_DIR="$HOME/data-gen-rand-abcd"
DATASET_DIR="${BASE_DIR}/OPENABC_DATASET"

# Try to load yosys module if available
module load 2025 2>/dev/null || true
if module avail 2>&1 | grep -qi yosys; then
    echo "Loading yosys module..."
    module load yosys 2>/dev/null || module load Yosys 2>/dev/null || true
fi

# Check if yosys-abc is available
if ! command -v yosys-abc &> /dev/null; then
    echo "=========================================="
    echo "✗ ERROR: yosys-abc not found in PATH"
    echo "=========================================="
    echo ""
    echo "yosys-abc is required for synthesis. Please:"
    echo ""
    echo "1. Check if yosys module exists:"
    echo "   module avail | grep -i yosys"
    echo ""
    echo "2. If module exists, load it:"
    echo "   module load yosys  # or Yosys"
    echo ""
    echo "3. If no module exists, you need to:"
    echo "   - Request installation from Snellius support"
    echo "   - OR compile from source: https://github.com/YosysHQ/yosys"
    echo ""
    echo "4. After installation, ensure 'yosys-abc' is in PATH"
    echo ""
    exit 1
fi

echo "✓ Using yosys-abc: $(which yosys-abc)"
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
