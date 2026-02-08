#!/bin/bash
#SBATCH --job-name=r_metadata
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/collect_metadata_%j.out

# Job 5: Collect Post-Synthesis Metadata
# Analyzes synthesis log files and AIG files to extract circuit statistics
# Run this after synthesis jobs (4a-4h) complete

set -e  # Exit on error

echo "=========================================="
echo "POST-SYNTHESIS METADATA COLLECTION"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

# Load required modules
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0

# Load ABC for circuit analysis (metadata script uses 'abc' command)
# Check if ABC module exists, otherwise script will use fallback methods
if module avail ABC 2>&1 | grep -q "ABC"; then
    module load ABC
    echo "✓ ABC module loaded"
else
    echo "⚠️  ABC module not available - using fallback AIG parsing methods"
fi

# Define paths
BASE_DIR="$HOME/data-gen-rand-abcd"
METADATA_SCRIPT="${BASE_DIR}/dataset_tools/collect_post_synthesis_metadata.py"

echo "Configuration:"
echo "  Base directory: ${BASE_DIR}"
echo "  Metadata script: ${METADATA_SCRIPT}"
echo ""

# Verify script exists
if [ ! -f "${METADATA_SCRIPT}" ]; then
    echo "✗ ERROR: Metadata collection script not found: ${METADATA_SCRIPT}"
    exit 1
fi

echo "✓ Metadata collection script found"
echo ""

# Collect metadata for all designs
echo "Collecting metadata for all designs..."
echo "This will analyze log files and zip files to extract circuit statistics"
echo ""

python3 "${METADATA_SCRIPT}" --home "${BASE_DIR}"

echo ""
echo "=========================================="
echo "Metadata Collection Complete"
echo "=========================================="
echo ""
echo "Metadata CSV files generated:"
for design in 128 256 512 1024 2048 4096 8192 16384; do
    csv_file="${BASE_DIR}/OPENABC_DATASET/bench/${design}/metadata/${design}.csv"
    if [ -f "$csv_file" ]; then
        line_count=$(wc -l < "$csv_file" 2>/dev/null || echo "0")
        echo "  • ${design}.csv: $((line_count - 1)) entries"
    else
        echo "  • ${design}.csv: Not found"
    fi
done
echo ""
echo "End time: $(date)"