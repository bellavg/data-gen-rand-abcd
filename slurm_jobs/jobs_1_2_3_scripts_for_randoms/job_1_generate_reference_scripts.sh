#!/bin/bash
#SBATCH --job-name=gen_ref_scripts
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/gen_ref_scripts_%j.out

set -e  # Exit on error

echo "=========================================="
echo "STEP 1: Generating Reference Scripts"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

# Load Python module
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

# Define paths
BASE_DIR="$HOME/data-gen-rand-abcd"
REFERENCE_SCRIPTS_DIR="${BASE_DIR}/referenceScripts"

# Generate 1500 reference scripts
echo "Generating 1500 reference scripts with L=20 transformations each..."
echo ""

python "${BASE_DIR}/generate_reference_scripts.py" \
    --output "${REFERENCE_SCRIPTS_DIR}" \
    --num 1500 \
    --length 20 \
    --seed 42

echo ""
echo "✓ Reference scripts generation complete"
echo ""

# Verify reference scripts
ref_count=$(find "${REFERENCE_SCRIPTS_DIR}" -name "abc*.script" 2>/dev/null | wc -l | tr -d ' ')
if [ "$ref_count" -eq 1500 ]; then
    echo "✓ Verified: 1500 reference scripts created"
else
    echo "✗ Error: Expected 1500 scripts, found ${ref_count}"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 1 Complete"
echo "=========================================="
echo ""
echo "Generated: 1500 reference scripts"
echo "Location: ${REFERENCE_SCRIPTS_DIR}"
echo ""
echo "Next step: Submit job_2_generate_design_scripts.sh"
echo ""
echo "End time: $(date)"
