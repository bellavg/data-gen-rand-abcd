#!/bin/bash
#SBATCH --job-name=gen_shell_scripts
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/gen_shell_scripts_%j.out

# Step 3: Generate Master Shell Scripts
# Creates shell scripts that will execute all 1500 synthesis runs per design

set -e  # Exit on error

echo "=========================================="
echo "STEP 3: Generating Bulk Synthesis Shell Scripts"
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
DATASET_DIR="${BASE_DIR}/OPENABC_DATASET"
AUTOMATION_DIR="${BASE_DIR}/OpenABC-master/datagen/automation"

# Define the 8 designs
DESIGNS=("128" "256" "512" "1024" "2048" "4096" "8192" "16384")

echo "Generating bulk synthesis shell scripts..."
echo ""

python "${AUTOMATION_DIR}/automate_bulkSynthesis.py" \
    --home "${BASE_DIR}"

echo ""
echo "✓ Shell scripts generated"
echo ""

# Make shell scripts executable
echo "Making shell scripts executable..."
for design in "${DESIGNS[@]}"; do
    script_file="${DATASET_DIR}/bench/synthesisBulk_${design}.sh"
    if [ -f "$script_file" ]; then
        chmod +x "$script_file"
        echo "  ✓ ${design}: synthesisBulk_${design}.sh"
    else
        echo "  ✗ ${design}: Shell script not found!"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "Step 3 Complete"
echo "=========================================="
echo ""
echo "Generated: 8 bulk synthesis shell scripts"
echo "Location: ${DATASET_DIR}/bench/"
echo ""
echo "Next step: Submit job_4a_synthesis_*.sh for synthesis runs"
echo "  You can run all 8 designs in parallel or one at a time"
echo ""
echo "End time: $(date)"
