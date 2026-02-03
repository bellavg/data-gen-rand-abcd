#!/bin/bash
#SBATCH --job-name=gen_design_scripts
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/gen_design_scripts_%j.out

# Step 2: Generate Design-Specific Synthesis Scripts
# Customizes the 1500 reference scripts for each of the 8 designs
# Creates 12,000 scripts total (8 designs × 1500 scripts)

set -e  # Exit on error

echo "=========================================="
echo "STEP 2: Generating Design-Specific Scripts"
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
REFERENCE_SCRIPTS_DIR="${BASE_DIR}/referenceScripts"
AUTOMATION_DIR="${BASE_DIR}/OpenABC-master/datagen/automation"

# Define the 8 designs
DESIGNS=("128" "256" "512" "1024" "2048" "4096" "8192" "16384")

echo "Generating customized synthesis scripts for 8 designs..."
echo "This creates 1500 scripts per design (12,000 total)"
echo ""

python "${AUTOMATION_DIR}/automate_synthesisScriptGen.py" \
    --home "${BASE_DIR}" \
    --script "${REFERENCE_SCRIPTS_DIR}"

echo ""
echo "✓ Design-specific scripts generated"
echo ""

# Verify generated scripts for each design
echo "Verifying generated scripts..."
total_scripts=0
for design in "${DESIGNS[@]}"; do
    script_count=$(find "${DATASET_DIR}/synScripts/${design}" -name "abc*.script" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$script_count" -eq 1500 ]; then
        echo "  ✓ ${design}: 1500 scripts"
        total_scripts=$((total_scripts + script_count))
    else
        echo "  ✗ ${design}: ${script_count} scripts (expected 1500)"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "Step 2 Complete"
echo "=========================================="
echo ""
echo "Generated: ${total_scripts} customized synthesis scripts"
echo "  8 designs × 1500 scripts each"
echo ""
echo "Next step: Submit job_3_generate_shell_scripts.sh"
echo ""
echo "End time: $(date)"
