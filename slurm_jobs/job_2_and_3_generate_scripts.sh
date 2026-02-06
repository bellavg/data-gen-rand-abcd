#!/bin/bash
#SBATCH --job-name=gen_scripts_and_bulk
#SBATCH --time=01:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/gen_scripts_and_bulk_%j.out

# Combined Job 2&3: Generate Design-Specific Scripts + Bulk Synthesis Scripts
# This replaces the separate job_2 and job_3 scripts
# Generates synthesis scripts with metadata collection, then creates shell scripts

set -e  # Exit on error

echo "=========================================="
echo "COMBINED JOB 2&3: Scripts Generation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

# Load Python module
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0

# Define paths
BASE_DIR="$HOME/data-gen-rand-abcd"
DATASET_DIR="${BASE_DIR}/OPENABC_DATASET"
REFERENCE_SCRIPTS_DIR="${BASE_DIR}/referenceScripts"
AUTOMATION_DIR="${BASE_DIR}/OpenABC-master/datagen/automation"
LIBRARY_FILE="/scratch-shared/$USER/openabc_full/OPENABC_DATASET/lib/nangate45.lib"

# Design list
DESIGNS=("128" "256" "512" "1024" "2048" "4096" "8192" "16384")

echo "Configuration:"
echo "  Base directory: ${BASE_DIR}"
echo "  Dataset directory: ${DATASET_DIR}"
echo "  Reference scripts: ${REFERENCE_SCRIPTS_DIR}"
echo "  Library file: ${LIBRARY_FILE}"
echo "  Designs: ${DESIGNS[*]}"
echo ""

# Verify prerequisites
echo "Verifying prerequisites..."
if [ ! -d "${REFERENCE_SCRIPTS_DIR}" ]; then
    echo "✗ ERROR: Reference scripts directory not found: ${REFERENCE_SCRIPTS_DIR}"
    echo "  Please run job_1_generate_reference_scripts.sh first"
    exit 1
fi

script_count=$(find "${REFERENCE_SCRIPTS_DIR}" -name "abc*.script" | wc -l | tr -d ' ')
if [ "$script_count" -ne 1500 ]; then
    echo "✗ ERROR: Expected 1500 reference scripts, found ${script_count}"
    echo "  Please run job_1_generate_reference_scripts.sh first"
    exit 1
fi

if [ ! -f "${LIBRARY_FILE}" ]; then
    echo "✗ ERROR: Library file not found: ${LIBRARY_FILE}"
    exit 1
fi

echo "✓ Prerequisites verified"
echo ""

# Create necessary directories
echo "Creating directory structure..."
mkdir -p "${DATASET_DIR}/synScripts"
mkdir -p "${DATASET_DIR}/bench"
for design in "${DESIGNS[@]}"; do
    mkdir -p "${DATASET_DIR}/bench/${design}/metadata"
done
echo "✓ Directories created"
echo ""

# STEP 2A: Generate design-specific synthesis scripts with metadata collection
echo "=========================================="
echo "STEP 2A: Generating Design-Specific Scripts"
echo "=========================================="

python "${AUTOMATION_DIR}/automate_synthesisScriptGen.py" \
    --home "${BASE_DIR}" \
    --script "${REFERENCE_SCRIPTS_DIR}" \
    --lib "${LIBRARY_FILE}"

echo ""
echo "✓ Design-specific scripts with metadata collection generated"
echo ""

# Verify generated scripts for each design
echo "Verifying generated synthesis scripts..."
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
echo "Generated: ${total_scripts} customized synthesis scripts with metadata collection"
echo ""

# STEP 2B: Generate bulk synthesis shell scripts  
echo "=========================================="
echo "STEP 2B: Generating Bulk Synthesis Scripts"
echo "=========================================="

python "${AUTOMATION_DIR}/automate_bulkSynthesis.py" \
    --home "${BASE_DIR}" \
    --lib "${LIBRARY_FILE}"

echo ""
echo "✓ Bulk synthesis shell scripts generated"
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
echo "Combined Job 2&3 Complete"
echo "=========================================="
echo ""
echo "Generated:"
echo "  • ${total_scripts} synthesis scripts with metadata collection"
echo "  • 8 bulk synthesis shell scripts"
echo ""
echo "Scripts include:"
echo "  • Statistics capture at each synthesis step"
echo "  • Automatic metadata CSV generation per design"
echo "  • CSV format: file_path,design,recipe_id,step_id,tier_id,nodes,edges,num_PI,num_PO,depth,avg_fanout,max_fanout"
echo "  • Focus on logical statistics (no area/delay - requires technology mapping)"
echo ""
echo "Next step: Submit job_4a_synthesis_*.sh for synthesis runs"
echo "  Synthesis will automatically generate metadata CSV files in:"
echo "  ${DATASET_DIR}/bench/{design}/metadata/{design}.csv"
echo ""
echo "End time: $(date)"