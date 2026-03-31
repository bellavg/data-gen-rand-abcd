#!/bin/bash
#SBATCH --job-name=syn_array
#SBATCH --time=04:00:00
#SBATCH --array=0-36
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=genoa
#SBATCH --constraint=scratch-node
#SBATCH --output=logs/syn_%A_%a.out

set -euo pipefail

# 1. Map the array ID to the specific design name
DESIGNS=(
    "128" "256" "512" "1024" "2048" "4096" "8192" "16384"
    "ac97_ctrl" "aes" "aes_secworks" "aes_xcrypt" "apex1" "bc0" "bp_be" "c1355"
    "c5315" "c6288" "c7552" "dalu" "des3_area" "dft" "div" "dynamic_node"
    "ethernet" "fir" "fpu" "hyp" "i10" "i2c" "idft" "iir"
    "jpeg" "k2" "log2" "mainpla" "max" "mem_ctrl" "multiplier" "pci"
    "picosoc" "sasc" "sha256" "simple_spi" "sin" "spi" "sqrt" "square"
    "ss_pcm" "tinyRocket" "tv80" "usb_phy" "vga_lcd" "wb_conmax" "wb_dma"
)

DESIGN=${DESIGNS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "JOB: Synthesis for Design $DESIGN"
echo "=========================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

# 2. Load required modules & set ABC path
module purge
module load 2025
module load foss/2025a
export PATH="$HOME/abc:$PATH"

if ! command -v abc &> /dev/null; then
    echo "✗ Error: abc not found in PATH"
    exit 1
fi

BASE_DIR="$HOME/data-gen-rand-abcd"
SYN_SCRIPTS_DIR="${BASE_DIR}/data/abc_scripts/synthesis_scripts"
DESIGN_DIR="${BASE_DIR}/data/designs/${DESIGN}"

# 3. Setup High-Speed Local Node Scratch Space
if [[ -n "${TMPDIR:-}" ]]; then
    LOCAL_SCRATCH="$TMPDIR"
else
    LOCAL_SCRATCH="/tmp/$USER"
fi
mkdir -p "$LOCAL_SCRATCH"

WORK_DIR=$(mktemp -d "$LOCAL_SCRATCH/syn_task_${DESIGN}_XXXXXX")

# Ensure scratch is thoroughly cleaned up when the script exits
trap 'rm -rf "$WORK_DIR"' EXIT

echo ">> Setting up scratch directory at $WORK_DIR..."
mkdir -p "$WORK_DIR/tier0"
mkdir -p "$WORK_DIR/design_metadata/raw_logs/synthesis_logs"

echo ">> Copying initial tier0 AIG to scratch..."
cp -a "$DESIGN_DIR/tier0/." "$WORK_DIR/tier0/"

# 4. Extract and Patch Scripts
echo ">> Unzipping scripts to scratch..."
unzip -q -o "${SYN_SCRIPTS_DIR}/${DESIGN}.zip" -d "$WORK_DIR/scripts"

echo ">> Patching scripts to use scratch paths..."
# Replace placeholder tokens baked in at generation time with actual scratch paths
find "$WORK_DIR/scripts/${DESIGN}" -type f -exec sed -i \
    -e "s|__SCRATCH__|${WORK_DIR}|g" \
    -e "s|__SCRIPTS__|${WORK_DIR}/scripts/${DESIGN}|g" \
    {} +

# 5. Execute Synthesis
echo ">> Executing 200 synthesis recipes for $DESIGN..."
cd "$WORK_DIR/scripts/${DESIGN}"
bash "run_synthesis_${DESIGN}.sh"

# 6. Verification: Check File Counts Before Archiving
echo ">> Verifying file counts before archiving..."
cd "$WORK_DIR"

# 1 base (synX_step0) + (200 recipes * 21 steps) = 4,201 AIGs
EXPECTED_AIGS=4201
EXPECTED_LOGS=200

ACTUAL_AIGS=$(find "tier0" -maxdepth 1 -name "*.aig" | wc -l)
ACTUAL_LOGS=$(find "design_metadata/raw_logs/synthesis_logs" -maxdepth 1 -name "*.log" | wc -l)

echo "   Expected AIGs: $EXPECTED_AIGS | Actual AIGs: $ACTUAL_AIGS"
echo "   Expected Logs: $EXPECTED_LOGS   | Actual Logs: $ACTUAL_LOGS"

if [ "$ACTUAL_AIGS" -ne "$EXPECTED_AIGS" ] || [ "$ACTUAL_LOGS" -ne "$EXPECTED_LOGS" ]; then
    echo "✗ ERROR: File count mismatch! Synthesis failed or aborted early."
    echo "         Aborting zip and cleanup. Permanent files remain untouched."
    exit 1
fi

echo "✓ File counts verified successfully!"

# 7. Zip Tier 0 and Logs locally on Scratch
echo ">> Zipping outputs in scratch..."
zip -r -q "${DESIGN}_tier0.zip" "tier0/"
zip -r -q "${DESIGN}_synthesis_logs.zip" "design_metadata/raw_logs/synthesis_logs/"

# 8. Move zipped files to Home and Cleanup loose files
echo ">> Moving zipped archives back to permanent storage..."
mv -f "${DESIGN}_tier0.zip" "$DESIGN_DIR/tier0.zip"
mkdir -p "$DESIGN_DIR/design_metadata/raw_logs/synthesis_logs"
mv -f "${DESIGN}_synthesis_logs.zip" "$DESIGN_DIR/design_metadata/raw_logs/synthesis_logs/synthesis_logs.zip"

echo ">> Deleting original loose files in home directory to free up space..."
rm -rf "$DESIGN_DIR/tier0"

echo ""
echo "=========================================="
echo "Job Successfully Finished for $DESIGN"
echo "=========================================="
echo "End time: $(date)"