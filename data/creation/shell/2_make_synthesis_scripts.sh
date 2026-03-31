#!/bin/bash
#SBATCH --job-name=gen_and_zip_syn_scripts
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/gen_and_zip_scripts_%j.out

# This job generates the synthesis scripts using automate_bulkSynthesis.py,
# then zips the resulting scripts per-design and deletes the loose files.

set -e  # Exit on error

echo "=========================================="
echo "JOB: Generate and Zip Synthesis Scripts"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

# Load required modules
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0

# Define paths
BASE_DIR="$HOME/data-gen-rand-abcd"
AUTOMATION_DIR="${BASE_DIR}/data/creation"
SYN_SCRIPTS_DIR="${BASE_DIR}/data/abc_scripts/synthesis_scripts"

# Define the designs list to iterate over for zipping
DESIGNS=(
    "128" "256" "512" "1024" "2048" "4096" "8192" "16384"
    "ac97_ctrl" "aes" "aes_secworks" "aes_xcrypt" "apex1" "bc0" "bp_be" "c1355"
    "c5315" "c6288" "c7552" "dalu" "des3_area" "dft" "div" "dynamic_node"
    "ethernet" "fir" "fpu" "hyp" "i10" "i2c" "idft" "iir"
    "jpeg" "k2" "log2" "mainpla" "max" "mem_ctrl" "multiplier" "pci"
    "picosoc" "sasc" "sha256" "simple_spi" "sin" "spi" "sqrt" "square"
    "ss_pcm" "tinyRocket" "tv80" "usb_phy" "vga_lcd" "wb_conmax" "wb_dma"
)

echo "Configuration:"
echo "  Base directory: ${BASE_DIR}"
echo "  Automation script: ${AUTOMATION_DIR}/automate_bulkSynthesis.py"
echo "  Output directory: ${SYN_SCRIPTS_DIR}"
echo ""

# STEP 1: Run the Python script to generate all loose synthesis scripts
echo "=========================================="
echo "STEP 1: Generating Scripts via Python"
echo "=========================================="
python "${AUTOMATION_DIR}/automate_bulkSynthesis.py" --home "${BASE_DIR}"
echo "✓ Python generation complete."
echo ""

# STEP 2: Zip the loose files per design and delete the loose directories
echo "=========================================="
echo "STEP 2: Zipping and Cleaning Loose Files"
echo "=========================================="

# Make sure we are in the directory where the design folders were created
if [ -d "${SYN_SCRIPTS_DIR}" ]; then
    cd "${SYN_SCRIPTS_DIR}"
else
    echo "✗ ERROR: Synthesis scripts directory not found: ${SYN_SCRIPTS_DIR}"
    exit 1
fi

for design in "${DESIGNS[@]}"; do
    if [ -d "$design" ]; then
        echo "Processing $design..."
        
        # Create a zip archive containing the directory and its contents
        zip -r -q "${design}.zip" "$design/"
        
        # Verify the zip was created successfully before deleting
        if [ -f "${design}.zip" ]; then
            # Remove the loose directory
            rm -rf "$design/"
            echo "  ✓ Zipped to ${design}.zip and removed loose files."
        else
            echo "  ✗ ERROR: Failed to create zip for $design!"
        fi
    else
        echo "  ⚠ Directory for $design not found, skipping..."
    fi
done

echo ""
echo "=========================================="
echo "Job Complete"
echo "=========================================="
echo "All synthesis scripts have been generated, zipped, and cleaned up."
echo "Zipped files are located in: ${SYN_SCRIPTS_DIR}"
echo ""
echo "End time: $(date)"