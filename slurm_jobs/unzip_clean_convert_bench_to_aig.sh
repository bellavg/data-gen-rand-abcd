#!/bin/bash
#SBATCH -p genoa
#SBATCH -t 24:00:00
#SBATCH --job-name=openabc_final_report
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/openabc_final_%j.out

# --- 1. SETUP & MODULES ---
set -e
module purge
module load 2025 foss/2025a p7zip/17.05-GCCcore-14.2.0

BASE_DIR="/scratch-shared/$USER/openabc_full"
DATA_ROOT="$BASE_DIR/OPENABC_DATASET"
ABC_BIN="$HOME/abc/abc"

cd "$BASE_DIR"

# --- 2. EXTRACT CORE METADATA FOLDERS ---
echo ">> Extracting lib, statistics, and synScripts..."
7z x OPENABC_DATASET.zip \
    "OPENABC_DATASET/lib/*" \
    "OPENABC_DATASET/statistics/*" \
    "OPENABC_DATASET/synScripts/*" \
    -o"$BASE_DIR" -y

# --- 3. CONVERSION FUNCTION ---
convert_bench_to_aig() {
    local bench_file="$1"
    local output_dir="$2"
    local base_name=$(basename "${bench_file%.bench}")
    $ABC_BIN -c "read_bench $bench_file; strash; write $output_dir/${base_name}.aig" > /dev/null 2>&1
}
export -f convert_bench_to_aig
export ABC_BIN

# --- 4. THE DESIGN LOOP ---
DESIGNS=("ac97_ctrl" "aes_secworks" "aes_xcrypt" "aes" "bp_be" "des3_area" "dft" "dynamic_node" "ethernet" "fir" "fpu" "i2c" "idft" "iir" "jpeg" "mem_ctrl" "pci" "picosoc" "sasc" "sha256" "simple_spi" "spi" "ss_pcm" "tinyRocket" "tv80" "usb_phy" "vga_lcd" "wb_conmax" "wb_dma")

for design in "${DESIGNS[@]}"; do
    echo "Processing: $design"
    DESIGN_DIR="$DATA_ROOT/bench/$design"
    mkdir -p "$DESIGN_DIR"
    
    TEMP_ZIPS="$BASE_DIR/temp_zips_$design"
    7z x OPENABC_DATASET.zip "OPENABC_DATASET/bench/$design/syn*.zip" -o"$TEMP_ZIPS" -y
    
    find "$TEMP_ZIPS" -name "syn*.zip" | while read szip; do
        WORK_DIR="$BASE_DIR/work_batch_$$"
        mkdir -p "$WORK_DIR"
        unzip -q -j "$szip" "*.bench" -d "$WORK_DIR"
        
        find "$WORK_DIR" -name "*.bench" | xargs -I {} -P 16 bash -c \
            "convert_bench_to_aig {} $DESIGN_DIR"
        
        rm -rf "$WORK_DIR"
    done
    rm -rf "$TEMP_ZIPS"
done

# --- 5. FINAL SIZE REPORT ---
echo ""
echo "========================================================="
echo "      OPENABC-D PROCESSING SUMMARY REPORT"
echo "========================================================="
echo "Location: $DATA_ROOT"
echo "---------------------------------------------------------"

# Function to safely get size even if folder is empty
get_size() {
    du -sh "$1" 2>/dev/null | cut -f1 || echo "0"
}

# Counting AIG files
TOTAL_AIGS=$(find "$DATA_ROOT/bench" -name "*.aig" | wc -l)

echo "Component Sizes:"
echo "  - Libraries (lib):          $(get_size "$DATA_ROOT/lib")"
echo "  - Statistics:               $(get_size "$DATA_ROOT/statistics")"
echo "  - Synthesis Scripts:        $(get_size "$DATA_ROOT/synScripts")"
echo "  - Bench/AIG Directory:      $(get_size "$DATA_ROOT/bench")"
echo "---------------------------------------------------------"
echo "Total AIG Files Created:      $TOTAL_AIGS"
echo "---------------------------------------------------------"
echo "Total Combined Space Used:    $(du -sh "$DATA_ROOT" | cut -f1)"
echo "========================================================="
echo "Report Generated: $(date)"