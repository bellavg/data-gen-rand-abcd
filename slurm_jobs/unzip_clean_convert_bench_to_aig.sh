#!/bin/bash
#SBATCH -p genoa
#SBATCH -t 24:00:00
#SBATCH --job-name=openabc_final_fixed
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

# --- 2. CORE METADATA EXTRACTION ---
echo ">> Extracting core metadata folders..."
7z x OPENABC_DATASET.zip -o"$BASE_DIR" -y \
    "OPENABC-D/lib/*" \
    "OPENABC-D/statistics/*" \
    "OPENABC-D/synScripts/*"

if [ -d "$BASE_DIR/OPENABC-D" ]; then
    echo ">> Renaming OPENABC-D to OPENABC_DATASET..."
    rm -rf "$DATA_ROOT"
    mv "$BASE_DIR/OPENABC-D" "$DATA_ROOT"
fi

# --- 3. CONSOLIDATE SYNSCRIPTS ---
echo ">> Consolidating synScripts into a zip..."
cd "$DATA_ROOT"
if [ -d "synScripts" ]; then
    zip -r -q -m synScripts.zip synScripts/
    echo ">> synScripts.zip created successfully."
fi
cd "$BASE_DIR"

# --- 4. CONVERSION ENGINE ---
convert_bench_to_aig() {
    local bench_file="$1"
    local output_dir="$2"
    local base_name=$(basename "${bench_file%.bench}")
    $ABC_BIN -c "read_bench $bench_file; strash; write $output_dir/${base_name}.aig" > /dev/null 2>&1
}
export -f convert_bench_to_aig
export ABC_BIN

# --- 5. THE DESIGN PROCESSING LOOP ---
# Using a hardcoded list to avoid the "x64)" header parsing error
DESIGNS=(
    "ac97_ctrl" "aes_secworks" "aes_xcrypt" "aes" "bp_be" "des3_area" 
    "dft" "dynamic_node" "ethernet" "fir" "fpu" "i2c" "idft" "iir" 
    "jpeg" "mem_ctrl" "pci" "picosoc" "sasc" "sha256" "simple_spi" 
    "spi" "ss_pcm" "tinyRocket" "tv80" "usb_phy" "vga_lcd" "wb_conmax" "wb_dma"
)

for design in "${DESIGNS[@]}"; do
    echo "------------------------------------------------"
    echo "Processing Design: $design"
    
    DESIGN_DIR="$DATA_ROOT/bench/$design"
    mkdir -p "$DESIGN_DIR"
    
    TEMP_DIR="$BASE_DIR/temp_$design"
    mkdir -p "$TEMP_DIR"
    
    echo "   -> Extracting OPENABC-D/bench/$design.zip from main archive..."
    # We extract the specific zip for this design
    7z x OPENABC_DATASET.zip -o"$TEMP_DIR" -y "OPENABC-D/bench/$design.zip"
    
    DESIGN_ZIP_PATH="$TEMP_DIR/OPENABC-D/bench/$design.zip"
    
    # Check if the file actually exists before trying to unzip it
    if [ -f "$DESIGN_ZIP_PATH" ]; then
        echo "   -> Unzipping design contents..."
        unzip -q "$DESIGN_ZIP_PATH" -d "$TEMP_DIR/contents"
        
        # 1. Process original bench file
        find "$TEMP_DIR/contents" -maxdepth 1 -name "*.bench" | while read obench; do
            convert_bench_to_aig "$obench" "$DESIGN_DIR"
        done

        # 2. Process nested synthesis zips
        if ls "$TEMP_DIR/contents"/syn*.zip >/dev/null 2>&1; then
            echo "   -> Converting 1500 synthesis recipes to AIG..."
            find "$TEMP_DIR/contents" -name "syn*.zip" | while read szip; do
                BATCH_DIR="$BASE_DIR/batch_$$"
                mkdir -p "$BATCH_DIR"
                unzip -q -j "$szip" "*.bench" -d "$BATCH_DIR"
                find "$BATCH_DIR" -name "*.bench" | xargs -I {} -P 16 bash -c \
                    "convert_bench_to_aig {} $DESIGN_DIR"
                rm -rf "$BATCH_DIR"
            done
        fi
    else
        echo "!! Warning: Could not find $DESIGN_ZIP_PATH after extraction."
    fi
    
    # Cleanup temp space
    rm -rf "$TEMP_DIR"
    echo "   -> Finished $design."
done

echo ">> ALL DONE <<"