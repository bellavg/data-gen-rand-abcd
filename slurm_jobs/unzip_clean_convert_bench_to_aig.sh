#!/bin/bash
#SBATCH -p genoa
#SBATCH -t 24:00:00
#SBATCH --job-name=openabc_resume
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/openabc_resume_%j.out

# --- 1. SETUP & MODULES ---
set -e
module purge
module load 2025 foss/2025a p7zip/17.05-GCCcore-14.2.0

BASE_DIR="/scratch-shared/$USER/openabc_full"
DATA_ROOT="$BASE_DIR/OPENABC_DATASET"
ABC_BIN="$HOME/abc/abc"

cd "$BASE_DIR"

# --- 2. SMART EXTRACTION (SKIP IF DONE) ---
# Only extract metadata if the folder doesn't exist yet
if [ ! -d "$DATA_ROOT/lib" ]; then
    echo ">> Extracting core metadata folders..."
    7z x OPENABC_DATASET.zip -o"$BASE_DIR" -y \
        "OPENABC-D/lib/*" \
        "OPENABC-D/statistics/*" \
        "OPENABC-D/synScripts/*"

    if [ -d "$BASE_DIR/OPENABC-D" ]; then
        mv "$BASE_DIR/OPENABC-D" "$DATA_ROOT"
    fi

    cd "$DATA_ROOT"
    if [ -d "synScripts" ]; then
        zip -r -q -m synScripts.zip synScripts/
    fi
    cd "$BASE_DIR"
else
    echo ">> Metadata already exists. Skipping Section 2."
fi

# --- 3. CONVERSION FUNCTION ---
convert_bench_to_aig() {
    local bench_file="$1"
    local output_dir="$2"
    local base_name=$(basename "${bench_file%.bench}")
    $ABC_BIN -c "read_bench $bench_file; strash; write $output_dir/${base_name}.aig" > /dev/null 2>&1
}
export -f convert_bench_to_aig
export ABC_BIN

# --- 4. THE RESUME PROCESSING LOOP ---
DESIGNS=(
    "ac97_ctrl" "aes_secworks" "aes_xcrypt" "aes" "bp_be" "des3_area" 
    "dft" "dynamic_node" "ethernet" "fir" "fpu" "i2c" "idft" "iir" 
    "jpeg" "mem_ctrl" "pci" "picosoc" "sasc" "sha256" "simple_spi" 
    "spi" "ss_pcm" "tinyRocket" "tv80" "usb_phy" "vga_lcd" "wb_conmax" "wb_dma"
)

for design in "${DESIGNS[@]}"; do
    DESIGN_DIR="$DATA_ROOT/bench/$design"
    
    # RESUME LOGIC: Check if this design is already finished (looking for ~31501 files)
    if [ -d "$DESIGN_DIR" ]; then
        AIG_COUNT=$(ls -1 "$DESIGN_DIR"/*.aig 2>/dev/null | wc -l)
        if [ "$AIG_COUNT" -ge 31501 ]; then
            echo ">> Skipping $design: Already complete ($AIG_COUNT AIGs)."
            continue
        fi
    fi

    echo "------------------------------------------------"
    echo "Processing Design: $design"
    mkdir -p "$DESIGN_DIR"
    
    TEMP_DIR="$BASE_DIR/temp_$design"
    rm -rf "$TEMP_DIR" # Clean any garbage from the crash
    mkdir -p "$TEMP_DIR"
    
    echo "   -> Extracting design archive..."
    7z x OPENABC_DATASET.zip -o"$TEMP_DIR" -y "OPENABC-D/bench/$design.zip"
    
    DESIGN_ZIP_PATH=$(find "$TEMP_DIR" -name "$design.zip")
    
    if [ -f "$DESIGN_ZIP_PATH" ]; then
        echo "   -> Opening design container..."
        # ADDED -o TO OVERWRITE AND PREVENT HANGING
        unzip -o -q "$DESIGN_ZIP_PATH" -d "$TEMP_DIR/contents"
        
        find "$TEMP_DIR/contents" -name "*_orig.bench" | while read obench; do
            convert_bench_to_aig "$obench" "$DESIGN_DIR"
        done

        echo "   -> Converting synthesis recipes..."
        find "$TEMP_DIR/contents" -name "syn*.zip" | while read szip; do
            BATCH_DIR="$BASE_DIR/batch_$(basename "$szip" .zip)_$$"
            mkdir -p "$BATCH_DIR"
            
            # ADDED -o HERE AS WELL
            unzip -o -q -j "$szip" "*.bench" -d "$BATCH_DIR"
            
            find "$BATCH_DIR" -name "*.bench" | xargs -I {} -P 16 bash -c \
                "convert_bench_to_aig {} $DESIGN_DIR"
            
            rm -rf "$BATCH_DIR"
        done
    fi
    
    rm -rf "$TEMP_DIR"
    FINAL_COUNT=$(ls -1 "$DESIGN_DIR"/*.aig 2>/dev/null | wc -l)
    echo "   -> Finished $design. (Success: $FINAL_COUNT AIG files created)"
done

echo ">> ALL DONE <<"