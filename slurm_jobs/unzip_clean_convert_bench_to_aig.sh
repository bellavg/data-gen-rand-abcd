#!/bin/bash
#SBATCH -p genoa
#SBATCH -t 24:00:00
#SBATCH --job-name=openabc_final_process
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
echo ">> Extracting lib, statistics, and synScripts..."
7z x OPENABC_DATASET.zip -o"$BASE_DIR" -y \
    "OPENABC-D/lib/*" \
    "OPENABC-D/statistics/*" \
    "OPENABC-D/synScripts/*"

if [ -d "$BASE_DIR/OPENABC-D" ]; then
    mv "$BASE_DIR/OPENABC-D" "$DATA_ROOT"
fi

# Zip the synScripts folder to save inodes
echo ">> Consolidating synScripts into a zip..."
cd "$DATA_ROOT"
zip -r -q synScripts.zip synScripts/
rm -rf synScripts/
cd "$BASE_DIR"

# --- 3. CONVERSION ENGINE ---
convert_bench_to_aig() {
    local bench_file="$1"
    local output_dir="$2"
    local base_name=$(basename "${bench_file%.bench}")
    # ABC: read -> strash (AIG) -> write
    $ABC_BIN -c "read_bench $bench_file; strash; write $output_dir/${base_name}.aig" > /dev/null 2>&1
}
export -f convert_bench_to_aig
export ABC_BIN

# --- 4. THE DESIGN PROCESSING LOOP ---
# Get the list of all .zip archives in the bench folder
DESIGN_ZIPS=$(7z l OPENABC_DATASET.zip OPENABC-D/bench/*.zip | grep ".zip" | awk '{print $NF}')

for dzip_path in $DESIGN_ZIPS; do
    design=$(basename "$dzip_path" .zip)
    echo "------------------------------------------------"
    echo "Processing Design Archive: $design"
    
    DESIGN_DIR="$DATA_ROOT/bench/$design"
    mkdir -p "$DESIGN_DIR"
    
    # Create unique temp space for this design
    TEMP_DIR="$BASE_DIR/temp_$design"
    mkdir -p "$TEMP_DIR"
    
    # Step A: Extract the design's main zip from the 1.4TB archive
    echo "   -> Extracting $design.zip from main archive..."
    7z x OPENABC_DATASET.zip -o"$TEMP_DIR" -y "$dzip_path"
    
    # Step B: Unzip the design's contents
    # This design zip (e.g., ac97_ctrl.zip) is now at $TEMP_DIR/$dzip_path
    echo "   -> Unzipping design contents..."
    unzip -q "$TEMP_DIR/$dzip_path" -d "$TEMP_DIR/contents"
    
    # Step C: Handle original bench file (usually in the root of the design zip)
    find "$TEMP_DIR/contents" -maxdepth 1 -name "*.bench" | while read obench; do
        convert_bench_to_aig "$obench" "$DESIGN_DIR"
    done

    # Step D: Handle nested syn*.zip files (the 1,500 recipes)
    if ls "$TEMP_DIR/contents"/syn*.zip >/dev/null 2>&1; then
        echo "   -> Converting 1500 synthesis recipes to AIG..."
        find "$TEMP_DIR/contents" -name "syn*.zip" | while read szip; do
            BATCH_DIR="$BASE_DIR/batch_$$"
            mkdir -p "$BATCH_DIR"
            
            # Extract only .bench files from the syn zip
            unzip -q -j "$szip" "*.bench" -d "$BATCH_DIR"
            
            # Convert bench to AIG in parallel
            find "$BATCH_DIR" -name "*.bench" | xargs -I {} -P 16 bash -c \
                "convert_bench_to_aig {} $DESIGN_DIR"
            
            rm -rf "$BATCH_DIR"
        done
    fi
    
    # Cleanup temp space for this design
    rm -rf "$TEMP_DIR"
    echo "   -> Finished $design."
done

# --- 5. FINAL SIZE REPORT ---
echo ""
echo "========================================================="
echo "      OPENABC_DATASET FINAL SUMMARY"
echo "========================================================="
echo "Final AIG count: $(find "$DATA_ROOT/bench" -name "*.aig" | wc -l)"
echo "Total Folder Size: $(du -sh "$DATA_ROOT" | cut -f1)"
echo "========================================================="