#!/bin/bash
#SBATCH -p genoa
#SBATCH -t 24:00:00
#SBATCH --job-name=openabc_fix_zip
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
# We use -o"$BASE_DIR" which creates $BASE_DIR/OPENABC-D/
7z x OPENABC_DATASET.zip -o"$BASE_DIR" -y \
    "OPENABC-D/lib/*" \
    "OPENABC-D/statistics/*" \
    "OPENABC-D/synScripts/*"

# Rename internal folder to your preference
if [ -d "$BASE_DIR/OPENABC-D" ]; then
    echo ">> Renaming OPENABC-D to OPENABC_DATASET..."
    # If OPENABC_DATASET already exists from a previous failed run, remove it first
    rm -rf "$DATA_ROOT" 
    mv "$BASE_DIR/OPENABC-D" "$DATA_ROOT"
fi

# --- 3. CONSOLIDATE SYNSCRIPTS ---
echo ">> Consolidating synScripts into a zip..."
# Change into the data root so the zip doesn't contain the whole path
cd "$DATA_ROOT"
if [ -d "synScripts" ]; then
    # -m deletes the files after they are added to the zip (moves them into the zip)
    zip -r -q -m synScripts.zip synScripts/
    echo ">> synScripts.zip created successfully."
else
    echo ">> Warning: synScripts folder not found in $DATA_ROOT"
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
# Re-list archives to ensure fresh state
DESIGN_ZIPS=$(7z l OPENABC_DATASET.zip OPENABC-D/bench/*.zip | grep ".zip" | awk '{print $NF}')

for dzip_path in $DESIGN_ZIPS; do
    design=$(basename "$dzip_path" .zip)
    [ "$design" == "synScripts" ] && continue # Skip the metadata folder if it shows up
    
    echo "------------------------------------------------"
    echo "Processing Design Archive: $design"
    
    DESIGN_DIR="$DATA_ROOT/bench/$design"
    mkdir -p "$DESIGN_DIR"
    
    TEMP_DIR="$BASE_DIR/temp_$design"
    mkdir -p "$TEMP_DIR"
    
    echo "   -> Extracting $design.zip from main archive..."
    7z x OPENABC_DATASET.zip -o"$TEMP_DIR" -y "$dzip_path"
    
    echo "   -> Unzipping design contents..."
    # The extracted file is at $TEMP_DIR/OPENABC-D/bench/$design.zip
    unzip -q "$TEMP_DIR/OPENABC-D/bench/$design.zip" -d "$TEMP_DIR/contents"
    
    # Process original bench
    find "$TEMP_DIR/contents" -maxdepth 1 -name "*.bench" | while read obench; do
        convert_bench_to_aig "$obench" "$DESIGN_DIR"
    done

    # Process syn zips
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
    
    rm -rf "$TEMP_DIR"
    echo "   -> Finished $design."
done

echo ">> JOB COMPLETE <<"