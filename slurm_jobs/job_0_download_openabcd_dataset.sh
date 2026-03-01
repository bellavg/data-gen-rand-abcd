#!/bin/bash
#SBATCH -p staging
#SBATCH -t 36:00:00
#SBATCH --job-name=openabc_to_fulldataset_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/openabc_to_fulldataset_bench_%j.out

set -euo pipefail

echo "=========================================="
echo "JOB 0: Download OpenABC-D into FULL_DATASET (BENCH mode)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-manual_run}"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

module purge
module load 2025
module load foss/2025a
module load p7zip/17.05-GCCcore-14.2.0

for cmd in curl 7z unzip find awk wc stat rsync zip; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: $cmd"
        exit 1
    fi
done

# Layout targets
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/scratch-shared/$USER/openabc_download}"
FULL_DATASET_DIR="${FULL_DATASET_DIR:-/scratch-shared/$USER/FULL_DATASET}"
TMP_EXTRACT_DIR="$DOWNLOAD_DIR/_tmp_extract"

# API Variables (NYU UltraViolet)
RECORD_ID="mw6q2-a8p15"
BASE_URL="https://ultraviolet.library.nyu.edu/api/records/$RECORD_ID/files"

mkdir -p "$DOWNLOAD_DIR" "$FULL_DATASET_DIR" "$TMP_EXTRACT_DIR"
mkdir -p "$FULL_DATASET_DIR/base_aigs" "$FULL_DATASET_DIR/metadata" "$FULL_DATASET_DIR/metadata/library" "$FULL_DATASET_DIR/metadata/openabc_raw_statistics" "$FULL_DATASET_DIR/synScripts"

echo "Download dir: $DOWNLOAD_DIR"
echo "FULL_DATASET dir: $FULL_DATASET_DIR"
echo ""

cd "$DOWNLOAD_DIR"

download_file() {
    local file_name="$1"
    local min_size_bytes="$2"

    if [ -f "$file_name" ] && [ "$(stat -c%s "$file_name")" -gt "$min_size_bytes" ]; then
        echo ">> $file_name already exists and looks complete. Skipping."
        return
    fi

    echo ">> Downloading $file_name ..."
    until curl -fL -X GET "$BASE_URL/$file_name/content" -o "$file_name"; do
        echo "!! Download failed for $file_name. Removing partial file and retrying in 15s..."
        rm -f "$file_name"
        sleep 15
    done
    echo ">> Downloaded $file_name"
}

echo ">> Step 1/5: Download multipart archive"
for i in {01..13}; do
    download_file "OPENABC_DATASET.z$i" 90000000000
done
download_file "OPENABC_DATASET.zip" 100000000

echo ""
echo ">> Step 2/5: Extract selected folders from master archive"
rm -rf "$TMP_EXTRACT_DIR"
mkdir -p "$TMP_EXTRACT_DIR"

extract_with_root() {
    local root_name="$1"
    7z x OPENABC_DATASET.zip -o"$TMP_EXTRACT_DIR" -y \
        "$root_name/bench/*.zip" \
        "$root_name/statistics/*" \
        "$root_name/lib/*" \
        "$root_name/synScripts/*" \
        "$root_name/synScripts.zip" \
        >/dev/null 2>&1 || true
}

extract_with_root "OPENABC_DATASET"
if [ ! -d "$TMP_EXTRACT_DIR/OPENABC_DATASET" ]; then
    extract_with_root "OPENABC-D"
fi

EXTRACTED_ROOT=""
if [ -d "$TMP_EXTRACT_DIR/OPENABC_DATASET" ]; then
    EXTRACTED_ROOT="$TMP_EXTRACT_DIR/OPENABC_DATASET"
elif [ -d "$TMP_EXTRACT_DIR/OPENABC-D" ]; then
    EXTRACTED_ROOT="$TMP_EXTRACT_DIR/OPENABC-D"
else
    echo "ERROR: Could not find extracted OpenABC root (OPENABC_DATASET or OPENABC-D)."
    exit 1
fi

echo ">> Using extracted root: $EXTRACTED_ROOT"

echo ""
echo ">> Step 3/5: Populate FULL_DATASET/base_aigs with BENCH-mode design folders"

BENCH_ZIP_DIR="$EXTRACTED_ROOT/bench"
if [ ! -d "$BENCH_ZIP_DIR" ]; then
    echo "ERROR: Bench folder not found under extracted root: $BENCH_ZIP_DIR"
    exit 1
fi

DESIGN_ZIP_COUNT=0
for design_zip in "$BENCH_ZIP_DIR"/*.zip; do
    [ -f "$design_zip" ] || continue
    DESIGN_ZIP_COUNT=$((DESIGN_ZIP_COUNT + 1))

    design_name="$(basename "$design_zip" .zip)"
    design_target="$FULL_DATASET_DIR/base_aigs/$design_name"
    mkdir -p "$design_target"

    if [ -f "$design_target/${design_name}_orig.bench" ] && [ "$(find "$design_target" -maxdepth 1 -name 'syn*.zip' | wc -l)" -ge 1500 ]; then
        echo "  - $design_name already populated. Skipping."
        continue
    fi

    tmp_design_dir="$TMP_EXTRACT_DIR/_design_${design_name}"
    rm -rf "$tmp_design_dir"
    mkdir -p "$tmp_design_dir"

    unzip -o -q "$design_zip" -d "$tmp_design_dir"

    find "$tmp_design_dir" -type f -name "*_orig.bench" | while read -r obench; do
        cp -f "$obench" "$design_target/${design_name}_orig.bench"
    done

    find "$tmp_design_dir" -type f -name "syn*.zip" | while read -r synzip; do
        cp -f "$synzip" "$design_target/$(basename "$synzip")"
    done

    orig_count="$(find "$design_target" -maxdepth 1 -name '*_orig.bench' | wc -l)"
    syn_count="$(find "$design_target" -maxdepth 1 -name 'syn*.zip' | wc -l)"
    echo "  - $design_name populated: orig.bench=$orig_count, syn.zip=$syn_count"

    rm -rf "$tmp_design_dir"
done

if [ "$DESIGN_ZIP_COUNT" -eq 0 ]; then
    echo "ERROR: No design zip archives found in $BENCH_ZIP_DIR"
    exit 1
fi

echo ""
echo ">> Step 4/5: Copy OpenABC metadata/library/synScripts as raw sources"

if [ -d "$EXTRACTED_ROOT/statistics" ]; then
    rsync -a --delete "$EXTRACTED_ROOT/statistics/" "$FULL_DATASET_DIR/metadata/openabc_raw_statistics/"
    echo "  - Copied statistics -> metadata/openabc_raw_statistics"
else
    echo "  - WARNING: statistics folder missing in extracted root"
fi

if [ -d "$EXTRACTED_ROOT/lib" ]; then
    rsync -a "$EXTRACTED_ROOT/lib/" "$FULL_DATASET_DIR/metadata/library/"
    echo "  - Copied library files -> metadata/library"
fi

if [ -f "$EXTRACTED_ROOT/synScripts.zip" ]; then
    cp -f "$EXTRACTED_ROOT/synScripts.zip" "$FULL_DATASET_DIR/synScripts/openabc_synScripts_master.zip"
    echo "  - Copied synScripts.zip -> synScripts/openabc_synScripts_master.zip"
elif [ -d "$EXTRACTED_ROOT/synScripts" ]; then
    tmp_syn_zip="$FULL_DATASET_DIR/synScripts/openabc_synScripts_master.zip"
    (cd "$EXTRACTED_ROOT" && zip -r -q "$tmp_syn_zip" synScripts)
    echo "  - Packed synScripts directory -> synScripts/openabc_synScripts_master.zip"
else
    echo "  - WARNING: No synScripts source found"
fi

echo ""
echo ">> Step 5/5: Final summary"
design_count="$(find "$FULL_DATASET_DIR/base_aigs" -mindepth 1 -maxdepth 1 -type d | wc -l)"
orig_bench_count="$(find "$FULL_DATASET_DIR/base_aigs" -name '*_orig.bench' | wc -l)"
syn_zip_count="$(find "$FULL_DATASET_DIR/base_aigs" -name 'syn*.zip' | wc -l)"

echo "FULL_DATASET/base_aigs design folders: $design_count"
echo "FULL_DATASET/base_aigs *_orig.bench files: $orig_bench_count"
echo "FULL_DATASET/base_aigs syn*.zip files: $syn_zip_count"

echo ""
echo ">> Verification: Checking expected OpenABC design completeness"

OPENABC_DESIGNS=(
    "ac97_ctrl" "aes_secworks" "aes_xcrypt" "aes" "bp_be" "des3_area"
    "dft" "dynamic_node" "ethernet" "fir" "fpu" "i2c" "idft" "iir"
    "jpeg" "mem_ctrl" "pci" "picosoc" "sasc" "sha256" "simple_spi"
    "spi" "ss_pcm" "tinyRocket" "tv80" "usb_phy" "vga_lcd" "wb_conmax" "wb_dma"
)

verify_failures=0

for design in "${OPENABC_DESIGNS[@]}"; do
    design_dir="$FULL_DATASET_DIR/base_aigs/$design"
    if [ ! -d "$design_dir" ]; then
        echo "  ✗ $design: missing design directory"
        verify_failures=$((verify_failures + 1))
        continue
    fi

    orig_file="$design_dir/${design}_orig.bench"
    if [ ! -f "$orig_file" ]; then
        echo "  ✗ $design: missing ${design}_orig.bench"
        verify_failures=$((verify_failures + 1))
        continue
    fi

    syn_zip_present=0
    for recipe_id in $(seq 0 1499); do
        if [ -f "$design_dir/syn${recipe_id}.zip" ]; then
            syn_zip_present=$((syn_zip_present + 1))
        fi
    done

    if [ "$syn_zip_present" -ne 1500 ]; then
        echo "  ✗ $design: expected 1500 syn*.zip, found $syn_zip_present"
        verify_failures=$((verify_failures + 1))
        continue
    fi

    echo "  ✓ $design: orig.bench present and 1500 syn*.zip found"
done

if [ "$verify_failures" -ne 0 ]; then
    echo ""
    echo "ERROR: Verification failed for $verify_failures design(s)."
    echo "OpenABC BENCH layout in FULL_DATASET is incomplete."
    exit 1
fi

echo "✓ Verification passed for all 29 OpenABC designs."

echo ""
echo "Done. OpenABC-D is now downloaded and laid out in FULL_DATASET BENCH mode (no AIG conversion)."
echo "End time: $(date)"
