#!/bin/bash
#SBATCH --job-name=opt_syn4_fix
#SBATCH --time=78:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=genoa
#SBATCH --constraint=scratch-node
#SBATCH --output=logs/8c_opt_syn4_fix_%j.out

set -euo pipefail

# --- Setup Directories & Environment ---
if [[ -n "${TMPDIR:-}" ]]; then : ; else export TMPDIR="/scratch-shared/$USER/tmp"; fi
mkdir -p "$TMPDIR"
echo "STEP 8c FIX start: job=${SLURM_JOB_ID:-local} host=$(hostname) time=$(date)"

module purge
module load 2025 foss/2025a Python/3.13.1-GCCcore-14.2.0
export PATH="$HOME/abc:$PATH"

ALGORITHM="Syn4"
SOURCE_LABEL="base"
OUTPUT_TIER="tier1"
export OPT_SCRIPT_PARALLELISM="${OPT_SCRIPT_PARALLELISM:-${SLURM_CPUS_PER_TASK:-72}}"

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
GEN_SCRIPT="${BASE_DIR}/dataset_tools/generate_optimization_bulk_scripts.py"
CONFIG_FILE="${BASE_DIR}/dataset_tools/optimization_config.json"
FULL_DATASET="/scratch-shared/$USER/FULL_DATASET"
MANIFEST_DIR="$FULL_DATASET/optimized_aigs/manifests"
SCRIPT_ZIP_ROOT="$FULL_DATASET/synScripts/optimization/$ALGORITHM"

# ==========================================
# EXPLICIT TARGET LIST
# Exactly the 15 designs that failed with 0 or 17801 files.
# Good designs (128, 256, i2c, etc.) are perfectly safe.
# ==========================================
EXPLICIT_DESIGNS="jpeg,mem_ctrl,pci,picosoc,sasc,sha256,simple_spi,spi,ss_pcm,tinyRocket,tv80,usb_phy,vga_lcd,wb_conmax,wb_dma"

echo "=========================================="
echo "PRE-RERUN CLEANUP: Clearing Broken Outputs"
echo "=========================================="
# Convert comma-separated string to an array and loop
IFS=',' read -r -a DESIGN_ARRAY <<< "$EXPLICIT_DESIGNS"

for design in "${DESIGN_ARRAY[@]}"; do
    echo "  - Wiping old outputs for $design"
    # Delete the fake summaries so the inner script actually runs
    rm -f "$FULL_DATASET/metadata/raw_logs/$design/$OUTPUT_TIER/$ALGORITHM/summary.json"
    # Delete the empty/partial zips and folders
    rm -f "$FULL_DATASET/optimized_aigs/$ALGORITHM/$OUTPUT_TIER/$design.zip"
    rm -rf "$FULL_DATASET/optimized_aigs/$ALGORITHM/$OUTPUT_TIER/$design"
done

echo ""
# --- Script Regeneration (Only for the explicitly broken designs) ---
echo "Regenerating optimization scripts for targeted designs..."
python3 "$GEN_SCRIPT" \
    --base-dir "$BASE_DIR" \
    --full-dataset "$FULL_DATASET" \
    --config "$CONFIG_FILE" \
    --design-group "all" \
    --algorithms "$ALGORITHM" \
    --input-source "base_aigs" \
    --designs "$EXPLICIT_DESIGNS"

# --- Read Manifest ---
latest_manifest="$MANIFEST_DIR/bulk_scripts_manifest.json"
designs_file="$(mktemp "${TMPDIR:-/tmp}/fix_designs_XXXXXX")"
trap 'rm -f "$designs_file"' EXIT

python3 - "$latest_manifest" > "$designs_file" <<'PY'
import json, sys
with open(sys.argv[1], "r") as fh:
    m = json.load(fh)
    for d in m.get("designs", []): print(d)
PY

processed=0

# --- Execution Loop ---
echo "=========================================="
echo "STARTING OPTIMIZATION"
echo "=========================================="
while IFS= read -r design_name; do
    [ -z "$design_name" ] && continue
    
    echo "STEP 8c: Processing design ${design_name} at $(date)"
    tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/opt_run_${design_name}_XXXXXX")"
    design_zip="$SCRIPT_ZIP_ROOT/${design_name}.zip"
    script_file="optimizeBulk_${ALGORITHM}_${design_name}_${SOURCE_LABEL}.sh"
    
    # Extract the fresh script and run it
    unzip -q "$design_zip" "$script_file" -d "$tmp_dir"
    chmod +x "$tmp_dir/$script_file"
    bash "$tmp_dir/$script_file"
    
    processed=$((processed + 1))
    echo "STEP 8c: Done ${design_name} at $(date)"
    rm -rf "$tmp_dir"
done < "$designs_file"

echo "=========================================="
echo "STEP 8c FIX COMPLETE: Processed exactly $processed targeted designs."
echo "Time: $(date)"
echo "=========================================="