#!/bin/bash
#SBATCH -p genoa
#SBATCH -t 24:00:00
#SBATCH --job-name=openabc_bench_to_aig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/0a_openabc_bench_to_aig_%j.out

set -euo pipefail

echo "=========================================="
echo "JOB 0A: Convert OpenABC BENCH -> AIG in FULL_DATASET"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-manual_run}"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

module purge
module load 2025
module load foss/2025a
module load p7zip/17.05-GCCcore-14.2.0

for cmd in unzip zip 7z find awk wc mktemp; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: $cmd"
        exit 1
    fi
done

FULL_DATASET_DIR="${FULL_DATASET_DIR:-/scratch-shared/$USER/FULL_DATASET}"
ABC_BIN="${ABC_BIN:-$HOME/abc/abc}"
WORKERS="${SLURM_CPUS_PER_TASK:-16}"

if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [ "$WORKERS" -lt 1 ]; then
    WORKERS=8
fi

if [ ! -d "$FULL_DATASET_DIR/base_aigs" ]; then
    echo "ERROR: base_aigs not found: $FULL_DATASET_DIR/base_aigs"
    exit 1
fi

if [ ! -x "$ABC_BIN" ]; then
    echo "ERROR: ABC binary not found or not executable: $ABC_BIN"
    echo "Set ABC_BIN=/path/to/abc when submitting if needed."
    exit 1
fi

echo "FULL_DATASET: $FULL_DATASET_DIR"
echo "ABC binary: $ABC_BIN"
echo "Workers: $WORKERS"
echo ""

convert_bench_to_aig() {
    local bench_file="$1"
    local aig_file="$2"
    "$ABC_BIN" -c "read_bench $bench_file; strash; write $aig_file" >/dev/null 2>&1
}

OPENABC_DESIGNS=(
    "ac97_ctrl" "aes_secworks" "aes_xcrypt" "aes" "bp_be" "des3_area"
    "dft" "dynamic_node" "ethernet" "fir" "fpu" "i2c" "idft" "iir"
    "jpeg" "mem_ctrl" "pci" "picosoc" "sasc" "sha256" "simple_spi"
    "spi" "ss_pcm" "tinyRocket" "tv80" "usb_phy" "vga_lcd" "wb_conmax" "wb_dma"
)

RANDOM_DESIGNS=("128" "256" "512" "1024" "2048" "4096" "8192" "16384")

converted_orig=0
converted_zip=0
skipped_zip_already_aig=0
missing_design_dirs=0

echo ">> Converting per-design BENCH artifacts to AIG artifacts..."

for design in "${OPENABC_DESIGNS[@]}"; do
    design_dir="$FULL_DATASET_DIR/base_aigs/$design"

    if [ ! -d "$design_dir" ]; then
        echo "  - WARNING: missing design directory: $design_dir"
        missing_design_dirs=$((missing_design_dirs + 1))
        continue
    fi

    echo "  - Processing $design"

    orig_bench="$design_dir/${design}_orig.bench"
    orig_aig="$design_dir/${design}_orig.aig"

    if [ -f "$orig_aig" ]; then
        :
    elif [ -f "$orig_bench" ]; then
        convert_bench_to_aig "$orig_bench" "$orig_aig"
        converted_orig=$((converted_orig + 1))
        echo "    * Converted ${design}_orig.bench -> ${design}_orig.aig"
    else
        echo "    * WARNING: no orig file found (${design}_orig.aig or .bench)"
    fi

    for syn_zip in "$design_dir"/syn*.zip; do
        [ -f "$syn_zip" ] || continue

        has_aig="no"
        has_bench="no"

        if unzip -Z1 "$syn_zip" | awk 'tolower($0) ~ /\.aig$/ {found=1} END {exit found?0:1}'; then
            has_aig="yes"
        fi
        if unzip -Z1 "$syn_zip" | awk 'tolower($0) ~ /\.bench$/ {found=1} END {exit found?0:1}'; then
            has_bench="yes"
        fi

        if [ "$has_aig" = "yes" ] && [ "$has_bench" = "no" ]; then
            skipped_zip_already_aig=$((skipped_zip_already_aig + 1))
            continue
        fi

        if [ "$has_bench" = "no" ]; then
            # Unknown zip payload; leave untouched.
            continue
        fi

        tmp_root="$(mktemp -d "$FULL_DATASET_DIR/.tmp_${design}_XXXXXX")"
        tmp_bench="$tmp_root/bench"
        tmp_aig="$tmp_root/aig"
        mkdir -p "$tmp_bench" "$tmp_aig"

        # Try unzip first (common case), then fallback to 7z for odd zip layouts.
        unzip -o -q "$syn_zip" "*.bench" -d "$tmp_bench" >/dev/null 2>&1 || true

        bench_count="$(find "$tmp_bench" -type f \( -iname "*.bench" \) | wc -l)"
        if [ "$bench_count" -eq 0 ]; then
            7z x "$syn_zip" -o"$tmp_bench" -y "*.bench" >/dev/null 2>&1 || true
            bench_count="$(find "$tmp_bench" -type f \( -iname "*.bench" \) | wc -l)"
        fi

        if [ "$bench_count" -eq 0 ]; then
            echo "    * WARNING: no .bench entries found in $(basename "$syn_zip")"
            rm -rf "$tmp_root"
            continue
        fi

        find "$tmp_bench" -type f \( -iname "*.bench" \) | while read -r bench_file; do
            bench_base="$(basename "$bench_file" .bench)"
            aig_file="$tmp_aig/${bench_base}.aig"
            convert_bench_to_aig "$bench_file" "$aig_file"
        done

        aig_count="$(find "$tmp_aig" -type f -name "*.aig" | wc -l)"
        if [ "$aig_count" -ne "$bench_count" ]; then
            echo "    * ERROR: conversion mismatch for $(basename "$syn_zip"): bench=$bench_count aig=$aig_count"
            rm -rf "$tmp_root"
            exit 1
        fi

        new_zip="$tmp_root/new.zip"
        (cd "$tmp_aig" && zip -q -j "$new_zip" ./*.aig)

        mv "$new_zip" "$syn_zip"
        rm -rf "$tmp_root"

        converted_zip=$((converted_zip + 1))
    done

done

echo ""
echo ">> Verification summary"
total_design_dirs="$(find "$FULL_DATASET_DIR/base_aigs" -mindepth 1 -maxdepth 1 -type d | wc -l)"
total_openabc_orig_aig=0
total_openabc_syn_zip=0

for design in "${OPENABC_DESIGNS[@]}"; do
    design_dir="$FULL_DATASET_DIR/base_aigs/$design"
    [ -d "$design_dir" ] || continue
    [ -f "$design_dir/${design}_orig.aig" ] && total_openabc_orig_aig=$((total_openabc_orig_aig + 1))
    total_openabc_syn_zip=$((total_openabc_syn_zip + $(find "$design_dir" -maxdepth 1 -type f -name "syn*.zip" | wc -l)))
done

echo "Design folders under base_aigs: $total_design_dirs"
echo "OpenABC orig AIG files present: $total_openabc_orig_aig / 29"
echo "OpenABC syn*.zip files present: $total_openabc_syn_zip"
echo "Converted orig.bench -> orig.aig this run: $converted_orig"
echo "Converted syn*.zip bench payloads this run: $converted_zip"
echo "Skipped syn*.zip already-AIG payloads this run: $skipped_zip_already_aig"
echo "Missing OpenABC design directories: $missing_design_dirs"

if [ "$total_openabc_orig_aig" -lt 29 ]; then
    echo "WARNING: Not all OpenABC original AIGs are present yet."
fi

echo ""
echo ">> Structure verification against desired FULL_DATASET layout"
verify_failures=0

for required_dir in \
    "$FULL_DATASET_DIR/base_aigs" \
    "$FULL_DATASET_DIR/synScripts" \
    "$FULL_DATASET_DIR/metadata" \
    "$FULL_DATASET_DIR/optimized_aigs"; do
    if [ ! -d "$required_dir" ]; then
        echo "  ✗ Missing required directory: $required_dir"
        verify_failures=$((verify_failures + 1))
    fi
done

# OpenABC designs should now be in desired AIG+zip form.
for design in "${OPENABC_DESIGNS[@]}"; do
    design_dir="$FULL_DATASET_DIR/base_aigs/$design"
    if [ ! -d "$design_dir" ]; then
        echo "  ✗ $design: missing design directory"
        verify_failures=$((verify_failures + 1))
        continue
    fi

    if [ ! -f "$design_dir/${design}_orig.aig" ]; then
        echo "  ✗ $design: missing ${design}_orig.aig"
        verify_failures=$((verify_failures + 1))
    fi

    syn_zip_count="$(find "$design_dir" -maxdepth 1 -type f -name "syn*.zip" | wc -l)"
    if [ "$syn_zip_count" -ne 1500 ]; then
        echo "  ✗ $design: expected 1500 syn*.zip, found $syn_zip_count"
        verify_failures=$((verify_failures + 1))
    fi
done

# Random designs should also follow the desired orig.aig + syn*.zip layout.
for design in "${RANDOM_DESIGNS[@]}"; do
    design_dir="$FULL_DATASET_DIR/base_aigs/$design"
    if [ ! -d "$design_dir" ]; then
        echo "  ✗ $design: missing random design directory"
        verify_failures=$((verify_failures + 1))
        continue
    fi

    if [ ! -f "$design_dir/${design}_orig.aig" ]; then
        echo "  ✗ $design: missing ${design}_orig.aig"
        verify_failures=$((verify_failures + 1))
    fi

    syn_zip_count="$(find "$design_dir" -maxdepth 1 -type f -name "syn*.zip" | wc -l)"
    if [ "$syn_zip_count" -ne 1500 ]; then
        echo "  ✗ $design: expected 1500 syn*.zip, found $syn_zip_count"
        verify_failures=$((verify_failures + 1))
    fi
done

if [ "$verify_failures" -ne 0 ]; then
    echo ""
    echo "ERROR: FULL_DATASET structure verification failed with $verify_failures issue(s)."
    exit 1
fi

echo "✓ FULL_DATASET structure matches desired layout (orig.aig + syn*.zip per design)."

echo ""
echo "JOB 0A complete."
echo "End time: $(date)"
