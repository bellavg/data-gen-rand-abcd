#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --job-name=0a_sopenabc_bench_to_aig
#SBATCH --output=logs/0a_openabc_bench_to_aig_%j.out

set -euo pipefail

mkdir -p logs

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

for cmd in unzip zip 7z find awk wc mktemp xargs sed; do
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

BASE_AIGS_DIR="$FULL_DATASET_DIR/base_aigs"
LOG_DIR="$FULL_DATASET_DIR/logs"
PRECHECK_REPORT="$LOG_DIR/job0a_synzip_precheck_${SLURM_JOB_ID:-manual_run}.txt"
CONVERT_ERRORS_LOG="$FULL_DATASET_DIR/convert_errors.log"

OPENABC_DESIGNS=(
    "ac97_ctrl" "aes_secworks" "aes_xcrypt" "aes" "bp_be" "des3_area"
    "dft" "dynamic_node" "ethernet" "fir" "fpu" "i2c" "idft" "iir"
    "jpeg" "mem_ctrl" "pci" "picosoc" "sasc" "sha256" "simple_spi"
    "spi" "ss_pcm" "tinyRocket" "tv80" "usb_phy" "vga_lcd" "wb_conmax" "wb_dma"
)
RANDOM_DESIGNS=("128" "256" "512" "1024" "2048" "4096" "8192" "16384")
ALL_DESIGNS=("${OPENABC_DESIGNS[@]}" "${RANDOM_DESIGNS[@]}")

converted_orig=0
converted_zip=0
skipped_zip_already_aig=0
removed_orig_bench=0
zip_content_issues=0
missing_openabc_design_dirs=0
verify_failures=0

ZIP_AIG_COUNT=0
ZIP_BENCH_COUNT=0
ZIP_CORRUPT=0

zip_counts() {
    local zip_file="$1"
    local zip_list

    zip_list="$(unzip -Z1 "$zip_file" 2>/dev/null || true)"
    if [ -z "$zip_list" ]; then
        ZIP_AIG_COUNT=0
        ZIP_BENCH_COUNT=0
        ZIP_CORRUPT=1
        return
    fi

    ZIP_AIG_COUNT="$(printf '%s\n' "$zip_list" | awk 'tolower($0) ~ /\.aig$/ {n++} END {print n+0}')"
    ZIP_BENCH_COUNT="$(printf '%s\n' "$zip_list" | awk 'tolower($0) ~ /\.bench$/ {n++} END {print n+0}')"
    ZIP_CORRUPT=0
}

convert_bench_to_aig() {
    local bench_file="$1"
    local aig_file="$2"
    "$ABC_BIN" -c "read_bench '$bench_file'; strash; write '$aig_file'" >/dev/null 2>&1
}

precheck_syn_zips() {
    local total=0
    local bad=0

    mkdir -p "$LOG_DIR"
    : >"$PRECHECK_REPORT"

    while IFS= read -r -d '' syn_zip; do
        total=$((total + 1))
        zip_counts "$syn_zip"

        if [ "$ZIP_CORRUPT" -eq 1 ]; then
            echo "CORRUPT|$syn_zip" >>"$PRECHECK_REPORT"
            bad=$((bad + 1))
            continue
        fi

        if [ "$ZIP_BENCH_COUNT" -gt 0 ] || [ "$ZIP_AIG_COUNT" -lt 20 ] || [ "$ZIP_AIG_COUNT" -gt 21 ]; then
            echo "BAD|$syn_zip|aig=$ZIP_AIG_COUNT|bench=$ZIP_BENCH_COUNT" >>"$PRECHECK_REPORT"
            bad=$((bad + 1))
        fi
    done < <(find "$BASE_AIGS_DIR" -type f -name 'syn*.zip' -print0)

    echo ">> Pre-check: scanning existing syn*.zip payloads before conversion"
    echo "Pre-check scanned: $total syn*.zip files"
    echo "Pre-check mismatches: $bad"
    echo "Pre-check report: $PRECHECK_REPORT"
    echo ""
}

convert_syn_zip_if_needed() {
    local design="$1"
    local syn_zip="$2"
    local tmp_root tmp_bench tmp_aig pairs_file convert_script new_zip
    local bench_count aig_count

    zip_counts "$syn_zip"

    if [ "$ZIP_CORRUPT" -eq 1 ]; then
        echo "    * WARNING: cannot read zip payload: $(basename "$syn_zip")"
        zip_content_issues=$((zip_content_issues + 1))
        return
    fi

    if [ "$ZIP_BENCH_COUNT" -eq 0 ] && [ "$ZIP_AIG_COUNT" -gt 0 ]; then
        if [ "$ZIP_AIG_COUNT" -lt 20 ] || [ "$ZIP_AIG_COUNT" -gt 21 ]; then
            echo "    * WARNING: $(basename "$syn_zip") has $ZIP_AIG_COUNT .aig files (expected 20-21)"
            zip_content_issues=$((zip_content_issues + 1))
        fi
        skipped_zip_already_aig=$((skipped_zip_already_aig + 1))
        return
    fi

    if [ "$ZIP_BENCH_COUNT" -eq 0 ]; then
        return
    fi

    tmp_root="$(mktemp -d "$FULL_DATASET_DIR/.tmp_${design}_XXXXXX")"
    tmp_bench="$tmp_root/bench"
    tmp_aig="$tmp_root/aig"
    mkdir -p "$tmp_bench" "$tmp_aig" "$tmp_root/abc_logs"

    unzip -o -q "$syn_zip" "*.bench" -d "$tmp_bench" >/dev/null 2>&1 || true
    bench_count="$(find "$tmp_bench" -type f -iname '*.bench' | wc -l)"
    if [ "$bench_count" -eq 0 ]; then
        7z x "$syn_zip" -o"$tmp_bench" -y "*.bench" >/dev/null 2>&1 || true
        bench_count="$(find "$tmp_bench" -type f -iname '*.bench' | wc -l)"
    fi

    if [ "$bench_count" -eq 0 ]; then
        echo "    * WARNING: no .bench entries found in $(basename "$syn_zip")"
        rm -rf "$tmp_root"
        return
    fi

    pairs_file="$tmp_root/pairs0"
    convert_script="$tmp_root/convert_one.sh"
    : >"$pairs_file"

    while IFS= read -r -d '' bench_file; do
        bench_base="$(basename "$bench_file" .bench)"
        aig_file="$tmp_aig/${bench_base}.aig"
        printf '%s\0%s\0' "$bench_file" "$aig_file" >>"$pairs_file"
    done < <(find "$tmp_bench" -type f -iname '*.bench' -print0)

    export ABC_BIN
    export TMP_ROOT="$tmp_root"

    cat >"$convert_script" <<'SH'
#!/bin/bash
bench="$1"
aig="$2"
logfile="$TMP_ROOT/abc_logs/$(basename "$bench").log"
"$ABC_BIN" -c "read_bench '$bench'; strash; write '$aig'" >"$logfile" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
    printf '%s %s %d\n' "$(date '+%Y-%m-%dT%H:%M:%S')" "$bench" "$rc" >>"$TMP_ROOT/failed.lst"
fi
exit 0
SH
    chmod +x "$convert_script"

    if [ -s "$pairs_file" ]; then
        xargs -0 -n2 -P "$WORKERS" "$convert_script" <"$pairs_file" || true
    fi

    if [ -s "$tmp_root/failed.lst" ]; then
        echo "FAILED conversions in $(basename "$syn_zip") (design=$design):" >>"$CONVERT_ERRORS_LOG"
        sed -n '1,100p' "$tmp_root/failed.lst" >>"$CONVERT_ERRORS_LOG"
        echo "logs at: $tmp_root/abc_logs" >>"$CONVERT_ERRORS_LOG"
    fi

    aig_count="$(find "$tmp_aig" -type f -name '*.aig' | wc -l)"
    if [ "$aig_count" -ne "$bench_count" ]; then
        echo "ERROR: conversion mismatch for $(basename "$syn_zip"): bench=$bench_count aig=$aig_count"
        rm -rf "$tmp_root"
        exit 1
    fi

    new_zip="$tmp_root/new.zip"
    (cd "$tmp_aig" && find . -type f -name '*.aig' -print0 | xargs -0 zip -q -j "$new_zip")

    zip_counts "$new_zip"
    if [ "$ZIP_CORRUPT" -eq 1 ] || [ "$ZIP_BENCH_COUNT" -ne 0 ] || [ "$ZIP_AIG_COUNT" -lt 20 ] || [ "$ZIP_AIG_COUNT" -gt 21 ]; then
        echo "ERROR: invalid rebuilt zip $(basename "$syn_zip") (.aig=$ZIP_AIG_COUNT, .bench=$ZIP_BENCH_COUNT)"
        rm -rf "$tmp_root"
        exit 1
    fi

    mv "$new_zip" "$syn_zip"
    rm -rf "$tmp_root"
    converted_zip=$((converted_zip + 1))
}

convert_openabc_design() {
    local design="$1"
    local design_dir="$BASE_AIGS_DIR/$design"
    local orig_bench orig_aig

    if [ ! -d "$design_dir" ]; then
        echo "  - WARNING: missing design directory: $design_dir"
        missing_openabc_design_dirs=$((missing_openabc_design_dirs + 1))
        return
    fi

    echo "  - Processing $design"

    orig_bench="$design_dir/${design}_orig.bench"
    orig_aig="$design_dir/${design}_orig.aig"

    if [ ! -f "$orig_aig" ] && [ -f "$orig_bench" ]; then
        convert_bench_to_aig "$orig_bench" "$orig_aig"
        converted_orig=$((converted_orig + 1))
        echo "    * Converted ${design}_orig.bench -> ${design}_orig.aig"
    elif [ ! -f "$orig_aig" ] && [ ! -f "$orig_bench" ]; then
        echo "    * WARNING: no orig file found (${design}_orig.aig or .bench)"
    fi

    if [ -f "$orig_aig" ] && [ -f "$orig_bench" ]; then
        rm -f "$orig_bench"
        removed_orig_bench=$((removed_orig_bench + 1))
        echo "    * Removed ${design}_orig.bench (AIG exists)"
    fi

    for syn_zip in "$design_dir"/syn*.zip; do
        [ -f "$syn_zip" ] || continue
        convert_syn_zip_if_needed "$design" "$syn_zip"
    done
}

verify_design_set() {
    local designs=("$@")
    local design_dir syn_zip_count

    for design in "${designs[@]}"; do
        design_dir="$BASE_AIGS_DIR/$design"

        if [ ! -d "$design_dir" ]; then
            echo "  ✗ $design: missing design directory"
            verify_failures=$((verify_failures + 1))
            continue
        fi

        if [ ! -f "$design_dir/${design}_orig.aig" ]; then
            echo "  ✗ $design: missing ${design}_orig.aig"
            verify_failures=$((verify_failures + 1))
        fi

        syn_zip_count="$(find "$design_dir" -maxdepth 1 -type f -name 'syn*.zip' | wc -l)"
        if [ "$syn_zip_count" -ne 1500 ]; then
            echo "  ✗ $design: expected 1500 syn*.zip, found $syn_zip_count"
            verify_failures=$((verify_failures + 1))
        fi
    done
}

deep_verify_syn_payloads() {
    local design design_dir syn_zip

    echo ""
    echo ">> Deep verification: each syn*.zip should contain 20-21 .aig files and zero .bench files"

    for design in "${ALL_DESIGNS[@]}"; do
        design_dir="$BASE_AIGS_DIR/$design"
        [ -d "$design_dir" ] || continue

        for syn_zip in "$design_dir"/syn*.zip; do
            [ -f "$syn_zip" ] || continue
            zip_counts "$syn_zip"

            if [ "$ZIP_CORRUPT" -eq 1 ] || [ "$ZIP_BENCH_COUNT" -ne 0 ] || [ "$ZIP_AIG_COUNT" -lt 20 ] || [ "$ZIP_AIG_COUNT" -gt 21 ]; then
                echo "  ✗ $design/$(basename "$syn_zip"): .aig=$ZIP_AIG_COUNT .bench=$ZIP_BENCH_COUNT"
                verify_failures=$((verify_failures + 1))
            fi
        done
    done
}

if [ ! -d "$BASE_AIGS_DIR" ]; then
    echo "ERROR: base_aigs not found: $BASE_AIGS_DIR"
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

precheck_syn_zips

echo ">> Converting per-design BENCH artifacts to AIG artifacts..."
for design in "${OPENABC_DESIGNS[@]}"; do
    convert_openabc_design "$design"
done

echo ""
echo ">> Verification summary"
total_design_dirs="$(find "$BASE_AIGS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
total_openabc_orig_aig=0
total_openabc_syn_zip=0
for design in "${OPENABC_DESIGNS[@]}"; do
    design_dir="$BASE_AIGS_DIR/$design"
    [ -d "$design_dir" ] || continue
    [ -f "$design_dir/${design}_orig.aig" ] && total_openabc_orig_aig=$((total_openabc_orig_aig + 1))
    total_openabc_syn_zip=$((total_openabc_syn_zip + $(find "$design_dir" -maxdepth 1 -type f -name 'syn*.zip' | wc -l)))
done

echo "Design folders under base_aigs: $total_design_dirs"
echo "OpenABC orig AIG files present: $total_openabc_orig_aig / 29"
echo "OpenABC syn*.zip files present: $total_openabc_syn_zip"
echo "Converted orig.bench -> orig.aig this run: $converted_orig"
echo "Removed orig.bench files this run: $removed_orig_bench"
echo "Converted syn*.zip bench payloads this run: $converted_zip"
echo "Skipped syn*.zip already-AIG payloads this run: $skipped_zip_already_aig"
echo "Syn zip content issues detected this run: $zip_content_issues"
echo "Missing OpenABC design directories: $missing_openabc_design_dirs"

if [ "$total_openabc_orig_aig" -lt 29 ]; then
    echo "WARNING: Not all OpenABC original AIGs are present yet."
fi

echo ""
echo ">> Structure verification against desired FULL_DATASET layout"
for required_dir in "$BASE_AIGS_DIR" "$FULL_DATASET_DIR/synScripts" "$FULL_DATASET_DIR/metadata"; do
    if [ ! -d "$required_dir" ]; then
        echo "  ✗ Missing required directory: $required_dir"
        verify_failures=$((verify_failures + 1))
    fi
done

verify_design_set "${OPENABC_DESIGNS[@]}"
verify_design_set "${RANDOM_DESIGNS[@]}"
deep_verify_syn_payloads

if [ "$verify_failures" -ne 0 ]; then
    echo ""
    echo "ERROR: FULL_DATASET structure verification failed with $verify_failures issue(s)."
    exit 1
fi

echo "✓ FULL_DATASET structure matches desired layout (orig.aig + syn*.zip per design)."

echo ""
echo "JOB 0A complete."
echo "End time: $(date)"
