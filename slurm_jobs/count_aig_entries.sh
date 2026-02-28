#!/bin/bash
#SBATCH -p genoa
#SBATCH -t 02:00:00
#SBATCH --job-name=count_aig_entries
#SBATCH --output=logs/count_aig_entries_%j.out

set -euo pipefail

# Default dataset dir; override when submitting with --export or env var
FULL_DATASET_DIR="${FULL_DATASET_DIR:-/scratch-shared/$USER/FULL_DATASET}"

OPENABC_DESIGNS=(
    "ac97_ctrl" "aes_secworks" "aes_xcrypt" "aes" "bp_be" "des3_area"
    "dft" "dynamic_node" "ethernet" "fir" "fpu" "i2c" "idft" "iir"
    "jpeg" "mem_ctrl" "pci" "picosoc" "sasc" "sha256" "simple_spi"
    "spi" "ss_pcm" "tinyRocket" "tv80" "usb_phy" "vga_lcd" "wb_conmax" "wb_dma"
)

EXPECTED_PER_ZIP=21
EXPECTED_ZIPS_PER_DESIGN=1500

total_zips=0
total_aigs=0
total_benches=0
problem_zips=0

echo "Scanning syn*.zip files under: $FULL_DATASET_DIR/base_aigs"

for design in "${OPENABC_DESIGNS[@]}"; do
    design_dir="$FULL_DATASET_DIR/base_aigs/$design"
    [ -d "$design_dir" ] || { echo "SKIP missing design dir: $design"; continue; }

    design_zips=0
    design_aigs=0
    design_benches=0
    declare -A seen_idx=()
    nonmatching=()

    while IFS= read -r -d '' zipfile; do
        design_zips=$((design_zips+1))
        total_zips=$((total_zips+1))

        # Count .aig and .bench entries (case-insensitive)
        aig_count=$(unzip -Z1 "$zipfile" 2>/dev/null | awk 'tolower($0) ~ /\.aig$/ {c++} END {print c+0}')
        bench_count=$(unzip -Z1 "$zipfile" 2>/dev/null | awk 'tolower($0) ~ /\.bench$/ {c++} END {print c+0}')

        design_aigs=$((design_aigs + aig_count))
        design_benches=$((design_benches + bench_count))
        total_aigs=$((total_aigs + aig_count))
        total_benches=$((total_benches + bench_count))

        if [ "$aig_count" -ne "$EXPECTED_PER_ZIP" ] || [ "$aig_count" -ne "$bench_count" ]; then
            problem_zips=$((problem_zips+1))
            echo "PROBLEM: $(basename "$zipfile") (design=$design): aig=$aig_count bench=$bench_count"
        fi

        # Record syn index if the zip filename matches syn<NUM>.zip
        name=$(basename "$zipfile")
        if [[ $name =~ ^syn([0-9]+)\.zip$ ]]; then
            idx=${BASH_REMATCH[1]}
            seen_idx[$idx]=1
        else
            nonmatching+=("$name")
        fi

    done < <(find "$design_dir" -maxdepth 1 -type f -name 'syn*.zip' -print0)

    echo "$design: zips=$design_zips aigs=$design_aigs benches=$design_benches"

    # Check continuity of syn indices for this design
    if [ "$design_zips" -ne "$EXPECTED_ZIPS_PER_DESIGN" ]; then
        missing_list=()
        for ((i=0;i<EXPECTED_ZIPS_PER_DESIGN;i++)); do
            if [ -z "${seen_idx[$i]:-}" ]; then
                missing_list+=("$i")
            fi
        done

        if [ ${#missing_list[@]} -gt 0 ]; then
            problem_zips=$((problem_zips+1))
            echo "  MISSING syn indices for $design: count=${#missing_list[@]}"
            # Print first 200 missing indices for brevity
            printf '    %s\n' "${missing_list[@]:0:200}"
        fi
    fi

    if [ ${#nonmatching[@]} -gt 0 ]; then
        echo "  WARNING: non-matching zip names in $design: ${nonmatching[*]:0:20}"
    fi

    # Unset associative array for next iteration
    unset seen_idx
done

echo ""
echo "TOTAL: zips=$total_zips aigs=$total_aigs benches=$total_benches problem_zips=$problem_zips"

if [ "$problem_zips" -ne 0 ]; then
    echo "Some syn*.zip files are missing expected AIG entries or have mismatched bench counts."
    exit 2
fi

echo "All syn*.zip files contain $EXPECTED_PER_ZIP .aig entries and match .bench counts."
exit 0
