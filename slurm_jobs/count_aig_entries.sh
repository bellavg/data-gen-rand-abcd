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
total_converted_only=0
total_contains_bench=0
total_mismatched_counts=0
total_missing_indices=0
total_nonmatching=0

echo "Scanning syn*.zip files under: $FULL_DATASET_DIR/base_aigs"

for design in "${OPENABC_DESIGNS[@]}"; do
    design_dir="$FULL_DATASET_DIR/base_aigs/$design"
    [ -d "$design_dir" ] || { echo "SKIP missing design dir: $design"; continue; }

    design_zips=0
    design_aigs=0
    design_benches=0
    declare -A seen_idx=()
    nonmatching=()
    converted_only=0
    contains_bench=0
    mismatched_counts=0
    missing_count=0
    nonmatching_count=0

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

        # Classify this zip for summary stats
        if [ "$aig_count" -eq "$EXPECTED_PER_ZIP" ] && [ "$bench_count" -eq 0 ]; then
            converted_only=$((converted_only+1))
        elif [ "$bench_count" -gt 0 ]; then
            contains_bench=$((contains_bench+1))
        else
            mismatched_counts=$((mismatched_counts+1))
        fi

        # Record syn index if the zip filename matches syn<NUM>.zip
        name=$(basename "$zipfile")
        if [[ $name =~ ^syn([0-9]+)\.zip$ ]]; then
            idx=${BASH_REMATCH[1]}
            seen_idx[$idx]=1
        else
            nonmatching+=("$name")
            nonmatching_count=$((nonmatching_count+1))
        fi

    done < <(find "$design_dir" -maxdepth 1 -type f -name 'syn*.zip' -print0)

    echo "$design: zips=$design_zips aigs=$design_aigs benches=$design_benches converted_only=$converted_only with_bench=$contains_bench mismatched=$mismatched_counts"

    # Check continuity of syn indices for this design
    if [ "$design_zips" -ne "$EXPECTED_ZIPS_PER_DESIGN" ]; then
        missing_list=()
        for ((i=0;i<EXPECTED_ZIPS_PER_DESIGN;i++)); do
            if [ -z "${seen_idx[$i]:-}" ]; then
                missing_list+=("$i")
            fi
        done

        if [ ${#missing_list[@]} -gt 0 ]; then
            missing_count=${#missing_list[@]}
            total_missing_indices=$((total_missing_indices+missing_count))
            echo "  MISSING syn indices for $design: count=${missing_count} (showing up to 200)"
            printf '    %s\n' "${missing_list[@]:0:200}"
        fi
    fi

    if [ ${#nonmatching[@]} -gt 0 ]; then
        echo "  WARNING: non-matching zip names in $design: ${nonmatching[*]:0:20}"
        total_nonmatching=$((total_nonmatching+nonmatching_count))
    fi

    # Unset associative array for next iteration
    unset seen_idx
    total_converted_only=$((total_converted_only+converted_only))
    total_contains_bench=$((total_contains_bench+contains_bench))
    total_mismatched_counts=$((total_mismatched_counts+mismatched_counts))
done

echo ""
echo "SUMMARY:" 
echo "  total zips:             $total_zips"
echo "  total .aig entries:     $total_aigs"
echo "  total .bench entries:   $total_benches"
echo "  converted-only zips:    $total_converted_only"
echo "  zips still with .bench: $total_contains_bench"
echo "  mismatched-counts zips: $total_mismatched_counts"
echo "  missing syn indices:    $total_missing_indices"
echo "  non-matching zip names: $total_nonmatching"

if [ "$total_missing_indices" -ne 0 ] || [ "$total_mismatched_counts" -ne 0 ]; then
    echo "Some issues were found (missing indices or mismatched counts). See per-design lines above."
    exit 2
fi

echo "All syn*.zip files look good (expected counts and continuity)."
exit 0
