#!/bin/bash
#SBATCH --job-name=opt_syn4
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/opt_syn4_%j.out

# Step 8c: Run Syn4 Optimization
# Executes generated bulk script for Syn4.

set -e

echo "=========================================="
echo "STEP 8c: Running Syn4 Optimization"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0
export PATH="$HOME/abc:$PATH"

FULL_DATASET="${FULL_DATASET:-/scratch-shared/$USER/FULL_DATASET}"
SCRIPT_ZIP_ROOT="${FULL_DATASET}/synScripts/optimization"

TIER="${TIER:-}"
INPUT_SOURCE="${INPUT_SOURCE:-}"
DRY_RUN="${DRY_RUN:-true}"
DESIGN_FILTER="${DESIGN_FILTER:-}"

if [ -n "$TIER" ] && [ "$TIER" != "1" ] && [ "$TIER" != "2" ] && [ "$TIER" != "3" ]; then
    echo "✗ ERROR: Invalid TIER=$TIER (must be 1, 2, or 3)"
    exit 1
fi

if [ -n "$INPUT_SOURCE" ] && [ "$INPUT_SOURCE" != "base_aigs" ] && [ "$INPUT_SOURCE" != "tier1" ] && [ "$INPUT_SOURCE" != "tier2" ]; then
    echo "✗ ERROR: Invalid INPUT_SOURCE=$INPUT_SOURCE (must be base_aigs|tier1|tier2)"
    exit 1
fi

if [ -n "$TIER" ]; then
    case "$TIER" in
        1) INPUT_SOURCE="base_aigs" ;;
        2) INPUT_SOURCE="tier1" ;;
        3) INPUT_SOURCE="tier2" ;;
    esac
fi

if [ -z "$INPUT_SOURCE" ]; then
    INPUT_SOURCE="base_aigs"
fi

if [ "$INPUT_SOURCE" = "base_aigs" ]; then
    SOURCE_LABEL="base"
elif [ "$INPUT_SOURCE" = "tier1" ]; then
    SOURCE_LABEL="tier1"
else
    SOURCE_LABEL="tier2"
fi

echo "Configuration:"
echo "  FULL_DATASET: $FULL_DATASET"
echo "  Script zip root: $SCRIPT_ZIP_ROOT"
echo "  TIER override: ${TIER:-<none>}"
echo "  INPUT_SOURCE: $INPUT_SOURCE"
echo "  DRY_RUN:      $DRY_RUN"
echo "  DESIGN_FILTER:${DESIGN_FILTER:-<none>}"
echo "  ARRAY_TASK_ID:${SLURM_ARRAY_TASK_ID:-<none>}"
echo ""

if [ ! -d "$SCRIPT_ZIP_ROOT" ]; then
    echo "✗ ERROR: Missing generated script zip root: $SCRIPT_ZIP_ROOT"
    echo "Run slurm_jobs/job_8_make_optimize_scripts.sh first."
    exit 1
fi

if ! command -v abc >/dev/null 2>&1 && [ "$DRY_RUN" != "true" ]; then
    echo "✗ ERROR: abc not found in PATH"
    exit 1
fi

script_count=$(find "$SCRIPT_ZIP_ROOT" -type f -name '*.zip' -exec sh -c 'for z in "$@"; do unzip -Z1 "$z" "$0" 2>/dev/null; done' "optimizeBulk_Syn4_*_${SOURCE_LABEL}.sh" {} + | wc -l | tr -d ' ')
if [ "$script_count" -eq 0 ]; then
    echo "✗ ERROR: No Syn4 shard scripts found in $SCRIPT_ZIP_ROOT for input source $INPUT_SOURCE"
    exit 1
fi

echo "Found ${script_count} shard scripts"

mapfile -t design_zips < <(find "$SCRIPT_ZIP_ROOT" -type f -name '*.zip' | sort)
if [ ${#design_zips[@]} -eq 0 ]; then
    echo "✗ ERROR: No design zip bundles found in $SCRIPT_ZIP_ROOT"
    exit 1
fi

if [ -n "$DESIGN_FILTER" ]; then
    filtered=()
    for z in "${design_zips[@]}"; do
        if [ "$(basename "$z")" = "${DESIGN_FILTER}.zip" ]; then
            filtered+=("$z")
        fi
    done
    design_zips=("${filtered[@]}")
    if [ ${#design_zips[@]} -eq 0 ]; then
        echo "✗ ERROR: DESIGN_FILTER did not match any zip: ${DESIGN_FILTER}.zip"
        exit 1
    fi
fi

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    if ! [[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]]; then
        echo "✗ ERROR: SLURM_ARRAY_TASK_ID must be numeric (got: ${SLURM_ARRAY_TASK_ID})"
        exit 1
    fi
    idx=$((SLURM_ARRAY_TASK_ID - 1))
    if [ "$idx" -lt 0 ] || [ "$idx" -ge "${#design_zips[@]}" ]; then
        echo "✗ ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for ${#design_zips[@]} design zip(s)"
        exit 1
    fi
    design_zips=("${design_zips[$idx]}")
fi

echo "Design zip bundles to process: ${#design_zips[@]}"

for design_zip in "${design_zips[@]}"; do
    tmp_extract_dir="$(mktemp -d "${TMPDIR:-/tmp}/opt_syn4_${SLURM_JOB_ID:-local}_XXXXXX")"

    if unzip -q "$design_zip" "optimizeBulk_Syn4_*_${SOURCE_LABEL}.sh" -d "$tmp_extract_dir" 2>/dev/null; then
        find "$tmp_extract_dir" -type f -name 'optimizeBulk_Syn4_*.sh' -exec chmod +x {} +
        while IFS= read -r script_file; do
            INPUT_SOURCE="$INPUT_SOURCE" DRY_RUN="$DRY_RUN" bash "$script_file"
        done < <(find "$tmp_extract_dir" -type f -name 'optimizeBulk_Syn4_*.sh' | sort)
    fi

    rm -rf "$tmp_extract_dir"
done

echo ""
echo "=========================================="
echo "Step 8c Complete"
echo "=========================================="
echo "End time: $(date)"
