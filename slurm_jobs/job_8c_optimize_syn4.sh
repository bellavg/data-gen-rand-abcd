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
DRY_RUN="${DRY_RUN:-false}"

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

while IFS= read -r design_zip; do
    tmp_extract_dir="$(mktemp -d "${TMPDIR:-/tmp}/opt_syn4_${SLURM_JOB_ID:-local}_XXXXXX")"

    if unzip -q "$design_zip" "optimizeBulk_Syn4_*_${SOURCE_LABEL}.sh" -d "$tmp_extract_dir" 2>/dev/null; then
        find "$tmp_extract_dir" -type f -name 'optimizeBulk_Syn4_*.sh' -exec chmod +x {} +
        mapfile -t script_list < <(find "$tmp_extract_dir" -type f -name 'optimizeBulk_Syn4_*.sh' | sort)
        if [ "${#script_list[@]}" -eq 0 ]; then
            rm -rf "$tmp_extract_dir"
            continue
        fi

        if [ -n "${PARALLELISM:-}" ]; then
            :
        elif [ -n "${SLURM_CPUS_ON_NODE:-}" ]; then
            PARALLELISM="$SLURM_CPUS_ON_NODE"
        elif [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
            PARALLELISM="$SLURM_CPUS_PER_TASK"
        elif command -v nproc >/dev/null 2>&1; then
            PARALLELISM=$(nproc)
        else
            PARALLELISM=4
        fi

        if command -v parallel >/dev/null 2>&1; then
            printf "%s\n" "${script_list[@]}" | parallel -j "$PARALLELISM" INPUT_SOURCE="$INPUT_SOURCE" DRY_RUN="$DRY_RUN" bash {}
        else
            for f in "${script_list[@]}"; do
                INPUT_SOURCE="$INPUT_SOURCE" DRY_RUN="$DRY_RUN" bash "$f" &
                while [ "$(jobs -rp | wc -l)" -ge "$PARALLELISM" ]; do sleep 1; done
            done
            wait
        fi
    fi

    rm -rf "$tmp_extract_dir"
done < <(find "$SCRIPT_ZIP_ROOT" -type f -name '*.zip' | sort)

echo ""
echo "=========================================="
echo "Step 8c Complete"
echo "=========================================="
echo "End time: $(date)"

# End of job
