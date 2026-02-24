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
export PATH="$HOME/abc:$PATH"

FULL_DATASET="${FULL_DATASET:-/scratch-shared/$USER/FULL_DATASET}"
SCRIPT_DIR="${FULL_DATASET}/optimized_aigs/scripts/Syn4"

TIER="${TIER:-1}"
DRY_RUN="${DRY_RUN:-true}"

if [ "$TIER" != "1" ] && [ "$TIER" != "2" ]; then
    echo "✗ ERROR: Invalid TIER=$TIER (must be 1 or 2)"
    echo "  Examples:"
    echo "    sbatch slurm_jobs/job_8c_optimize_syn4.sh"
    echo "    sbatch --export=ALL,TIER=2 slurm_jobs/job_8c_optimize_syn4.sh"
    exit 1
fi

echo "Configuration:"
echo "  FULL_DATASET: $FULL_DATASET"
echo "  Script dir:   $SCRIPT_DIR"
echo "  TIER:         $TIER"
echo "  DRY_RUN:      $DRY_RUN"
echo ""

if [ ! -d "$SCRIPT_DIR" ]; then
    echo "✗ ERROR: Missing generated script directory: $SCRIPT_DIR"
    echo "Run job_7_optimize.sh first."
    exit 1
fi

if ! command -v abc >/dev/null 2>&1 && [ "$DRY_RUN" != "true" ]; then
    echo "✗ ERROR: abc not found in PATH"
    exit 1
fi

script_count=$(find "$SCRIPT_DIR" -type f -name 'optimizeBulk_Syn4_*.sh' | wc -l | tr -d ' ')
if [ "$script_count" -eq 0 ]; then
    echo "✗ ERROR: No Syn4 shard scripts found in $SCRIPT_DIR"
    exit 1
fi

echo "Found ${script_count} shard scripts"

while IFS= read -r script_file; do
    TIER="$TIER" DRY_RUN="$DRY_RUN" bash "$script_file"
done < <(find "$SCRIPT_DIR" -type f -name 'optimizeBulk_Syn4_*.sh' | sort)

echo ""
echo "=========================================="
echo "Step 8c Complete"
echo "=========================================="
echo "End time: $(date)"
