#!/bin/bash
#SBATCH --job-name=opt_deepsyn
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/opt_deepsyn_%j.out

# Step 8b: Run Deepsyn Optimization
# Executes generated bulk script for Deepsyn.

set -e

echo "=========================================="
echo "STEP 8b: Running Deepsyn Optimization"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

module purge
module load 2025
module load foss/2025a

BASE_DIR="$HOME/data-gen-rand-abcd"
FULL_DATASET="${FULL_DATASET:-/scratch-shared/$USER/FULL_DATASET}"
SCRIPT_DIR="${FULL_DATASET}/optimized_aigs/scripts/Deepsyn"

TIER="${TIER:-1}"
DRY_RUN="${DRY_RUN:-true}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"

if [ "$TIER" != "1" ] && [ "$TIER" != "2" ]; then
    echo "✗ ERROR: Invalid TIER=$TIER (must be 1 or 2)"
    echo "  Examples:"
    echo "    sbatch slurm_jobs/job_8b_optimize_deepsyn.sh"
    echo "    sbatch --export=ALL,TIER=2 slurm_jobs/job_8b_optimize_deepsyn.sh"
    exit 1
fi

echo "Configuration:"
echo "  FULL_DATASET: $FULL_DATASET"
echo "  Script dir:   $SCRIPT_DIR"
echo "  TIER:         $TIER"
echo "  DRY_RUN:      $DRY_RUN"
echo "  TIMEOUT_SECONDS: $TIMEOUT_SECONDS"
echo ""

if ! [[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || [ "$TIMEOUT_SECONDS" -le 0 ]; then
    echo "✗ ERROR: TIMEOUT_SECONDS must be a positive integer (got: $TIMEOUT_SECONDS)"
    echo "  Example: sbatch --export=ALL,TIMEOUT_SECONDS=600 slurm_jobs/job_8b_optimize_deepsyn.sh"
    exit 1
fi

if [ ! -d "$SCRIPT_DIR" ]; then
    echo "✗ ERROR: Missing generated script directory: $SCRIPT_DIR"
    echo "Run slurm_jobs/job_8_make_optimize_scripts.sh first."
    exit 1
fi

script_count=$(find "$SCRIPT_DIR" -type f -name 'optimizeBulk_Deepsyn_*.sh' | wc -l | tr -d ' ')
if [ "$script_count" -eq 0 ]; then
    echo "✗ ERROR: No Deepsyn shard scripts found in $SCRIPT_DIR"
    exit 1
fi

echo "Found ${script_count} shard scripts"

while IFS= read -r script_file; do
    TIER="$TIER" DRY_RUN="$DRY_RUN" TIMEOUT_SECONDS="$TIMEOUT_SECONDS" bash "$script_file"
done < <(find "$SCRIPT_DIR" -type f -name 'optimizeBulk_Deepsyn_*.sh' | sort)

echo ""
echo "=========================================="
echo "Step 8b Complete"
echo "=========================================="
echo "End time: $(date)"
