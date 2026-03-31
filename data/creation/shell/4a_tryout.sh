#!/bin/bash
#SBATCH --job-name=tryout_4a
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --partition=genoa
#SBATCH --constraint=scratch-node
#SBATCH --output=logs/tryout_4a_%j.out

set -euo pipefail

if [[ -z "${TMPDIR:-}" ]]; then
    export TMPDIR="/scratch-shared/$USER/tmp"
fi
mkdir -p "$TMPDIR"

echo "STEP 4a (TRYOUT): start job=${SLURM_JOB_ID} host=$(hostname) time=$(date)"

module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0
export PATH="$HOME/abc:$PATH"

# ==========================================
# HARDCODED FOR TRYOUT
# ==========================================
ALGORITHM="Orchestrate"
DESIGN="128"

BASE_DIR="$HOME/data-gen-rand-abcd"
PERM_TIER0_DIR="$BASE_DIR/data/designs/$DESIGN"
PERM_TIER1_DIR="$BASE_DIR/data/designs/$DESIGN/tier1"
PERM_LOG_DIR="$BASE_DIR/data/designs/$DESIGN/design_metadata/raw_logs/optimization_logs/tier1/${ALGORITHM}"

JOB_SCRATCH="$(mktemp -d "$TMPDIR/tier1_${ALGORITHM}_${DESIGN}_XXXXXX")"
trap 'rm -rf "$JOB_SCRATCH"' EXIT

SCRATCH_IN="$JOB_SCRATCH/in"
SCRATCH_OUT="$JOB_SCRATCH/out"
SCRATCH_LOGS="$JOB_SCRATCH/logs"
SCRATCH_SCRIPTS="$JOB_SCRATCH/scripts"

mkdir -p "$SCRATCH_IN" "$SCRATCH_OUT" "$SCRATCH_LOGS" "$SCRATCH_SCRIPTS" "$PERM_LOG_DIR" "$PERM_TIER1_DIR"

echo "=================================================="
echo " TRYOUT MODE: Tier-1 [${ALGORITHM}] for: $DESIGN"
echo " Cores: $SLURM_CPUS_PER_TASK | Scratch: $JOB_SCRATCH"
echo "=================================================="

# 1. GENERATE SCRIPTS
echo ">>> Generating runner scripts for $DESIGN..."
python3 "$BASE_DIR/automate_bulkOptimization.py" --home "$BASE_DIR" --design "$DESIGN"

unzip -q "$BASE_DIR/data/abc_scripts/optimization_scripts/${DESIGN}.zip" -d "$SCRATCH_SCRIPTS"
chmod +x "$SCRATCH_SCRIPTS/$DESIGN/"*.sh

# 2. STAGE AIGS
echo ">>> Staging inputs to local scratch..."
if [ -f "$PERM_TIER0_DIR/tier0.zip" ]; then
    unzip -q "$PERM_TIER0_DIR/tier0.zip" -d "$SCRATCH_IN"
elif [ -f "$PERM_TIER0_DIR/tier0/tier0.zip" ]; then
    unzip -q "$PERM_TIER0_DIR/tier0/tier0.zip" -d "$SCRATCH_IN"
fi

find "$SCRATCH_IN" -mindepth 2 -name "*.aig" -exec mv -t "$SCRATCH_IN" {} +
find "$SCRATCH_IN" -mindepth 1 -type d -delete 2>/dev/null || true

INPUT_COUNT=$(find "$SCRATCH_IN" -maxdepth 1 -name "*.aig" | wc -l)
if [[ "$INPUT_COUNT" -eq 0 ]]; then
    echo "✗ ERROR: No input AIGs found for $DESIGN in $SCRATCH_IN!" >&2
    exit 1
fi

echo ">>> Successfully staged $INPUT_COUNT files. Starting execution..."
echo ">>> (You can cancel this job anytime using: scancel $SLURM_JOB_ID)"

# 3. EXECUTE ALGORITHM
bash "$SCRATCH_SCRIPTS/$DESIGN/${ALGORITHM}.sh" "$SCRATCH_IN" "$SCRATCH_OUT" "$SCRATCH_LOGS" 192

echo ">>> Execution finished (if you didn't cancel it first). Zipping..."
# (Validation & Zipping logic omitted here to keep the tryout lightweight, 
# but the real scripts will handle it normally!)