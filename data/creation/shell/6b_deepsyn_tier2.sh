#!/bin/bash
#SBATCH --job-name=T2_Deep
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --partition=genoa
#SBATCH --constraint=scratch-node
#SBATCH --array=0-54
#SBATCH --output=logs/tier2/opt_deep/T2_Deep_%A_%a.out

set -euo pipefail

# --- Environment Setup ---
module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0
export PATH="$HOME/abc:$PATH"

# --- CONFIGURATION ---
ALGO_TARGET="Deepsyn"
# Algorithms to pull Tier-1 inputs from (excluding itself)
SOURCE_ALGOS=("Orchestrate" "Syn4" "C2RS")

DESIGNS=("128" "256" "512" "1024" "2048" "4096" "8192" "16384" "ac97_ctrl" "aes" "aes_secworks" "aes_xcrypt" "apex1" "bc0" "bp_be" "c1355" "c5315" "c6288" "c7552" "dalu" "des3_area" "dft" "div" "dynamic_node" "ethernet" "fir" "fpu" "hyp" "i10" "i2c" "idft" "iir" "jpeg" "k2" "log2" "mainpla" "max" "mem_ctrl" "multiplier" "pci" "picosoc" "sasc" "sha256" "simple_spi" "sin" "spi" "sqrt" "square" "ss_pcm" "tinyRocket" "tv80" "usb_phy" "vga_lcd" "wb_conmax" "wb_dma")

DESIGN="${DESIGNS[$SLURM_ARRAY_TASK_ID]}"
BASE_DIR="$HOME/data-gen-rand-abcd"
SCRATCH_BASE="/scratch-shared/$USER/data-gen-rand-abcd/tier2_aigs/$DESIGN/$ALGO_TARGET"
PERM_LOG_DIR="$BASE_DIR/data/designs/$DESIGN/design_metadata/raw_logs/optimization_logs/tier2/$ALGO_TARGET"

# Ensure directories exist
mkdir -p logs/tier2/opt_deep
mkdir -p "$PERM_LOG_DIR"
mkdir -p "$SCRATCH_BASE"

# --- Setup Local Compute Scratch ---
# Automatically route this to Slurm's approved local scratch using dynamic design name
JOB_SCRATCH="$(mktemp -d -t final_t2_${DESIGN}_${ALGO_TARGET}_XXXXXX)"
trap 'rm -rf "$JOB_SCRATCH"' EXIT

mkdir -p "$JOB_SCRATCH/in" "$JOB_SCRATCH/tier2" "$JOB_SCRATCH/logs" "$JOB_SCRATCH/scripts"

echo "=================================================="
echo " >>> Starting Tier-2 [$ALGO_TARGET] for $DESIGN"
echo " Time: $(date)"
echo "=================================================="

# 1. Stage Runner Script
python3 "$BASE_DIR/data/creation/automate_bulkOptimization.py" --home "$BASE_DIR" --design "$DESIGN"
unzip -q "$BASE_DIR/data/abc_scripts/optimization_scripts/${DESIGN}.zip" -d "$JOB_SCRATCH/scripts"
chmod +x "$JOB_SCRATCH/scripts/$DESIGN/$ALGO_TARGET.sh"

# 2. Stage Tier-1 Inputs (From the OTHER algorithms)
echo ">>> Staging Tier-1 inputs from ${SOURCE_ALGOS[*]}..."
for s_algo in "${SOURCE_ALGOS[@]}"; do
    ZIP_FILE="$BASE_DIR/data/designs/$DESIGN/tier1/${DESIGN}_${s_algo}.zip"
    if [ -f "$ZIP_FILE" ]; then
        echo "    Unzipping $s_algo Tier-1 outputs..."
        unzip -q "$ZIP_FILE" -d "$JOB_SCRATCH/in"
    else
        echo "    ❌ ERROR: $ZIP_FILE missing! Run Tier-1 for $s_algo first." >&2
        exit 1
    fi
done

# 3. RUN ABC (Using 192 cores)
echo ">>> Running Tier-2 Optimization..."
bash "$JOB_SCRATCH/scripts/$DESIGN/$ALGO_TARGET.sh" "$JOB_SCRATCH/in" "$JOB_SCRATCH/tier2" "$JOB_SCRATCH/logs" 192

# 4. ZIP LOGS to Permanent Storage
echo ">>> Archiving Tier-2 logs to $PERM_LOG_DIR"
(cd "$JOB_SCRATCH/logs" && zip -q -r "$PERM_LOG_DIR/opt_t2_${ALGO_TARGET}_${DESIGN}.zip" .)

# 5. ZIP AIGS to Shared Scratch (Safety Net)
if [ -d "$JOB_SCRATCH/tier2" ]; then
    echo ">>> Zipping Tier-2 AIGs to Shared Scratch..."
    (cd "$JOB_SCRATCH/tier2" && zip -q -r "$SCRATCH_BASE/${DESIGN}_t2_${ALGO_TARGET}_AIGS.zip" .)
else
    echo "✗ ERROR: Tier-2 AIG directory not found in scratch!" >&2
    exit 1
fi

echo "=================================================="
echo " ✅ Complete for $DESIGN: Tier-2 $ALGO_TARGET finished."
echo "=================================================="