#!/bin/bash
#SBATCH --job-name=T2_Orch
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --partition=genoa
#SBATCH --constraint=scratch-node
#SBATCH --array=0-54
#SBATCH --output=logs/tier2/opt_orch/T2_Orch_%A_%a.out

set -euo pipefail

# --- CONFIGURATION ---
ALGO_TARGET="Orchestrate"
# Algorithms to pull Tier-1 inputs from (excluding itself)
SOURCE_ALGOS=("Deepsyn" "Syn4" "C2RS")

DESIGNS=("128" "256" "512" "1024" "2048" "4096" "8192" "16384" "ac97_ctrl" "aes" "aes_secworks" "aes_xcrypt" "apex1" "bc0" "bp_be" "c1355" "c5315" "c6288" "c7552" "dalu" "des3_area" "dft" "div" "dynamic_node" "ethernet" "fir" "fpu" "hyp" "i10" "i2c" "idft" "iir" "jpeg" "k2" "log2" "mainpla" "max" "mem_ctrl" "multiplier" "pci" "picosoc" "sasc" "sha256" "simple_spi" "sin" "spi" "sqrt" "square" "ss_pcm" "tinyRocket" "tv80" "usb_phy" "vga_lcd" "wb_conmax" "wb_dma")

DESIGN="${DESIGNS[$SLURM_ARRAY_TASK_ID]}"
BASE_DIR="$HOME/data-gen-rand-abcd"
SCRATCH_BASE="/scratch-shared/$USER/data-gen-rand-abcd/tier2_aigs/$DESIGN/$ALGO_TARGET"
PERM_LOG_DIR="$BASE_DIR/data/designs/$DESIGN/design_metadata/raw_logs/optimization_logs/tier2/$ALGO_TARGET"

# --- Setup Local Compute Scratch ---
JOB_SCRATCH="$(mktemp -d "/scratch-node/$USER/t2_${ALGO_TARGET}_${DESIGN}_XXXXXX")"
trap 'rm -rf "$JOB_SCRATCH"' EXIT

mkdir -p "$JOB_SCRATCH/in" "$JOB_SCRATCH/tier2" "$JOB_SCRATCH/logs"
mkdir -p "$SCRATCH_BASE" "$PERM_LOG_DIR"

echo ">>> Starting Tier-2 [$ALGO_TARGET] for $DESIGN"

# 1. Stage Runner Script
python3 "$BASE_DIR/data/creation/automate_bulkOptimization.py" --home "$BASE_DIR" --design "$DESIGN"
unzip -q "$BASE_DIR/data/abc_scripts/optimization_scripts/${DESIGN}.zip" -d "$JOB_SCRATCH/scripts"
chmod +x "$JOB_SCRATCH/scripts/$DESIGN/$ALGO_TARGET.sh"

# 2. Stage Tier-1 Inputs (From the OTHER algorithms)
echo ">>> Staging Tier-1 inputs from ${SOURCE_ALGOS[*]}..."
for s_algo in "${SOURCE_ALGOS[@]}"; do
    ZIP_FILE="$BASE_DIR/data/designs/$DESIGN/tier1/${DESIGN}_${s_algo}.zip"
    if [ -f "$ZIP_FILE" ]; then
        unzip -q "$ZIP_FILE" -d "$JOB_SCRATCH/in"
    fi
done

# 3. RUN ABC (Using 192 cores)
bash "$JOB_SCRATCH/scripts/$DESIGN/$ALGO_TARGET.sh" "$JOB_SCRATCH/in" "$JOB_SCRATCH/tier2" "$JOB_SCRATCH/logs" 192
# 4. ZIP LOGS to Permanent Storage
(cd "$JOB_SCRATCH/logs" && zip -q -r "$PERM_LOG_DIR/opt_t2_${ALGO_TARGET}_${DESIGN}.zip" .)

# 5. ZIP AIGS to Shared Scratch (Safety Net)
if [ -d "$JOB_SCRATCH/tier2" ]; then
    echo ">>> Zipping Tier-2 AIGs to Shared Scratch..."
    (cd "$JOB_SCRATCH/tier2" && zip -q -r "$SCRATCH_BASE/${DESIGN}_t2_${ALGO_TARGET}_AIGS.zip" .)
else
    echo "✗ ERROR: Tier-2 AIG directory not found in scratch!" >&2
    exit 1
fi

echo ">>> COMPLETED Tier-2 $ALGO_TARGET for $DESIGN"