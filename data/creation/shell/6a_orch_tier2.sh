#!/bin/bash
#SBATCH --job-name=T2_Orch
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --partition=genoa
#SBATCH --constraint=scratch-node
#SBATCH --array=1-54
#SBATCH --output=logs/tier2/opt_orch/T2_Orch_%A_%a.out

set -euo pipefail

# --- Environment Setup ---
module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0
export PATH="$HOME/abc:$PATH"

# --- CONFIGURATION ---
ALGO_TARGET="Orchestrate"
SOURCE_ALGOS=("Deepsyn" "Syn4" "C2RS")

DESIGNS=("128" "256" "512" "1024" "2048" "4096" "8192" "16384" "ac97_ctrl" "aes" "aes_secworks" "aes_xcrypt" "apex1" "bc0" "bp_be" "c1355" "c5315" "c6288" "c7552" "dalu" "des3_area" "dft" "div" "dynamic_node" "ethernet" "fir" "fpu" "hyp" "i10" "i2c" "idft" "iir" "jpeg" "k2" "log2" "mainpla" "max" "mem_ctrl" "multiplier" "pci" "picosoc" "sasc" "sha256" "simple_spi" "sin" "spi" "sqrt" "square" "ss_pcm" "tinyRocket" "tv80" "usb_phy" "vga_lcd" "wb_conmax" "wb_dma")

DESIGN="${DESIGNS[$SLURM_ARRAY_TASK_ID]}"
BASE_DIR="$HOME/data-gen-rand-abcd"
SCRATCH_BASE="/scratch-shared/$USER/data-gen-rand-abcd/tier2_aigs/$DESIGN/$ALGO_TARGET"
PERM_LOG_DIR="$BASE_DIR/data/designs/$DESIGN/design_metadata/raw_logs/optimization_logs/tier2/$ALGO_TARGET"

mkdir -p logs/tier2/opt_orch "$PERM_LOG_DIR" "$SCRATCH_BASE"

# --- Setup Local Compute Scratch ---
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

# 2. Stage Tier-1 Inputs
echo ">>> Staging Tier-1 inputs from ${SOURCE_ALGOS[*]}..."
for s_algo in "${SOURCE_ALGOS[@]}"; do
    ZIP_FILE="$BASE_DIR/data/designs/$DESIGN/tier1/${DESIGN}_${s_algo}.zip"
    if [ -f "$ZIP_FILE" ]; then
        unzip -q "$ZIP_FILE" -d "$JOB_SCRATCH/in"
    else
        echo "❌ ERROR: $ZIP_FILE missing!" >&2; exit 1
    fi
done

# 3. RUN ABC
echo ">>> Running Tier-2 Optimization..."
bash "$JOB_SCRATCH/scripts/$DESIGN/$ALGO_TARGET.sh" "$JOB_SCRATCH/in" "$JOB_SCRATCH/tier2" "$JOB_SCRATCH/logs" 192

# 4. VALIDATION AND CONTENT CHECK
echo ">>> Validating output counts..."
IN_COUNT=$(ls "$JOB_SCRATCH/in"/*.aig | wc -l)
OUT_COUNT=$(ls "$JOB_SCRATCH/tier2"/*.aig | wc -l)
LOG_COUNT=$(ls "$JOB_SCRATCH/logs"/*.log | wc -l)

echo "    Inputs found: $IN_COUNT"
echo "    AIGs created: $OUT_COUNT"
echo "    Logs created: $LOG_COUNT"

if [ "$OUT_COUNT" -ne "$IN_COUNT" ]; then
    echo "❌ ERROR: Count mismatch! Some files were not processed." >&2
    # Optional: don't exit if you want to keep partial results, 
    # but for a 'Final' run, we should probably exit 1.
    exit 1
fi

echo ">>> Inspecting sample log content..."
SAMPLE_LOG=$(ls "$JOB_SCRATCH/logs"/*.log | shuf -n 1)
echo "    Sample Log: $(basename "$SAMPLE_LOG")"
echo "--------------------------------------------------"
cat "$SAMPLE_LOG"
echo "--------------------------------------------------"

# 5. ZIP LOGS
echo ">>> Archiving Tier-2 logs to $PERM_LOG_DIR"
(cd "$JOB_SCRATCH/logs" && zip -q -r "$PERM_LOG_DIR/opt_t2_${ALGO_TARGET}_${DESIGN}.zip" .)

# 6. ZIP AIGS
echo ">>> Archiving Tier-2 AIGs to $SCRATCH_BASE"
(cd "$JOB_SCRATCH/tier2" && zip -q -r "$SCRATCH_BASE/${DESIGN}_t2_${ALGO_TARGET}_AIGS.zip" .)

echo "=================================================="
echo " ✅ Complete for $DESIGN: Tier-2 $ALGO_TARGET finished."
echo "=================================================="