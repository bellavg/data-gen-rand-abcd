#!/bin/bash
#SBATCH --job-name=5_make_csv
#SBATCH --time=00:59:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=genoa
#SBATCH --array=0-54
#SBATCH --output=logs/5_csv/5_csv_%A_%a.out

# --- Safety and Environment ---
set -euo pipefail

module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0

# --- Design List (0-54) ---
DESIGNS=(
    "128" "256" "512" "1024" "2048" "4096" "8192" "16384"
    "ac97_ctrl" "aes" "aes_secworks" "aes_xcrypt" "apex1" "bc0" "bp_be"
    "c1355" "c5315" "c6288" "c7552" "dalu" "des3_area" "dft" "div"
    "dynamic_node" "ethernet" "fir" "fpu" "hyp" "i10" "i2c" "idft"
    "iir" "jpeg" "k2" "log2" "mainpla" "max" "mem_ctrl" "multiplier"
    "pci" "picosoc" "sasc" "sha256" "simple_spi" "sin" "spi" "sqrt"
    "square" "ss_pcm" "tinyRocket" "tv80" "usb_phy" "vga_lcd" "wb_conmax"
    "wb_dma"
)

DESIGN="${DESIGNS[$SLURM_ARRAY_TASK_ID]}"
BASE_DIR="$HOME/data-gen-rand-abcd"
DESIGN_DIR="$BASE_DIR/data/designs/$DESIGN"

# Ensure the log directory for SLURM exists
mkdir -p logs/5_csv

echo "=================================================="
echo " JOB 5: Generating CSV for $DESIGN"
echo " Time: $(date)"
echo " CPUs: $SLURM_CPUS_PER_TASK (1/8th Genoa Node)"
echo "=================================================="

# Execute the Multiprocessing Python Parser
# Note: We pass --cpus 24 to match our SLURM allocation
python3 "$BASE_DIR/data/creation/generate_csv.py" \
    --design-dir "$DESIGN_DIR" \
    --design-name "$DESIGN" \
    --cpus 24

echo ">>> COMPLETED: CSV saved to $DESIGN_DIR/design_metadata/${DESIGN}.csv"