#!/bin/bash
#SBATCH --job-name=tryout_5
#SBATCH --time=00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=genoa
#SBATCH --output=logs/5_csv/tryout_5_%j.out

# --- Safety and Environment ---
set -euo pipefail

module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0

# --- Hardcoded for Tryout ---
DESIGN="128"
BASE_DIR="$HOME/data-gen-rand-abcd"
DESIGN_DIR="$BASE_DIR/data/designs/$DESIGN"

# Ensure the log directory for SLURM exists
mkdir -p logs/5_csv

echo "=================================================="
echo " JOB 5 (TRYOUT): Generating CSV for $DESIGN"
echo " Time: $(date)"
echo " CPUs: $SLURM_CPUS_PER_TASK (1/8th Genoa Node)"
echo "=================================================="

# Execute the Multiprocessing Python Parser
# Note: Ensure generate_csv.py is actually located in dataset_tools/
python3 "$BASE_DIR/data/creation/generate_csv.py" \
    --design-dir "$DESIGN_DIR" \
    --design-name "$DESIGN" \
    --cpus 24

echo ">>> COMPLETED: CSV saved to $DESIGN_DIR/design_metadata/${DESIGN}.csv"