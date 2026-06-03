#!/bin/bash
#SBATCH --job-name=mass_delete
#SBATCH --time=06:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=genoa
#SBATCH --output=logs/mass_delete_%j.out

set -euo pipefail

echo "=========================================="
echo "MASS DELETION TASK (MULTI-THREADED)"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

TARGET_DIR="/scratch-shared/$USER/aig_train_run"

echo "Target to delete: $TARGET_DIR"

if [ ! -d "$TARGET_DIR" ]; then
    echo "ERROR: Directory $TARGET_DIR does not exist. Exiting."
    exit 1
fi

echo "Step 1: Deleting files using 24 parallel workers..."
# -print0 and -0 handle weird filenames safely
# -P 24 runs 24 parallel deletion threads
# -n 5000 feeds 5,000 files to each thread at a time
find "$TARGET_DIR" -type f -print0 | xargs -0 -P 24 -n 5000 rm -f

echo "Step 2: Cleaning up the remaining empty directories..."
rm -rf "$TARGET_DIR"

echo "=========================================="
echo "Deletion finished: $(date)"
echo "=========================================="