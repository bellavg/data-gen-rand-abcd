#!/bin/bash
#SBATCH --job-name=metadata_fix_6c
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=genoa
#SBATCH --output=logs/6c_fix_metadata_schema_%j.out

set -euo pipefail

echo "=========================================="
echo "JOB 6C: Metadata Schema Fix"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-manual_run}"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0
module load SciPy-bundle/2025.06-gfbf-2025a

echo "Loaded modules: 2025, foss/2025a, Python/3.13.1, SciPy-bundle/2025.06"
echo ""

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
OUTPUT_DATASET="${OUTPUT_DATASET:-/scratch-shared/$USER/FULL_DATASET}"
METADATA_DIR="${METADATA_DIR:-$OUTPUT_DATASET/metadata/stats}"

echo "Base directory: $BASE_DIR"
echo "Target FULL_DATASET: $OUTPUT_DATASET"
echo "Metadata directory: $METADATA_DIR"
echo ""

if [ ! -d "$METADATA_DIR" ]; then
    echo "ERROR: Metadata directory missing: $METADATA_DIR"
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: Required command not found: python3"
    exit 1
fi

python3 -c "import pandas; print('✓ pandas available')" || {
    echo "ERROR: pandas is required but not available"
    exit 1
}

python3 - <<PY
import glob
import os
import sys

import pandas as pd

metadata_dir = os.path.normpath("""$METADATA_DIR""")

canonical_columns = [
    "file_path",
    "design",
    "recipe_id",
    "step_id",
    "tier_id",
    "algorithm",
    "nodes",
    "edges",
    "num_PI",
    "num_PO",
    "depth",
    "avg_fanout",
    "max_fanout",
]

csv_files = sorted(glob.glob(os.path.join(metadata_dir, "*.csv")))
if not csv_files:
    print(f"ERROR: No CSV files found in {metadata_dir}")
    sys.exit(1)

files_processed = 0
files_changed = 0
algorithm_columns_added = 0
tier_blanks_fixed = 0

for csv_path in csv_files:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to read {csv_path}: {exc}")
        sys.exit(1)

    files_processed += 1
    changed = False

    if "tier_id" not in df.columns:
        print(f"ERROR: Missing required column tier_id in {csv_path}")
        sys.exit(1)

    if "algorithm" not in df.columns:
        tier_idx = df.columns.get_loc("tier_id")
        df.insert(tier_idx + 1, "algorithm", "")
        algorithm_columns_added += 1
        changed = True

    tier_series = df["tier_id"]
    empty_mask = tier_series.isna() | tier_series.astype(str).str.strip().eq("")
    blanks_in_file = int(empty_mask.sum())
    if blanks_in_file > 0:
        df.loc[empty_mask, "tier_id"] = 0
        tier_blanks_fixed += blanks_in_file
        changed = True

    ordered = [col for col in canonical_columns if col in df.columns]
    extras = [col for col in df.columns if col not in canonical_columns]
    new_order = ordered + extras
    if list(df.columns) != new_order:
        df = df[new_order]
        changed = True

    if changed:
        df.to_csv(csv_path, index=False)
        files_changed += 1

print("✓ Metadata schema fix complete")
print(f"✓ CSV files processed: {files_processed}")
print(f"✓ CSV files changed: {files_changed}")
print(f"✓ Added missing algorithm column in: {algorithm_columns_added} file(s)")
print(f"✓ Replaced blank tier_id values with 0: {tier_blanks_fixed}")
PY

echo ""
echo "JOB 6C completed successfully."
echo "End time: $(date)"