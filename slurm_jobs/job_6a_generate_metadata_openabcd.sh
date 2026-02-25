#!/bin/bash
#SBATCH --job-name=6a_generate_metadata_openabc
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=genoa
#SBATCH --output=logs/6a_generate_metadata_openabc_%j.out

set -euo pipefail

mkdir -p logs

echo "=========================================="
echo "JOB 6A: Generate metadata for OpenABC-converted AIGs"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-manual_run}"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
DATASET_TOOLS_DIR="${DATASET_TOOLS_DIR:-$BASE_DIR/dataset_tools}"
FULL_DATASET_DIR="${FULL_DATASET_DIR:-/scratch-shared/$USER/FULL_DATASET}"
OPENABC_RAW_STATS_DIR="$FULL_DATASET_DIR/metadata/openabc_raw_statistics"

for cmd in python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: $cmd"
        exit 1
    fi
done

echo "Dataset tools: $DATASET_TOOLS_DIR"
echo "FULL_DATASET: $FULL_DATASET_DIR"
echo "OpenABC raw stats (expected): $OPENABC_RAW_STATS_DIR"

if [ ! -d "$DATASET_TOOLS_DIR" ]; then
    echo "ERROR: dataset_tools directory missing: $DATASET_TOOLS_DIR"
    exit 1
fi

if [ ! -d "$FULL_DATASET_DIR" ]; then
    echo "ERROR: FULL_DATASET dir missing: $FULL_DATASET_DIR"
    exit 1
fi

WORKERS="${SLURM_CPUS_PER_TASK:-24}"
if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [ "$WORKERS" -lt 1 ]; then
    WORKERS=4
fi

cd "$DATASET_TOOLS_DIR"

STATS_ROOT="$FULL_DATASET_DIR/.openabc_stats_root"
cleanup_stats_root() {
    [ -L "$STATS_ROOT/statistics" ] && rm -f "$STATS_ROOT/statistics" || true
    [ -d "$STATS_ROOT" ] && rmdir --ignore-fail-on-non-empty "$STATS_ROOT" 2>/dev/null || true
}

# Create a small wrapper so generate_metadata.py can find a 'statistics' folder
if [ -d "$OPENABC_RAW_STATS_DIR" ]; then
    mkdir -p "$STATS_ROOT"
    ln -sfn "$OPENABC_RAW_STATS_DIR" "$STATS_ROOT/statistics"
    trap cleanup_stats_root EXIT
    OPENABC_ARG="$STATS_ROOT"
else
    echo "WARNING: OpenABC raw statistics directory not found at $OPENABC_RAW_STATS_DIR"
    echo "Attempting to run metadata generation without explicit OpenABC source..."
    OPENABC_ARG=""
fi

CMD=( python3 generate_metadata.py "$FULL_DATASET_DIR" --workers "$WORKERS" --source-scope openabc --validate --summary )

if [ -n "$OPENABC_ARG" ]; then
    CMD+=(--openabc-source "$OPENABC_ARG")
else
    echo "ERROR: OpenABC statistics source not found at $OPENABC_RAW_STATS_DIR"
    echo "Place the OpenABC 'statistics' folder at $OPENABC_RAW_STATS_DIR or set OPENABC_RAW_STATS_DIR to point to it."
    cleanup_stats_root
    exit 1
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "Post-processing metadata CSVs to ensure canonical tier/algorithm for base AIGs"

python3 - <<'PY'
import glob, os, pandas as pd

metadata_dir = os.path.join(os.path.normpath("$FULL_DATASET_DIR"), "metadata", "stats")
header = "file_path,design,recipe_id,step_id,tier_id,algorithm,nodes,edges,num_PI,num_PO,depth,avg_fanout,max_fanout".split(',')

if not os.path.isdir(metadata_dir):
    print(f"ERROR: metadata dir missing: {metadata_dir}")
    raise SystemExit(1)

changed = 0
for csv in glob.glob(os.path.join(metadata_dir, '*.csv')):
    if os.path.basename(csv).startswith('dataset_summary'):
        continue
    df = pd.read_csv(csv)
    # Ensure canonical columns exist
    for col in header:
        if col not in df.columns:
            df[col] = "" if col in ("file_path","design","algorithm") else 0
    df = df[header]

    # For base AIGs (orig), set tier_id to 0 and algorithm to empty string
    is_base = df['file_path'].astype(str).str.endswith('_orig.aig') | df['file_path'].astype(str).str.contains('/base_aigs/.*/.*_orig.aig')
    if is_base.any():
        df.loc[is_base, 'tier_id'] = 0
        df.loc[is_base, 'algorithm'] = ""
        changed += int(is_base.sum())

    # Ensure header types/ordering and write back
    # Preserve integer/numeric columns where possible
    numeric_cols = ['nodes','edges','num_PI','num_PO','depth','avg_fanout','max_fanout']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df.to_csv(csv, index=False)

print(f"Post-processed metadata CSVs: updated {changed} base rows")
PY

echo ""
echo "JOB 6A complete." 
echo "End time: $(date)"
