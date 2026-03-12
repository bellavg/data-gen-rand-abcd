#!/bin/bash
#SBATCH --job-name=opt_deepsyn
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=genoa
#SBATCH --constraint=scratch-node
#SBATCH --output=logs/8b_opt_deepsyn_%j.out

# Resource notes:
# - Default CPUs per task is set to 72 for higher throughput.
# - Snellius billing is in 24-core chunks on shared Genoa nodes.
# - Override at submit time, e.g.:
#   sbatch --cpus-per-task=24  slurm_jobs/jobs_8_optimization/job_8b_optimize_deepsyn.sh
#   sbatch --cpus-per-task=192 slurm_jobs/jobs_8_optimization/job_8b_optimize_deepsyn.sh

set -euo pipefail

if [[ -n "${TMPDIR:-}" ]]; then
    :
else
    export TMPDIR="/scratch-shared/$USER/tmp"
fi
mkdir -p "$TMPDIR"
echo "TMPDIR=$TMPDIR"

echo "STEP 8b start: job=${SLURM_JOB_ID:-local} host=$(hostname) time=$(date)"

module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0
export PATH="$HOME/abc:$PATH"

ALGORITHM="Deepsyn"
SOURCE_LABEL="base"
OUTPUT_TIER="tier1"
TARGET_DESIGN="${TARGET_DESIGN:-${DESIGN:-${DESIGNS:-}}}"

# Worker count used inside generated optimizeBulk scripts.
export OPT_SCRIPT_PARALLELISM="${OPT_SCRIPT_PARALLELISM:-${SLURM_CPUS_PER_TASK:-72}}"

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
GEN_SCRIPT="${GEN_SCRIPT:-$BASE_DIR/dataset_tools/generate_optimization_bulk_scripts.py}"
CONFIG_FILE="${CONFIG_FILE:-$BASE_DIR/dataset_tools/optimization_config.json}"

FULL_DATASET="${FULL_DATASET:-/scratch-shared/$USER/FULL_DATASET}"
SCRIPT_ZIP_ROOT="$FULL_DATASET/synScripts/optimization/$ALGORITHM"
MANIFEST_DIR="$FULL_DATASET/optimized_aigs/manifests"

if [ ! -f "$GEN_SCRIPT" ]; then
    echo "✗ ERROR: Optimization generator script not found: $GEN_SCRIPT"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "✗ ERROR: Optimization config not found: $CONFIG_FILE"
    exit 1
fi

if ! command -v abc >/dev/null 2>&1; then
    echo "✗ ERROR: abc not found in PATH"
    exit 1
fi

if [ -z "$TARGET_DESIGN" ]; then
    echo "✗ ERROR: TARGET_DESIGN is required for per-design mode."
    echo "  Example: TARGET_DESIGN=1024 sbatch slurm_jobs/jobs_8_optimization/job_8b_optimize_deepsyn.sh"
    exit 1
fi

echo "STEP 8b: regenerating optimization scripts (algorithm=${ALGORITHM}, input_source=base_aigs, design=${TARGET_DESIGN})"
GEN_ARGS=(
    --base-dir "$BASE_DIR"
    --full-dataset "$FULL_DATASET"
    --config "$CONFIG_FILE"
    --design-group all
    --algorithms "$ALGORITHM"
    --input-source base_aigs
    --designs "$TARGET_DESIGN"
)
python3 "$GEN_SCRIPT" "${GEN_ARGS[@]}"

if [ ! -d "$SCRIPT_ZIP_ROOT" ]; then
    echo "✗ ERROR: Missing generated script zip root after regeneration: $SCRIPT_ZIP_ROOT"
    exit 1
fi

latest_manifest="$MANIFEST_DIR/bulk_scripts_manifest.json"
if [ ! -f "$latest_manifest" ]; then
    latest_manifest=$(ls -1t "$MANIFEST_DIR"/bulk_scripts_manifest_*.json 2>/dev/null | head -n 1 || true)
fi
if [ -z "$latest_manifest" ]; then
    echo "✗ ERROR: No manifest found under: $MANIFEST_DIR"
    exit 1
fi

python3 - "$latest_manifest" "$ALGORITHM" "$TARGET_DESIGN" <<'PY'
import json
import sys

manifest_path, algorithm, target_design = sys.argv[1], sys.argv[2], sys.argv[3]
with open(manifest_path, "r", encoding="utf-8") as fh:
    m = json.load(fh)

if algorithm not in m.get("algorithms", []):
    raise SystemExit(f"manifest missing algorithm: {algorithm}")
if "base_aigs" not in m.get("input_sources", []):
    raise SystemExit("manifest missing input_source: base_aigs")

designs = m.get("designs", [])
if designs != [target_design]:
    raise SystemExit(f"manifest design mismatch: expected [{target_design}], got {designs}")
PY
echo "STEP 8b config: parallelism=${OPT_SCRIPT_PARALLELISM}"

design_zip="$SCRIPT_ZIP_ROOT/${TARGET_DESIGN}.zip"
if [ ! -f "$design_zip" ]; then
    echo "✗ ERROR: Missing design zip: $design_zip"
    exit 1
fi

summary_path="$FULL_DATASET/metadata/raw_logs/${TARGET_DESIGN}/${OUTPUT_TIER}/${ALGORITHM}/summary.json"
output_tier_dir="$FULL_DATASET/optimized_aigs/$ALGORITHM/$OUTPUT_TIER"
output_zip="$output_tier_dir/${TARGET_DESIGN}.zip"
legacy_output_dir="$output_tier_dir/${TARGET_DESIGN}"

mkdir -p "$output_tier_dir"
rm -f "$output_zip"
rm -rf "$legacy_output_dir"

echo "STEP 8b: starting design ${TARGET_DESIGN}"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/opt_deepsyn_${SLURM_JOB_ID:-local}_XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT
script_file="optimizeBulk_${ALGORITHM}_${TARGET_DESIGN}_${SOURCE_LABEL}.sh"
unzip -q "$design_zip" "$script_file" -d "$tmp_dir"
chmod +x "$tmp_dir/$script_file"
bash "$tmp_dir/$script_file"

if [ ! -f "$summary_path" ]; then
    echo "✗ ERROR: Missing per-design summary: $summary_path"
    exit 1
fi

python3 - "$summary_path" "$output_zip" <<'PY'
import json
import sys
import zipfile
from pathlib import Path

summary_path = sys.argv[1]
output_zip_path = Path(sys.argv[2])
with open(summary_path, "r", encoding="utf-8") as fh:
    payload = json.load(fh)

if int(payload.get("failed", 0)) != 0:
    raise SystemExit(f"failed>0 in {summary_path}")

if not output_zip_path.is_file():
    raise SystemExit(f"missing output zip: {output_zip_path}")

with zipfile.ZipFile(output_zip_path, "r") as zf:
    aig_count = sum(1 for name in zf.namelist() if name.lower().endswith(".aig"))

expected = int(payload.get("created", -1))
if expected <= 0:
    raise SystemExit(f"non-positive created count in {summary_path}: {expected}")
if aig_count != expected:
    raise SystemExit(
        f"output zip aig count mismatch for {output_zip_path}: zip_aigs={aig_count}, summary_created={expected}"
    )
PY

echo "STEP 8b: done design ${TARGET_DESIGN}"
echo "STEP 8b complete: processed=1 skipped=0 total=1 output_zip=${output_zip} time=$(date)"
