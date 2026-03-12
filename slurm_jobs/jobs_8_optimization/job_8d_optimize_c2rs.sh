#!/bin/bash
#SBATCH --job-name=opt_c2rs
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --partition=genoa
#SBATCH --constraint=scratch-node
#SBATCH --output=logs/8d_opt_c2rs_%j.out

# Resource notes:
# - Default CPUs per task is set to 192 for max node-level throughput.
# - Snellius billing is in 24-core chunks on shared Genoa nodes.
# - Override at submit time, e.g.:
#   sbatch --cpus-per-task=24  slurm_jobs/jobs_8_optimization/job_8d_optimize_c2rs.sh
#   sbatch --cpus-per-task=192 slurm_jobs/jobs_8_optimization/job_8d_optimize_c2rs.sh

set -euo pipefail

if [[ -n "${TMPDIR:-}" ]]; then
    :
else
    export TMPDIR="/scratch-shared/$USER/tmp"
fi
mkdir -p "$TMPDIR"
echo "TMPDIR=$TMPDIR"

echo "STEP 8d start: job=${SLURM_JOB_ID:-local} host=$(hostname) time=$(date)"

module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0
export PATH="$HOME/abc:$PATH"

ALGORITHM="C2RS"
SOURCE_LABEL="base"
OUTPUT_TIER="tier1"
DESIGN_GROUP="${DESIGN_GROUP:-all}"
DESIGNS="${DESIGNS:-}"

# Worker count used inside generated optimizeBulk scripts.
export OPT_SCRIPT_PARALLELISM="${OPT_SCRIPT_PARALLELISM:-${SLURM_CPUS_PER_TASK:-192}}"

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
GEN_SCRIPT="${GEN_SCRIPT:-$BASE_DIR/dataset_tools/generate_optimization_bulk_scripts.py}"
CONFIG_FILE="${CONFIG_FILE:-$BASE_DIR/dataset_tools/optimization_config.json}"

FULL_DATASET="${FULL_DATASET:-/scratch-shared/$USER/FULL_DATASET}"
SCRIPT_ZIP_ROOT="$FULL_DATASET/synScripts/optimization/$ALGORITHM"
MANIFEST_DIR="$FULL_DATASET/optimized_aigs/manifests"

zip_loose_outputs_for_design() {
    local design_name="$1"
    local out_dir="$FULL_DATASET/optimized_aigs/$ALGORITHM/$OUTPUT_TIER/$design_name"

    if [ ! -d "$out_dir" ]; then
        return 0
    fi

    python3 - "$out_dir" "$ALGORITHM" "$design_name" <<'PY'
import os
import sys
import zipfile
from pathlib import Path

out_dir = Path(sys.argv[1])
algorithm = sys.argv[2]
design = sys.argv[3]

loose_aigs = sorted(p for p in out_dir.rglob("*.aig") if p.is_file())
if not loose_aigs:
    print(f"STEP zip: no loose outputs to consolidate for design={design}, algorithm={algorithm}")
    raise SystemExit(0)

zip_path = out_dir / "syn_migrated.zip"
existing_names: set[str] = set()
if zip_path.exists():
    with zipfile.ZipFile(zip_path, "r") as zf:
        existing_names = set(zf.namelist())

with zipfile.ZipFile(
    zip_path,
    "a",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=1,
    allowZip64=True,
) as zf:
    for aig_path in loose_aigs:
        arcname = aig_path.relative_to(out_dir).as_posix()
        if arcname in existing_names:
            stem, ext = os.path.splitext(arcname)
            idx = 1
            candidate = f"{stem}__dup{idx}{ext}"
            while candidate in existing_names:
                idx += 1
                candidate = f"{stem}__dup{idx}{ext}"
            arcname = candidate
        zf.write(aig_path, arcname=arcname)
        existing_names.add(arcname)

for aig_path in loose_aigs:
    aig_path.unlink(missing_ok=True)

for maybe_dir in sorted((p for p in out_dir.rglob("*") if p.is_dir()), reverse=True):
    try:
        maybe_dir.rmdir()
    except OSError:
        pass

print(
    f"STEP zip: consolidated {len(loose_aigs)} loose outputs into {zip_path} "
    f"for design={design}, algorithm={algorithm}"
)
PY
}

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

echo "STEP 8d: regenerating optimization scripts (algorithm=${ALGORITHM}, input_source=base_aigs, design_group=${DESIGN_GROUP})"
GEN_ARGS=(
    --base-dir "$BASE_DIR"
    --full-dataset "$FULL_DATASET"
    --config "$CONFIG_FILE"
    --design-group "$DESIGN_GROUP"
    --algorithms "$ALGORITHM"
    --input-source base_aigs
)
if [ -n "$DESIGNS" ]; then
    GEN_ARGS+=(--designs "$DESIGNS")
fi
python3 "$GEN_SCRIPT" "${GEN_ARGS[@]}"

if [ ! -d "$SCRIPT_ZIP_ROOT" ]; then
    echo "✗ ERROR: Missing generated script zip root after regeneration: $SCRIPT_ZIP_ROOT"
    exit 1
fi

latest_manifest=$(ls -1t "$MANIFEST_DIR"/bulk_scripts_manifest_*.json 2>/dev/null | head -n 1 || true)
if [ -z "$latest_manifest" ]; then
    echo "✗ ERROR: No manifest found under: $MANIFEST_DIR"
    exit 1
fi

expected_designs=$(python3 - "$latest_manifest" "$ALGORITHM" <<'PY'
import json
import sys

manifest_path, algorithm = sys.argv[1], sys.argv[2]
with open(manifest_path, "r", encoding="utf-8") as fh:
    m = json.load(fh)

if algorithm not in m.get("algorithms", []):
    raise SystemExit(f"manifest missing algorithm: {algorithm}")
if "base_aigs" not in m.get("input_sources", []):
    raise SystemExit("manifest missing input_source: base_aigs")

designs = m.get("designs", [])
if not designs:
    raise SystemExit("manifest has zero designs")

print(len(designs))
PY
)

script_count=$(python3 - "$latest_manifest" "$SCRIPT_ZIP_ROOT" "$ALGORITHM" "$SOURCE_LABEL" <<'PY'
import json
import sys
import zipfile
from pathlib import Path

manifest_path = Path(sys.argv[1])
zip_root = Path(sys.argv[2])
algorithm = sys.argv[3]
source_label = sys.argv[4]

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
count = 0
for design in manifest.get("designs", []):
    zip_path = zip_root / f"{design}.zip"
    if not zip_path.exists():
        continue
    script_name = f"optimizeBulk_{algorithm}_{design}_{source_label}.sh"
    with zipfile.ZipFile(zip_path, "r") as archive:
        if script_name in archive.namelist():
            count += 1
print(count)
PY
)
if [ "$script_count" -ne "$expected_designs" ]; then
    echo "✗ ERROR: Expected $expected_designs ${ALGORITHM} shard scripts, found $script_count"
    exit 1
fi

processed_designs=0
skipped_designs=0

# Temporary hardcoded resume list for designs already completed in a prior run.
SKIP_DESIGNS=(
    "16384"
    "2048"
    "256"
    "4096"
    "512"
    "8192"
    "ac97_ctrl"
    "aes"
    "aes_secworks"
    "aes_xcrypt"
    "bp_be"
)

designs_file="$(mktemp "${TMPDIR:-/tmp}/opt8d_designs_${SLURM_JOB_ID:-local}_XXXXXX")"
trap 'rm -f "$designs_file"' EXIT
python3 - "$latest_manifest" > "$designs_file" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    m = json.load(fh)
for design in m.get("designs", []):
    print(design)
PY

while IFS= read -r design_name; do
    [ -z "$design_name" ] && continue
    design_zip="$SCRIPT_ZIP_ROOT/${design_name}.zip"
    if [ ! -f "$design_zip" ]; then
        echo "✗ ERROR: Missing design zip: $design_zip"
        exit 1
    fi

    summary_path="$FULL_DATASET/metadata/raw_logs/${design_name}/${OUTPUT_TIER}/${ALGORITHM}/summary.json"

    if [[ " ${SKIP_DESIGNS[*]} " == *" ${design_name} "* ]]; then
        if [ ! -f "$summary_path" ]; then
            echo "✗ ERROR: Missing per-design summary for skipped design: $summary_path"
            exit 1
        fi

        python3 - "$summary_path" <<'PY'
import json
import sys

summary_path = sys.argv[1]
with open(summary_path, "r", encoding="utf-8") as fh:
    payload = json.load(fh)

if int(payload.get("failed", 0)) != 0:
    raise SystemExit(f"failed>0 in {summary_path}")
PY

    zip_loose_outputs_for_design "$design_name"

        skipped_designs=$((skipped_designs + 1))
        echo "STEP 8d: skipping design ${design_name} (hardcoded completed, summary verified)"
        continue
    fi

    if [ -f "$summary_path" ]; then
        if python3 - "$summary_path" <<'PY'
import json
import sys

summary_path = sys.argv[1]
with open(summary_path, "r", encoding="utf-8") as fh:
    payload = json.load(fh)

if int(payload.get("failed", 0)) != 0:
    raise SystemExit(1)
PY
        then
            zip_loose_outputs_for_design "$design_name"
            skipped_designs=$((skipped_designs + 1))
            echo "STEP 8d: skipping design ${design_name} (summary already clean)"
            continue
        fi
    fi

    echo "STEP 8d: starting design ${design_name}"

    tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/opt_c2rs_${SLURM_JOB_ID:-local}_XXXXXX")"
    script_file="optimizeBulk_${ALGORITHM}_${design_name}_${SOURCE_LABEL}.sh"
    unzip -q "$design_zip" "$script_file" -d "$tmp_dir"
    chmod +x "$tmp_dir/$script_file"
    bash "$tmp_dir/$script_file"

    if [ ! -f "$summary_path" ]; then
        echo "✗ ERROR: Missing per-design summary: $summary_path"
        rm -rf "$tmp_dir"
        exit 1
    fi

    python3 - "$summary_path" <<'PY'
import json
import sys

summary_path = sys.argv[1]
with open(summary_path, "r", encoding="utf-8") as fh:
    payload = json.load(fh)

if int(payload.get("failed", 0)) != 0:
    raise SystemExit(f"failed>0 in {summary_path}")
PY

    processed_designs=$((processed_designs + 1))
    echo "STEP 8d: done design ${design_name}"
    rm -rf "$tmp_dir"
done < "$designs_file"

if [ $((processed_designs + skipped_designs)) -ne "$expected_designs" ]; then
    echo "✗ ERROR: Expected total $expected_designs designs, processed=$processed_designs skipped=$skipped_designs"
    exit 1
fi

echo "STEP 8d complete: processed=${processed_designs} skipped=${skipped_designs} total=${expected_designs} time=$(date)"
