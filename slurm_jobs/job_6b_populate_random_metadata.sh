#!/bin/bash
#SBATCH --job-name=populate_random_metadata
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=genoa
#SBATCH --output=logs/6b_populate_random_metadata_%j.out

set -euo pipefail

echo "=========================================="
echo "JOB 6B: Populate Random Metadata"
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DATASET_TOOLS_DIR="${BASE_DIR}/dataset_tools"
RANDOM_DATASET="${RANDOM_DATASET:-${BASE_DIR}/OPENABC_DATASET}"
OPENABC_DATASET="${OPENABC_DATASET:-/scratch-shared/igardner1/openabc_full/OPENABC_DATASET}"
OUTPUT_DATASET="${OUTPUT_DATASET:-/scratch-shared/$USER/FULL_DATASET}"
SOURCE_SCOPE="${SOURCE_SCOPE:-random}"  # random | all
BACKUP_DIR="${BACKUP_DIR:-/scratch-shared/$USER/dataset_backups}"
ARCHIVE_RANDOM_SOURCE="${ARCHIVE_RANDOM_SOURCE:-true}"  # true | false

if [[ "$SOURCE_SCOPE" != "all" && "$SOURCE_SCOPE" != "random" ]]; then
    echo "ERROR: SOURCE_SCOPE must be 'all' or 'random' (got: $SOURCE_SCOPE)"
    exit 1
fi

if [[ "$ARCHIVE_RANDOM_SOURCE" != "true" && "$ARCHIVE_RANDOM_SOURCE" != "false" ]]; then
    echo "ERROR: ARCHIVE_RANDOM_SOURCE must be 'true' or 'false' (got: $ARCHIVE_RANDOM_SOURCE)"
    exit 1
fi

echo "Base directory: $BASE_DIR"
echo "Random dataset source: $RANDOM_DATASET"
echo "OpenABC source: $OPENABC_DATASET"
echo "Target FULL_DATASET: $OUTPUT_DATASET"
echo "Metadata source scope: $SOURCE_SCOPE"
echo "Archive directory: $BACKUP_DIR"
echo "Archive+delete random source: $ARCHIVE_RANDOM_SOURCE"
echo ""

for required_dir in "$DATASET_TOOLS_DIR" "$RANDOM_DATASET" "$OUTPUT_DATASET"; do
    if [ ! -d "$required_dir" ]; then
        echo "ERROR: Required directory missing: $required_dir"
        exit 1
    fi
done

RANDOM_DESIGNS=("128" "256" "512" "1024" "2048" "4096" "8192" "16384")
missing_random_metadata=()
for design in "${RANDOM_DESIGNS[@]}"; do
    metadata_csv="$RANDOM_DATASET/bench/$design/metadata/$design.csv"
    if [ ! -f "$metadata_csv" ]; then
        missing_random_metadata+=("$metadata_csv")
    fi
done

if [ "${#missing_random_metadata[@]}" -gt 0 ]; then
    echo "ERROR: Random source metadata is incomplete."
    echo "Expected metadata CSV files were not found for ${#missing_random_metadata[@]} design(s)."
    printf '  - %s\n' "${missing_random_metadata[@]}"
    echo "Run metadata collection for the random source first (Job 5) or set RANDOM_DATASET correctly."
    exit 1
fi

OPENABC_AVAILABLE=false
if [ -d "$OPENABC_DATASET" ]; then
    OPENABC_AVAILABLE=true
elif [ "$SOURCE_SCOPE" = "all" ]; then
    echo "ERROR: Required directory missing for SOURCE_SCOPE=all: $OPENABC_DATASET"
    echo "Set SOURCE_SCOPE=random to run Random AIG metadata only."
    exit 1
else
    echo "OpenABC source not found; continuing with Random-only scope."
fi

mkdir -p "$BACKUP_DIR"

for cmd in python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: $cmd"
        exit 1
    fi
done

python3 -c "import pandas, numpy; print('✓ pandas and numpy available')" || {
    echo "ERROR: Required Python packages not available"
    exit 1
}

WORKERS="${SLURM_CPUS_PER_TASK:-24}"
if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [ "$WORKERS" -lt 1 ]; then
    WORKERS=4
fi

if [ "$SOURCE_SCOPE" = "random" ]; then
    echo "Populating metadata for Random AIG designs only..."
else
    echo "Populating metadata for ALL designs (random + OpenABC)..."
fi
echo "Workers: $WORKERS"
echo ""

cd "$DATASET_TOOLS_DIR"

CMD=(
    python3 generate_metadata.py
    "$OUTPUT_DATASET"
    --workers "$WORKERS"
    --validate
    --summary
    --source-scope "$SOURCE_SCOPE"
    --random-source "$RANDOM_DATASET"
)

if [ "$OPENABC_AVAILABLE" = true ] && [ "$SOURCE_SCOPE" = "all" ]; then
    CMD+=(--openabc-source "$OPENABC_DATASET")
fi

"${CMD[@]}"

echo ""
echo "=========================================="
echo "FINAL VERIFICATION: Metadata Integrity"
echo "=========================================="

VERIFY_SCOPE="$SOURCE_SCOPE"
if [ "$VERIFY_SCOPE" = "all" ] && [ "$OPENABC_AVAILABLE" != "true" ]; then
    echo "ERROR: SOURCE_SCOPE=all selected but OpenABC source was unavailable during run."
    exit 1
fi

python3 - <<PY
import json
import os
import sys

import pandas as pd

output_dataset = os.path.realpath("""$OUTPUT_DATASET""")
scope = """$VERIFY_SCOPE"""

metadata_dir = os.path.join(output_dataset, "metadata", "stats")
if not os.path.isdir(metadata_dir):
    print(f"ERROR: Metadata directory missing: {metadata_dir}")
    sys.exit(1)

canonical_header = [
    "file_path",
    "design",
    "recipe_id",
    "step_id",
    "tier_id",
    "nodes",
    "edges",
    "num_PI",
    "num_PO",
    "depth",
    "avg_fanout",
    "max_fanout",
]

random_designs = ["128", "256", "512", "1024", "2048", "4096", "8192", "16384"]
openabc_designs = [
    "i2c",
    "spi",
    "des3_area",
    "ss_pcm",
    "usb_phy",
    "sasc",
    "wb_dma",
    "simple_spi",
    "dynamic_node",
    "aes",
    "pci",
    "ac97_ctrl",
    "mem_ctrl",
    "tv80",
    "fpu",
    "wb_conmax",
    "tinyRocket",
    "aes_xcrypt",
    "aes_secworks",
    "jpeg",
    "bp_be",
    "ethernet",
    "vga_lcd",
    "picosoc",
    "dft",
    "idft",
    "fir",
    "iir",
    "sha256",
]

if scope == "random":
    expected_designs = random_designs
elif scope == "all":
    expected_designs = random_designs + openabc_designs
else:
    print(f"ERROR: Unsupported verification scope: {scope}")
    sys.exit(1)

required_numeric = ["nodes", "edges", "num_PI", "num_PO", "depth", "max_fanout"]
errors = []
total_rows = 0

for design in expected_designs:
    csv_path = os.path.join(metadata_dir, f"{design}.csv")
    if not os.path.isfile(csv_path):
        errors.append(f"Missing expected metadata file: {csv_path}")
        continue

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Unreadable CSV for {design}: {exc}")
        continue

    if list(df.columns) != canonical_header:
        errors.append(f"Header mismatch in {csv_path}")
        continue

    if df.empty:
        errors.append(f"Empty metadata CSV: {csv_path}")
        continue

    bad_design_rows = df["design"].astype(str) != design
    if bad_design_rows.any():
        errors.append(f"Design column mismatch in {csv_path}")

    for col in required_numeric:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.isna().any():
            errors.append(f"Non-numeric values in {csv_path}:{col}")
        elif (coerced < 0).any():
            errors.append(f"Negative values in {csv_path}:{col}")

    if "file_path" in df.columns:
        bad_paths = ~df["file_path"].astype(str).str.startswith(f"base_aigs/{design}/")
        if bad_paths.any():
            errors.append(f"Invalid file_path prefix in {csv_path}")

    total_rows += len(df)

summary_json = os.path.join(metadata_dir, "dataset_summary.json")
if not os.path.isfile(summary_json):
    errors.append(f"Missing summary JSON: {summary_json}")
else:
    try:
        with open(summary_json, "r", encoding="utf-8") as f:
            summary = json.load(f)
        reported_designs = int(summary.get("totals", {}).get("designs", -1))
        if reported_designs < len(expected_designs):
            errors.append(
                f"Summary reports too few designs ({reported_designs} < {len(expected_designs)})"
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Unreadable summary JSON: {exc}")

if errors:
    print("ERROR: Metadata integrity verification failed:")
    for item in errors:
        print(f"  - {item}")
    sys.exit(1)

print(f"✓ Metadata integrity verified for scope={scope}")
print(f"✓ Verified designs: {len(expected_designs)}")
print(f"✓ Verified metadata rows: {total_rows}")
print(f"✓ Metadata directory: {metadata_dir}")

report_path = os.path.join(metadata_dir, "metadata_verification_report.txt")
report_lines = [
    f"timestamp={pd.Timestamp.utcnow().isoformat()}",
    f"scope={scope}",
    f"expected_designs={len(expected_designs)}",
    f"verified_rows={total_rows}",
    f"summary_json={summary_json}",
    "status=PASS",
]
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")

print(f"✓ Verification report written: {report_path}")
PY

echo ""
echo "=========================================="
echo "POST-RUN ARCHIVE: Random source -> ZIP"
echo "=========================================="

if [ "$ARCHIVE_RANDOM_SOURCE" != "true" ]; then
    echo "ARCHIVE_RANDOM_SOURCE=false, skipping archive and delete of random source."
    echo ""
    echo "JOB 6B completed successfully."
    echo "End time: $(date)"
    exit 0
fi

RANDOM_DATASET_REAL="$(cd "$RANDOM_DATASET" && pwd)"
BASE_DIR_REAL="$(cd "$BASE_DIR" && pwd)"
OUTPUT_DATASET_REAL="$(cd "$OUTPUT_DATASET" && pwd)"
BACKUP_DIR_REAL="$(cd "$BACKUP_DIR" && pwd)"

if [ "$RANDOM_DATASET_REAL" = "/" ] || [ "$RANDOM_DATASET_REAL" = "$HOME" ] || [ "$RANDOM_DATASET_REAL" = "$BASE_DIR_REAL" ]; then
    echo "ERROR: Refusing to archive/delete unsafe RANDOM_DATASET path: $RANDOM_DATASET_REAL"
    exit 1
fi

if [[ "$OUTPUT_DATASET_REAL" == "$RANDOM_DATASET_REAL" || "$OUTPUT_DATASET_REAL" == "$RANDOM_DATASET_REAL/"* ]]; then
    echo "ERROR: OUTPUT_DATASET is inside RANDOM_DATASET."
    echo "Refusing archive/delete because it would remove output data: $OUTPUT_DATASET_REAL"
    exit 1
fi

if [[ "$BACKUP_DIR_REAL" == "$RANDOM_DATASET_REAL" || "$BACKUP_DIR_REAL" == "$RANDOM_DATASET_REAL/"* ]]; then
    echo "ERROR: BACKUP_DIR is inside RANDOM_DATASET."
    echo "Refusing archive/delete because ZIP could recursively include itself: $BACKUP_DIR_REAL"
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RANDOM_BASENAME="$(basename "$RANDOM_DATASET_REAL")"
RANDOM_PARENT="$(dirname "$RANDOM_DATASET_REAL")"
RANDOM_ZIP_PATH="$BACKUP_DIR/${RANDOM_BASENAME}_backup_${TIMESTAMP}.zip"

echo "Creating archive: $RANDOM_ZIP_PATH"
python3 - <<PY
import os
import sys
import zipfile

source_dir = os.path.realpath("""$RANDOM_DATASET_REAL""")
zip_path = os.path.realpath("""$RANDOM_ZIP_PATH""")
parent_dir = os.path.realpath("""$RANDOM_PARENT""")
base_name = os.path.basename(source_dir)

if not os.path.isdir(source_dir):
    print(f"ERROR: Source directory not found: {source_dir}")
    sys.exit(1)

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for root, _, files in os.walk(source_dir):
        for name in files:
            abs_path = os.path.join(root, name)
            rel_to_parent = os.path.relpath(abs_path, start=parent_dir)
            zf.write(abs_path, arcname=rel_to_parent)

with zipfile.ZipFile(zip_path, "r") as zf:
    bad_file = zf.testzip()
    if bad_file is not None:
        print(f"ERROR: ZIP integrity check failed at: {bad_file}")
        sys.exit(1)

print(f"✓ ZIP created and verified: {zip_path}")
PY

if [ ! -f "$RANDOM_ZIP_PATH" ]; then
    echo "ERROR: ZIP file was not created: $RANDOM_ZIP_PATH"
    exit 1
fi

echo "Removing original Random dataset directory: $RANDOM_DATASET_REAL"
rm -rf "$RANDOM_DATASET_REAL"

if [ -d "$RANDOM_DATASET_REAL" ]; then
    echo "ERROR: Failed to remove original Random dataset directory"
    exit 1
fi

echo "✓ Original Random dataset removed"
echo "✓ Archive kept at: $RANDOM_ZIP_PATH"

echo ""
echo "JOB 6B completed successfully."
echo "End time: $(date)"
