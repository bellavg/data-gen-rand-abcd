#!/bin/bash
#SBATCH --job-name=validate_full_dataset
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/validate_full_dataset_%j.out

set -euo pipefail

echo "=========================================="
echo "JOB 6A: Validate Full Dataset"
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

BASE_DIR="$HOME/data-gen-rand-abcd"
DEFAULT_DATASET_PATH="/scratch-shared/$USER/FULL_DATASET"
FULL_DATASET="${1:-${DATASET_PATH:-$DEFAULT_DATASET_PATH}}"
export DEFAULT_DATASET_PATH
export FULL_DATASET

echo "Expected default location: $DEFAULT_DATASET_PATH"
echo "Dataset location to validate: $FULL_DATASET"
echo ""

if [ "$FULL_DATASET" != "$DEFAULT_DATASET_PATH" ]; then
    echo "WARNING: Validating a non-default dataset path."
    echo "         This is allowed, but confirm this is intentional."
    echo ""
fi

if [ ! -d "$FULL_DATASET" ]; then
    echo "ERROR: FULL_DATASET directory not found at: $FULL_DATASET"
    exit 1
fi

if [ ! -d "$BASE_DIR" ]; then
    echo "ERROR: Base repo directory not found at: $BASE_DIR"
    exit 1
fi

python3 - <<'PY'
import csv
import json
import os
import sys
from datetime import datetime

FULL_DATASET = os.environ.get("FULL_DATASET")
DEFAULT_DATASET_PATH = os.environ.get("DEFAULT_DATASET_PATH")

required_dirs = [
    "base_aigs",
    "synScripts",
    "metadata",
    "metadata/stats",
    "metadata/library",
    "optimized_aigs",
]
required_files = [
    "dataset_manifest.json",
    "metadata/stats/dataset_summary.json",
]

expected_columns = [
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

errors = []
warnings = []

report = {
    "timestamp": datetime.now().isoformat(),
    "dataset_path": FULL_DATASET,
    "default_expected_path": DEFAULT_DATASET_PATH,
    "using_default_path": os.path.abspath(FULL_DATASET) == os.path.abspath(DEFAULT_DATASET_PATH),
    "checks": {},
}

# Directory checks
missing_dirs = []
for rel in required_dirs:
    p = os.path.join(FULL_DATASET, rel)
    if not os.path.isdir(p):
        missing_dirs.append(rel)

report["checks"]["required_dirs"] = {
    "required": required_dirs,
    "missing": missing_dirs,
    "ok": len(missing_dirs) == 0,
}
if missing_dirs:
    errors.append(f"Missing required directories: {', '.join(missing_dirs)}")

# File checks
missing_files = []
for rel in required_files:
    p = os.path.join(FULL_DATASET, rel)
    if not os.path.isfile(p):
        missing_files.append(rel)

report["checks"]["required_files"] = {
    "required": required_files,
    "missing": missing_files,
    "ok": len(missing_files) == 0,
}
if missing_files:
    errors.append(f"Missing required files: {', '.join(missing_files)}")

# AIG file count
base_aigs_path = os.path.join(FULL_DATASET, "base_aigs")
aig_count = 0
if os.path.isdir(base_aigs_path):
    for root, _, files in os.walk(base_aigs_path):
        aig_count += sum(1 for f in files if f.endswith(".aig"))

report["checks"]["aig_files"] = {
    "count": aig_count,
    "ok": aig_count > 0,
}
if aig_count == 0:
    errors.append("No .aig files found under base_aigs")

# Load manifest/summary if present
manifest = {}
summary = {}
manifest_path = os.path.join(FULL_DATASET, "dataset_manifest.json")
summary_path = os.path.join(FULL_DATASET, "metadata", "stats", "dataset_summary.json")

if os.path.isfile(manifest_path):
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Failed to read dataset_manifest.json: {exc}")

if os.path.isfile(summary_path):
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Failed to read dataset_summary.json: {exc}")

# CSV checks
stats_dir = os.path.join(FULL_DATASET, "metadata", "stats")
csv_files = []
if os.path.isdir(stats_dir):
    for name in sorted(os.listdir(stats_dir)):
        if name.endswith(".csv"):
            csv_files.append(os.path.join(stats_dir, name))

csv_report = {
    "csv_files_found": len(csv_files),
    "csv_files": [],
    "total_rows": 0,
    "valid_files": 0,
    "invalid_files": 0,
}

if len(csv_files) == 0:
    errors.append("No CSV files found in metadata/stats")

for csv_path in csv_files:
    name = os.path.basename(csv_path)
    entry = {
        "file": name,
        "rows": 0,
        "ok": True,
        "issues": [],
    }

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []

            if header != expected_columns:
                entry["ok"] = False
                entry["issues"].append(
                    f"Header mismatch. Expected {expected_columns}, got {header}"
                )

            sample_paths = []
            row_count = 0
            non_numeric_issues = 0

            for row in reader:
                row_count += 1
                if row_count <= 3:
                    sample_paths.append(row.get("file_path", ""))

                for numeric_col in ["nodes", "edges", "num_PI", "num_PO", "depth", "max_fanout"]:
                    val = row.get(numeric_col, "")
                    try:
                        ival = int(float(val))
                        if ival < 0:
                            non_numeric_issues += 1
                    except (TypeError, ValueError):
                        non_numeric_issues += 1

                fanout_val = row.get("avg_fanout", "")
                try:
                    fval = float(fanout_val)
                    if fval < 0:
                        non_numeric_issues += 1
                except (TypeError, ValueError):
                    non_numeric_issues += 1

            entry["rows"] = row_count
            csv_report["total_rows"] += row_count

            if row_count == 0:
                entry["ok"] = False
                entry["issues"].append("CSV has no data rows")

            if non_numeric_issues > 0:
                entry["ok"] = False
                entry["issues"].append(
                    f"Found {non_numeric_issues} numeric-type/sanity issues"
                )

            # Sample path existence check to avoid full heavy scan
            missing_sample_paths = []
            for rel_path in sample_paths:
                if not rel_path:
                    missing_sample_paths.append("<empty_file_path>")
                    continue
                abs_path = os.path.join(FULL_DATASET, rel_path)
                if not os.path.exists(abs_path):
                    missing_sample_paths.append(rel_path)

            if missing_sample_paths:
                entry["ok"] = False
                entry["issues"].append(
                    f"Missing sampled file_path targets: {missing_sample_paths[:5]}"
                )

    except OSError as exc:
        entry["ok"] = False
        entry["issues"].append(f"Read error: {exc}")

    if entry["ok"]:
        csv_report["valid_files"] += 1
    else:
        csv_report["invalid_files"] += 1
        errors.append(f"CSV validation failed: {name} -> {'; '.join(entry['issues'])}")

    csv_report["csv_files"].append(entry)

report["checks"]["metadata_csvs"] = csv_report

# Consistency checks against summary/manifest
consistency = {
    "summary_totals_match_csv": None,
    "manifest_total_designs_match_csv_count": None,
    "manifest_expected_base_aigs_vs_found": None,
}

if summary:
    summary_totals = summary.get("totals", {})
    summary_designs = summary_totals.get("designs")
    summary_files = summary_totals.get("files")
    summary_ok = (summary_designs == len(csv_files)) and (summary_files == csv_report["total_rows"])
    consistency["summary_totals_match_csv"] = {
        "ok": summary_ok,
        "summary_designs": summary_designs,
        "actual_csv_count": len(csv_files),
        "summary_files": summary_files,
        "actual_csv_rows": csv_report["total_rows"],
    }
    if not summary_ok:
        errors.append("dataset_summary.json totals do not match discovered CSV metrics")

if manifest:
    mstats = manifest.get("statistics", {})
    total_designs = mstats.get("total_designs")
    expected_base_aigs = mstats.get("expected_base_aigs")

    designs_match = (total_designs == len(csv_files))
    consistency["manifest_total_designs_match_csv_count"] = {
        "ok": designs_match,
        "manifest_total_designs": total_designs,
        "actual_csv_count": len(csv_files),
    }
    if not designs_match:
        errors.append("dataset_manifest.json total_designs does not match CSV design count")

    if expected_base_aigs is not None:
        base_aigs_match = (aig_count == expected_base_aigs)
        consistency["manifest_expected_base_aigs_vs_found"] = {
            "ok": base_aigs_match,
            "manifest_expected_base_aigs": expected_base_aigs,
            "actual_aig_count": aig_count,
        }
        if not base_aigs_match:
            warnings.append(
                "AIG count differs from manifest expected_base_aigs (this may be valid for partial/non-canonical builds)"
            )

report["checks"]["consistency"] = consistency

report["result"] = {
    "status": "PASS" if len(errors) == 0 else "FAIL",
    "error_count": len(errors),
    "warning_count": len(warnings),
    "errors": errors,
    "warnings": warnings,
}

print("==========================================")
print("JOB 6A VALIDATION SUMMARY")
print("==========================================")
print(f"Dataset path: {FULL_DATASET}")
print(f"AIG files found: {aig_count}")
print(f"Metadata CSV files found: {len(csv_files)}")
print(f"Metadata CSV rows total: {csv_report['total_rows']}")
print("")
print("Full validation report (JSON):")
print(json.dumps(report, indent=2))

if warnings:
    print("Warnings:")
    for warning in warnings:
        print(f"  - {warning}")

if errors:
    print("Errors:")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)

print("All critical validation checks passed.")
PY

echo ""
echo "End time: $(date)"
echo "Job 6a completed successfully."
