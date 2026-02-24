#!/bin/bash
#SBATCH --job-name=validate_full_dataset
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/7_validate_full_dataset_%j.out

set -euo pipefail

echo "=========================================="
echo "JOB 7: Validate Full Dataset"
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
RANDOM_SOURCE_PATH="${2:-${RANDOM_SOURCE_PATH:-$BASE_DIR/OPENABC_DATASET}}"
OPENABC_SOURCE_PATH="${3:-${OPENABC_SOURCE_PATH:-/scratch-shared/igardner1/openabc_full/OPENABC_DATASET}}"
VALIDATION_SCOPE="${4:-${VALIDATION_SCOPE:-all}}"

if [[ "$VALIDATION_SCOPE" != "random" && "$VALIDATION_SCOPE" != "openabcd" && "$VALIDATION_SCOPE" != "all" ]]; then
    echo "ERROR: VALIDATION_SCOPE must be one of: random, openabcd, all"
    echo "Got: $VALIDATION_SCOPE"
    exit 1
fi

export DEFAULT_DATASET_PATH
export FULL_DATASET
export RANDOM_SOURCE_PATH
export OPENABC_SOURCE_PATH
export VALIDATION_SCOPE

echo "Expected default location: $DEFAULT_DATASET_PATH"
echo "Dataset location to validate: $FULL_DATASET"
echo "Random source for comparison: $RANDOM_SOURCE_PATH"
echo "OpenABC source for comparison: $OPENABC_SOURCE_PATH"
echo "Validation scope: $VALIDATION_SCOPE"
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
import re
import sys
import zipfile
from datetime import datetime

FULL_DATASET = os.environ.get("FULL_DATASET")
DEFAULT_DATASET_PATH = os.environ.get("DEFAULT_DATASET_PATH")
RANDOM_SOURCE_PATH = os.environ.get("RANDOM_SOURCE_PATH")
OPENABC_SOURCE_PATH = os.environ.get("OPENABC_SOURCE_PATH")
VALIDATION_SCOPE = os.environ.get("VALIDATION_SCOPE", "all")

RANDOM_DESIGNS = ["128", "256", "512", "1024", "2048", "4096", "8192", "16384"]
OPENABC_DESIGNS = [
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
    "algorithm",
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
    "validation_scope": VALIDATION_SCOPE,
    "using_default_path": os.path.abspath(FULL_DATASET) == os.path.abspath(DEFAULT_DATASET_PATH),
    "checks": {},
}

if VALIDATION_SCOPE == "random":
    expected_designs = list(RANDOM_DESIGNS)
elif VALIDATION_SCOPE == "openabcd":
    expected_designs = list(OPENABC_DESIGNS)
else:
    expected_designs = list(RANDOM_DESIGNS) + list(OPENABC_DESIGNS)

expected_designs_set = set(expected_designs)

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
zip_count = 0
if os.path.isdir(base_aigs_path):
    for root, _, files in os.walk(base_aigs_path):
        aig_count += sum(1 for f in files if f.endswith(".aig"))
        zip_count += sum(1 for f in files if f.endswith(".zip"))

report["checks"]["aig_files"] = {
    "count": aig_count,
    "zip_count": zip_count,
    "ok": (aig_count > 0) or (zip_count > 0),
}
if (aig_count == 0) and (zip_count == 0):
    errors.append("No .aig or .zip files found under base_aigs")


def resolve_bench_root(source_path):
    candidates = [
        os.path.join(source_path, "bench"),
        os.path.join(source_path, "OPENABC_DATASET", "bench"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def is_zip_sane(path):
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return zf.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def compare_design_source_to_target(design, source_design_dir, target_design_dir):
    result = {
        "design": design,
        "source_dir": source_design_dir,
        "target_dir": target_design_dir,
        "source_top_level_files": 0,
        "missing_files": [],
        "size_mismatches": [],
        "corrupt_targets": [],
        "ok": True,
    }

    if not os.path.isdir(source_design_dir):
        result["ok"] = False
        result["missing_source_dir"] = True
        return result

    os.makedirs(target_design_dir, exist_ok=True)

    for name in sorted(os.listdir(source_design_dir)):
        source_file = os.path.join(source_design_dir, name)
        if not os.path.isfile(source_file):
            continue

        result["source_top_level_files"] += 1
        target_file = os.path.join(target_design_dir, name)

        if not os.path.exists(target_file):
            result["missing_files"].append(name)
            continue

        try:
            source_size = os.path.getsize(source_file)
            target_size = os.path.getsize(target_file)
        except OSError:
            result["size_mismatches"].append(name)
            continue

        if source_size != target_size:
            result["size_mismatches"].append(name)
            continue

        if name.endswith(".zip") and (not is_zip_sane(target_file)):
            result["corrupt_targets"].append(name)
        elif name.endswith(".aig") and target_size == 0:
            result["corrupt_targets"].append(name)

    result["ok"] = (
        len(result["missing_files"]) == 0
        and len(result["size_mismatches"]) == 0
        and len(result["corrupt_targets"]) == 0
    )
    return result


source_compare = {
    "random_source": RANDOM_SOURCE_PATH,
    "openabc_source": OPENABC_SOURCE_PATH,
    "random_bench_root": None,
    "openabc_bench_root": None,
    "checked_designs": 0,
    "source_designs_found": 0,
    "missing_source_design_dirs": [],
    "missing_files_total": 0,
    "size_mismatches_total": 0,
    "corrupt_targets_total": 0,
    "designs": [],
}

random_bench_root = resolve_bench_root(RANDOM_SOURCE_PATH) if RANDOM_SOURCE_PATH else None
openabc_bench_root = resolve_bench_root(OPENABC_SOURCE_PATH) if OPENABC_SOURCE_PATH else None
source_compare["random_bench_root"] = random_bench_root
source_compare["openabc_bench_root"] = openabc_bench_root

if not random_bench_root:
    if VALIDATION_SCOPE in ("random", "all"):
        warnings.append("Random source bench root not found; skipped random source-to-target integrity comparison")
if not openabc_bench_root:
    if VALIDATION_SCOPE in ("openabcd", "all"):
        warnings.append("OpenABC source bench root not found; skipped OpenABC source-to-target integrity comparison")

if VALIDATION_SCOPE in ("random", "all"):
    for design in RANDOM_DESIGNS:
        source_compare["checked_designs"] += 1
        source_design_dir = os.path.join(random_bench_root, design) if random_bench_root else None
        target_design_dir = os.path.join(FULL_DATASET, "base_aigs", design)

        if (not source_design_dir) or (not os.path.isdir(source_design_dir)):
            source_compare["missing_source_design_dirs"].append(design)
            continue

        source_compare["source_designs_found"] += 1
        result = compare_design_source_to_target(design, source_design_dir, target_design_dir)
        source_compare["missing_files_total"] += len(result["missing_files"])
        source_compare["size_mismatches_total"] += len(result["size_mismatches"])
        source_compare["corrupt_targets_total"] += len(result["corrupt_targets"])
        source_compare["designs"].append(result)

if VALIDATION_SCOPE in ("openabcd", "all"):
    for design in OPENABC_DESIGNS:
        source_compare["checked_designs"] += 1
        source_design_dir = os.path.join(openabc_bench_root, design) if openabc_bench_root else None
        target_design_dir = os.path.join(FULL_DATASET, "base_aigs", design)

        if (not source_design_dir) or (not os.path.isdir(source_design_dir)):
            source_compare["missing_source_design_dirs"].append(design)
            continue

        source_compare["source_designs_found"] += 1
        result = compare_design_source_to_target(design, source_design_dir, target_design_dir)
        source_compare["missing_files_total"] += len(result["missing_files"])
        source_compare["size_mismatches_total"] += len(result["size_mismatches"])
        source_compare["corrupt_targets_total"] += len(result["corrupt_targets"])
        source_compare["designs"].append(result)

report["checks"]["source_integrity"] = source_compare

if source_compare["missing_files_total"] > 0:
    errors.append(
        f"Source integrity check: {source_compare['missing_files_total']} source files are missing in FULL_DATASET/base_aigs"
    )
if source_compare["size_mismatches_total"] > 0:
    errors.append(
        f"Source integrity check: {source_compare['size_mismatches_total']} files have source/target size mismatches"
    )
if source_compare["corrupt_targets_total"] > 0:
    errors.append(
        f"Source integrity check: {source_compare['corrupt_targets_total']} target files look corrupted"
    )
if source_compare["source_designs_found"] == 0:
    warnings.append("Source integrity check found zero source design directories to compare")


def sampled_path_exists(rel_path):
    if not rel_path:
        return False

    normalized_rel = rel_path.strip().lstrip("/")
    candidates = [
        os.path.join(FULL_DATASET, normalized_rel),
        os.path.join(FULL_DATASET, "base_aigs", normalized_rel),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return True

    parts = normalized_rel.split("/")

    # Handle zip-preserved logical paths like:
    #   <design>/syn<recipe>.zip/<design>_syn<recipe>_step<step>.aig
    # or
    #   base_aigs/<design>/syn<recipe>.zip/<design>_syn<recipe>_step<step>.aig
    if len(parts) >= 3 and parts[-2].endswith(".zip"):
        zip_parts = parts[:-1]
        zip_rel = os.path.join(*zip_parts)
        zip_candidates = [
            os.path.join(FULL_DATASET, zip_rel),
            os.path.join(FULL_DATASET, "base_aigs", zip_rel),
        ]
        for zip_candidate in zip_candidates:
            if os.path.exists(zip_candidate):
                return True

    # If file_path points to extracted step AIG, allow zip-preserved layout fallback:
    # <design>/<design>_syn<recipe>_step<step>.aig -> base_aigs/<design>/syn<recipe>.zip
    # base_aigs/<design>/<design>_syn<recipe>_step<step>.aig -> base_aigs/<design>/syn<recipe>.zip
    base_name = os.path.basename(normalized_rel)
    match = re.match(r"(.+)_syn(\d+)_step\d+\.aig$", base_name)
    if not match:
        return False

    recipe_id = match.group(2)
    design_dir_rel = os.path.dirname(normalized_rel)
    fallback_candidates = [
        os.path.join(FULL_DATASET, design_dir_rel, f"syn{recipe_id}.zip"),
        os.path.join(FULL_DATASET, "base_aigs", design_dir_rel, f"syn{recipe_id}.zip"),
    ]
    for fallback_zip in fallback_candidates:
        if os.path.exists(fallback_zip):
            return True

    return False

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
ignored_csv_files = []
if os.path.isdir(stats_dir):
    for name in sorted(os.listdir(stats_dir)):
        if name.endswith(".csv"):
            design_name = os.path.splitext(name)[0]
            full_csv_path = os.path.join(stats_dir, name)
            if design_name in expected_designs_set:
                csv_files.append(full_csv_path)
            else:
                ignored_csv_files.append(name)

csv_report = {
    "scope": VALIDATION_SCOPE,
    "expected_designs": len(expected_designs),
    "expected_design_list": expected_designs,
    "csv_files_found": len(csv_files),
    "ignored_csv_files": ignored_csv_files,
    "csv_files": [],
    "total_rows": 0,
    "valid_files": 0,
    "invalid_files": 0,
}

if len(csv_files) == 0:
    errors.append(
        f"No CSV files found in metadata/stats for validation scope={VALIDATION_SCOPE}"
    )

found_designs = {
    os.path.splitext(os.path.basename(path))[0]
    for path in csv_files
}
missing_expected_design_csvs = sorted(expected_designs_set - found_designs)
csv_report["missing_expected_design_csvs"] = missing_expected_design_csvs
if missing_expected_design_csvs:
    errors.append(
        "Missing expected metadata CSVs for scope "
        f"{VALIDATION_SCOPE}: {', '.join(missing_expected_design_csvs)}"
    )

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
                if not sampled_path_exists(rel_path):
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
    if VALIDATION_SCOPE == "all":
        summary_ok = (summary_designs == len(csv_files)) and (summary_files == csv_report["total_rows"])
    else:
        summary_ok = None
    consistency["summary_totals_match_csv"] = {
        "ok": summary_ok,
        "skipped": VALIDATION_SCOPE != "all",
        "reason": "Scope-specific validation only" if VALIDATION_SCOPE != "all" else "",
        "summary_designs": summary_designs,
        "actual_csv_count": len(csv_files),
        "summary_files": summary_files,
        "actual_csv_rows": csv_report["total_rows"],
    }
    if summary_ok is False:
        errors.append("dataset_summary.json totals do not match discovered CSV metrics")

if manifest:
    mstats = manifest.get("statistics", {})
    total_designs = mstats.get("total_designs")
    expected_base_aigs = mstats.get("expected_base_aigs")

    if VALIDATION_SCOPE == "all":
        designs_match = (total_designs == len(csv_files))
    else:
        designs_match = None
    consistency["manifest_total_designs_match_csv_count"] = {
        "ok": designs_match,
        "skipped": VALIDATION_SCOPE != "all",
        "reason": "Scope-specific validation only" if VALIDATION_SCOPE != "all" else "",
        "manifest_total_designs": total_designs,
        "actual_csv_count": len(csv_files),
    }
    if designs_match is False:
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
print("JOB 7 VALIDATION SUMMARY")
print("==========================================")
print(f"Dataset path: {FULL_DATASET}")
print(f"AIG files found: {aig_count}")
print(f"ZIP files found under base_aigs: {zip_count}")
print(f"Metadata CSV files found: {len(csv_files)}")
print(f"Metadata CSV rows total: {csv_report['total_rows']}")
print(
    "Source comparison: "
    f"checked={source_compare['checked_designs']}, "
    f"found={source_compare['source_designs_found']}, "
    f"missing_files={source_compare['missing_files_total']}, "
    f"size_mismatches={source_compare['size_mismatches_total']}, "
    f"corrupt_targets={source_compare['corrupt_targets_total']}"
)
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
echo "Job 7 completed successfully."
