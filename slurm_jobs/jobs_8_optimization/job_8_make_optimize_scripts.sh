#!/bin/bash
#SBATCH --job-name=gen_opt_scripts
#SBATCH --time=00:45:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/8_gen_opt_scripts_%j.out

# Job 8: Generate Optimization Bulk Scripts
# Mirrors the Job 2/3 pattern:
#   - Use dataset_tools to generate bulk shell scripts
#   - Then run per-algorithm jobs (Job 8a-8d)

set -e

# ================= USER SETTINGS (EDIT HERE) =================
# Design scope: random | openabcd | all
DEFAULT_DESIGN_GROUP="${DEFAULT_DESIGN_GROUP:-all}"
# Algorithms: all | Orchestrate,Deepsyn,Syn4,C2RS
DEFAULT_ALGORITHMS="${DEFAULT_ALGORITHMS:-all}"
# Optional explicit design override: e.g. "128,256"
DEFAULT_DESIGNS="${DEFAULT_DESIGNS:-}"
# Input source: base_aigs | tier1 | tier2 | all
DEFAULT_INPUT_SOURCE="${DEFAULT_INPUT_SOURCE:-base_aigs}"
# ============================================================

DESIGN_GROUP="${DESIGN_GROUP:-$DEFAULT_DESIGN_GROUP}"
DESIGNS="${DESIGNS:-}"
if [[ -z "$DESIGNS" && -n "$DEFAULT_DESIGNS" ]]; then
    DESIGNS="$DEFAULT_DESIGNS"
fi
ALGORITHMS="${ALGORITHMS:-$DEFAULT_ALGORITHMS}"
INPUT_SOURCE="${INPUT_SOURCE:-$DEFAULT_INPUT_SOURCE}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --design-group)
            DESIGN_GROUP="$2"
            shift 2
            ;;
        --designs)
            DESIGNS="$2"
            shift 2
            ;;
        --algorithms)
            ALGORITHMS="$2"
            shift 2
            ;;
        --input-source)
            INPUT_SOURCE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: sbatch slurm_jobs/jobs_8_optimization/job_8_make_optimize_scripts.sh [--design-group all|random|openabc|openabcd] [--designs 'd1,d2,...'] [--algorithms all|Orchestrate,Deepsyn,Syn4,C2RS] [--input-source all|base_aigs|tier1|tier2]"
            echo ""
            echo "Examples:"
            echo "  sbatch slurm_jobs/jobs_8_optimization/job_8_make_optimize_scripts.sh --design-group all"
            echo "  sbatch slurm_jobs/jobs_8_optimization/job_8_make_optimize_scripts.sh --designs '128,256,512'"
            echo "  sbatch slurm_jobs/jobs_8_optimization/job_8_make_optimize_scripts.sh --design-group random --designs '128,256'"
            echo "  sbatch slurm_jobs/jobs_8_optimization/job_8_make_optimize_scripts.sh --design-group openabc --algorithms C2RS"
            echo "  sbatch slurm_jobs/jobs_8_optimization/job_8_make_optimize_scripts.sh --input-source tier1 --algorithms 'C2RS,Syn4'"
            echo "  sbatch slurm_jobs/jobs_8_optimization/job_8_make_optimize_scripts.sh --input-source all --algorithms C2RS"
            exit 0
            ;;
        *)
            echo "✗ ERROR: Unknown argument: $1"
            echo "Run with --help for usage"
            exit 1
            ;;
    esac
done

if [[ "$DESIGN_GROUP" == "openabcd" ]]; then
    DESIGN_GROUP="openabc"
fi

if [[ "$DESIGN_GROUP" != "all" && "$DESIGN_GROUP" != "random" && "$DESIGN_GROUP" != "openabc" ]]; then
    echo "✗ ERROR: Invalid --design-group '$DESIGN_GROUP' (expected all|random|openabc|openabcd)"
    exit 1
fi

if [[ "$INPUT_SOURCE" != "all" && "$INPUT_SOURCE" != "base_aigs" && "$INPUT_SOURCE" != "tier1" && "$INPUT_SOURCE" != "tier2" ]]; then
    echo "✗ ERROR: Invalid --input-source '$INPUT_SOURCE' (expected all|base_aigs|tier1|tier2)"
    exit 1
fi

if [[ "$ALGORITHMS" != "all" ]]; then
    IFS=',' read -r -a requested_algorithms <<< "$ALGORITHMS"
    if [[ ${#requested_algorithms[@]} -eq 0 ]]; then
        echo "✗ ERROR: --algorithms provided but empty"
        exit 1
    fi
    for alg in "${requested_algorithms[@]}"; do
        trimmed="$(echo "$alg" | xargs)"
        if [[ "$trimmed" != "Orchestrate" && "$trimmed" != "Deepsyn" && "$trimmed" != "Syn4" && "$trimmed" != "C2RS" ]]; then
            echo "✗ ERROR: Invalid algorithm '$trimmed' in --algorithms (allowed: Orchestrate, Deepsyn, Syn4, C2RS)"
            exit 1
        fi
    done
fi

echo "STEP 8 start: job=${SLURM_JOB_ID:-local} host=$(hostname) time=$(date)"

module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0

BASE_DIR="$HOME/data-gen-rand-abcd"
FULL_DATASET="${FULL_DATASET:-/scratch-shared/$USER/FULL_DATASET}"
GEN_SCRIPT="${BASE_DIR}/dataset_tools/generate_optimization_bulk_scripts.py"
CONFIG_FILE="${BASE_DIR}/dataset_tools/optimization_config.json"
SCRIPTS_ZIP_ROOT="${FULL_DATASET}/synScripts/optimization"

if [ ! -d "$FULL_DATASET" ]; then
    echo "✗ ERROR: FULL_DATASET not found: $FULL_DATASET"
    echo "  Override path: FULL_DATASET=/path/to/FULL_DATASET sbatch slurm_jobs/jobs_8_optimization/job_8_make_optimize_scripts.sh"
    exit 1
fi

if [ ! -d "$FULL_DATASET/base_aigs" ]; then
    echo "✗ ERROR: Missing base_aigs directory: $FULL_DATASET/base_aigs"
    exit 1
fi

if [ ! -f "$GEN_SCRIPT" ]; then
    echo "✗ ERROR: Generator script not found: $GEN_SCRIPT"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "✗ ERROR: Optimization config not found: $CONFIG_FILE"
    exit 1
fi

GEN_ARGS=(
    --base-dir "$BASE_DIR"
    --full-dataset "$FULL_DATASET"
    --config "$CONFIG_FILE"
    --design-group "$DESIGN_GROUP"
    --algorithms "$ALGORITHMS"
    --input-source "$INPUT_SOURCE"
)

if [[ -n "$DESIGNS" ]]; then
    GEN_ARGS+=(--designs "$DESIGNS")
fi

python3 "$GEN_SCRIPT" "${GEN_ARGS[@]}"

if [ ! -d "$SCRIPTS_ZIP_ROOT" ]; then
    echo "✗ ERROR: Script zip root not found: $SCRIPTS_ZIP_ROOT"
    exit 1
fi

MANIFEST_DIR="${FULL_DATASET}/optimized_aigs/manifests"
latest_manifest=$(ls -1t "$MANIFEST_DIR"/bulk_scripts_manifest_*.json 2>/dev/null | head -n 1 || true)
if [ -z "$latest_manifest" ]; then
    echo "✗ ERROR: No manifest found under: $MANIFEST_DIR"
    exit 1
fi

verify_result=$(python3 - "$latest_manifest" "$SCRIPTS_ZIP_ROOT" <<'PY'
import json
import sys
import zipfile
from pathlib import Path

manifest_path = Path(sys.argv[1])
zip_root = Path(sys.argv[2])

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
designs = manifest.get("designs", [])
algorithms = manifest.get("algorithms", [])
input_sources = manifest.get("input_sources", [])
label_map = {"base_aigs": "base", "tier1": "tier1", "tier2": "tier2"}

expected = len(designs) * len(algorithms) * len(input_sources)
actual = 0
missing = []
missing_zips = []

for design in designs:
    zip_path = zip_root / f"{design}.zip"
    if not zip_path.exists():
        missing_zips.append(str(zip_path))
        continue

    with zipfile.ZipFile(zip_path, "r") as archive:
        members = set(archive.namelist())

    for algorithm in algorithms:
        for source in input_sources:
            label = label_map[source]
            script_name = f"optimizeBulk_{algorithm}_{design}_{label}.sh"
            if script_name in members:
                actual += 1
            else:
                missing.append(script_name)

print(f"expected={expected} actual={actual} designs={len(designs)} algs={len(algorithms)} sources={len(input_sources)}")

if missing_zips:
    print("missing_design_zips=" + str(len(missing_zips)))
if missing:
    print("missing_scripts=" + str(len(missing)))

if missing_zips or missing or actual != expected:
    sys.exit(1)
PY
)

echo "Verification: ${verify_result}"

echo "STEP 8 complete: manifest=${latest_manifest} time=$(date)"
