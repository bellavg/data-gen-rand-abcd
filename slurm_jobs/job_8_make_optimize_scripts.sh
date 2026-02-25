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
DEFAULT_DESIGN_GROUP="${DEFAULT_DESIGN_GROUP:-random}"
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
            echo "Usage: sbatch slurm_jobs/job_8_make_optimize_scripts.sh [--design-group all|random|openabc|openabcd] [--designs 'd1,d2,...'] [--algorithms all|Orchestrate,Deepsyn,Syn4,C2RS] [--input-source all|base_aigs|tier1|tier2]"
            echo ""
            echo "Examples:"
            echo "  sbatch slurm_jobs/job_8_make_optimize_scripts.sh --design-group random"
            echo "  sbatch slurm_jobs/job_8_make_optimize_scripts.sh --designs '128,256,512'"
            echo "  sbatch slurm_jobs/job_8_make_optimize_scripts.sh --design-group random --designs '128,256'"
            echo "  sbatch slurm_jobs/job_8_make_optimize_scripts.sh --design-group random --algorithms C2RS"
            echo "  sbatch slurm_jobs/job_8_make_optimize_scripts.sh --input-source tier1 --algorithms 'C2RS,Syn4'"
            echo "  sbatch slurm_jobs/job_8_make_optimize_scripts.sh --input-source all --algorithms C2RS"
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

echo "=========================================="
echo "STEP 8: Generating Optimization Bulk Scripts"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0

echo "Loaded modules: 2025, foss/2025a, Python/3.13.1"
echo ""

BASE_DIR="$HOME/data-gen-rand-abcd"
FULL_DATASET="${FULL_DATASET:-/scratch-shared/$USER/FULL_DATASET}"
GEN_SCRIPT="${BASE_DIR}/dataset_tools/generate_optimization_bulk_scripts.py"
CONFIG_FILE="${BASE_DIR}/dataset_tools/optimization_config.json"
SCRIPTS_ZIP_ROOT="${FULL_DATASET}/synScripts/optimization"

echo "Configuration:"
echo "  Base directory: $BASE_DIR"
echo "  Full dataset:   $FULL_DATASET"
echo "  Generator:      $GEN_SCRIPT"
echo "  Config source:  $CONFIG_FILE"
echo "  Design group:   $DESIGN_GROUP"
echo "  Designs:        ${DESIGNS:-<all in selected group>}"
echo "  Algorithms:     $ALGORITHMS"
echo "  Input source:   $INPUT_SOURCE"
echo "  Script zip root:$SCRIPTS_ZIP_ROOT"
echo ""

if [ ! -d "$FULL_DATASET" ]; then
    echo "✗ ERROR: FULL_DATASET not found: $FULL_DATASET"
    echo "  Override path: FULL_DATASET=/path/to/FULL_DATASET sbatch slurm_jobs/job_8_make_optimize_scripts.sh"
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

echo "Generating optimization bulk scripts..."
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

echo ""
echo "Verifying generated script zip bundles..."
if [[ "$ALGORITHMS" == "all" ]]; then
    algorithms_to_check=(Orchestrate Deepsyn Syn4 C2RS)
else
    IFS=',' read -r -a algorithms_to_check <<< "$ALGORITHMS"
fi

if [[ "$INPUT_SOURCE" == "all" ]]; then
    source_labels=(base tier1 tier2)
elif [[ "$INPUT_SOURCE" == "base_aigs" ]]; then
    source_labels=(base)
elif [[ "$INPUT_SOURCE" == "tier1" ]]; then
    source_labels=(tier1)
else
    source_labels=(tier2)
fi

if [ ! -d "$SCRIPTS_ZIP_ROOT" ]; then
    echo "✗ ERROR: Script zip root not found: $SCRIPTS_ZIP_ROOT"
    exit 1
fi

zip_count=$(find "$SCRIPTS_ZIP_ROOT" -type f -name '*.zip' | wc -l | tr -d ' ')
if [ "$zip_count" -eq 0 ]; then
    echo "✗ ERROR: No design zip bundles found in: $SCRIPTS_ZIP_ROOT"
    exit 1
fi

echo "Found ${zip_count} design zip bundles"

for algorithm in "${algorithms_to_check[@]}"; do
    algorithm="$(echo "$algorithm" | xargs)"
    for source_label in "${source_labels[@]}"; do
        pattern="optimizeBulk_${algorithm}_*_${source_label}.sh"
        match_count=$(find "$SCRIPTS_ZIP_ROOT" -type f -name '*.zip' -exec sh -c 'for z in "$@"; do unzip -Z1 "$z" "$0" 2>/dev/null; done' "$pattern" {} + | wc -l | tr -d ' ')
        if [ "$match_count" -gt 0 ]; then
            echo "  ✓ ${algorithm} (${source_label}): ${match_count} scripts in zip bundles"
        else
            echo "  ✗ ${algorithm} (${source_label}): No scripts found in zip bundles"
            exit 1
        fi
    done
done

echo ""
echo "=========================================="
echo "Step 8 Complete"
echo "=========================================="
echo ""
total_scripts=$(find "$SCRIPTS_ZIP_ROOT" -type f -name '*.zip' -exec sh -c 'for z in "$@"; do unzip -Z1 "$z" "optimizeBulk_*.sh" 2>/dev/null; done' _ {} + | wc -l | tr -d ' ')
echo "Generated: ${total_scripts} optimization shard scripts"
echo "Location: $SCRIPTS_ZIP_ROOT"
echo ""
echo "Next step: Submit per-algorithm jobs"
echo ""
echo "Tier 1 (default):"
echo "  sbatch slurm_jobs/job_8a_optimize_orchestrate.sh"
echo "  sbatch slurm_jobs/job_8b_optimize_deepsyn.sh"
echo "  sbatch slurm_jobs/job_8c_optimize_syn4.sh"
echo "  sbatch slurm_jobs/job_8d_optimize_c2rs.sh"
echo ""
echo "Tier 2 (same jobs, set TIER=2):"
echo "  sbatch --export=ALL,TIER=2 slurm_jobs/job_8a_optimize_orchestrate.sh"
echo "  sbatch --export=ALL,TIER=2 slurm_jobs/job_8b_optimize_deepsyn.sh"
echo "  sbatch --export=ALL,TIER=2 slurm_jobs/job_8c_optimize_syn4.sh"
echo "  sbatch --export=ALL,TIER=2 slurm_jobs/job_8d_optimize_c2rs.sh"
echo ""
echo "End time: $(date)"
