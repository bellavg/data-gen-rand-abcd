#!/bin/bash
#SBATCH --job-name=gen_opt_scripts
#SBATCH --time=00:45:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/gen_opt_scripts_%j.out

# Job 8: Generate Optimization Bulk Scripts
# Mirrors the Job 2/3 pattern:
#   - Use dataset_tools to generate bulk shell scripts
#   - Then run per-algorithm jobs (Job 8a-8d)

set -e

DESIGN_GROUP="${DESIGN_GROUP:-all}"
DESIGNS="${DESIGNS:-}"

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
        -h|--help)
            echo "Usage: sbatch slurm_jobs/job_8_make_optimize_scripts.sh [--design-group all|random|openabc] [--designs 'd1,d2,...']"
            echo ""
            echo "Examples:"
            echo "  sbatch slurm_jobs/job_8_make_optimize_scripts.sh --design-group random"
            echo "  sbatch slurm_jobs/job_8_make_optimize_scripts.sh --designs '128,256,512'"
            echo "  sbatch slurm_jobs/job_8_make_optimize_scripts.sh --design-group random --designs '128,256'"
            exit 0
            ;;
        *)
            echo "✗ ERROR: Unknown argument: $1"
            echo "Run with --help for usage"
            exit 1
            ;;
    esac
done

if [[ "$DESIGN_GROUP" != "all" && "$DESIGN_GROUP" != "random" && "$DESIGN_GROUP" != "openabc" ]]; then
    echo "✗ ERROR: Invalid --design-group '$DESIGN_GROUP' (expected all|random|openabc)"
    exit 1
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
SCRIPTS_DIR="${FULL_DATASET}/optimized_aigs/scripts"

echo "Configuration:"
echo "  Base directory: $BASE_DIR"
echo "  Full dataset:   $FULL_DATASET"
echo "  Generator:      $GEN_SCRIPT"
echo "  Config source:  $CONFIG_FILE"
echo "  Design group:   $DESIGN_GROUP"
echo "  Designs:        ${DESIGNS:-<all in selected group>}"
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
)

if [[ -n "$DESIGNS" ]]; then
    GEN_ARGS+=(--designs "$DESIGNS")
fi

python3 "$GEN_SCRIPT" "${GEN_ARGS[@]}"

echo ""
echo "Verifying generated scripts..."
for algorithm in Orchestrate Deepsyn Syn4 C2RS; do
    algorithm_dir="$SCRIPTS_DIR/${algorithm}"
    if [ ! -d "$algorithm_dir" ]; then
        echo "  ✗ ${algorithm}: Script directory not found: $algorithm_dir"
        exit 1
    fi

    script_count=$(find "$algorithm_dir" -type f -name "optimizeBulk_${algorithm}_*.sh" | wc -l | tr -d ' ')
    if [ "$script_count" -gt 0 ]; then
        chmod +x "$algorithm_dir"/*.sh
        echo "  ✓ ${algorithm}: ${script_count} shard scripts"
    else
        echo "  ✗ ${algorithm}: No shard scripts found"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "Step 8 Complete"
echo "=========================================="
echo ""
total_scripts=$(find "$SCRIPTS_DIR" -type f -name "optimizeBulk_*.sh" | wc -l | tr -d ' ')
echo "Generated: ${total_scripts} optimization shard scripts"
echo "Location: $SCRIPTS_DIR"
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
