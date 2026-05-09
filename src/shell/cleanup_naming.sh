#!/bin/bash
#SBATCH --job-name=cleanup_naming
#SBATCH --time=08:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=genoa
#SBATCH --output=logs/cleanup_naming_%j.out


set -euo pipefail

echo "=========================================="
echo "NAMING CLEANUP"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "=========================================="
echo "SAFETY: default is dry-run. Set APPLY=1 to perform real renames."
echo "=========================================="

# Optional module load on cluster.
if command -v module >/dev/null 2>&1; then
	module purge || true
	module load 2025 || true
	module load foss/2025a || true
	module load Python/3.13.1-GCCcore-14.2.0 || true
fi

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
ISSUES_DIR="${ISSUES_DIR:-$HOME/data-gen-rand-abcd/logs/inspect_issues_22600850}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
APPLY="${APPLY:-1}"
PHASES="${PHASES:-}"      # e.g. "pt csv" to restrict phases; empty = all
NO_VERIFY="${NO_VERIFY:-0}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "$VENV_PATH/bin/python" ]]; then
	PYTHON_BIN="$VENV_PATH/bin/python"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "ERROR: python executable not found: $PYTHON_BIN" >&2
	exit 1
fi

if [[ -z "$ISSUES_DIR" ]]; then
	echo "ERROR: ISSUES_DIR must be set (path to inspect_data issues output directory)." >&2
	echo "  e.g.  ISSUES_DIR=/home/\$USER/data-gen-rand-abcd/logs/inspect_issues_<jobid>" >&2
	exit 1
fi

if [[ ! -d "$ISSUES_DIR" ]]; then
	echo "ERROR: ISSUES_DIR does not exist: $ISSUES_DIR" >&2
	exit 1
fi

ARGS=(
	"$BASE_DIR/src/data/cleanup_naming.py"
	--issues-dir "$ISSUES_DIR"
	--workers "$WORKERS"
)

if [[ "$APPLY" == "1" ]]; then
	ARGS+=(--apply)
	echo "WARNING: APPLY=1 — real renames will be performed."
else
	echo "INFO: Dry-run mode (pass APPLY=1 to perform actual renames)."
fi

if [[ -n "$PHASES" ]]; then
	for phase in $PHASES; do
		ARGS+=(--phase "$phase")
	done
fi

if [[ "$NO_VERIFY" == "1" ]]; then
	ARGS+=(--no-verify)
fi

echo ""
echo "Cleanup configuration:"
echo "  BASE_DIR=$BASE_DIR"
echo "  ISSUES_DIR=$ISSUES_DIR"
echo "  WORKERS=$WORKERS"
echo "  APPLY=$APPLY"
echo "  PHASES=${PHASES:-all}"
echo "  NO_VERIFY=$NO_VERIFY"
echo "  PYTHON_BIN=$PYTHON_BIN"
echo ""

"$PYTHON_BIN" -u "${ARGS[@]}"

echo "=========================================="
echo "Cleanup finished: $(date)"
echo "=========================================="
