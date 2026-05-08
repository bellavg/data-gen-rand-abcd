#!/bin/bash
#SBATCH --job-name=inspect_data
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=genoa
#SBATCH --output=logs/inspect_data_%j.out

set -euo pipefail

echo "=========================================="
echo "READ-ONLY DATA AUDIT"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# Optional module setup when running on cluster.
if command -v module >/dev/null 2>&1; then
	module purge || true
	module load 2025 || true
	module load foss/2025a || true
	module load Python/3.13.1-GCCcore-14.2.0 || true
fi

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"

AIG_ROOT="${AIG_ROOT:-$BASE_DIR/data/designs}"
PT_ROOT="${PT_ROOT:-/scratch-shared/$USER}"
TIER2_AIG_ROOT="${TIER2_AIG_ROOT:-/scratch-shared/$USER/data-gen-rand-abcd/tier2_aigs}"

WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-24}}"
EXAMPLES="${EXAMPLES:-5}"
INCLUDE_PER_DESIGN_CSV="${INCLUDE_PER_DESIGN_CSV:-0}"

CSV_GLOBS=(
	"$BASE_DIR/data/designs/design_metadata/algo_*_ml.csv"
	"$BASE_DIR/data/designs/design_metadata/full_master*.csv"
)

if [[ "$INCLUDE_PER_DESIGN_CSV" == "1" ]]; then
	CSV_GLOBS+=("$BASE_DIR/data/designs/*/design_metadata/*.csv")
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "$VENV_PATH/bin/python" ]]; then
	PYTHON_BIN="$VENV_PATH/bin/python"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "ERROR: python executable not found: $PYTHON_BIN" >&2
	exit 1
fi

ARGS=(
	"$BASE_DIR/src/data/inspect_data.py"
	--aig-root "$AIG_ROOT"
	--pt-root "$PT_ROOT"
	--workers "$WORKERS"
	--examples "$EXAMPLES"
)

for g in "${CSV_GLOBS[@]}"; do
	ARGS+=(--csv-glob "$g")
done

if [[ -d "$TIER2_AIG_ROOT" ]]; then
	ARGS+=(--tier2-aig-root "$TIER2_AIG_ROOT")
else
	echo "[info] Tier2 AIG root not found, skipping: $TIER2_AIG_ROOT"
fi

# Colon-separated optional extras.
if [[ -n "${EXTRA_TIER2_AIG_ROOTS:-}" ]]; then
	IFS=':' read -r -a extra_t2_roots <<< "$EXTRA_TIER2_AIG_ROOTS"
	for p in "${extra_t2_roots[@]}"; do
		if [[ -n "$p" ]]; then
			ARGS+=(--tier2-aig-root "$p")
		fi
	done
fi

if [[ -n "${EXTRA_CSV_FILES:-}" ]]; then
	IFS=':' read -r -a extra_csv_files <<< "$EXTRA_CSV_FILES"
	for p in "${extra_csv_files[@]}"; do
		if [[ -n "$p" ]]; then
			ARGS+=(--csv-file "$p")
		fi
	done
fi

if [[ -n "${EXTRA_CSV_GLOBS:-}" ]]; then
	IFS=':' read -r -a extra_csv_globs <<< "$EXTRA_CSV_GLOBS"
	for p in "${extra_csv_globs[@]}"; do
		if [[ -n "$p" ]]; then
			ARGS+=(--csv-glob "$p")
		fi
	done
fi

echo "Audit configuration:"
echo "  BASE_DIR=$BASE_DIR"
echo "  AIG_ROOT=$AIG_ROOT"
echo "  PT_ROOT=$PT_ROOT"
echo "  TIER2_AIG_ROOT=$TIER2_AIG_ROOT"
echo "  WORKERS=$WORKERS"
echo "  EXAMPLES=$EXAMPLES"
echo "  INCLUDE_PER_DESIGN_CSV=$INCLUDE_PER_DESIGN_CSV"
echo "  PYTHON_BIN=$PYTHON_BIN"
echo ""
echo "NOTE: This script is read-only. It only scans files and prints a report."
echo ""

"$PYTHON_BIN" -u "${ARGS[@]}"

echo "=========================================="
echo "Audit finished: $(date)"
echo "=========================================="
