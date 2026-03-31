#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=genoa
#SBATCH --job-name=abc_bench_to_aig
#SBATCH --output=logs/abc_bench_to_aig_%j.out

set -euo pipefail

# 1. Setup Environment & Modules
module purge
module load 2025
module load foss/2025a
# Add any other modules you need for ABC or basic utilities

mkdir -p logs

echo "=========================================="
echo "JOB: Convert BENCH -> AIG using ABC"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-manual_run}"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [design...]

Converts <design>/tier0/<design>_synX_step0.bench -> .aig using ABC.
Defaults:
  DATA_DIR: data/designs
  ABC_BIN:  $HOME/abc/abc

Options:
  --data-dir DIR   Directory containing per-design folders (default: data/designs)
  --abc PATH       Path to ABC binary (default: $HOME/abc/abc)
  --workers N      Parallel workers (defaults to SLURM_CPUS_PER_TASK)
  -h, --help       Show this help

If no designs are provided, script will attempt to process all subdirectories in DATA_DIR.
EOF
}

# 2. Map Workers to Slurm Allocation
DATA_DIR="${DATA_DIR:-data/designs}"
ABC_BIN="${ABC_BIN:-$HOME/abc/abc}"
WORKERS="${SLURM_CPUS_PER_TASK:-4}" # Dynamically grab Slurm CPUs

# simple arg parsing
while [[ ${1:-} != "" ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2;;
    --abc) ABC_BIN="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    --) shift; break;;
    -*) echo "Unknown option: $1"; usage; exit 1;;
    *) break;;
  esac
done

if [ $# -gt 0 ]; then
  DESIGNS=("$@")
else
  if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: data dir not found: $DATA_DIR" >&2
    exit 1
  fi
  mapfile -t DESIGNS < <(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null || true)
fi

if [ ! -x "$ABC_BIN" ]; then
  echo "ERROR: ABC binary not found or not executable: $ABC_BIN" >&2
  echo "Set ABC_BIN=/path/to/abc or use --abc PATH" >&2
  exit 1
fi

convert_one() {
  local design="$1"
  local design_dir="$DATA_DIR/$design"
  local bench="$design_dir/tier0/${design}_synX_step0.bench"
  local aig="$design_dir/tier0/${design}_synX_step0.aig"

  if [ ! -d "$design_dir" ]; then
    echo "Skipping missing design dir: $design_dir"
    return
  fi

  if [ -f "$aig" ]; then
    echo "Already present: $aig (skipping)"
    return
  fi

  if [ ! -f "$bench" ]; then
    echo "No bench for $design: $bench (skipping)"
    return
  fi

  echo "Converting: $bench -> $aig"
  if ! "$ABC_BIN" -c "read_bench '$bench'; strash; write '$aig'" >/dev/null 2>&1; then
    echo "ERROR: conversion failed for $bench" >&2
    return 1
  fi
  echo "Done: $aig"
}

# Export functions and variables so xargs can use them
export DATA_DIR
export ABC_BIN
export -f convert_one

echo "Processing ${#DESIGNS[@]} designs using $WORKERS workers..."

# 3. Parallel Execution Using xargs
printf "%s\n" "${DESIGNS[@]}" | xargs -n 1 -P "$WORKERS" bash -c 'convert_one "$0"'

echo "=========================================="
echo "JOB complete."
echo "End time: $(date)"