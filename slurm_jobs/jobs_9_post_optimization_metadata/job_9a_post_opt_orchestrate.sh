#!/bin/bash
#SBATCH --job-name=9a_post_orch
#SBATCH --time=96:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --output=logs/9a_post_orchestrate_%j.out

set -euo pipefail

export ALGORITHM="Orchestrate"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/job_9_worker_post_optimization_metadata.sh"
