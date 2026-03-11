#!/bin/bash
#SBATCH --job-name=9c_post_syn4
#SBATCH --time=96:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --output=logs/9c_post_syn4_%j.out

set -euo pipefail

export ALGORITHM="Syn4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/job_9_worker_post_optimization_metadata.sh"
