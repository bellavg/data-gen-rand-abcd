#!/bin/bash
#SBATCH --job-name=9a_post_orch
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --output=logs/9a_post_orchestrate_%j.out

set -euo pipefail

export ALGORITHM="Orchestrate"

bash "$SLURM_SUBMIT_DIR/slurm_jobs/jobs_9_post_optimization_metadata/job_9_worker_post_optimization_metadata.sh"
