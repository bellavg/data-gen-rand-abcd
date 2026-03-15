#!/bin/bash
#SBATCH --job-name=9d_post_c2rs
#SBATCH --time=96:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --output=logs/9d_post_c2rs_%j.out

set -euo pipefail

export ALGORITHM="C2RS"

bash "$SLURM_SUBMIT_DIR/slurm_jobs/jobs_9_post_optimization_metadata/job_9_worker_post_optimization_metadata.sh"