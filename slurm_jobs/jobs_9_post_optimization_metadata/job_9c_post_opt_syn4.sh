#!/bin/bash
#SBATCH --job-name=9c_post_syn4
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --output=logs/9c_post_syn4_%j.out

set -euo pipefail

export ALGORITHM="Syn4"

bash "$SLURM_SUBMIT_DIR/slurm_jobs/jobs_9_post_optimization_metadata/job_9_worker_post_optimization_metadata.sh"