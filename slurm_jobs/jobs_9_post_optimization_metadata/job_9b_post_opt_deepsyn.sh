#!/bin/bash
#SBATCH --job-name=9b_post_deepsyn
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/9b_post_deepsyn_%j.out

set -euo pipefail

export ALGORITHM="Deepsyn"

export DESIGN_GROUP="openabc"

bash "$SLURM_SUBMIT_DIR/slurm_jobs/jobs_9_post_optimization_metadata/job_9_worker_post_optimization_metadata.sh"
