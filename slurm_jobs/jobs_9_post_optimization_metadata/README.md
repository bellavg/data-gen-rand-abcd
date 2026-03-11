# Job 9: Post-Optimization Metadata (Tier0 + Tier1)

This folder contains per-algorithm Job 9 scripts to run **after Job 8** finishes.

## What Each Job 9 Script Does

For one algorithm at a time, the pipeline does:

1. Pre-check Tier-1 completeness:
   - reads per-design summary at `metadata/raw_logs/{design}/tier1/{algorithm}/summary.json`
   - verifies `failed == 0`, `processed == discovered`, `created == discovered`
   - verifies file count in `optimized_aigs/{algorithm}/tier1/{design}` matches `discovered`
2. Populate metadata CSVs:
   - runs `dataset_tools/update_optimization_metadata.py` for each selected design
   - appends Tier-1 rows into `metadata/stats/{design}.csv`
3. Post-check CSV correctness:
   - checks Tier-1 CSV row count equals Tier-1 graph file count for this algorithm
   - checks Tier-0 rows still exist in each CSV
4. Print and save full dataset stats:
   - writes `metadata/stats/job9_full_stats_<timestamp>.json`
   - writes `metadata/stats/job9_full_stats_<timestamp>.txt`
5. Create timestamped backup ZIP:
   - writes `/scratch-shared/$USER/dataset_backups/FULL_DATASET_<timestamp>.zip`

## Scripts

- `job_9a_post_opt_orchestrate.sh`
- `job_9b_post_opt_deepsyn.sh`
- `job_9c_post_opt_syn4.sh`
- `job_9d_post_opt_c2rs.sh`
- `job_9_worker_post_optimization_metadata.sh` (shared implementation)

## Submit Examples

```bash
sbatch slurm_jobs/jobs_9_post_optimization_metadata/job_9a_post_opt_orchestrate.sh
sbatch slurm_jobs/jobs_9_post_optimization_metadata/job_9b_post_opt_deepsyn.sh
sbatch slurm_jobs/jobs_9_post_optimization_metadata/job_9c_post_opt_syn4.sh
sbatch slurm_jobs/jobs_9_post_optimization_metadata/job_9d_post_opt_c2rs.sh
```

## Useful Overrides

All overrides are environment variables at submit time:

```bash
sbatch \
  --cpus-per-task=72 \
  --export=ALL,FULL_DATASET=/scratch-shared/$USER/FULL_DATASET,DESIGN_GROUP=all,METADATA_WORKERS=24,ARCHIVE_FULL_DATASET=true \
  slurm_jobs/jobs_9_post_optimization_metadata/job_9b_post_opt_deepsyn.sh
```

Supported variables:

- `FULL_DATASET` (default: `/scratch-shared/$USER/FULL_DATASET`)
- `BASE_DIR` (default: `~/data-gen-rand-abcd`)
- `DESIGN_GROUP` (`all` | `random` | `openabc`, default: `all`)
- `DESIGNS` (optional explicit list, comma/space separated)
- `METADATA_WORKERS` (default: `SLURM_CPUS_PER_TASK`)
- `ARCHIVE_FULL_DATASET` (`true` | `false`, default: `true`)
- `BACKUP_DIR` (default: `/scratch-shared/$USER/dataset_backups`)

## Notes

- Metadata population can be very long for large Tier-1 outputs.
- Re-running is safe: existing rows are deduplicated by `(file_path, algorithm, tier_id)`.
- If you only want checks + stats (no ZIP backup), set `ARCHIVE_FULL_DATASET=false`.
