#!/bin/bash
#SBATCH -p genoa
#SBATCH -t 00:30:00
#SBATCH --job-name=0b_inspect_openabc_stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/0b_inspect_openabc_stats_%j.out

set -euo pipefail

mkdir -p logs

echo "=========================================="
echo "JOB 0B: Inspect OpenABC 'statistics' contents"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-manual_run}"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

FULL_DATASET_DIR="${FULL_DATASET_DIR:-/scratch-shared/$USER/FULL_DATASET}"
STAT_DIR="$FULL_DATASET_DIR/metadata/openabc_raw_statistics"

if [ ! -d "$STAT_DIR" ]; then
    echo "ERROR: OpenABC statistics directory not found: $STAT_DIR"
    exit 1
fi

echo "Statistics root: $STAT_DIR"
echo "Subdirectories and file counts:"
find "$STAT_DIR" -maxdepth 2 -type d -print -exec bash -c 'echo -n "  files: "; find "$0" -maxdepth 1 -type f | wc -l' {} \;

echo "\nSample listing (first 20 files under statistics):"
find "$STAT_DIR" -type f | head -n 20

echo "\nInspecting CSV headers and sample content to detect per-recipe/step columns..."

python3 - <<'PY'
import glob, os, pandas as pd

stat_dir = os.path.normpath(os.environ.get('STAT_DIR'))
inspect_paths = []

for sub in ('finalAig','adp'):
    p = os.path.join(stat_dir, sub)
    if os.path.isdir(p):
        csvs = sorted(glob.glob(os.path.join(p, '*.csv')))
        inspect_paths += csvs[:5]

if not inspect_paths:
    print('No CSVs found in finalAig/adp to inspect.')
    raise SystemExit(0)

keywords = {'recipe','recipe_id','sid','synth_id','step','step_id','file','filename','path'}
summary = {'files_checked':0,'files_with_recipe_or_step_cols':0,'files_with_filecols_containing_step_strings':0,'details':[]}

for f in inspect_paths:
    try:
        df = pd.read_csv(f, nrows=50)
    except Exception as e:
        summary['details'].append((os.path.basename(f),'unreadable',str(e)))
        continue

    cols = [c.lower() for c in df.columns.tolist()]
    has_kw = any(any(kw in c for kw in keywords) for c in cols)

    filecol_contains_step = False
    for c in df.columns:
        if c.lower() in ('file','filename','path','file_path'):
            sample = df[c].astype(str).head(20).tolist()
            for s in sample:
                if any(sub in s for sub in ['_step','_syn','syn','step']):
                    filecol_contains_step = True
                    break
        if filecol_contains_step:
            break

    summary['files_checked'] += 1
    if has_kw:
        summary['files_with_recipe_or_step_cols'] += 1
    if filecol_contains_step:
        summary['files_with_filecols_containing_step_strings'] += 1

    summary['details'].append((os.path.basename(f), cols[:10], has_kw, filecol_contains_step))

print('\nInspection summary:')
print('  files checked:', summary['files_checked'])
print('  files with recipe/step-like columns:', summary['files_with_recipe_or_step_cols'])
print('  files whose file columns reference step/syn strings:', summary['files_with_filecols_containing_step_strings'])

print('\nPer-file details (name, first columns, has_recipe/step_col, filecol_has_step):')
for d in summary['details']:
    print(' ', d)

if summary['files_with_recipe_or_step_cols']>0 or summary['files_with_filecols_containing_step_strings']>0:
    print('\nConclusion: statistics likely contain per-recipe or per-step information (intermediate AIGs).')
else:
    print('\nConclusion: statistics appear to describe final AIGs only (no per-recipe/step columns detected).')
PY

echo "\nJOB 0B complete."
echo "End time: $(date)"
