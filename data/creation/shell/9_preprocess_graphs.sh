#!/bin/bash
#SBATCH --job-name=9_preprocess_graphs
#SBATCH --time=12:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --constraint=scratch-node
#SBATCH --partition=genoa
#SBATCH --output=logs/preprocess_%j.out

set -euo pipefail

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
AIG_ROOT="${AIG_ROOT:-$HOME/data-gen-rand-abcd/data/designs}"
FINAL_OUT="${FINAL_OUT:-/scratch-shared/$USER/}"
VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-96}}"
AIG_DEBUG_PATH_COUNTS="${AIG_DEBUG_PATH_COUNTS:-0}"
FAIL_FAST="${FAIL_FAST:-1}"
STAGE_TO_SCRATCH="${STAGE_TO_SCRATCH:-1}"
STAGE_ONLY_MISSING="${STAGE_ONLY_MISSING:-0}"
STAGE_WORKERS="${STAGE_WORKERS:-12}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10000}"


# Load cluster environment directly (SLURM node).
module load 2025

source "$VENV_PATH/bin/activate"

echo "=========================================="
echo "JOB: PyG Graph Preprocessing"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-manual_run}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Base dir: $BASE_DIR"
echo "AIG root: $AIG_ROOT"
echo "Final out: $FINAL_OUT"
echo "Venv: $VENV_PATH"
echo "Workers: $WORKERS"
echo "Debug path counts: $AIG_DEBUG_PATH_COUNTS"
echo "Fail fast: $FAIL_FAST"
echo "Stage to scratch: $STAGE_TO_SCRATCH"
echo "Stage only missing: $STAGE_ONLY_MISSING"
echo "Stage workers: $STAGE_WORKERS"
echo "Progress every: $PROGRESS_EVERY"
echo "=========================================="

if [[ -n "${TMPDIR:-}" ]]; then
  LOCAL_SCRATCH="$TMPDIR"
else
  LOCAL_SCRATCH="/scratch-shared/$USER/tmp"
fi
mkdir -p "$LOCAL_SCRATCH"

WORK_DIR=$(mktemp -d "$LOCAL_SCRATCH/preprocess_aigs_${SLURM_JOB_ID:-manual}_XXXXXX")
trap 'rm -rf "$WORK_DIR"' EXIT

STAGED_AIG_ROOT="$WORK_DIR/designs"

if [[ "$STAGE_TO_SCRATCH" == "1" ]]; then
  echo "Staging tier0/tier1 AIG zips into local scratch: $STAGED_AIG_ROOT"
  mkdir -p "$STAGED_AIG_ROOT"

  export PYTHONPATH="$BASE_DIR/src:${PYTHONPATH:-}"
  export STAGE_AIG_ROOT="$AIG_ROOT"
  export STAGE_FINAL_OUT="$FINAL_OUT"
  export STAGE_DST_ROOT="$STAGED_AIG_ROOT"
  export STAGE_ONLY_MISSING_ENV="$STAGE_ONLY_MISSING"
    export STAGE_WORKERS_ENV="$STAGE_WORKERS"

  python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shutil

from data.data_utils import parse_aig_name

src_root = Path(os.environ["STAGE_AIG_ROOT"])
final_out = Path(os.environ["STAGE_FINAL_OUT"])
dst_root = Path(os.environ["STAGE_DST_ROOT"])
only_missing = os.environ.get("STAGE_ONLY_MISSING_ENV", "1") == "1"
stage_workers = max(1, int(os.environ.get("STAGE_WORKERS_ENV", "8")))

def output_path(tier_id: int, algo: str, design: str, filename: str) -> Path:
    stem = Path(filename).stem
    if tier_id == 0:
        return final_out / "graphs" / "tier0" / design / f"{stem}.pt"
    if tier_id == 1:
        return final_out / "graphs" / "tier1" / algo / design / f"{stem}.pt"
    return final_out / "graphs" / f"tier{tier_id}" / design / f"{stem}.pt"

def zip_aig_entries(zip_path: Path):
    with ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".aig"):
                continue
            name = Path(info.filename).name
            parsed = parse_aig_name(name)
            if parsed is None:
                continue
            yield info.filename, name, parsed


def should_stage_zip(design: str, zip_path: Path) -> tuple[bool, int, int]:
    valid_entries = list(zip_aig_entries(zip_path))
    if not valid_entries:
        return False, 0, 0

    if not only_missing:
        return True, len(valid_entries), 0

    existing = 0
    for _member, name, (tier_id, algo, parsed_design) in valid_entries:
        design_name = parsed_design or design
        if output_path(tier_id, algo, design_name, name).exists():
            existing += 1
        else:
            return True, len(valid_entries), existing

    return False, len(valid_entries), existing


def stage_design(design_dir: Path) -> dict:
    design = design_dir.name
    result = {
        "design": design,
        "zip_files": 0,
        "zip_aig_seen": 0,
        "staged": 0,
        "skipped_existing": 0,
        "errors": [],
    }

    zip_paths = []
    t0a = design_dir / "tier0.zip"
    t0b = design_dir / "tier0" / "tier0.zip"
    if t0a.exists():
        zip_paths.append((t0a, 0))
    elif t0b.exists():
        zip_paths.append((t0b, 0))

    t1_dir = design_dir / "tier1"
    if t1_dir.is_dir():
        for z in sorted(t1_dir.glob("*.zip")):
            zip_paths.append((z, 1))

    out_tier0 = dst_root / design / "tier0"
    out_tier1 = dst_root / design / "tier1"
    out_tier0.mkdir(parents=True, exist_ok=True)
    out_tier1.mkdir(parents=True, exist_ok=True)

    for zip_path, default_tier in zip_paths:
        result["zip_files"] += 1
        try:
            needed, valid_count, existing = should_stage_zip(design, zip_path)
            result["zip_aig_seen"] += valid_count
            if not needed:
                result["skipped_existing"] += valid_count
                continue

            out_dir = out_tier1 if default_tier == 1 else out_tier0
            with ZipFile(zip_path, "r") as zf:
                zf.extractall(path=out_dir)
            result["staged"] += valid_count
            result["skipped_existing"] += existing
        except Exception as exc:
            result["errors"].append(f"{zip_path}: {exc}")

    return result


design_dirs = sorted(
    [p for p in src_root.iterdir() if p.is_dir() and p.name not in {"design_metadata", "logs"}]
)
print(f"staging: parallel design workers={stage_workers} designs={len(design_dirs)}")

stats = {
    "zip_files": 0,
    "zip_aig_seen": 0,
    "staged": 0,
    "skipped_existing": 0,
    "errors": 0,
}

with ThreadPoolExecutor(max_workers=stage_workers) as ex:
    futures = {ex.submit(stage_design, d): d.name for d in design_dirs}
    completed = 0
    for fut in as_completed(futures):
        design = futures[fut]
        completed += 1
        try:
            res = fut.result()
        except Exception as exc:
            stats["errors"] += 1
            print(f"staging: design failed {design}: {exc}")
            continue

        stats["zip_files"] += res["zip_files"]
        stats["zip_aig_seen"] += res["zip_aig_seen"]
        stats["staged"] += res["staged"]
        stats["skipped_existing"] += res["skipped_existing"]
        if res["errors"]:
            stats["errors"] += len(res["errors"])
            for err in res["errors"][:3]:
                print(f"staging: warning {design}: {err}")

        print(
            f"staging: completed design [{completed}/{len(design_dirs)}] {design} "
            f"staged={res['staged']} skipped_existing={res['skipped_existing']}"
        )

if stats["errors"] > 0:
    raise RuntimeError(f"staging encountered {stats['errors']} error(s)")

print(
    "staging: summary "
    f"zip_files={stats['zip_files']} zip_aig_seen={stats['zip_aig_seen']} "
    f"staged={stats['staged']} skipped_existing={stats['skipped_existing']}"
)
PY

  STAGED_COUNT=$(find "$STAGED_AIG_ROOT" -type f -name "*.aig" | wc -l)
  echo "Staging complete: staged_aigs=$STAGED_COUNT"
  EFFECTIVE_AIG_ROOT="$STAGED_AIG_ROOT"
else
  EFFECTIVE_AIG_ROOT="$AIG_ROOT"
fi

if [[ "$FAIL_FAST" == "1" ]]; then
  FAIL_FAST_FLAG="--fail-fast"
else
  FAIL_FAST_FLAG="--no-fail-fast"
fi

AIG_DEBUG_PATH_COUNTS="$AIG_DEBUG_PATH_COUNTS" \
python -u -m data.preprocess_data \
  --aig-root "$EFFECTIVE_AIG_ROOT" \
  --final-out "$FINAL_OUT" \
  --workers "$WORKERS" \
  --progress-every "$PROGRESS_EVERY" \
  --allow-unmatched-names \
  --overwrite \
  "$FAIL_FAST_FLAG"

echo "Finished: $(date)"