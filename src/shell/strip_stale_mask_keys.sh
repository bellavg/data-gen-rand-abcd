#!/bin/bash
#SBATCH --job-name=strip_stale_masks
#SBATCH --time=08:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=genoa
#SBATCH --output=logs/strip_stale_masks_%j.out

# ===========================================================================
# strip_stale_mask_keys.sh
#
# One-time cleanup: removes leftover _dynamic_mask / _dynamic_num_partitions
# attributes from cached .pt graph files in shared_tier0_cache.
#
# Strategy:
#   - Small files (< LARGE_MB threshold) → batched threads (fast)
#   - Large files (>= LARGE_MB threshold) → sequential (no OOM)
#   - Already-clean files are skipped quickly on re-runs (resumable)
#
# Submit with:
#   sbatch src/shell/strip_stale_mask_keys.sh
# ===========================================================================

set -euo pipefail

echo "=========================================="
echo "STRIP STALE MASK KEYS"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

if command -v module > /dev/null 2>&1; then
    module purge || true
    module load 2025 || true
    module load foss/2025a || true
    module load Python/3.13.1-GCCcore-14.2.0 || true
fi

VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/$USER/aig_train_run/shared_tier0_cache}"
WORKERS="${WORKERS:-8}"
LARGE_MB="${LARGE_MB:-50}"   # files >= this MB are processed sequentially

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "$VENV_PATH/bin/python" ]]; then
    PYTHON_BIN="$VENV_PATH/bin/python"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

echo "CACHE_ROOT=$CACHE_ROOT"
echo "WORKERS=$WORKERS  LARGE_MB=$LARGE_MB"
echo "PYTHON_BIN=$PYTHON_BIN"
echo ""

N=$(find "$CACHE_ROOT" -maxdepth 1 -name "*.pt" | wc -l)
echo "Found $N .pt files to scan."
if [[ "$N" -eq 0 ]]; then echo "Nothing to clean. Exiting."; exit 0; fi

"$PYTHON_BIN" -u - "$CACHE_ROOT" "$WORKERS" "$LARGE_MB" << 'PYEOF'
import sys, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch, torch.serialization

try:
    from torch_geometric.data import Data
    from torch_geometric.data import storage as _s
    torch.serialization.add_safe_globals([Data, _s.GlobalStorage])
    try:
        import torch_geometric.data.data as _m
        for _n in ("DataTensorAttr", "DataEdgeAttr"):
            _c = getattr(_m, _n, None)
            if _c:
                torch.serialization.add_safe_globals([_c])
    except Exception:
        pass
except ImportError:
    pass

STALE_SUFFIXES = ("_dynamic_mask", "_dynamic_num_partitions")


def clean_file(path: str) -> tuple:
    try:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True, mmap=False)
        except Exception:
            obj = torch.load(path, map_location="cpu", weights_only=False)

        removed = [k for k in list(obj.keys()) if any(k.endswith(s) for s in STALE_SUFFIXES)]
        for k in removed:
            delattr(obj, k)

        if removed:
            tmp = path + ".tmp_strip"
            torch.save(obj, tmp)
            os.replace(tmp, path)
            del obj
            return path, True, removed

        del obj
        return path, False, []
    except Exception as exc:
        return path, False, [f"ERROR: {exc}"]


def process_batch(batch: list[str], n_workers: int) -> tuple[int, int]:
    """Run a bounded batch through the thread pool. Returns (changed, errors)."""
    changed = errors = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for path, was_changed, keys in ex.map(clean_file, batch):
            if keys and str(keys[0]).startswith("ERROR"):
                print(f"  FAIL  {path}: {keys[0]}", flush=True)
                errors += 1
            elif was_changed:
                print(f"  FIXED {path}: removed {keys}", flush=True)
                changed += 1
    return changed, errors


def main(cache_root: str, n_workers: int, large_threshold_mb: float) -> None:
    large_threshold = int(large_threshold_mb * 1024 * 1024)

    all_paths = sorted(Path(cache_root).glob("*.pt"))
    total = len(all_paths)

    # Split by size: small → fast threaded batches, large → sequential
    small = [str(p) for p in all_paths if p.stat().st_size < large_threshold]
    large = [str(p) for p in all_paths if p.stat().st_size >= large_threshold]
    # Process largest files last (most memory-safe order)
    large.sort(key=lambda p: os.path.getsize(p))

    print(f"Total: {total} files  |  small (<{large_threshold_mb:.0f}MB): {len(small)}  |  large: {len(large)}", flush=True)
    print(f"Small files: {n_workers} threads, batch={n_workers * 4}  |  Large files: sequential", flush=True)

    changed = errors = processed = 0

    # --- Small files: threaded in bounded batches ----------------------------
    BATCH = n_workers * 4  # at most this many files in memory at once
    for batch_start in range(0, len(small), BATCH):
        batch = small[batch_start:batch_start + BATCH]
        c, e = process_batch(batch, n_workers)
        changed += c
        errors += e
        processed += len(batch)
        if processed % 10000 < BATCH:
            print(f"  ... {processed}/{total} done", flush=True)

    # --- Large files: sequential ---------------------------------------------
    print(f"\nProcessing {len(large)} large files sequentially ...", flush=True)
    for p in large:
        size_mb = os.path.getsize(p) / 1024 / 1024
        print(f"  [{size_mb:.1f}MB] {p}", flush=True)
        path, was_changed, keys = clean_file(p)
        if keys and str(keys[0]).startswith("ERROR"):
            print(f"  FAIL  {path}: {keys[0]}", flush=True)
            errors += 1
        elif was_changed:
            print(f"  FIXED {path}: removed {keys}", flush=True)
            changed += 1
        processed += 1

    print(f"\nDone. {changed} files cleaned, {errors} errors out of {total} scanned.", flush=True)


if __name__ == "__main__":
    cache_root = sys.argv[1]
    n_workers = int(sys.argv[2])
    large_mb = float(sys.argv[3])
    main(cache_root, n_workers, large_mb)
PYEOF

echo "=========================================="
echo "Strip finished: $(date)"
echo "=========================================="
