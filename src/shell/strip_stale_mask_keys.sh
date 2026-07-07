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
#   - Sequential loop, no stat() calls on all files upfront.
#   - mmap=True for fast key-check on clean files (lazy, no full read).
#   - Only dirty files (stale keys found) are fully reloaded + saved.
#   - Resumable: already-clean files skip in milliseconds.
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

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "$VENV_PATH/bin/python" ]]; then
    PYTHON_BIN="$VENV_PATH/bin/python"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

echo "CACHE_ROOT=$CACHE_ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo ""

N=$(find "$CACHE_ROOT" -maxdepth 1 -name "*.pt" | wc -l)
echo "Found $N .pt files to scan."
if [[ "$N" -eq 0 ]]; then echo "Nothing to clean. Exiting."; exit 0; fi

"$PYTHON_BIN" -u - "$CACHE_ROOT" << 'PYEOF'
import sys, os
from pathlib import Path
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


def has_stale_keys(obj) -> list:
    """Return list of stale key names found on obj, or empty list."""
    if not hasattr(obj, "keys"):
        return []
    return [k for k in obj.keys() if any(k.endswith(s) for s in STALE_SUFFIXES)]


def main(cache_root: str) -> None:
    paths = sorted(Path(cache_root).glob("*.pt"))  # no stat() calls here
    total = len(paths)
    print(f"Processing {total} files ...", flush=True)

    changed = errors = 0
    for i, p in enumerate(paths, 1):
        path = str(p)
        try:
            # Fast path: mmap=True is lazy — only file metadata is accessed
            # to check keys. Clean files (no stale keys) cost ~1 file open.
            obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            stale = has_stale_keys(obj)
            del obj  # release mmap immediately

            if stale:
                # Dirty file: reload fully (mmap=False so we can save it)
                try:
                    obj = torch.load(path, map_location="cpu", weights_only=True, mmap=False)
                except Exception:
                    obj = torch.load(path, map_location="cpu", weights_only=False)

                for k in stale:
                    delattr(obj, k)

                tmp = path + ".tmp_strip"
                torch.save(obj, tmp)
                os.replace(tmp, path)
                del obj
                print(f"  FIXED {path}: removed {stale}", flush=True)
                changed += 1

        except Exception as exc:
            print(f"  FAIL  {path}: {exc}", flush=True)
            errors += 1

        if i % 10000 == 0:
            print(f"  ... {i}/{total} done  ({changed} fixed, {errors} errors)", flush=True)

    print(f"\nDone. {changed} files cleaned, {errors} errors out of {total} scanned.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
PYEOF

echo "=========================================="
echo "Strip finished: $(date)"
echo "=========================================="
