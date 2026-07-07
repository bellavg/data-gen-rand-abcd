#!/bin/bash
#SBATCH --job-name=strip_stale_masks
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=genoa
#SBATCH --output=logs/strip_stale_masks_%j.out

# ===========================================================================
# strip_stale_mask_keys.sh
#
# One-time cleanup: removes leftover _dynamic_mask / _dynamic_num_partitions
# attributes from all cached .pt graph files.  These were written by a failed
# partitioning run and cause KeyError in Batch.from_data_list when only SOME
# graphs in a batch carry the attribute.
#
# Submit with:
#   sbatch src/shell/strip_stale_mask_keys.sh
#
# Or run interactively (set CACHE_ROOT override if needed):
#   bash src/shell/strip_stale_mask_keys.sh
# ===========================================================================

set -euo pipefail

echo "=========================================="
echo "STRIP STALE MASK KEYS"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# ---- Module / environment setup (mirrors cleanup_naming.sh) ----------------
if command -v module > /dev/null 2>&1; then
    module purge || true
    module load 2025 || true
    module load foss/2025a || true
    module load Python/3.13.1-GCCcore-14.2.0 || true
fi

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
CACHE_ROOT="${CACHE_ROOT:-/scratch-shared/$USER/aig_train_run}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "$VENV_PATH/bin/python" ]]; then
    PYTHON_BIN="$VENV_PATH/bin/python"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

echo ""
echo "Configuration:"
echo "  CACHE_ROOT=$CACHE_ROOT"
echo "  WORKERS=$WORKERS"
echo "  PYTHON_BIN=$PYTHON_BIN"
echo ""

# ---- Count files -----------------------------------------------------------
N=$(find "$CACHE_ROOT" -name "*.pt" | wc -l)
echo "Found $N .pt files to scan."

if [[ "$N" -eq 0 ]]; then
    echo "Nothing to clean. Exiting."
    exit 0
fi

# ---- Inline Python cleanup script ------------------------------------------
"$PYTHON_BIN" -u - "$CACHE_ROOT" "$WORKERS" << 'PYEOF'
import sys, os, glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch, torch.serialization

# Register PyG safe globals so weights_only=True works on cached Data objects.
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
    print("WARNING: torch_geometric not importable — using weights_only=False", flush=True)

STALE_SUFFIXES = ("_dynamic_mask", "_dynamic_num_partitions")


def clean_file(path: str) -> tuple:
    """Load, strip stale attrs, save atomically. Returns (path, changed, removed_keys)."""
    try:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True, mmap=False)
        except Exception:
            obj = torch.load(path, map_location="cpu", weights_only=False)

        removed = []
        if hasattr(obj, "keys"):
            for key in list(obj.keys()):
                if any(key.endswith(s) for s in STALE_SUFFIXES):
                    delattr(obj, key)
                    removed.append(key)

        if removed:
            tmp = path + ".tmp_strip"
            torch.save(obj, tmp)
            os.replace(tmp, path)  # atomic on POSIX
            return path, True, removed

        return path, False, []
    except Exception as exc:
        return path, False, [f"ERROR: {exc}"]


def main(cache_root: str, n_workers: int) -> None:
    paths = list(glob.glob(os.path.join(cache_root, "**", "*.pt"), recursive=True))
    total = len(paths)
    print(f"Processing {total} files with {n_workers} workers ...", flush=True)

    changed = errors = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(clean_file, p): p for p in paths}
        for i, fut in enumerate(as_completed(futs), 1):
            path, was_changed, keys = fut.result()
            if keys and str(keys[0]).startswith("ERROR"):
                print(f"  FAIL  {path}: {keys[0]}", flush=True)
                errors += 1
            elif was_changed:
                print(f"  FIXED {path}: removed {keys}", flush=True)
                changed += 1
            if i % 5000 == 0:
                print(f"  ... {i}/{total} processed", flush=True)

    print(f"\nDone. {changed} files cleaned, {errors} errors out of {total} scanned.", flush=True)


if __name__ == "__main__":
    cache_root = sys.argv[1]
    n_workers = int(sys.argv[2])
    main(cache_root, n_workers)
PYEOF

echo "=========================================="
echo "Strip finished: $(date)"
echo "=========================================="
