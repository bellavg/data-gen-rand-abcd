#!/bin/bash
#SBATCH --job-name=10_algorithm_csv
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=genoa
#SBATCH --output=logs/10_algorithm_csv_%j.out

set -euo pipefail

module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0
module load SciPy-bundle/2025.06-gfbf-2025a

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
DESIGNS_DIR="$BASE_DIR/data/designs"
MASTER_CSV="${MASTER_CSV:-$DESIGNS_DIR/design_metadata/full_master.csv}"
OUT_DIR="${OUT_DIR:-$DESIGNS_DIR/design_metadata}"
# Root where preprocess_data.py saved the .pt graph files (FINAL_OUT from job 9).
GRAPHS_ROOT="${GRAPHS_ROOT:-/scratch-shared/$USER}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-24}}"

echo "=================================================="
echo " JOB 10: Build per-algorithm CSVs (no tier-0, with pre/post stats)"
echo " Time: $(date)"
echo " Base dir:    $BASE_DIR"
echo " Master CSV:  $MASTER_CSV"
echo " Output dir:  $OUT_DIR"
echo " Graphs root: $GRAPHS_ROOT"
echo " Workers:     $WORKERS"
echo "=================================================="

mkdir -p "$OUT_DIR"

python3 - "$MASTER_CSV" "$OUT_DIR" "$GRAPHS_ROOT" "$WORKERS" <<'PY'
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

master_csv = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
graphs_root = Path(sys.argv[3])
workers = max(1, int(sys.argv[4]))

algorithms = ["Orchestrate", "Deepsyn", "Syn4", "C2RS"]

# Output columns — unoptimized_graph_path is the PRE-optimisation .pt used as model input;
# targets are optimizability and depth_optimizability.
out_columns = [
    "unoptimized_graph_path",
    "design",
    "recipe_id",
    "step_id",
    "tier_id",
    "algorithm",
    "pre_nodes",
    "post_nodes",
    "edges",
    "pre_num_PI",
    "post_num_PI",
    "pre_num_PO",
    "post_num_PO",
    "pre_depth",
    "post_depth",
    "optimizability",
    "depth_optimizability",
]


# ---------------------------------------------------------------------------
# Path helpers — mirror artifact_output_base_path() in preprocess_data.py
# ---------------------------------------------------------------------------

def _recipe_stem(recipe_id: int, step_id: int) -> str:
    """synX_step0 for the original AIG, syn{r}_step{s} for synthesised ones."""
    if recipe_id == 0 and step_id == 0:
        return "synX_step0"
    return f"syn{recipe_id}_step{step_id}"


def _t0_path(design: str, recipe_id: int, step_id: int) -> str:
    stem = f"{design}_{_recipe_stem(recipe_id, step_id)}"
    return str(graphs_root / "graphs" / "tier0" / design / f"{stem}.pt")


def check_exists_parallel(paths: list[str], n_workers: int) -> list[bool]:
    """Return a bool list indicating whether each .pt path exists on disk."""
    if not paths:
        return []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(lambda p: Path(p).exists(), paths))


# ---------------------------------------------------------------------------
# Load master CSV — all columns as str so merges stay clean
# ---------------------------------------------------------------------------
print(f"Loading {master_csv} ...")
df = pd.read_csv(master_csv, dtype=str).fillna("")
df.columns = df.columns.str.strip()

# Normalise algorithm and whitespace in key columns
for col in ("algorithm", "recipe_id", "step_id", "tier_id", "design", "file_path"):
    if col in df.columns:
        df[col] = df[col].str.strip()

# Parse tier_id to int for filtering
def _to_tier(s: str) -> int:
    if not s:
        return 0
    try:
        return int(float(s))
    except ValueError:
        return -1

df["_tier"] = df["tier_id"].map(_to_tier)

# Derive root design from file_path ("base_aigs/{design}/...").
# For tier-2 rows the "design" column may be "{base}_{tier1_algo}".
df["_root_design"] = (
    df["file_path"]
    .str.extract(r"^base_aigs/([^/]+)/", expand=False)
    .fillna("")
    .str.strip()
)
df.loc[df["_root_design"] == "", "_root_design"] = df.loc[df["_root_design"] == "", "design"]

# Integer recipe/step for join keys
df["_recipe_id"] = pd.to_numeric(df["recipe_id"], errors="coerce").fillna(-1).astype(int)
df["_step_id"]   = pd.to_numeric(df["step_id"],   errors="coerce").fillna(-1).astype(int)

t0 = df[df["_tier"] == 0].copy()
t1 = df[(df["_tier"] == 1) & df["algorithm"].isin(algorithms)].copy()
t2 = df[(df["_tier"] == 2) & df["algorithm"].isin(algorithms)].copy()

print(f"Master CSV loaded: {len(df)} rows  (tier0={len(t0)}, tier1={len(t1)}, tier2={len(t2)})")


# ---------------------------------------------------------------------------
# Tier-1 rows
# Input graph  = tier-0 .pt (the synthesis-output AIG before optimisation)
# Targets      = tier-1 optimizability / depth_optimizability
# ---------------------------------------------------------------------------
t0_lookup = (
    t0[["_root_design", "_recipe_id", "_step_id", "nodes", "num_PI", "num_PO", "depth"]]
    .rename(columns={"nodes": "pre_nodes", "num_PI": "pre_num_PI",
                     "num_PO": "pre_num_PO", "depth": "pre_depth"})
    .drop_duplicates(subset=["_root_design", "_recipe_id", "_step_id"])
)

t1m = t1.merge(t0_lookup, on=["_root_design", "_recipe_id", "_step_id"], how="inner")
t1_missing_parent = len(t1) - len(t1m)

t1m["unoptimized_graph_path"] = [
    _t0_path(d, r, s)
    for d, r, s in zip(t1m["_root_design"], t1m["_recipe_id"], t1m["_step_id"])
]
t1_exists = check_exists_parallel(t1m["unoptimized_graph_path"].tolist(), workers)
t1_graph_missing = t1_exists.count(False)
t1m = t1m[t1_exists].copy()
print(f"Tier-1: total={len(t1)}  missing_parent={t1_missing_parent}  "
      f"graph_missing={t1_graph_missing}  ok={len(t1m)}")


# ---------------------------------------------------------------------------
# Tier-2 rows
# Input graph  = tier-1 .pt (the AIG BEFORE tier-2 optimisation)
# Targets      = tier-2 optimizability / depth_optimizability
# Pre-stats    = from the tier-1 parent row (stats of the input graph)
# Post-stats   = from the tier-2 row itself (stats of the post-optimisation graph)
# ---------------------------------------------------------------------------
# Extract the tier-1 source algorithm from the filename.
# Filename: {root_design}_{tier1_algo}_{tier2_algo}_tier2_syn{r}_step{s}.aig
t2["_fname"] = t2["file_path"].apply(lambda x: Path(x).name)
t2["_name_after_design"] = [
    fname[len(design) + 1:] if fname.startswith(design + "_") else ""
    for fname, design in zip(t2["_fname"], t2["_root_design"])
]
t2["_tier1_algo"] = ""
for algo in algorithms:
    mask = t2["_name_after_design"].str.startswith(f"{algo}_") & (t2["_tier1_algo"] == "")
    t2.loc[mask, "_tier1_algo"] = algo

t2_no_source = (t2["_tier1_algo"] == "").sum()
t2 = t2[t2["_tier1_algo"] != ""].copy()

t1_lookup = (
    t1[["_root_design", "algorithm", "_recipe_id", "_step_id",
        "nodes", "num_PI", "num_PO", "depth", "file_path"]]
    .rename(columns={"algorithm": "_tier1_algo",
                     "nodes": "pre_nodes", "num_PI": "pre_num_PI",
                     "num_PO": "pre_num_PO", "depth": "pre_depth",
                     "file_path": "_t1_file_path"})
    .drop_duplicates(subset=["_root_design", "_tier1_algo", "_recipe_id", "_step_id"])
)

t2m = t2.merge(
    t1_lookup,
    on=["_root_design", "_tier1_algo", "_recipe_id", "_step_id"],
    how="inner",
)
t2_missing_parent = len(t2) - len(t2m)

# Build unoptimized_graph_path from the actual tier-1 AIG filename stem stored in
# file_path — the real filenames contain extra components (algo, design, random hash)
# that cannot be reconstructed from (design, algo, recipe_id, step_id) alone.
t2m["unoptimized_graph_path"] = [
    str(graphs_root / "graphs" / "tier1" / algo / design / (Path(fp).stem + ".pt"))
    for algo, design, fp in zip(
        t2m["_tier1_algo"], t2m["_root_design"], t2m["_t1_file_path"]
    )
]
t2_exists = check_exists_parallel(t2m["unoptimized_graph_path"].tolist(), workers)
t2_graph_missing = t2_exists.count(False)
t2m = t2m[t2_exists].copy()
print(f"Tier-2: total={len(t2)}  no_source_algo={t2_no_source}  "
      f"missing_parent={t2_missing_parent}  graph_missing={t2_graph_missing}  ok={len(t2m)}")


# ---------------------------------------------------------------------------
# Build final output DataFrames and write per-algorithm CSVs
# ---------------------------------------------------------------------------
def _build_out(merged: pd.DataFrame, tier_id: int) -> pd.DataFrame:
    return pd.DataFrame({
        "unoptimized_graph_path": merged["unoptimized_graph_path"],
        "design":               merged["_root_design"],
        "recipe_id":            merged["recipe_id"],
        "step_id":              merged["step_id"],
        "tier_id":              str(tier_id),
        "algorithm":            merged["algorithm"],
        "pre_nodes":            merged["pre_nodes"].astype(str).str.strip(),
        "post_nodes":           merged["nodes"].astype(str).str.strip(),
        "edges":                merged["edges"].astype(str).str.strip(),
        "pre_num_PI":           merged["pre_num_PI"].astype(str).str.strip(),
        "post_num_PI":          merged["num_PI"].astype(str).str.strip(),
        "pre_num_PO":           merged["pre_num_PO"].astype(str).str.strip(),
        "post_num_PO":          merged["num_PO"].astype(str).str.strip(),
        "pre_depth":            merged["pre_depth"].astype(str).str.strip(),
        "post_depth":           merged["depth"].astype(str).str.strip(),
        "optimizability":       merged["optimizability"].astype(str).str.strip(),
        "depth_optimizability": merged["depth_optimizability"].astype(str).str.strip(),
    })


combined = pd.concat([_build_out(t1m, 1), _build_out(t2m, 2)], ignore_index=True)

for algo in algorithms:
    subset = combined[combined["algorithm"] == algo][out_columns].reset_index(drop=True)
    out_path = out_dir / f"algo_{algo}_ml.csv"
    subset.to_csv(out_path, index=False)
    print(f"Wrote {len(subset):>8,} rows  →  {out_path}")

print("Done.")
PY

echo "Finished: $(date)"
