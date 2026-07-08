#!/usr/bin/env python
"""
Measure actual edge/node retention for each sparsification method.

Run on the cluster (needs access to the mask cache):
    python src/data/measure_sparsity.py

Reports per-method statistics so you can calibrate parameters
for a fair comparison across methods.
"""
import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.sparsification import (
    _SPARSE_PREFIX,
    random_edge_dropout,
    spanning_forest_sparsification,
    pagerank_sparsification,
    and_gate_only_sparsification,
)

# Try to use the safe globals registration if available
try:
    from data.sparsification import _register_pyg_safe_globals
    _register_pyg_safe_globals()
except ImportError:
    pass


def measure_from_precomputed_masks(mask_cache_dirs: list[Path], max_samples: int = 500):
    """Scan precomputed mask index files and report retention stats."""
    stats: dict[str, list[float]] = defaultdict(list)

    KNOWN_ALGOS = ("random_edge_dropout", "spanning_forest", "pagerank")

    # Collect all index files across all cache dirs
    all_index_files: list[Path] = []
    for cache_dir in mask_cache_dirs:
        if not cache_dir.is_dir():
            print(f"  [skip] Directory does not exist: {cache_dir}")
            continue
        found = sorted(cache_dir.rglob(f"{_SPARSE_PREFIX}*.pt"))
        print(f"  Found {len(found)} mask index files in {cache_dir}")
        all_index_files.extend(found)

    if not all_index_files:
        print("  No mask index files found in any directory!")
        return stats

    # Group files by algorithm
    algo_files: dict[str, list[Path]] = defaultdict(list)
    unrecognized: list[str] = []
    for index_path in all_index_files:
        fname = index_path.stem  # e.g. _sparse_spanning_forest_1720000000_abcd
        matched = False
        for candidate in KNOWN_ALGOS:
            # Check if the filename contains the algo name after the prefix
            # Files are named: _sparse_{algo_name}_{chunk_id}_{uuid}.pt
            if f"{_SPARSE_PREFIX}{candidate}" in index_path.name:
                algo_files[candidate].append(index_path)
                matched = True
                break
        if not matched:
            unrecognized.append(index_path.name)

    print(f"\n  Files per algorithm:")
    for algo in KNOWN_ALGOS:
        print(f"    {algo}: {len(algo_files[algo])} index files")
    if unrecognized:
        print(f"    unrecognized: {len(unrecognized)} files (e.g. {unrecognized[0]})")
    print()

    # Process each algorithm
    for algo_name in KNOWN_ALGOS:
        files = algo_files[algo_name]
        if not files:
            print(f"  [{algo_name}] No precomputed mask files found — skipping.")
            continue

        for index_path in files:
            if len(stats[algo_name]) >= max_samples:
                break
            try:
                chunk = torch.load(index_path, map_location="cpu", weights_only=True)
            except Exception as exc:
                print(f"  [skip] {index_path.name}: {exc}")
                continue

            for basename, entry in chunk.items():
                if len(stats[algo_name]) >= max_samples:
                    break
                mask = entry["mask"]
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask)
                total = mask.numel()
                kept = int(mask.sum())
                if total > 0:
                    retention = kept / total
                    stats[algo_name].append(retention)

        print(f"  [{algo_name}] Collected {len(stats[algo_name])} mask samples.")

    return stats


def measure_from_graphs(graph_dir: Path, max_samples: int = 100):
    """Load actual graph .pt files and compute masks on-the-fly."""
    import config
    import time

    stats: dict[str, list[dict]] = defaultdict(list)
    pt_files = sorted(graph_dir.rglob("*.pt"))[:max_samples]

    print(f"Measuring sparsity on {len(pt_files)} graph files...")

    for i, pt_path in enumerate(pt_files):
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
        except Exception:
            continue

        if not hasattr(data, "edge_index") or not hasattr(data, "x"):
            continue

        n_nodes = data.x.size(0)
        n_edges = data.edge_index.size(1)

        if n_edges == 0 or n_nodes == 0:
            continue

        # Random edge dropout
        t0 = time.perf_counter()
        mask_re = random_edge_dropout(data, dropout_rate=config.SPARSIFICATION_RANDOM_DROPOUT_RATE, seed=42)
        t1 = time.perf_counter()
        stats["random_edge_dropout"].append({
            "edge_retention": mask_re.sum().item() / n_edges,
            "node_retention": 1.0,  # all nodes kept
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "time_s": t1 - t0,
        })

        # Spanning Forest
        try:
            t0 = time.perf_counter()
            mask_sf = spanning_forest_sparsification(data, seed=42)
            t1 = time.perf_counter()
            stats["spanning_forest"].append({
                "edge_retention": mask_sf.sum().item() / n_edges,
                "node_retention": 1.0,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "time_s": t1 - t0,
            })
        except Exception as exc:
            print(f"  Spanning Forest failed on {pt_path.name}: {exc}")

        # PageRank
        try:
            t0 = time.perf_counter()
            mask_pr = pagerank_sparsification(
                data,
                keep_ratio=config.SPARSIFICATION_PAGERANK_KEEP_RATIO,
                alpha=config.SPARSIFICATION_PAGERANK_ALPHA,
            )
            t1 = time.perf_counter()
            nodes_kept = mask_pr.sum().item()
            # Estimate edge retention: edges where both src and dst are kept
            src_kept = mask_pr[data.edge_index[0]]
            dst_kept = mask_pr[data.edge_index[1]]
            edges_kept = (src_kept & dst_kept).sum().item()
            stats["pagerank"].append({
                "edge_retention": edges_kept / n_edges,
                "node_retention": nodes_kept / n_nodes,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "time_s": t1 - t0,
            })
        except Exception as exc:
            print(f"  PageRank failed on {pt_path.name}: {exc}")

        # And-gate-only
        try:
            t0 = time.perf_counter()
            is_pi = (data.x[:, 1] == 1.0)
            is_po = (data.x[:, 3] == 1.0)
            n_removed = (is_pi | is_po).sum().item()
            nodes_kept_ago = n_nodes - n_removed
            out = and_gate_only_sparsification(data)
            t1 = time.perf_counter()
            edges_kept_ago = out.edge_index.size(1)
            stats["and_gate_only"].append({
                "edge_retention": edges_kept_ago / n_edges,
                "node_retention": nodes_kept_ago / n_nodes,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "time_s": t1 - t0,
            })
        except Exception as exc:
            print(f"  And-gate-only failed on {pt_path.name}: {exc}")

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(pt_files)} graphs")

    return stats


def print_stats(stats: dict):
    print("\n" + "=" * 70)
    print("SPARSIFICATION RETENTION STATISTICS")
    print("=" * 70)

    for algo, entries in sorted(stats.items()):
        if not entries:
            continue

        if isinstance(entries[0], dict):
            edge_rets = [e["edge_retention"] for e in entries]
            node_rets = [e["node_retention"] for e in entries]
            times = [e.get("time_s", 0) for e in entries]
            print(f"\n  {algo} ({len(entries)} graphs):")
            print(f"    Edge retention: mean={np.mean(edge_rets):.1%}  "
                  f"std={np.std(edge_rets):.1%}  "
                  f"min={np.min(edge_rets):.1%}  max={np.max(edge_rets):.1%}")
            print(f"    Node retention: mean={np.mean(node_rets):.1%}  "
                  f"std={np.std(node_rets):.1%}  "
                  f"min={np.min(node_rets):.1%}  max={np.max(node_rets):.1%}")
            print(f"    → Effective edge REDUCTION: ~{1 - np.mean(edge_rets):.1%}")
            if sum(times) > 0:
                print(f"    Avg time per graph: {np.mean(times)*1000:.2f} ms")
        else:
            # Simple retention ratios from precomputed masks
            arr = np.array(entries)
            # Pagerank masks are node masks; the others are edge masks.
            mask_type = "NODE" if algo == "pagerank" else "EDGE"
            print(f"\n  {algo} ({len(entries)} masks, {mask_type} mask):")
            print(f"    {mask_type.capitalize()} retention: mean={arr.mean():.1%}  "
                  f"std={arr.std():.1%}  "
                  f"min={arr.min():.1%}  max={arr.max():.1%}")
            print(f"    → Effective {mask_type} REDUCTION: ~{1 - arr.mean():.1%}")
            if algo == "pagerank":
                print(f"    NOTE: This is node retention. Actual edge reduction is higher")
                print(f"          (removing a node also removes all its incident edges).")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Measure sparsification retention")
    parser.add_argument("--mask-cache-dirs", nargs="*",
                        help="Directories containing precomputed _sparse_*.pt index files")
    parser.add_argument("--graph-dir",
                        help="Directory containing graph .pt files to measure on-the-fly")
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()

    user = os.environ.get("USER", "")

    if args.graph_dir:
        stats = measure_from_graphs(Path(args.graph_dir), max_samples=args.max_samples)
        print_stats(stats)
    elif args.mask_cache_dirs:
        dirs = [Path(d) for d in args.mask_cache_dirs]
        stats = measure_from_precomputed_masks(dirs, max_samples=args.max_samples)
        print_stats(stats)
    else:
        # Default: try the standard mask cache location
        default_dirs = [
            Path(f"/scratch-shared/{user}/aig_mask_cache"),
            Path(f"/scratch-shared/{user}/aig_train_run/shared_tier0_cache"),
            Path(f"/scratch-shared/{user}/aig_train_run/shared_tier1_cache"),
        ]
        existing = [d for d in default_dirs if d.is_dir()]
        if existing:
            print(f"Scanning precomputed masks in: {[str(d) for d in existing]}")
            stats = measure_from_precomputed_masks(existing, max_samples=args.max_samples)
            print_stats(stats)
        else:
            print("No mask cache or graph directory found.")
            print("Usage:")
            print("  python measure_sparsity.py --graph-dir /path/to/cached/graphs")
            print("  python measure_sparsity.py --mask-cache-dirs /path/to/mask/cache")
