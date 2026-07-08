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
    spanner_sparsification,
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

    for cache_dir in mask_cache_dirs:
        if not cache_dir.is_dir():
            continue
        for index_path in sorted(cache_dir.rglob(f"{_SPARSE_PREFIX}*.pt")):
            algo_name = index_path.stem.split("_")[1]  # _sparse_{algo}_{chunk}.pt
            # Determine algo from filename
            for candidate in ("random_edge_dropout", "spanner", "pagerank"):
                if candidate in index_path.stem:
                    algo_name = candidate
                    break
            else:
                continue

            try:
                chunk = torch.load(index_path, map_location="cpu", weights_only=True)
            except Exception as exc:
                print(f"  [skip] {index_path.name}: {exc}")
                continue

            for basename, entry in chunk.items():
                mask = entry["mask"]
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask)
                total = mask.numel()
                kept = mask.sum().item()
                if total > 0:
                    retention = kept / total
                    stats[algo_name].append(retention)
                    if len(stats[algo_name]) >= max_samples:
                        break

            # Stop early if we have enough samples for all algos
            if all(len(v) >= max_samples for v in stats.values()):
                break

    return stats


def measure_from_graphs(graph_dir: Path, max_samples: int = 100):
    """Load actual graph .pt files and compute masks on-the-fly."""
    import config

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
        mask_re = random_edge_dropout(data, dropout_rate=config.SPARSIFICATION_RANDOM_DROPOUT_RATE, seed=42)
        stats["random_edge_dropout"].append({
            "edge_retention": mask_re.sum().item() / n_edges,
            "node_retention": 1.0,  # all nodes kept
            "n_nodes": n_nodes,
            "n_edges": n_edges,
        })

        # Spanner
        try:
            mask_sp = spanner_sparsification(data, stretch=config.SPARSIFICATION_SPANNER_STRETCH, seed=42)
            stats["spanner"].append({
                "edge_retention": mask_sp.sum().item() / n_edges,
                "node_retention": 1.0,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
            })
        except Exception as exc:
            print(f"  Spanner failed on {pt_path.name}: {exc}")

        # PageRank
        try:
            mask_pr = pagerank_sparsification(
                data,
                keep_ratio=config.SPARSIFICATION_PAGERANK_KEEP_RATIO,
                alpha=config.SPARSIFICATION_PAGERANK_ALPHA,
            )
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
            })
        except Exception as exc:
            print(f"  PageRank failed on {pt_path.name}: {exc}")

        # And-gate-only
        try:
            is_pi = (data.x[:, 1] == 1.0)
            is_po = (data.x[:, 3] == 1.0)
            n_removed = (is_pi | is_po).sum().item()
            nodes_kept_ago = n_nodes - n_removed
            out = and_gate_only_sparsification(data)
            edges_kept_ago = out.edge_index.size(1)
            stats["and_gate_only"].append({
                "edge_retention": edges_kept_ago / n_edges,
                "node_retention": nodes_kept_ago / n_nodes,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
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
            print(f"\n  {algo} ({len(entries)} graphs):")
            print(f"    Edge retention: mean={np.mean(edge_rets):.1%}  "
                  f"std={np.std(edge_rets):.1%}  "
                  f"min={np.min(edge_rets):.1%}  max={np.max(edge_rets):.1%}")
            print(f"    Node retention: mean={np.mean(node_rets):.1%}  "
                  f"std={np.std(node_rets):.1%}  "
                  f"min={np.min(node_rets):.1%}  max={np.max(node_rets):.1%}")
            print(f"    → Effective edge REDUCTION: ~{1 - np.mean(edge_rets):.1%}")
        else:
            # Simple retention ratios from precomputed masks
            arr = np.array(entries)
            print(f"\n  {algo} ({len(entries)} masks):")
            print(f"    Retention: mean={arr.mean():.1%}  "
                  f"std={arr.std():.1%}  "
                  f"min={arr.min():.1%}  max={arr.max():.1%}")
            print(f"    → Effective REDUCTION: ~{1 - arr.mean():.1%}")

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
