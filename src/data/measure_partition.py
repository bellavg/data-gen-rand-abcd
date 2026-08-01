#!/usr/bin/env python
"""
Measure the offline cost of the partitioning algorithms themselves.

Mirrors measure_sparsity.py's structure and CLI, but partitioning never
removes nodes or edges (only cuts cross-partition edges for message
passing), so the numbers that matter are different: edge-cut ratio (not
node/edge retention), the dynamic partition count k, and how evenly the
nodes actually land across the k parts, alongside per-graph wall-clock time.

Run on the cluster (needs access to the graph cache):
    python src/data/measure_partition.py --graph-dirs /path/to/tier0 /path/to/tier1
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from data.partition import (
        _register_pyg_safe_globals as _register_partition_safe_globals,
    )
    _register_partition_safe_globals()
except ImportError:
    pass


def process_single_graph(pt_path: Path):
    import time

    import config
    from data.compute_levels import compute_node_levels
    from data.partition import (
        compute_dynamic_k,
        run_level_slicing,
        run_metis,
        run_random,
        run_span_weighted_metis,
    )

    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
    except Exception:
        return None

    if not hasattr(data, "edge_index") or not hasattr(data, "x"):
        return None

    n_nodes = data.x.size(0)
    n_edges = data.edge_index.size(1)
    if n_nodes == 0 or n_edges == 0:
        return None

    if not hasattr(data, "level") or data.level is None:
        data.level = compute_node_levels(data)

    k = compute_dynamic_k(
        n_nodes,
        getattr(config, "TARGET_NODES_PER_PART", 10_000),
        getattr(config, "MIN_K", 2),
        getattr(config, "MAX_K", 32),
    )
    seed = getattr(config, "PARTITION_SEED", 42)
    src, dst = data.edge_index[0], data.edge_index[1]

    def _entry(assignment: torch.Tensor, elapsed: float) -> dict:
        cut = (assignment[src] != assignment[dst]).sum().item()
        # minlength=k so a partitioner that leaves a part empty still
        # contributes its zero, rather than shortening the vector and
        # flattering its own spread.
        sizes = torch.bincount(assignment, minlength=k).to(torch.float64)
        return {
            "graph_id": pt_path.name,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "num_partitions": k,
            "avg_nodes_per_partition": n_nodes / k,
            # avg_nodes_per_partition is n_nodes / k, so it is identical for
            # all four partitioners on a given graph and says nothing about
            # how evenly any of them actually split it. These two do.
            # Population std, not sample: the k partitions are the whole set,
            # not a draw from a larger one.
            "std_nodes_per_partition": float(sizes.std(unbiased=False)),
            "max_nodes_per_partition": int(sizes.max()),
            "edge_cut_ratio": cut / n_edges,
            "time_s": elapsed,
        }

    local_stats: dict[str, dict | str] = {
        "random": None,
        "metis": None,
        "level_slicing": None,
        "span_weighted_metis": None,
    }

    t0 = time.perf_counter()
    assignment = run_random(data, k, seed=seed)
    local_stats["random"] = _entry(assignment, time.perf_counter() - t0)

    try:
        t0 = time.perf_counter()
        assignment = run_metis(data, k)
        local_stats["metis"] = _entry(assignment, time.perf_counter() - t0)
    except Exception as exc:
        local_stats["metis"] = str(exc)

    try:
        t0 = time.perf_counter()
        assignment = run_level_slicing(data, k)
        local_stats["level_slicing"] = _entry(assignment, time.perf_counter() - t0)
    except Exception as exc:
        local_stats["level_slicing"] = str(exc)

    try:
        t0 = time.perf_counter()
        assignment = run_span_weighted_metis(data, k)
        local_stats["span_weighted_metis"] = _entry(assignment, time.perf_counter() - t0)
    except Exception as exc:
        local_stats["span_weighted_metis"] = str(exc)

    return local_stats


def measure_from_graphs(graph_dirs: list[Path], max_samples: int = 100):
    """Load actual graph .pt files and compute partitions on-the-fly."""
    import concurrent.futures
    import csv
    import multiprocessing
    import os
    import random
    from collections import defaultdict

    stats: dict[str, list[dict]] = defaultdict(list)

    all_files: list[Path] = []
    for gdir in graph_dirs:
        if not gdir.is_dir():
            print(f"  [skip] Graph directory does not exist: {gdir}")
            continue
        all_files.extend(gdir.rglob("*.pt"))

    all_files = sorted(all_files)

    # Deterministic seeded random sampling — same convention used everywhere
    # else in this codebase (dataset.py's num_samples, measure_sparsity.py).
    rng = random.Random(42)
    if len(all_files) > max_samples:
        pt_files = rng.sample(all_files, max_samples)
    else:
        pt_files = all_files

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", min(multiprocessing.cpu_count(), 24)))
    print(f"Measuring partitioning cost on {len(pt_files)} randomly sampled graph files (seeded) from {len(graph_dirs)} directories...")
    print(f"Running in parallel with {num_workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_graph, pt_path): pt_path for pt_path in pt_files}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            pt_path = futures[future]
            try:
                res = future.result()
                if res is not None:
                    for algo, s in res.items():
                        if isinstance(s, dict):
                            stats[algo].append(s)
                        elif isinstance(s, str):
                            print(f"  {algo} failed on {pt_path.name}: {s}")
            except Exception as exc:
                print(f"  Worker failed on {pt_path.name}: {exc}")

            if (i + 1) % max(1, len(pt_files) // 20) == 0:
                print(f"  Processed {i + 1}/{len(pt_files)} graphs")

    os.makedirs("logs", exist_ok=True)
    for algo, entries in stats.items():
        if not entries:
            continue
        csv_path = f"logs/partition_stats_{algo}.csv"
        try:
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=entries[0].keys())
                writer.writeheader()
                writer.writerows(entries)
            print(f"  [CSV EXPORT] Saved offline analysis data to {csv_path}")
        except Exception as exc:
            print(f"  [CSV EXPORT] Failed to save {csv_path}: {exc}")

    return stats


def print_stats(stats: dict):
    print("\n" + "=" * 70)
    print("PARTITIONING OFFLINE COST STATISTICS")
    print("=" * 70)

    for algo, entries in sorted(stats.items()):
        if not entries:
            continue
        cut_ratios = [e["edge_cut_ratio"] for e in entries]
        k_vals = [e["num_partitions"] for e in entries]
        avg_part_sizes = [e["avg_nodes_per_partition"] for e in entries]
        size_stds = [e["std_nodes_per_partition"] for e in entries]
        max_part_sizes = [e["max_nodes_per_partition"] for e in entries]
        times = [e.get("time_s", 0) for e in entries]
        print(f"\n  {algo} ({len(entries)} graphs):")
        print(f"    Edge cut ratio: mean={np.mean(cut_ratios):.1%}  "
              f"std={np.std(cut_ratios):.1%}  "
              f"min={np.min(cut_ratios):.1%}  max={np.max(cut_ratios):.1%}")
        print(f"    Num partitions (k): mean={np.mean(k_vals):.1f}  "
              f"min={np.min(k_vals)}  max={np.max(k_vals)}")
        print(f"    Avg nodes/partition: mean={np.mean(avg_part_sizes):.1f}")
        print(f"    Imbalance (sd nodes/partition): mean={np.mean(size_stds):.1f}  "
              f"max part: mean={np.mean(max_part_sizes):.1f}")
        if sum(times) > 0:
            print(f"    Avg time per graph: {np.mean(times) * 1000:.2f} ms")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Measure offline partitioning cost")
    parser.add_argument("--graph-dirs", nargs="*", required=True,
                        help="Directories containing the actual .pt graph files to partition on the fly")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Maximum number of files to sample (seeded)")
    args = parser.parse_args()

    dirs = [Path(d) for d in args.graph_dirs if d.strip()]
    if not dirs:
        print("Usage:\n  python measure_partition.py --graph-dirs /path/to/tier0 /path/to/tier1")
        raise SystemExit(1)

    stats = measure_from_graphs(dirs, max_samples=args.max_samples)
    print_stats(stats)
