from __future__ import annotations

import os
import uuid
import torch
import functools
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
# Ensure safe unpickling globals are registered by importing dataset first
from data.dataset import AIGGraphRegressionDataset


# =====================================================================
# DYNAMIC K HEURISTIC
# =====================================================================

def compute_dynamic_k(
    num_nodes: int,
    target_nodes_per_part: int,
    min_k: int,
    max_k: int,
) -> int:
    """Compute the number of partitions for a graph using the heuristic:

        k = clamp(num_nodes // target_nodes_per_part, min_k, max_k)

    Args:
        num_nodes:             Number of nodes in the graph.
        target_nodes_per_part: Desired average nodes per partition.
        min_k:                 Minimum allowed k (>= 1).
        max_k:                 Maximum allowed k.

    Returns:
        An integer k in ``[min_k, max_k]``.
    """
    k = num_nodes // max(1, target_nodes_per_part)
    return max(min_k, min(max_k, k))


# =====================================================================
# PARTITIONING ALGORITHMS
#
# Each function receives a concrete ``num_partitions`` integer that has
# already been computed by the pipeline using ``compute_dynamic_k``.
# The functions are pure implementations — they do not read config or
# decide k themselves.
# =====================================================================

def run_metis(data_obj, num_partitions: int) -> torch.Tensor:
    """Computes standard METIS partitions using pymetis.

    Args:
        data_obj:       A PyG ``Data`` object.
        num_partitions: Number of partitions (pre-computed by the pipeline).

    Returns:
        A 1-D ``torch.long`` tensor of shape ``[num_nodes]`` with values in
        ``{0, …, num_partitions - 1}``.
    """
    import pymetis
    from torch_geometric.utils import to_scipy_sparse_matrix

    num_nodes = data_obj.num_nodes

    # 1. METIS requires an undirected structure
    undirected_edges = to_undirected(data_obj.edge_index, num_nodes=num_nodes)

    # 2. Convert to CSR format for pymetis
    adj_sparse = to_scipy_sparse_matrix(
        edge_index=undirected_edges, 
        num_nodes=num_nodes
    ).tocsr()

    # 3. Invoke pymetis with CSRAdjacency (unweighted)
    adjacency = pymetis.CSRAdjacency(
        adj_starts=adj_sparse.indptr,
        adjacent=adj_sparse.indices
    )
    _, part_labels = pymetis.part_graph(
        nparts=num_partitions,
        adjacency=adjacency
    )

    return torch.tensor(part_labels, dtype=torch.long, device="cpu")


def run_level_slicing(data_obj, num_partitions: int) -> torch.Tensor:
    """Partitions a graph into equal buckets by node level (topological depth).

    Nodes are sorted by their ``level`` attribute and divided into
    ``num_partitions`` equal-sized buckets.  This preserves the natural DAG
    layering of AIG/circuit graphs rather than using a graph-topology method
    like METIS.

    Strategy
    --------
    1. Read ``data_obj.level`` — a 1-D integer tensor of shape ``[num_nodes]``.
    2. Sort the nodes by their level value.
    3. Divide the sorted order into ``num_partitions`` equal buckets and assign
       each bucket a partition index in ``{0, …, num_partitions - 1}``.

    Args:
        data_obj:       A PyG ``Data`` object carrying a ``level`` node attribute
                        (shape ``[num_nodes]`` or ``[num_nodes, 1]``, integer/float dtype).
        num_partitions: Number of equal-depth buckets (pre-computed by the pipeline).

    Returns:
        A 1-D ``torch.long`` tensor of shape ``[num_nodes]`` with values in
        ``{0, …, num_partitions - 1}``.

    Raises:
        AttributeError: If ``data_obj`` has no ``level`` attribute.
        ValueError:     If ``num_partitions`` < 1.
    """
    if num_partitions < 1:
        raise ValueError(f"num_partitions must be >= 1, got {num_partitions}")

    if not hasattr(data_obj, "level") or data_obj.level is None:
        raise AttributeError(
            "run_level_slicing requires a 'level' node attribute on the Data object, "
            "but none was found. Ensure the cached .pt files include the 'level' tensor."
        )

    level = data_obj.level
    if not isinstance(level, torch.Tensor):
        level = torch.tensor(level, dtype=torch.long)
    level = level.to(dtype=torch.long, device="cpu").view(-1)

    num_nodes = level.size(0)

    # Sort nodes by level, then assign equal-sized buckets.
    sort_idx = torch.argsort(level, stable=True)

    # Each sorted position i maps to bucket floor(i * num_partitions / num_nodes).
    positions = torch.arange(num_nodes, dtype=torch.long)
    bucket_for_position = torch.div(
        positions * num_partitions, num_nodes, rounding_mode="floor"
    ).clamp(max=num_partitions - 1)  # guard against edge at i == num_nodes

    assignment_mask = torch.empty(num_nodes, dtype=torch.long, device="cpu")
    assignment_mask[sort_idx] = bucket_for_position

    return assignment_mask


def run_random(data_obj, num_partitions: int, seed: int = 0) -> torch.Tensor:
    """Assigns each node a uniformly random partition label using a fixed seed.

    Producing the mask offline ensures every training run sees **identical**
    partition assignments for the same graph, making random partitioning
    comparable to deterministic algorithms like METIS or level-slicing.

    Args:
        data_obj:       A PyG ``Data`` object.
        num_partitions: Number of partitions (pre-computed by the pipeline).
        seed:           Integer RNG seed for reproducibility (default 0).
                        Pass ``config.PARTITION_SEED`` from the call-site.

    Returns:
        A 1-D ``torch.long`` tensor of shape ``[num_nodes]`` with values in
        ``{0, …, num_partitions - 1}``.
    """
    num_nodes = data_obj.num_nodes
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randint(0, num_partitions, (num_nodes,), dtype=torch.long, generator=generator)


def run_span_weighted_metis(data_obj, num_partitions: int, alpha: float = 10.0) -> torch.Tensor:
    """Computes a Span-Aware METIS partition by penalizing cuts on long edges.

    Args:
        data_obj: PyG Data object carrying a 'level' node attribute.
        num_partitions: Number of partitions to create.
        alpha: Penalty multiplier. Higher alpha means METIS is more 
               reluctant to cut edges spanning multiple levels.
    """
    import pymetis
    from torch_geometric.utils import to_scipy_sparse_matrix

    num_nodes = data_obj.num_nodes

    # 1. METIS requires an undirected structure
    undirected_edges = to_undirected(data_obj.edge_index, num_nodes=num_nodes)

    # 2. Compute edge spans
    # Ensure levels are parsed as a flat tensor of floats for distance calculation
    levels = data_obj.level.view(-1).to(dtype=torch.float32)
    src, dst = undirected_edges

    # Span is the absolute difference in topological level between the two nodes
    spans = torch.abs(levels[src] - levels[dst])

    # 3. Formulate the edge weights
    # METIS seeks to *minimize* the sum of the weights of cut edges. 
    # High weight = do not cut. Low weight = safe to cut.
    # METIS strictly requires integer weights > 0.
    edge_weights = 1 + (alpha * spans)
    edge_weights = edge_weights.to(torch.int32)

    # 4. Convert to CSR format for pymetis using PyG's Scipy utility
    # Scipy's COO->CSR conversion naturally handles summing weights of duplicate/parallel edges
    adj_sparse = to_scipy_sparse_matrix(
        edge_index=undirected_edges, 
        edge_attr=edge_weights, 
        num_nodes=num_nodes
    ).tocsr()

    # 5. Invoke the PyMetis wrapper
    # Using CSRAdjacency to avoid deprecation warnings.
    adjacency = pymetis.CSRAdjacency(
        adj_starts=adj_sparse.indptr,
        adjacent=adj_sparse.indices
    )
    _, part_labels = pymetis.part_graph(
        nparts=num_partitions,
        adjacency=adjacency,
        eweights=adj_sparse.data.astype(int)
    )

    # 6. Return as a PyTorch tensor to match your pipeline
    return torch.tensor(part_labels, dtype=torch.long, device="cpu")


# =====================================================================
# WORKER TASK FOR PARALLEL EXECUTION
# =====================================================================

def _process_single_cache_file(
    cache_path: Path,
    target_nodes: int,
    min_k: int,
    max_k: int,
    algo_names: list[str],
    seed: int,
) -> None:
    if not cache_path.is_file():
        return

    # 1. Load the existing .pt file.
    with open(cache_path, "rb") as fh:
        data_obj = torch.load(fh, map_location="cpu", weights_only=True)

    # 2. Compute k once per graph (shared across all algorithms).
    k = compute_dynamic_k(data_obj.num_nodes, target_nodes, min_k, max_k)

    # 3. Run each requested algorithm with the pre-computed k.
    for algo_name in algo_names:
        if algo_name == "metis":
            mask_tensor = run_metis(data_obj, k)
        elif algo_name == "span_weighted_metis":
            mask_tensor = run_span_weighted_metis(data_obj, k)
        elif algo_name == "level_slicing":
            mask_tensor = run_level_slicing(data_obj, k)
        elif algo_name == "random":
            mask_tensor = run_random(data_obj, k, seed=seed)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        if not isinstance(mask_tensor, torch.Tensor):
            mask_tensor = torch.tensor(mask_tensor, dtype=torch.long)

        setattr(data_obj, f"{algo_name}_dynamic_mask",
                mask_tensor.to(dtype=torch.long, device="cpu"))
        setattr(data_obj, f"{algo_name}_dynamic_num_partitions",
                torch.tensor([k], dtype=torch.long, device="cpu"))

    # 4. Atomically overwrite the file on disk.
    temp_file = cache_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
    torch.save(data_obj, temp_file)
    os.replace(temp_file, cache_path)


# =====================================================================
# CORE UPDATE PIPELINE
# =====================================================================

def update_existing_cache_with_masks(
    directories: list[str | Path],
    algo_names: list[str],
    seed: int = 0,
) -> None:
    """Loads pre-cached graph files from specified directories, computes partition masks in parallel, and saves them back.

    Searches recursively for all ``*.pt`` files in the provided directories.
    Deduplicates the file paths so that each file is processed exactly once,
    avoiding redundant computation in shared/overlapping cache folders.

    For every graph the number of partitions ``k`` is determined dynamically
    by ``compute_dynamic_k`` using the ``TARGET_NODES_PER_PART``, ``MIN_K``,
    and ``MAX_K`` values from ``config.py``.

    Stored attributes (per graph, per algorithm):
        ``{algo_name}_dynamic_mask``            – 1-D long tensor, shape [num_nodes]
        ``{algo_name}_dynamic_num_partitions``  – scalar long tensor, value = k
    """
    import config as _cfg
    import concurrent.futures

    target_nodes = getattr(_cfg, "TARGET_NODES_PER_PART", 10_000)
    min_k        = getattr(_cfg, "MIN_K", 2)
    max_k        = getattr(_cfg, "MAX_K", 32)

    print(
        f"[Mask Precomputation] Dynamic-k heuristic: "
        f"TARGET_NODES_PER_PART={target_nodes}, MIN_K={min_k}, MAX_K={max_k}"
    )
    print(f"[Mask Precomputation] Scanning directories for cached graph files: {directories}")

    unique_cache_paths = []
    for d in directories:
        d_path = Path(d)
        if d_path.is_dir():
            unique_cache_paths.extend(d_path.rglob("*.pt"))
        elif d_path.is_file() and d_path.suffix == ".pt":
            unique_cache_paths.append(d_path)

    unique_cache_paths = sorted(set(p.resolve() for p in unique_cache_paths))
    total_files = len(unique_cache_paths)
    print(f"[Mask Precomputation] Found {total_files} unique graph cache files to process.")

    if total_files == 0:
        print("[Mask Precomputation] No graph cache files found. Exiting.")
        return

    # Respect SLURM allocated CPUs
    try:
        num_workers = len(os.sched_getaffinity(0))
    except AttributeError:
        num_workers = os.cpu_count() or 1

    print(f"[Mask Precomputation] Using {num_workers} parallel worker processes...")

    success_count = 0
    worker_fn = functools.partial(
        _process_single_cache_file,
        target_nodes=target_nodes,
        min_k=min_k,
        max_k=max_k,
        algo_names=algo_names,
        seed=seed,
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker_fn, path): path for path in unique_cache_paths}

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=total_files,
            desc="Appending masks to cache"
        ):
            path = futures[future]
            try:
                future.result()
                success_count += 1
            except Exception as e:
                print(f"\n[ERROR] Failed to process {path.name}: {e}")

    print(f"\n[Mask Precomputation] Complete! Successfully updated {success_count} files.")
    print("All other properties (features, edge layouts, positional encodings) were preserved untouched.")


if __name__ == "__main__":
    import argparse
    import config

    # -----------------------------------------------------------------------
    # Settings are driven by config.py (except algorithm/dirs which are CLI args).
    # -----------------------------------------------------------------------
    _seed = getattr(config, "PARTITION_SEED", 0)

    parser = argparse.ArgumentParser(
        description="Precompute dynamic-k partition masks for cached graphs in parallel."
    )
    parser.add_argument(
        "algorithm",
        type=str,
        choices=["metis", "span_weighted_metis", "level_slicing", "random", "all"],
        help="Partition algorithm to run, or 'all' to run all available partition algorithms."
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="One or more directory paths (or individual .pt files) to search recursively for cached graphs."
    )
    args = parser.parse_args()

    if args.algorithm == "all":
        algo_names = ["metis", "span_weighted_metis", "level_slicing", "random"]
    else:
        algo_names = [args.algorithm]

    print(
        f"[partition.py] Running for algorithm(s)={sorted(algo_names)}, dynamic-k heuristic\n"
        f"  seed={_seed}\n"
        f"  dirs={args.dirs}"
    )

    update_existing_cache_with_masks(
        directories=args.dirs,
        algo_names=algo_names,
        seed=_seed,
    )