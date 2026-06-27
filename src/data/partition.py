from __future__ import annotations

import os
import time
import uuid
import torch
import functools
import itertools
from pathlib import Path

# Prefix for index files — excluded from scanning so they are never treated
# as graph cache files.
_MASKS_PREFIX = "_masks_"
CHECKPOINT_EVERY = 50_000   # atomic index save cadence (number of completed files)


def _register_pyg_safe_globals() -> None:
    """Register the minimal set of PyG classes needed for ``weights_only=True``
    torch.load calls.  Doing this inline avoids importing the heavy
    ``data.dataset`` module (which pulls in pandas, models, etc.) in every
    spawned worker process, which caused a thundering-herd NFS stall when 48
    workers all imported simultaneously.
    """
    import torch.serialization
    from torch_geometric.data import Data
    import torch_geometric.data.data as _pyg_data_mod
    import torch_geometric.data.storage as _pyg_storage
    from data.partition_utils import PartitionedData

    safe_globals: list = [Data, _pyg_storage.GlobalStorage, PartitionedData]
    for _name in ("DataTensorAttr", "DataEdgeAttr"):
        _cls = getattr(_pyg_data_mod, _name, None)
        if _cls is not None:
            safe_globals.append(_cls)
    torch.serialization.add_safe_globals(safe_globals)

_LEVELS_INDEX_CACHE: dict[str, dict] = {}

def _get_level_for_file(cache_path: Path, data_obj) -> torch.Tensor:
    cache_dir_str = str(cache_path.parent)
    if cache_dir_str not in _LEVELS_INDEX_CACHE:
        index_path = cache_path.parent / "_levels.pt"
        if index_path.is_file():
            try:
                _LEVELS_INDEX_CACHE[cache_dir_str] = torch.load(
                    index_path, map_location="cpu", weights_only=True, mmap=True
                )
            except TypeError:
                _LEVELS_INDEX_CACHE[cache_dir_str] = torch.load(
                    index_path, map_location="cpu", weights_only=True
                )
        else:
            _LEVELS_INDEX_CACHE[cache_dir_str] = {}
            
    levels = _LEVELS_INDEX_CACHE[cache_dir_str].get(cache_path.name)
    if levels is not None:
        return levels
        
    # Fallback if not in index
    from data.compute_levels import compute_node_levels
    return compute_node_levels(data_obj)



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
    """Computes standard METIS partitions using pymetis."""
    import pymetis
    from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected

    num_nodes = data_obj.num_nodes
    undirected_edges = to_undirected(data_obj.edge_index, num_nodes=num_nodes)
    adj_sparse = to_scipy_sparse_matrix(
        edge_index=undirected_edges,
        num_nodes=num_nodes
    ).tocsr()
    adjacency = pymetis.CSRAdjacency(
        adj_starts=adj_sparse.indptr,
        adjacent=adj_sparse.indices
    )
    _, part_labels = pymetis.part_graph(nparts=num_partitions, adjacency=adjacency)
    return torch.tensor(part_labels, dtype=torch.long, device="cpu")


def run_level_slicing(data_obj, num_partitions: int) -> torch.Tensor:
    """Partitions a graph into equal buckets by node level (topological depth)."""
    if num_partitions < 1:
        raise ValueError(f"num_partitions must be >= 1, got {num_partitions}")
    if not hasattr(data_obj, "level") or data_obj.level is None:
        raise AttributeError(
            "run_level_slicing requires a 'level' node attribute on the Data object."
        )
    level = data_obj.level
    if not isinstance(level, torch.Tensor):
        level = torch.tensor(level, dtype=torch.long)
    level = level.to(dtype=torch.long, device="cpu").view(-1)
    num_nodes = level.size(0)
    sort_idx = torch.argsort(level, stable=True)
    positions = torch.arange(num_nodes, dtype=torch.long)
    bucket_for_position = torch.div(
        positions * num_partitions, num_nodes, rounding_mode="floor"
    ).clamp(max=num_partitions - 1)
    assignment_mask = torch.empty(num_nodes, dtype=torch.long, device="cpu")
    assignment_mask[sort_idx] = bucket_for_position
    return assignment_mask


def run_random(data_obj, num_partitions: int, seed: int = 0) -> torch.Tensor:
    """Assigns each node a uniformly random partition label using a fixed seed."""
    num_nodes = data_obj.num_nodes
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randint(0, num_partitions, (num_nodes,), dtype=torch.long, generator=generator)


def run_span_weighted_metis(data_obj, num_partitions: int, alpha: float = 10.0) -> torch.Tensor:
    """Computes a Span-Aware METIS partition by penalizing cuts on long edges."""
    import pymetis
    from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected

    num_nodes = data_obj.num_nodes
    undirected_edges = to_undirected(data_obj.edge_index, num_nodes=num_nodes)
    levels = data_obj.level.view(-1).to(dtype=torch.float32)
    src, dst = undirected_edges
    spans = torch.abs(levels[src] - levels[dst])
    edge_weights = (1 + alpha * spans).to(torch.int32)
    adj_sparse = to_scipy_sparse_matrix(
        edge_index=undirected_edges,
        edge_attr=edge_weights,
        num_nodes=num_nodes
    ).tocsr()
    adjacency = pymetis.CSRAdjacency(
        adj_starts=adj_sparse.indptr,
        adjacent=adj_sparse.indices
    )
    _, part_labels = pymetis.part_graph(
        nparts=num_partitions,
        adjacency=adjacency,
        eweights=adj_sparse.data.astype(int)
    )
    return torch.tensor(part_labels, dtype=torch.long, device="cpu")


# =====================================================================
# INDEX FILE HELPERS
# =====================================================================

def _get_index_path(cache_dir: Path, algo_name: str) -> Path:
    """Return the path of the per-directory, per-algorithm mask index file.

    Index files are named ``_masks_{algo_name}.pt`` and live directly inside
    the cache directory alongside the graph ``.pt`` files.  The leading
    underscore distinguishes them from graph files so the scanner skips them.
    """
    return cache_dir / f"{_MASKS_PREFIX}{algo_name}.pt"


def _load_mask_index(cache_dir: Path, algo_name: str) -> dict:
    """Load an existing mask index from disk, returning an empty dict on miss/error.

    Index format::

        {
            "abc123.pt": {
                "mask": torch.Tensor,  # shape [num_nodes], dtype long
                "k":    torch.Tensor,  # shape [1],         dtype long
            },
            ...
        }
    """
    index_path = _get_index_path(cache_dir, algo_name)
    if not index_path.is_file():
        return {}
    try:
        return torch.load(index_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"[WARNING] Could not load existing index {index_path}: {exc}. Starting fresh.")
        return {}


def _save_mask_index(cache_dir: Path, algo_name: str, index: dict) -> None:
    """Atomically persist the mask index for one directory + algorithm."""
    index_path = _get_index_path(cache_dir, algo_name)
    temp_file = index_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
    torch.save(index, temp_file)
    os.replace(temp_file, index_path)


# =====================================================================
# WORKER TASK FOR PARALLEL EXECUTION
# =====================================================================

def _worker_initializer() -> None:
    """Called once per worker process at pool startup.

    - Registers PyG safe globals for ``weights_only=True`` torch.load.
    - Pins PyTorch to 1 intra-op thread to prevent thread-thrashing when
      many workers run concurrently (OMP_NUM_THREADS is set too late in
      spawned processes to help on its own).
    """
    import torch as _torch
    _torch.set_num_threads(1)
    _register_pyg_safe_globals()


def _process_single_cache_file(
    cache_path: Path,
    target_nodes: int,
    min_k: int,
    max_k: int,
    algo_names: list[str],
    seed: int,
) -> tuple[str, str, dict] | None:
    """Compute partition masks for one cache file and return them as tensors.

    **No disk writes are performed here.**  The caller (main process) is
    responsible for accumulating results and flushing the index to disk.

    Returns:
        ``(cache_dir_str, basename, mask_entry_dict)`` where *mask_entry_dict* maps
        ``algo_name`` → ``{"mask": tensor, "k": tensor}``.
        Returns ``None`` if the file does not exist.
    """
    if not cache_path.is_file():
        return None

    with open(cache_path, "rb") as fh:
        data_obj = torch.load(fh, map_location="cpu", weights_only=True)

    needs_level = any(a in ("span_weighted_metis", "level_slicing") for a in algo_names)
    if needs_level:
        if not hasattr(data_obj, "level") or data_obj.level is None:
            data_obj.level = _get_level_for_file(cache_path, data_obj)

    k = compute_dynamic_k(data_obj.num_nodes, target_nodes, min_k, max_k)

    result: dict[str, dict] = {}
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

        result[algo_name] = {
            "mask": mask_tensor.to(dtype=torch.long, device="cpu").numpy(),
            "k":    k,
        }

    return str(cache_path.parent), cache_path.name, result


# =====================================================================
# CORE UPDATE PIPELINE
# =====================================================================

def update_existing_cache_with_masks(
    directories: list[str | Path],
    algo_names: list[str],
    seed: int = 0,
    out_directories: list[str | Path] | None = None,
) -> None:
    """Compute partition masks for all cached graph files and save them in
    per-directory index files.

    Instead of embedding masks inside each individual graph ``.pt`` file
    (which requires loading and re-saving the full graph for every file),
    this pipeline writes one lightweight index file per *(cache directory,
    algorithm)* pair::

        {cache_dir}/_masks_{algo_name}.pt

    Each index file maps graph basenames to their precomputed mask tensors.

    **Inode cost**: 1 index file per (directory × algorithm).  With 2 cache
    directories and 4 algorithms that is 8 new inodes — a negligible addition.

    **Resume support**: On startup the existing indices are loaded and files
    that already have all requested masks are skipped automatically.

    **Streaming**: Files are submitted to the worker pool immediately as
    ``os.scandir`` discovers them — workers start within seconds rather than
    waiting for a full directory scan to complete.  This is critical when
    reading 984K+ file names from NFS takes many minutes.

    **Checkpointing**: The indices are flushed to disk atomically every
    ``CHECKPOINT_EVERY`` completed files so that a SLURM time-limit kill
    preserves as much work as possible.

    Args:
        directories: Flat cache directories to scan for graph ``.pt`` files.
                     All files that are *not* index files (``_masks_*.pt``)
                     are submitted to the worker pool.
        algo_names:  Partition algorithm names to compute.
        seed:        RNG seed forwarded to ``run_random``.
    """
    import config as _cfg
    import concurrent.futures

    _register_pyg_safe_globals()

    target_nodes = getattr(_cfg, "TARGET_NODES_PER_PART", 10_000)
    min_k        = getattr(_cfg, "MIN_K", 2)
    max_k        = getattr(_cfg, "MAX_K", 32)

    print(
        f"[Mask Precomputation] Dynamic-k heuristic: "
        f"TARGET_NODES_PER_PART={target_nodes}, MIN_K={min_k}, MAX_K={max_k}"
    )

    # ------------------------------------------------------------------
    # 1. LOAD EXISTING INDICES
    #    The top-level cache directories are known upfront (they are the
    #    arguments passed in).  We load their existing mask indices so we
    #    can skip already-computed files without any per-file stat calls.
    # ------------------------------------------------------------------
    top_dirs = [Path(d).absolute() for d in directories]
    if out_directories is None:
        out_dirs_list = top_dirs
    else:
        out_dirs_list = [Path(d).absolute() for d in out_directories]
        if len(top_dirs) != len(out_dirs_list):
            raise ValueError("--dirs and --out-dirs must have the same length")
    
    dir_map = {str(d): str(o) for d, o in zip(top_dirs, out_dirs_list)}

    accumulated: dict[str, dict[str, dict]] = {}
    done_by_dir: dict[str, set[str]] = {}

    for top_dir in top_dirs:
        d_str = str(top_dir)
        out_d_str = dir_map[d_str]
        accumulated[d_str] = {a: _load_mask_index(Path(out_d_str), a) for a in algo_names}
        
        # A basename is fully done only when it appears in ALL algo indices.
        if algo_names:
            all_done = [set(accumulated[d_str][a].keys()) for a in algo_names]
            done_by_dir[d_str] = (
                all_done[0].intersection(*all_done[1:]) if len(all_done) > 1 else all_done[0]
            )
        else:
            done_by_dir[d_str] = set()
        print(f"  -> {out_d_str}: {len(done_by_dir[d_str])} entries already in index")

    # ------------------------------------------------------------------
    # 2. STREAMING PATH GENERATOR
    # ------------------------------------------------------------------
    def _path_stream():
        for top_dir in top_dirs:
            d_str    = str(top_dir)
            done_set = done_by_dir[d_str]
            try:
                with os.scandir(str(top_dir)) as scanner:
                    for entry in scanner:
                        if (entry.is_file(follow_symlinks=False)
                                and entry.name.endswith(".pt")
                                and not entry.name.startswith(_MASKS_PREFIX)
                                and entry.name not in done_set):
                            yield d_str, Path(entry.path)
            except PermissionError as exc:
                print(f"[WARNING] Cannot scan {top_dir}: {exc}")

    # ------------------------------------------------------------------
    # 3. PARALLEL COMPUTATION — streaming producer / consumer
    # ------------------------------------------------------------------
    try:
        all_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        all_cpus = os.cpu_count() or 1
    num_workers = max(1, all_cpus - 1)
    print(f"[Mask Precomputation] Using {num_workers}/{all_cpus} parallel worker processes...")

    worker_fn = functools.partial(
        _process_single_cache_file,
        target_nodes=target_nodes,
        min_k=min_k,
        max_k=max_k,
        algo_names=algo_names,
        seed=seed,
    )

    success_count = 0
    error_count   = 0

    def _flush_indices() -> None:
        """Atomically write all accumulated index dicts to disk."""
        for d_str, algo_map in accumulated.items():
            out_d_str = dir_map[d_str]
            for algo_name, index in algo_map.items():
                if index:
                    _save_mask_index(Path(out_d_str), algo_name, index)

    path_stream = (p for p in _path_stream())

    import torch.multiprocessing as mp
    mp_ctx = mp.get_context("spawn")

    with mp_ctx.Pool(
        processes=num_workers,
        initializer=_worker_initializer,
        maxtasksperchild=50
    ) as pool:
        
        results_iter = pool.imap_unordered(
            worker_fn, 
            (path for d_str, path in path_stream),
            chunksize=10
        )
        
        from tqdm import tqdm
        with tqdm(desc="Computing partition masks", unit=" files") as pbar:
            for result in results_iter:
                if result is not None:
                    d_str, basename, algo_results = result
                    for algo_name, entry in algo_results.items():
                        accumulated[d_str][algo_name][basename] = {
                            "mask": torch.from_numpy(entry["mask"]).clone(),
                            "k": torch.tensor([entry["k"]], dtype=torch.long, device="cpu"),
                        }
                    success_count += 1
                
                pbar.update(1)

                # Periodic checkpoint — preserve progress across SLURM kills.
                if success_count > 0 and success_count % CHECKPOINT_EVERY == 0:
                    print(f"\n[Checkpoint] {success_count} done — flushing indices...")
                    _flush_indices()

    # ------------------------------------------------------------------
    # 4. FINAL SAVE
    # ------------------------------------------------------------------
    print(f"\n[Mask Precomputation] Saving final indices to disk...")
    _flush_indices()

    for d_str, algo_map in accumulated.items():
        for algo_name, index in algo_map.items():
            idx_path = _get_index_path(Path(dir_map[d_str]), algo_name)
            print(f"  {idx_path}  ({len(index)} entries)")

    print(
        f"\n[Mask Precomputation] Complete! "
        f"Processed {success_count} files, {error_count} errors."
    )
    print("Graph .pt files were NOT modified — masks live in index files only.")


if __name__ == "__main__":
    import argparse
    import config

    _seed = getattr(config, "PARTITION_SEED", 0)

    parser = argparse.ArgumentParser(
        description="Precompute dynamic-k partition masks for cached graphs in parallel."
    )
    parser.add_argument(
        "algorithm",
        type=str,
        choices=["metis", "span_weighted_metis", "level_slicing", "random", "all"],
        help="Partition algorithm to run, or 'all' to run all available algorithms.",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="One or more flat cache directories (or individual .pt files) to process.",
    )
    parser.add_argument(
        "--out-dirs",
        nargs="+",
        required=False,
        help="Corresponding directories to save the index files.",
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
        out_directories=args.out_dirs,
    )