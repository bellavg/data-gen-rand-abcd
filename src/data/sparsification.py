import os
import uuid
import functools
import concurrent.futures
from pathlib import Path
from tqdm import tqdm

import torch
import networkx as nx
from torch_geometric.utils import to_networkx

# =====================================================================
# DUMMY ALGORITHMS
# =====================================================================

def random_edge_dropout(data_obj, dropout_rate=0.5, seed=0):
    """Sparsification 1: Randomly drop a percentage of edges."""
    torch.manual_seed(seed)
    num_edges = data_obj.edge_index.shape[1]
    # keep edges where random value is greater than dropout_rate
    return torch.rand(num_edges) >= dropout_rate

def spanner_sparsification(data_obj, stretch=3.0, seed=0):
    """Sparsification 2: NetworkX spanner algorithm."""
    # Convert to an undirected networkx graph
    G = to_networkx(data_obj, to_undirected=True)
    
    # Compute the spanner
    H = nx.spanner(G, stretch=stretch, seed=seed)
    
    # We map the undirected edges in H back to the directed edge_index
    # Enforce lower index to higher index direction
    h_edges = set()
    for u, v in H.edges():
        lower_idx, higher_idx = min(u, v), max(u, v)
        h_edges.add((lower_idx, higher_idx))
        
    num_edges = data_obj.edge_index.shape[1]
    mask = torch.zeros(num_edges, dtype=torch.bool)
    
    edge_index_list = data_obj.edge_index.t().tolist()
    for i, (u, v) in enumerate(edge_index_list):
        if (u, v) in h_edges:
            mask[i] = True
            
    return mask

# =====================================================================

def _process_single_cache_file(
    cache_path: Path,
    algo_names: list[str],
    dropout_rate: float,
    stretch: float,
    seed: int,
) -> None:
    if not cache_path.is_file():
        return

    # 1. Load the existing .pt file.
    # PyG data objects often require weights_only=False or safe globals
    with open(cache_path, "rb") as fh:
        data_obj = torch.load(fh, map_location="cpu", weights_only=False)

    missing_algos = []
    for algo_name in algo_names:
        if not hasattr(data_obj, f"{algo_name}_sparsification_mask"):
            missing_algos.append(algo_name)
    
    if not missing_algos:
        # All requested algorithms are already computed.
        return

    for algo_name in missing_algos:
        if algo_name == "random_edge_dropout":
            mask_tensor = random_edge_dropout(data_obj, dropout_rate=dropout_rate, seed=seed)
        elif algo_name == "spanner":
            mask_tensor = spanner_sparsification(data_obj, stretch=stretch, seed=seed)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        if not isinstance(mask_tensor, torch.Tensor):
            mask_tensor = torch.tensor(mask_tensor, dtype=torch.bool)

        setattr(data_obj, f"{algo_name}_sparsification_mask",
                mask_tensor.to(dtype=torch.bool, device="cpu"))

    # 3. Atomically overwrite the file on disk.
    temp_file = cache_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
    torch.save(data_obj, temp_file)
    os.replace(temp_file, cache_path)


# =====================================================================
# CORE UPDATE PIPELINE
# =====================================================================

def update_existing_cache_with_masks(
    directories: list[str | Path],
    algo_names: list[str],
    dropout_rate: float,
    stretch: float,
    seed: int,
) -> None:
    """Loads pre-cached graph files from specified directories, computes sparsification masks in parallel, and saves them back.

    Searches recursively for all ``*.pt`` files in the provided directories.
    Deduplicates the file paths so that each file is processed exactly once.

    Stored attributes (per graph, per algorithm):
        ``{algo_name}_sparsification_mask`` – 1-D bool tensor, shape [num_edges]
    """
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
        algo_names=algo_names,
        dropout_rate=dropout_rate,
        stretch=stretch,
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
    import config
    import argparse

    _seed = getattr(config, "SPARSIFICATION_SEED", 0)
    _dropout_rate = getattr(config, "SPARSIFICATION_RANDOM_DROPOUT_RATE", 0.5)
    _stretch = getattr(config, "SPARSIFICATION_SPANNER_STRETCH", 3.0)

    parser = argparse.ArgumentParser(
        description="Precompute sparsification edge masks for cached graphs in parallel."
    )
    parser.add_argument(
        "algorithm",
        type=str,
        choices=["random_edge_dropout", "spanner", "all"],
        help="Sparsification algorithm to run, or 'all' to run all available sparsification algorithms."
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="One or more directory paths (or individual .pt files) to search recursively for cached graphs."
    )
    args = parser.parse_args()

    if args.algorithm == "all":
        algo_names = ["random_edge_dropout", "spanner"]
    else:
        algo_names = [args.algorithm]

    print(
        f"[sparsification.py] Running for algorithm(s)={sorted(algo_names)}\n"
        f"  dropout_rate={_dropout_rate}\n"
        f"  stretch={_stretch}\n"
        f"  seed={_seed}\n"
        f"  dirs={args.dirs}"
    )

    update_existing_cache_with_masks(
        directories=args.dirs,
        algo_names=algo_names,
        dropout_rate=_dropout_rate,
        stretch=_stretch,
        seed=_seed,
    )
