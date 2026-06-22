from __future__ import annotations

import os
import uuid
import torch
from pathlib import Path
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData
from torch_geometric.utils import to_undirected
# Ensure safe unpickling globals are registered by importing dataset first
from data.dataset import AIGGraphRegressionDataset

# =====================================================================
# PLACEHOLDERS FOR YOUR PARTITIONING ALGORITHMS
# Replace the contents of these functions with your actual library calls.
# =====================================================================

def run_metis(data_obj, num_partitions: int) -> torch.Tensor:
    """Computes METIS partitions using PyTorch Geometric's ClusterData.
    
    Safely handles directed AIG/circuit graphs by mapping them to an
    undirected skeleton structure before executing the METIS engine.
    """
    num_nodes = data_obj.num_nodes

    # 1. METIS requires an undirected structure. Symmetrize the edge indices 
    # to avoid a Segmentation Fault / Core Dump from METIS library binaries.
    undirected_edges = to_undirected(data_obj.edge_index, num_nodes=num_nodes)

    # 2. Create a lightweight skeleton Data container.
    # Bypassing node/edge features avoids redundant deep-copies in memory.
    skeleton_data = Data(edge_index=undirected_edges, num_nodes=num_nodes)

    # 3. Invoke PyG's METIS implementation wrapper
    # Setting recursive=False uses multi-level k-way partitioning (best for small k)
    cluster_data = ClusterData(
        skeleton_data, 
        num_parts=num_partitions, 
        recursive=False, 
        log=False
    )

    # 4. Extract the underlying partition assignment map.
    # ClusterData tracks partitions internally using a flat permutation tensor (node_perm)
    # and boundary index pointers (partptr) indicating where each cluster begins/ends.
    node_perm = cluster_data.partition.node_perm
    partptr = cluster_data.partition.partptr

    # 5. Unpack the layout boundaries into a flat [num_nodes] assignment mask
    assignment_mask = torch.empty(num_nodes, dtype=torch.long, device="cpu")
    
    for part_id in range(num_partitions):
        start_idx = int(partptr[part_id])
        end_idx = int(partptr[part_id + 1])
        
        # Pull global node IDs assigned to the current partition group
        allocated_nodes = node_perm[start_idx:end_idx]
        assignment_mask[allocated_nodes] = part_id

    return assignment_mask


def run_level_bisect(data_obj, num_partitions: int = 2) -> torch.Tensor:
    """Partitions a graph into ``num_partitions`` equal buckets by node level.

    Nodes are sorted by their ``level`` attribute (topological depth) and
    divided into ``num_partitions`` equal-sized buckets.  This preserves the
    natural DAG layering of AIG/circuit graphs rather than using a
    graph-topology method like METIS.

    Strategy
    --------
    1. Read ``data_obj.level`` — a 1-D or 2-D integer tensor of shape
       ``[num_nodes]`` or ``[num_nodes, 1]``.
    2. Sort the nodes by their level value.
    3. Divide the sorted order into ``num_partitions`` equal buckets and
       assign each bucket a partition index in ``{0, …, num_partitions - 1}``.

    Args:
        data_obj:       A PyG ``Data`` object carrying a ``level`` node attribute
                        (shape ``[num_nodes]`` or ``[num_nodes, 1]``, integer/float dtype).
        num_partitions: Number of equal-depth buckets to create (default 2).

    Returns:
        A 1-D ``torch.long`` tensor of shape ``[num_nodes]`` with values in
        ``{0, …, num_partitions - 1}`` representing partition membership.

    Raises:
        AttributeError: If ``data_obj`` has no ``level`` attribute.
        ValueError:     If ``num_partitions`` is less than 1.
    """
    if num_partitions < 1:
        raise ValueError(f"num_partitions must be >= 1, got {num_partitions}")

    if not hasattr(data_obj, "level") or data_obj.level is None:
        raise AttributeError(
            "run_level_bisect requires a 'level' node attribute on the Data object, "
            "but none was found. Ensure the cached .pt files include the 'level' tensor."
        )

    # Access the per-node level tensor — shape [num_nodes], integer dtype.
    level = data_obj.level
    if not isinstance(level, torch.Tensor):
        level = torch.tensor(level, dtype=torch.long)
    level = level.to(dtype=torch.long, device="cpu").view(-1)

    num_nodes = level.size(0)

    # Sort nodes by level, then assign equal-sized buckets.
    sort_idx = torch.argsort(level, stable=True)

    # Use torch.div with floor rounding to assign bucket IDs to sorted positions.
    # Each position i maps to bucket floor(i * num_partitions / num_nodes).
    positions = torch.arange(num_nodes, dtype=torch.long)
    bucket_for_position = torch.div(
        positions * num_partitions, num_nodes, rounding_mode="floor"
    ).clamp(max=num_partitions - 1)  # guard against floating-point edge at i==num_nodes

    assignment_mask = torch.empty(num_nodes, dtype=torch.long, device="cpu")
    assignment_mask[sort_idx] = bucket_for_position

    return assignment_mask


def run_kahip(data_obj: torch.geometric.data.Data, num_partitions: int) -> torch.Tensor:
    """Computes KaHIP partitions.
    Should return a 1D LongTensor of shape [num_nodes] filled with values 0 to num_partitions-1.
    """
    # TODO: Integrate your KaHIP wrapper here
    raise NotImplementedError("Integrate your KaHIP library call here.")


# =====================================================================
# CORE UPDATE PIPELINE
# =====================================================================

def update_existing_cache_with_masks(
    csv_paths: str | Path | list[str | Path],
    cache_dir: str | Path,
    tier0_cache_dir: str | Path | None = None,
    tier1_cache_dir: str | Path | None = None,
    partition_configs: list[tuple[str, int, callable]] | None = None,
) -> None:
    """Loads pre-cached graph files, computes requested partition masks, 
    and saves them back directly into the existing files.
    """
    if partition_configs is None:
        raise ValueError(
            "partition_configs must be provided explicitly. "
            "Pass a list of (algo_name, num_partitions, callable) tuples, e.g.:\n"
            "  [(\"metis\", 4, run_metis), (\"level_bisect\", 4, run_level_bisect)]\n"
            "Or run this module directly via __main__ to have it read from config.py."
        )

    print("[Mask Precomputation] Initializing dataset to discover file paths...")
    # Instantiate the dataset with split=None to resolve every graph across train/val/test
    dataset = AIGGraphRegressionDataset(
        csv_paths=csv_paths,
        cache_dir=cache_dir,
        tier0_cache_dir=tier0_cache_dir,
        tier1_cache_dir=tier1_cache_dir,
        split=None,
    )

    # Extract all distinct cached file paths from the manifest map
    unique_cache_paths = sorted(set(dataset._graph_cache_path_map.values()))
    print(f"[Mask Precomputation] Found {len(unique_cache_paths)} unique graph cache files to process.")

    success_count = 0
    
    # Iterate through each .pt file with a progress bar
    for cache_path in tqdm(unique_cache_paths, desc="Appending masks to cache"):
        if not cache_path.is_file():
            continue

        # 1. Load the existing file using your project's secure deserialization settings
        with open(cache_path, "rb") as fh:
            data_obj = torch.load(fh, map_location="cpu", weights_only=True)

        # 2. Iterate through all algorithm configurations and attach them as attributes
        for algo_name, num_partitions, algo_fn in partition_configs:
            mask_attr_name = f"{algo_name}_{num_partitions}_mask"
            num_attr_name = f"{algo_name}_{num_partitions}_num_partitions"

            # Execute your partitioning function
            mask_tensor = algo_fn(data_obj, num_partitions)
            
            if not isinstance(mask_tensor, torch.Tensor):
                mask_tensor = torch.tensor(mask_tensor, dtype=torch.long)
                
            # Direct attribute assignment on the existing object
            setattr(data_obj, mask_attr_name, mask_tensor.to(dtype=torch.long, device="cpu"))
            setattr(data_obj, num_attr_name, torch.tensor([num_partitions], dtype=torch.long, device="cpu"))

        # 3. Atomically overwrite the file on disk using your dataset's temp pattern
        # This prevents file corruption if the script is forcefully interrupted midway
        temp_file = cache_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
        torch.save(data_obj, temp_file)
        os.replace(temp_file, cache_path)
        success_count += 1

    print(f"\n[Mask Precomputation] Complete! Successfully updated {success_count} files.")
    print("All other properties (features, edge layouts, positional encodings) were preserved untouched.")


if __name__ == "__main__":
    import config

    # ---------------------------------------------------------------------------
    # All settings are driven by config.py — edit that file, not this one.
    # ---------------------------------------------------------------------------
    _ALGO_MAP = {
        "metis": run_metis,
        "level_bisect": run_level_bisect,
        "kahip": run_kahip,
    }

    _algo_name    = getattr(config, "PARTITION", "metis")
    _num_parts    = getattr(config, "NUM_PARTITIONS", 2)

    if _algo_name not in _ALGO_MAP:
        raise ValueError(
            f"config.PARTITION='{_algo_name}' is not a known algorithm. "
            f"Choose from: {sorted(_ALGO_MAP)}"
        )

    _partition_configs = [(_algo_name, _num_parts, _ALGO_MAP[_algo_name])]

    print(
        f"[partition.py] Using algorithm='{_algo_name}', "
        f"num_partitions={_num_parts} (from config.py)"
    )

    update_existing_cache_with_masks(
        csv_paths=getattr(config, "CSV_PATHS", "data/raw/your_dataset_manifest.csv"),
        cache_dir=getattr(config, "CACHE_DIR", "data/cache"),
        tier0_cache_dir=getattr(config, "TIER0_CACHE_DIR", None),
        tier1_cache_dir=getattr(config, "TIER1_CACHE_DIR", None),
        partition_configs=_partition_configs,
    )