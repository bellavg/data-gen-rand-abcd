from __future__ import annotations

import torch
from pathlib import Path
from torch_geometric.data import Data as _PyGData


class PartitionedData(_PyGData):
    """PyG Data subclass that carries per-node partition labels.

    The only difference from the base ``Data`` class is the ``__inc__``
    override for ``partition_id``: during ``Batch.from_data_list`` each
    graph's ``partition_id`` values are offset by the **previous** graph's
    ``num_partitions``, making partition IDs globally unique across the batch.

    Example (batch of two graphs, each split into 2 partitions)::

        batch.partition_id   → [0, 0, 1, 1, 2, 2, 3, 3]   # globally unique
        batch.num_partitions → [2, 2]                       # per-graph scalar
        batch.batch          → [0, 0, 0, 0, 1, 1, 1, 1]   # standard PyG

    You can then pool per partition with::

        from torch_scatter import scatter
        partition_repr = scatter(batch.x, batch.partition_id, dim=0, reduce="mean")
    """

    def __inc__(self, key: str, value, *args, **kwargs):
        if key == "partition_id":
            # Offset by this graph's num_partitions so IDs are globally unique.
            return self.num_partitions
        return super().__inc__(key, value, *args, **kwargs)


# =====================================================================
# MASK INDEX CACHE
#
# Each dataloader worker process loads the per-directory mask index
# exactly once (lazily, on first access) and reuses it for all
# subsequent ``get()`` calls.  The OS page cache lets multiple workers
# share the underlying memory pages when the index is opened with
# ``mmap=True``, keeping per-worker resident memory low.
# =====================================================================

# Key: (str(cache_dir), algo_name) → index dict
_MASK_INDEX_CACHE: dict[tuple[str, str], dict] = {}

_MASKS_PREFIX = "_masks_"


def _get_mask_entry(
    cache_dir: Path,
    algo_name: str,
    basename: str,
) -> dict | None:
    """Return the index entry for *basename* in *cache_dir* for *algo_name*.

    The index is loaded from ``{cache_dir}/_masks_{algo_name}.pt`` on first
    access and then kept in the module-level ``_MASK_INDEX_CACHE``.  Entries
    have the form ``{"mask": torch.Tensor, "k": torch.Tensor}``.

    Returns ``None`` if the index file does not exist or the basename is not
    present in it.
    """
    cache_key = (str(cache_dir), algo_name)
    if cache_key not in _MASK_INDEX_CACHE:
        _MASK_INDEX_CACHE[cache_key] = {}
        for index_path in cache_dir.glob(f"{_MASKS_PREFIX}{algo_name}*.pt"):
            try:
                # mmap=True: tensor data is lazily paged from disk.
                chunk = torch.load(
                    index_path,
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
                _MASK_INDEX_CACHE[cache_key].update(chunk)
            except TypeError:
                chunk = torch.load(
                    index_path,
                    map_location="cpu",
                    weights_only=True,
                )
                _MASK_INDEX_CACHE[cache_key].update(chunk)
            except Exception as exc:
                print(f"[partition_utils] WARNING: could not load chunk {index_path}: {exc}")

    return _MASK_INDEX_CACHE[cache_key].get(basename)


def clear_mask_index_cache() -> None:
    """Drop all cached mask indices (useful between trials in Optuna studies)."""
    _MASK_INDEX_CACHE.clear()


def partition_by_assignment(
    data_obj: _PyGData,
    partition_id: torch.Tensor,
    num_partitions: int,
) -> PartitionedData:
    """Partition a PyG Data object based on the provided partition assignments.

    Args:
        data_obj:       A PyG ``Data`` object.
        partition_id:   A 1D long tensor of shape ``[num_nodes]`` mapping each
                        node to a partition index in ``{0, …, num_partitions - 1}``.
        num_partitions: Number of partitions.

    Returns:
        A ``PartitionedData`` object with all original attributes but with:
        * Nodes sorted so that partitions are contiguous.
        * Node-level attributes permuted to match the sorted node order.
        * Cross-partition edges removed/zeroed out.
        * ``partition_id`` and ``num_partitions`` added/updated.
    """
    result = PartitionedData(**{k: v for k, v in data_obj})

    n = data_obj.num_nodes
    device = data_obj.x.device  # Ensure strict device consistency
    
    # 1. Sort partition IDs to ensure contiguous partitions
    sort_idx = torch.argsort(partition_id)
    sorted_partition_id = partition_id[sort_idx]
    
    # 2. Map old node indices to new contiguous indices (Canonical Permutation Inversion)
    map_tensor = torch.empty(n, dtype=torch.long, device=device)
    map_tensor[sort_idx] = torch.arange(n, device=device)
    
    # 3. Update all node-level attributes safely using PyG's native inspection
    for key, value in result:
        if result.is_node_attr(key) and torch.is_tensor(value) and value.size(0) == n:
            result[key] = value[sort_idx]
            
    result.partition_id = sorted_partition_id
    result.num_partitions = torch.tensor([num_partitions], dtype=torch.long, device=device)

    # 4. Map and filter edges (removing cross-partition edges)
    mapped_edge_index = map_tensor[result.edge_index]
    src, dst = mapped_edge_index
    intra_mask = result.partition_id[src] == result.partition_id[dst]
    
    result.edge_index = mapped_edge_index[:, intra_mask]
    if "edge_attr" in result and result.edge_attr is not None:
        result.edge_attr = result.edge_attr[intra_mask]
    if hasattr(result, "edge_weight") and result.edge_weight is not None:
        result.edge_weight = result.edge_weight[intra_mask]

    return result


def random_partitioning(
    data_obj: _PyGData,
    num_partitions: int = 2,
) -> PartitionedData:
    """Label each node with a random partition assignment.

    All nodes and edges are **retained** (no filtering).  Each node receives
    a partition label in ``{0, …, num_partitions - 1}`` stored in the
    ``partition_id`` attribute (shape ``[num_nodes]``, dtype long).

    Because ``PartitionedData`` overrides ``__inc__``, calling
    ``Batch.from_data_list`` on a list of these objects produces a batch
    where ``partition_id`` values are globally unique — graph *i*'s labels
    are offset by the sum of ``num_partitions`` of all preceding graphs.

    Args:
        data_obj:       A PyG ``Data`` object.
        num_partitions: Number of partitions to assign nodes to (default 2).

    Returns:
        A ``PartitionedData`` object with all original attributes plus:

        * ``partition_id``   – shape ``[num_nodes]``, long, values in
                                ``{0, …, num_partitions - 1}``.
        * ``num_partitions`` – shape ``[1]``, long.
    """
    n = data_obj.num_nodes
    device = data_obj.x.device  # Ensure strict device consistency
    
    # Assign partition IDs randomly
    partition_id = torch.randint(0, num_partitions, (n,), dtype=torch.long, device=device)
    
    return partition_by_assignment(data_obj, partition_id, num_partitions)


def precomputed_partitioning(
    data_obj: _PyGData,
    algo_name: str,
    cache_path: str | Path | None = None,
) -> PartitionedData:
    """Apply a precomputed partitioning assignment to a graph.

    Lookup order:

    1. **Embedded attributes** (backward-compatible): checks for
       ``{algo_name}_dynamic_mask`` directly on *data_obj*.  This covers
       graphs whose masks were written into the ``.pt`` file by an older
       version of the precompute pipeline.

    2. **Directory index file**: looks up the mask in the per-directory
       ``_masks_{algo_name}.pt`` index file written by the current pipeline.
       Requires *cache_path* to be provided so the directory can be derived.

    Args:
        data_obj:    A PyG ``Data`` object.
        algo_name:   The partition algorithm name (e.g. ``"metis"``).
        cache_path:  Absolute path to the ``.pt`` file that *data_obj* was
                     loaded from.  Required for the index-file lookup path.
                     May be omitted when embedded attributes are guaranteed
                     to be present (e.g. in unit tests).

    Returns:
        A ``PartitionedData`` object with partition-contiguous nodes and
        cross-partition edges removed.

    Raises:
        AttributeError: If neither embedded attributes nor the index file
                        contain a mask for this graph + algorithm.
    """
    key     = f"{algo_name}_dynamic_mask"
    num_key = f"{algo_name}_dynamic_num_partitions"

    partition_id: torch.Tensor | None = None
    num_partitions: int | None        = None

    # ------------------------------------------------------------------
    # 1. Try embedded attributes (backward compat with old pipeline)
    # ------------------------------------------------------------------
    if hasattr(data_obj, key) or key in data_obj.keys():
        partition_id   = data_obj[key]
        num_partitions = int(data_obj[num_key].item())

    # ------------------------------------------------------------------
    # 2. Try the per-directory index file
    # ------------------------------------------------------------------
    if partition_id is None and cache_path is not None:
        p = Path(cache_path)
        entry = _get_mask_entry(p.parent, algo_name, p.name)
        if entry is not None:
            partition_id   = entry["mask"]
            num_partitions = int(entry["k"].item())

    if partition_id is None:
        raise AttributeError(
            f"Precomputed partition mask for algorithm '{algo_name}' not found.\n"
            f"  Checked embedded attribute '{key}' on data_obj: not present.\n"
            f"  Checked index file '_masks_{algo_name}.pt' in cache directory"
            + (f" '{Path(cache_path).parent}': not present." if cache_path else ": cache_path not provided.")
            + f"\nPrecompute masks by running:\n"
            f"  python -m data.partition {algo_name} --dirs <cache_dir>"
        )

    # Cast to long tensor on the correct device
    device = data_obj.x.device
    if not isinstance(partition_id, torch.Tensor):
        partition_id = torch.tensor(partition_id, dtype=torch.long, device=device)
    else:
        partition_id = partition_id.to(dtype=torch.long, device=device)

    return partition_by_assignment(data_obj, partition_id, num_partitions)


__all__ = [
    "PartitionedData",
    "random_partitioning",
    "partition_by_assignment",
    "precomputed_partitioning",
    "clear_mask_index_cache",
]
