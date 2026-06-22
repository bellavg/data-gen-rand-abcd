from __future__ import annotations

import torch
from torch_geometric.data import Data as _PyGData


class PartitionedData(_PyGData):
    """PyG Data subclass that carries per-node partition labels.

    The only difference from the base ``Data`` class is the ``__inc__``
    override for ``partition_id``: during ``Batch.from_data_list`` each
    graph's ``partition_id`` values are offset by the **previous** graph's
    ``num_partitions``, making partition IDs globally unique across the batch.

    Example (batch of two graphs, each split into 2 partitions):

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
    num_partitions: int = 2,
) -> PartitionedData:
    """Apply a precomputed partitioning assignment stored in the Data object.

    Args:
        data_obj:       A PyG ``Data`` object.
        algo_name:      The base attribute name (string) where the precomputed
                        partition assignments are stored (e.g., "metis").
        num_partitions: The number of partitions to retrieve.

    Returns:
        A ``PartitionedData`` object with partition-contiguous nodes and
        cross-partition edges zeroed out.
    """
    key = f"{algo_name}_{num_partitions}_mask"
    if not hasattr(data_obj, key) and key not in data_obj.keys():
        raise AttributeError(
            f"Precomputed partition assignment for '{algo_name}' with {num_partitions} "
            f"partitions was not found in Data object (tried '{key}')."
        )
    
    partition_id = data_obj[key]
    
    # Cast to long tensor if not already
    if not isinstance(partition_id, torch.Tensor):
        partition_id = torch.tensor(partition_id, dtype=torch.long, device=data_obj.x.device)
    else:
        partition_id = partition_id.to(dtype=torch.long, device=data_obj.x.device)
        
    return partition_by_assignment(data_obj, partition_id, num_partitions)


__all__ = [
    "PartitionedData",
    "random_partitioning",
    "partition_by_assignment",
    "precomputed_partitioning",
]
