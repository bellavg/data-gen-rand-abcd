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


def random_partitioning(
    data_obj: _PyGData,
    *,
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
    result = PartitionedData(**{k: v for k, v in data_obj})

    n = data_obj.num_nodes
    device = data_obj.x.device  # Ensure strict device consistency
    
    # 1. Assign partition IDs randomly
    partition_id = torch.randint(0, num_partitions, (n,), dtype=torch.long, device=device)
    
    # 2. Sort partition IDs to ensure contiguous partitions
    sort_idx = torch.argsort(partition_id)
    sorted_partition_id = partition_id[sort_idx]
    
    # 3. Map old node indices to new contiguous indices (Canonical Permutation Inversion)
    map_tensor = torch.empty(n, dtype=torch.long, device=device)
    map_tensor[sort_idx] = torch.arange(n, device=device)
    
    # 4. Update all node-level attributes safely using PyG's native inspection
    for key, value in result:
        if result.is_node_attr(key) and torch.is_tensor(value) and value.size(0) == n:
            result[key] = value[sort_idx]
            
    result.partition_id = sorted_partition_id
    result.num_partitions = torch.tensor([num_partitions], dtype=torch.long, device=device)

    # 5. Map and filter edges (removing cross-partition edges)
    mapped_edge_index = map_tensor[result.edge_index]
    src, dst = mapped_edge_index
    intra_mask = result.partition_id[src] == result.partition_id[dst]
    
    result.edge_index = mapped_edge_index[:, intra_mask]
    result.edge_attr = result.edge_attr[intra_mask]
    if hasattr(result, "edge_weight") and result.edge_weight is not None:
        result.edge_weight = result.edge_weight[intra_mask]

    return result

__all__ = ["PartitionedData", "random_partitioning"]
