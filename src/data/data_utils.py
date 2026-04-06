import os
import re
from pathlib import Path
from typing import Optional, Tuple, Union

# aigverse.adapters is required to attach the .to_networkx() method to Aig objects
import aigverse.adapters  # noqa: F401
import networkx as nx
import numpy as np
import torch
from aigverse import (
    read_aiger_into_aig,
)
from torch_geometric.data import Data


def default_workers() -> int:
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK", "").strip()
    if slurm_cpus.isdigit():
        return max(1, int(slurm_cpus))

    detected = os.cpu_count() or 1
    if detected >= 24:
        return 24
    return max(1, detected)


TIER1_NAME_RE = re.compile(
    r"^(?P<design>.+?)_(?P<algorithm>Orchestrate|Deepsyn|Syn4|C2RS)_tier1_syn(?P<recipe>[0-9X]+)_step(?P<step>\d+)\.aig$"
)
TIER0_NAME_RE = re.compile(
    r"^(?P<design>.+?)_syn(?P<recipe>[0-9X]+)_step(?P<step>\d+)\.aig$"
)


def parse_aig_name(name: str) -> Optional[Tuple[int, str, str]]:
    """Return (tier_id, algorithm, design) based on standardized AIG filename."""
    # Tier2 outputs exist in some storage layouts; never treat them as tier0/tier1 inputs.
    if "_tier2_" in name:
        return None

    match_t1 = TIER1_NAME_RE.match(name)
    if match_t1:
        return 1, match_t1.group("algorithm"), match_t1.group("design")

    # If the name claims tier1 but does not match the strict tier1 pattern, skip it.
    if "_tier1_" in name:
        return None

    match_t0 = TIER0_NAME_RE.match(name)
    if match_t0:
        return 0, "", match_t0.group("design")

    return None


def aig_to_pytorch_geometric(
    aig_path: Union[str, Path], neighborhood_cutoff: int = 2, max_path_depth: int = 10
) -> Data:
    """
    Transforms an aigverse Aig object into a PyTorch Geometric Data object.

    Node Features (x):
        - One-hot encoded node type (Const, PI, Gate, PO)

    Positional Encodings (stored as separate attributes):
        - level: Logic level
        - pi_paths: Number of paths from Primary Inputs to the node (ignoring edges that skip > max_path_depth levels)
        - local_sp_sum: Sum of directed shortest path distances within a local neighborhood cutoff

    Relative Distance Encodings:
        - rel_edge_index: Expanded edge index for nodes within neighborhood_cutoff
        - edge_rel_dist: The shortest path distance for the corresponding edge in rel_edge_index

    Edge Features (edge_attr):
        - One-hot encoded edge type (Regular, Inverted)
    """
    # Read the AIGER file into an Aig object using aigverse.
    aig_path = str(aig_path)
    aig = read_aiger_into_aig(aig_path)

    # 1. Convert AIG to a NetworkX DiGraph.
    # Using np.float32 generates the one-hot arrays directly compatible with PyTorch FloatTensors.
    # Node types are encoded as [const, pi, gate, po]
    # Edge types are encoded as [regular, inverted]
    G = aig.to_networkx(levels=True, dtype=np.float32)
    # Prefer counts from the Aig object when available (avoid re-parsing semantics).
    num_pis_aig = aig.num_pis()
    num_pos_aig = aig.num_pos()
    # Fallback to the NetworkX graph if Aig did not expose counts
    num_nodes = G.number_of_nodes()
    # NetworkX labels may not be 0..N-1; keep a stable topological order and map
    # node labels to contiguous tensor indices.
    node_order = list(nx.topological_sort(G))
    node_to_idx = {node: idx for idx, node in enumerate(node_order)}
    # Initialize path counts to 0 for all nodes
    path_counts = {n: 0 for n in G.nodes()}
    local_sp_feature = {}

    # New lists to store the relative encoding edges and their distances
    rel_edge_indices = []
    rel_edge_dists = []

    # 2. Single pass over the nodes in topological order
    # This correctly computes paths and local shortest paths efficiently
    for n in node_order:
        # Index 1 of the 'type' array corresponds to 'pi' (Primary Input)
        if G.nodes[n]["type"][1] == 1.0:
            path_counts[n] = 1

        # Propagate paths: Add current node's path count to successors
        # Using G.adj[n] view for faster successor iteration (avoiding method call overhead)
        for successor in G.adj[n]:
            # Only propagate if the difference in logic level (depth) is within the max cap
            if (
                G.nodes[successor].get("level", 0) - G.nodes[n].get("level", 0)
                <= max_path_depth
            ):
                path_counts[successor] += path_counts[n]

        # Compute local neighborhood distances
        lengths = nx.single_source_shortest_path_length(
            G, n, cutoff=neighborhood_cutoff
        )
        local_sp_feature[n] = sum(lengths.values())

        # Store the relative distances as new edges (ignoring self-loops where d=0)
        for tgt, d in lengths.items():
            if d > 0:
                rel_edge_indices.append([node_to_idx[n], node_to_idx[tgt]])
                rel_edge_dists.append([float(d)])

    # 3. Build Node Features Matrix (x) and separate Positional Encodings
    x_features = []
    level_features = []
    pi_path_features = []
    local_sp_features_list = []
    max_path_node = max(path_counts, key=path_counts.get) if path_counts else None
    max_path_value = path_counts[max_path_node] if max_path_node is not None else 0
    debug_path_counts = os.getenv("AIG_DEBUG_PATH_COUNTS", "0") == "1"
    if debug_path_counts and max_path_node is not None:
        max_bit_length = int(max_path_value).bit_length()
        approx_log10 = (
            (max_bit_length - 1) * 0.3010299956639812 if max_bit_length > 0 else 0.0
        )
        print(
            "debug:path_counts "
            f"aig_path={aig_path} num_nodes={num_nodes} num_edges={G.number_of_edges()} "
            f"max_path_node={max_path_node} max_path_bit_length={max_bit_length} "
            f"max_path_approx_log10={approx_log10:.2f}"
        )

    # Iterate over node_order to keep row ordering aligned with node_to_idx.
    for n in node_order:
        data = G.nodes[n]

        # Native aigverse one-hot encoded node type -> list of 4 floats
        x_features.append(data["type"].tolist())

        # Separate positional encodings
        level_features.append([float(data.get("level", 0.0))])
        try:
            pi_paths_value = float(path_counts[n])
        except OverflowError as exc:
            value = path_counts[n]
            bit_length = int(value).bit_length()
            approx_log10 = (
                (bit_length - 1) * 0.3010299956639812 if bit_length > 0 else 0.0
            )
            raise OverflowError(
                "pi_paths overflow while converting path-count to float; "
                f"aig_path={aig_path}; node_label={n}; node_index={node_to_idx.get(n)}; "
                f"level={data.get('level', 0)}; in_degree={G.in_degree(n)}; out_degree={G.out_degree(n)}; "
                f"path_count_bit_length={bit_length}; path_count_approx_log10={approx_log10:.2f}; "
                f"max_path_node={max_path_node}; max_path_bit_length={int(max_path_value).bit_length() if max_path_node is not None else 0}; "
                f"num_nodes={num_nodes}; num_edges={G.number_of_edges()}; max_path_depth={max_path_depth}"
            ) from exc
        pi_path_features.append([pi_paths_value])
        local_sp_features_list.append([float(local_sp_feature[n])])

    x = torch.tensor(x_features, dtype=torch.float32)
    level_tensor = torch.tensor(level_features, dtype=torch.float32)
    pi_paths_tensor = torch.tensor(pi_path_features, dtype=torch.float32)
    local_sp_tensor = torch.tensor(local_sp_features_list, dtype=torch.float32)

    # 4. Build Edge Indices (edge_index) and Edge Attributes (edge_attr)
    edge_indices = []
    edge_attr_features = []

    # Using edges.data('type') view to avoid extracting unused edge dictionary keys
    for u, v, e_type in G.edges.data("type"):
        edge_indices.append([node_to_idx[u], node_to_idx[v]])

        # Native aigverse one-hot encoded edge type: [regular, inverted]
        edge_attr_features.append(e_type.tolist())

    # Format tensors for PyTorch Geometric (edge_index must be shape [2, E])
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_features, dtype=torch.float32)
    else:
        # Fallback for an empty graph
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)

    # Format relative distances tensors
    if len(rel_edge_indices) > 0:
        rel_edge_index = (
            torch.tensor(rel_edge_indices, dtype=torch.long).t().contiguous()
        )
        edge_rel_dist = torch.tensor(rel_edge_dists, dtype=torch.float32)
    else:
        rel_edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_rel_dist = torch.empty((0, 1), dtype=torch.float32)

    # 5. Create PyTorch Geometric Data object with separated features
    data_obj = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        level=level_tensor,
        pi_paths=pi_paths_tensor,
        local_sp_sum=local_sp_tensor,
        rel_edge_index=rel_edge_index,
        edge_rel_dist=edge_rel_dist,
    )
    # Annotate nodes/edges and PI/PO indices for consumers
    data_obj.num_nodes = num_nodes
    data_obj.num_edges = edge_index.size(1)
    data_obj.num_pis = num_pis_aig
    data_obj.num_pos = num_pos_aig

    return data_obj
