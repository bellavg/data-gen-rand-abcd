import math
import collections
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# aigverse.adapters is required to attach the .to_networkx() method to Aig objects
import aigverse.adapters  # noqa: F401
import torch
from aigverse import (
    read_aiger_into_aig,
)
from torch_geometric.data import Data


def _safe_log1p_int(value: int) -> float:
    """Numerically-stable log(1 + value) for very large Python integers."""
    if value < 0:
        raise ValueError(f"expected non-negative path count, got {value}")

    try:
        return math.log1p(value)
    except OverflowError:
        # For very large ints, avoid int->float overflow by estimating log(value)
        # from the top-most bits: value ~= mantissa * 2**shift.
        if value == 0:
            return 0.0
        bits = value.bit_length()
        shift = max(0, bits - 53)
        mantissa = value >> shift
        return math.log(float(mantissa)) + shift * math.log(2.0)


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

    # If the name claims tier1 but does not match the strict canonical pattern, skip it.
    # Messy names (with mktemp junk token) must be cleaned by cleanup_naming.py first.
    if "_tier1_" in name:
        return None

    match_t0 = TIER0_NAME_RE.match(name)
    if match_t0:
        return 0, "", match_t0.group("design")

    return None


def _extract_topology(
    aig, depth_aig
) -> Tuple[
    int, List[List[float]], List[List[float]], List[tuple], Dict[int, List[int]]
]:
    """
    Extracts the graph topology, creating stable indices for base nodes and synthetic POs.
    Returns node count, raw features, and edge representations.
    """
    base_nodes = list(aig.nodes())

    # Add a check if constant 0 is there; if it is not, add it at the start.
    if 0 not in base_nodes:
        base_nodes.insert(0, 0)

    # Ensure deterministic topological progression for path propagation.
    # Keep constant node first when present.
    other_nodes_original = [n for n in base_nodes if n != 0]

    # Cheap numeric sort: AIG node IDs are expected to be in ascending
    # creation/order-of-assignment. Check and assert this invariant so we
    # catch unexpected ordering during development.
    other_nodes_sorted = sorted(other_nodes_original)
    assert other_nodes_original == other_nodes_sorted, (
        "AIG nodes are not in ascending ID order; expected creation-order IDs."
    )

    base_nodes = ([0] if 0 in base_nodes else []) + other_nodes_sorted

    # Assert that node labels are contiguous 0..(n-1). We rely on this
    # invariant for stable indexing and topological propagation.
    num_base_nodes = len(base_nodes)
    assert base_nodes == list(range(num_base_nodes)), (
        f"AIG node labels must be contiguous 0..n-1, got: {base_nodes}"
    )
    num_nodes = num_base_nodes + aig.num_pos()

    node_to_idx = {n: i for i, n in enumerate(base_nodes)}

    x_features = []
    level_features = []
    edges = []
    successors = collections.defaultdict(list)

    # Process Base Nodes (Constants, PIs, Gates)
    for n in base_nodes:
        idx = node_to_idx[n]

        # Determine One-hot node type [const, pi, gate, po]
        if aig.is_constant(n):
            ntype = [1.0, 0.0, 0.0, 0.0]
        elif aig.is_pi(n):
            ntype = [0.0, 1.0, 0.0, 0.0]
        else:
            ntype = [0.0, 0.0, 1.0, 0.0]

        x_features.append(ntype)
        level_features.append([float(depth_aig.level(n))])

        # Process incoming edges from fanins
        for f_sig in aig.fanins(n):
            # Use the documented pyaigverse API: AigSignal exposes
            # `get_index()` and `get_complement()` methods.
            try:
                fanin_node = f_sig.get_index()
                inv = 1.0 if f_sig.get_complement() else 0.0
            except AttributeError:
                raise AttributeError(
                    f"Unexpected fanin signal object {type(f_sig)}; expected get_index()/get_complement()"
                )

            u_idx = node_to_idx[fanin_node]
            v_idx = idx
            e_type = [1.0 - inv, inv]  # [regular, inverted]

            edges.append((u_idx, v_idx, e_type))
            successors[u_idx].append(v_idx)

    # Process Synthetic PO Nodes
    # Iterating over the PO signals ensures we never merge two POs that point
    # to the same node in the AIG.
    for i, po_sig in enumerate(aig.pos()):
        idx = num_base_nodes + i

        ntype = [0.0, 0.0, 0.0, 1.0]  # PO node type
        x_features.append(ntype)

        # PO depth is +1 to the node driving it
        driver_idx = node_to_idx[po_sig.get_index()]
        level_features.append([level_features[driver_idx][0] + 1.0])

        inv = 1.0 if po_sig.get_complement() else 0.0
        e_type = [1.0 - inv, inv]

        edges.append((driver_idx, idx, e_type))
        successors[driver_idx].append(idx)

    return num_nodes, x_features, level_features, edges, successors


def _compute_paths_and_distances(
    num_nodes: int,
    x_features: List[List[float]],
    level_features: List[List[float]],
    successors: Dict[int, List[int]],
    max_path_depth: int,
    neighborhood_cutoff: int,
) -> Tuple[List[int], List[List[float]]]:
    """
    Computes PI-to-node path counts and local shortest paths via BFS.
    """
    path_counts = [0] * num_nodes
    local_sp_features_list = [0.0] * num_nodes

    # Base nodes are naturally topological, and synthetic POs appear at the end.
    for n_idx in range(num_nodes):
        # Initialize paths for PIs
        if x_features[n_idx][1] == 1.0:
            path_counts[n_idx] = 1

        # Propagate paths downstream
        for succ_idx in successors[n_idx]:
            if level_features[succ_idx][0] - level_features[n_idx][0] <= max_path_depth:
                path_counts[succ_idx] += path_counts[n_idx]

        # BFS replacing NetworkX's single_source_shortest_path_length
        distances = {n_idx: 0}
        queue = collections.deque([(n_idx, 0)])

        while queue:
            curr, dist = queue.popleft()
            if dist < neighborhood_cutoff:
                for nxt in successors[curr]:
                    if nxt not in distances:
                        distances[nxt] = dist + 1
                        queue.append((nxt, dist + 1))

        local_sp_features_list[n_idx] = [float(sum(distances.values()))]

    return path_counts, local_sp_features_list


def _build_tensors(
    x_features: List[List[float]],
    level_features: List[List[float]],
    path_counts: List[int],
    local_sp_features_list: List[List[float]],
    edges: List[tuple],
):
    """
    Converts raw Python lists into PyTorch tensors for PyTorch Geometric.
    """
    x = torch.tensor(x_features, dtype=torch.float32)
    level_tensor = torch.tensor(level_features, dtype=torch.float32)
    pi_paths_tensor = torch.tensor(
        [[_safe_log1p_int(pc)] for pc in path_counts], dtype=torch.float32
    )
    local_sp_tensor = torch.tensor(local_sp_features_list, dtype=torch.float32)

    if len(edges) > 0:
        edge_indices = [[u, v] for u, v, _ in edges]
        edge_attr_features = [e_type for _, _, e_type in edges]
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_features, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)

    return x, level_tensor, pi_paths_tensor, local_sp_tensor, edge_index, edge_attr


def aig_to_pytorch_geometric(
    aig_path: Union[str, Path], neighborhood_cutoff: int = 2, max_path_depth: int = 10
) -> Data:
    """
    Transforms an aigverse Aig object into a PyTorch Geometric Data object
    directly, avoiding aig.to_networkx() and aig.to_edge_list() bugs.
    """
    # Ensure the path is safely converted to a string for C++ bindings
    path_str = str(aig_path)

    # Read the AIGER file into an Aig object
    aig = read_aiger_into_aig(path_str)

    # Use DepthAig view to calculate level (depth) properties efficiently.
    depth_aig = aigverse.DepthAig(aig)
    # 1. Extract Topology from real Aig
    num_nodes, x_features, level_features, edges, successors = _extract_topology(
        aig, depth_aig
    )

    # 2. Compute Paths and Distances
    path_counts, local_sp_features = _compute_paths_and_distances(
        num_nodes,
        x_features,
        level_features,
        successors,
        max_path_depth,
        neighborhood_cutoff,
    )

    # 3. Build Tensors
    x, level_tensor, pi_paths_tensor, local_sp_tensor, edge_index, edge_attr = (
        _build_tensors(
            x_features, level_features, path_counts, local_sp_features, edges
        )
    )

    # 4. Create PyTorch Geometric Data object
    data_obj = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        level=level_tensor,
        pi_paths=pi_paths_tensor,
        local_sp_sum=local_sp_tensor,
    )

    data_obj.num_nodes = num_nodes
    data_obj.num_edges = edge_index.size(1)
    data_obj.num_pis = aig.num_pis()
    data_obj.num_pos = aig.num_pos()

    return data_obj
