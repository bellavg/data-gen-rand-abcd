import os
import uuid
import functools
import concurrent.futures
from pathlib import Path
from tqdm import tqdm

import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

# =====================================================================
# ALGORITHMS
# =====================================================================

# Apply the node mask to create the sparsified graph
# sparsified_data = data_obj.subgraph(node_mask)

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

def pagerank_sparsification(data_obj, keep_ratio=0.8, alpha=0.85):
    """
    Identical functionality to the above, written for maximum tensor operation speed.
    """
    G = to_networkx(data_obj, to_undirected=False)
    pr_scores = nx.pagerank(G, alpha=alpha)
    
    sorted_nodes = sorted(pr_scores, key=pr_scores.get, reverse=True)
    num_to_keep = max(1, int(len(sorted_nodes) * keep_ratio))
    
    node_mask = torch.zeros(data_obj.num_nodes, dtype=torch.bool)
    node_mask[sorted_nodes[:num_to_keep]] = True
    
    return node_mask


def and_gate_only_sparsification(data_obj) -> Data:
    """AIG-specific graph transformation: remove PI and PO nodes, replace with self-loops.

    Operates on a pre-loaded PyG ``Data`` object (from a cached ``.pt`` file).
    Uses the one-hot node-type feature stored in ``x``:
        col 0 = constant, col 1 = PI, col 2 = AND gate, col 3 = PO

    Rules applied:
    * **PI → gate** edges  → self-loop on the **receiving gate** with the same
      ``edge_attr`` (inversion encoding preserved).
    * **gate → PO** edges  → self-loop on the **driving gate** with the same
      ``edge_attr`` (inversion encoding preserved).
    * **gate → gate** edges → kept unchanged, with node indices remapped to a
      new contiguous range.
    * Self-loops with identical ``(node, edge_attr)`` are deduplicated.

    All node-level tensors (``x``, ``level``, ``pi_paths``, ``local_sp_sum``)
    are sliced to the kept nodes.  The original ``num_pis`` / ``num_pos``
    counts are preserved as metadata attributes.

    Returns a new ``Data`` object; the input is not modified.
    """
    x = data_obj.x                          # shape [N, 4]
    edge_index = data_obj.edge_index        # shape [2, E]
    edge_attr = data_obj.edge_attr          # shape [E, 2]

    # -----------------------------------------------------------------------
    # Classify nodes from the one-hot type vector
    # -----------------------------------------------------------------------
    is_pi = x[:, 1] == 1.0   # bool tensor, shape [N]
    is_po = x[:, 3] == 1.0

    pi_set: set[int] = set(is_pi.nonzero(as_tuple=True)[0].tolist())
    po_set: set[int] = set(is_po.nonzero(as_tuple=True)[0].tolist())
    removed: set[int] = pi_set | po_set

    n = x.size(0)
    kept: list[int] = [i for i in range(n) if i not in removed]
    old_to_new: dict[int, int] = {old: new for new, old in enumerate(kept)}

    # -----------------------------------------------------------------------
    # Build new edge list with self-loop substitutions
    # -----------------------------------------------------------------------
    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()
    attr_list = edge_attr.tolist()          # list of [regular, inverted] pairs

    new_src: list[int] = []
    new_dst: list[int] = []
    new_attr: list[list[float]] = []
    self_loop_seen: set[tuple] = set()     # (node_new, attr_tuple) dedup key

    for u, v, ea in zip(src_list, dst_list, attr_list):
        u_pi = u in pi_set
        v_po = v in po_set

        if u_pi and v not in removed:
            # PI → gate  →  self-loop on receiving gate
            v_new = old_to_new[v]
            key = (v_new, tuple(ea))
            if key not in self_loop_seen:
                new_src.append(v_new)
                new_dst.append(v_new)
                new_attr.append(ea)
                self_loop_seen.add(key)

        elif not u_pi and u not in removed and v_po:
            # gate → PO  →  self-loop on driving gate
            u_new = old_to_new[u]
            key = (u_new, tuple(ea))
            if key not in self_loop_seen:
                new_src.append(u_new)
                new_dst.append(u_new)
                new_attr.append(ea)
                self_loop_seen.add(key)

        elif u not in removed and v not in removed:
            # gate → gate  →  keep, remap indices
            new_src.append(old_to_new[u])
            new_dst.append(old_to_new[v])
            new_attr.append(ea)

        # Edges entirely inside PI/PO space are silently dropped.

    # -----------------------------------------------------------------------
    # Assemble tensors
    # -----------------------------------------------------------------------
    kept_t = torch.tensor(kept, dtype=torch.long)
    new_x = x[kept_t]

    if new_src:
        new_edge_index = torch.tensor(
            [new_src, new_dst], dtype=torch.long
        )
        new_edge_attr = torch.tensor(new_attr, dtype=edge_attr.dtype)
    else:
        new_edge_index = torch.empty((2, 0), dtype=torch.long)
        new_edge_attr = torch.empty((0, edge_attr.size(1)), dtype=edge_attr.dtype)

    out = Data(
        x=new_x,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
    )
    out.num_nodes = len(kept)
    out.num_edges = new_edge_index.size(1)

    # Slice any optional per-node tensors that may be present
    for attr_name in ("level", "pi_paths", "local_sp_sum"):
        val = getattr(data_obj, attr_name, None)
        if val is not None and isinstance(val, torch.Tensor) and val.size(0) == n:
            setattr(out, attr_name, val[kept_t])

    # Preserve original PI/PO counts as informational metadata
    for meta in ("num_pis", "num_pos"):
        val = getattr(data_obj, meta, None)
        if val is not None:
            setattr(out, meta, val)

    return out


# =====================================================================

def _process_single_cache_file(
    cache_path: Path,
    algo_names: list[str],
    dropout_rate: float,
    stretch: float,
    keep_ratio: float = 0.8,
    alpha: float = 0.85,
    seed: int = 42,
) -> None:
    if not cache_path.is_file():
        return

    # 1. Load the existing .pt file.
    # PyG data objects often require weights_only=False or safe globals
    with open(cache_path, "rb") as fh:
        data_obj = torch.load(fh, map_location="cpu", weights_only=False)

    missing_algos = []
    for algo_name in algo_names:
        # and_gate_only stores a full Data object, not a mask
        attr_key = "and_gate_only_graph" if algo_name == "and_gate_only" else f"{algo_name}_sparsification_mask"
        if not hasattr(data_obj, attr_key):
            missing_algos.append(algo_name)
    
    if not missing_algos:
        # All requested algorithms are already computed.
        return

    for algo_name in missing_algos:
        if algo_name == "and_gate_only":
            # Structural transform: store the reduced Data object directly
            data_obj.and_gate_only_graph = and_gate_only_sparsification(data_obj)
            continue

        if algo_name == "random_edge_dropout":
            mask_tensor = random_edge_dropout(data_obj, dropout_rate=dropout_rate, seed=seed)
        elif algo_name == "spanner":
            mask_tensor = spanner_sparsification(data_obj, stretch=stretch, seed=seed)
        elif algo_name == "pagerank":
            mask_tensor = pagerank_sparsification(data_obj, keep_ratio=keep_ratio, alpha=alpha)
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
    keep_ratio: float,
    alpha: float,
    seed: int,
) -> None:
    """Loads pre-cached graph files from specified directories, computes sparsification masks in parallel, and saves them back.

    Searches recursively for all ``*.pt`` files in the provided directories.
    Deduplicates the file paths so that each file is processed exactly once.

    Stored attributes (per graph, per algorithm):
        ``{algo_name}_sparsification_mask`` – 1-D bool tensor, shape [num_edges] (or [num_nodes] for node-based masks)
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
        keep_ratio=keep_ratio,
        alpha=alpha,
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
    _keep_ratio = getattr(config, "SPARSIFICATION_PAGERANK_KEEP_RATIO", 0.8)
    _alpha = getattr(config, "SPARSIFICATION_PAGERANK_ALPHA", 0.85)

    parser = argparse.ArgumentParser(
        description="Precompute sparsification edge masks for cached graphs in parallel."
    )
    parser.add_argument(
        "algorithm",
        type=str,
        choices=["random_edge_dropout", "spanner", "pagerank", "and_gate_only", "all"],
        help=(
            "Sparsification algorithm to run, or 'all' to run all available algorithms.\n"
            "  and_gate_only: strip PI/PO nodes; replace their edges with self-loops on adjacent gates."
        ),
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="One or more directory paths (or individual .pt files) to search recursively for cached graphs."
    )
    args = parser.parse_args()

    if args.algorithm == "all":
        algo_names = ["random_edge_dropout", "spanner", "pagerank", "and_gate_only"]
    else:
        algo_names = [args.algorithm]

    print(
        f"[sparsification.py] Running for algorithm(s)={sorted(algo_names)}\n"
        f"  dropout_rate={_dropout_rate}\n"
        f"  stretch={_stretch}\n"
        f"  keep_ratio={_keep_ratio}\n"
        f"  alpha={_alpha}\n"
        f"  seed={_seed}\n"
        f"  dirs={args.dirs}"
    )

    update_existing_cache_with_masks(
        directories=args.dirs,
        algo_names=algo_names,
        dropout_rate=_dropout_rate,
        stretch=_stretch,
        keep_ratio=_keep_ratio,
        alpha=_alpha,
        seed=_seed,
    )
