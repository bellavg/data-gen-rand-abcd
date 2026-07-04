from __future__ import annotations

import os
import time
import uuid
import json
import functools
from collections import defaultdict
import torch
import numpy as np
import networkx as nx
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

# Prefix for index files — excluded from scanning so they are never treated
# as graph cache files.
_SPARSE_PREFIX = "_sparse_"
_MASKS_PREFIX = "_masks_"
CHECKPOINT_EVERY = 50_000   # atomic index save cadence (number of completed files)


# =====================================================================
# REGISTRATION
# =====================================================================

def _register_pyg_safe_globals() -> None:
    """Register PyG classes for ``weights_only=True`` torch.load in workers."""
    import torch.serialization
    import torch_geometric.data.data as _pyg_data_mod
    import torch_geometric.data.storage as _pyg_storage
    safe_globals: list = [Data, _pyg_storage.GlobalStorage]
    for _name in ("DataTensorAttr", "DataEdgeAttr"):
        _cls = getattr(_pyg_data_mod, _name, None)
        if _cls is not None:
            safe_globals.append(_cls)
    torch.serialization.add_safe_globals(safe_globals)


# =====================================================================
# ALGORITHMS
# =====================================================================

def random_edge_dropout(data_obj, dropout_rate: float = 0.5, seed: int = 0) -> torch.Tensor:
    """Returns a bool edge mask (True = keep edge)."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    num_edges = data_obj.edge_index.shape[1]
    return torch.rand(num_edges, generator=generator) >= dropout_rate


def spanner_sparsification(data_obj, stretch: float = 3.0, seed: int = 0) -> torch.Tensor:
    """Returns a bool edge mask via NetworkX spanner."""
    G = to_networkx(data_obj, to_undirected=True)
    H = nx.spanner(G, stretch=stretch, seed=seed)
    
    h_edges: set[tuple[int, int]] = set()
    for u, v in H.edges():
        h_edges.add((min(u, v), max(u, v)))
        
    num_edges = data_obj.edge_index.shape[1]
    mask = torch.zeros(num_edges, dtype=torch.bool)
    
    for i, (u, v) in enumerate(data_obj.edge_index.t().tolist()):
        # FIX: Normalize the queried edge to match the canonical set representation
        if (min(u, v), max(u, v)) in h_edges:
            mask[i] = True
            
    return mask


def pagerank_sparsification(data_obj, keep_ratio: float = 0.8, alpha: float = 0.85) -> torch.Tensor:
    """Returns a bool node mask (True = keep node)."""
    G = to_networkx(data_obj, to_undirected=False)
    pr_scores = nx.pagerank(G, alpha=alpha)
    sorted_nodes = sorted(pr_scores, key=pr_scores.get, reverse=True)
    num_to_keep = max(1, int(len(sorted_nodes) * keep_ratio))
    node_mask = torch.zeros(data_obj.num_nodes, dtype=torch.bool)
    node_mask[sorted_nodes[:num_to_keep]] = True
    return node_mask


def and_gate_only_sparsification(data_obj) -> Data:
    x = data_obj.x
    edge_index = data_obj.edge_index
    edge_attr = data_obj.edge_attr

    is_pi = x[:, 1] == 1.0
    is_po = x[:, 3] == 1.0

    pi_set: set[int] = set(is_pi.nonzero(as_tuple=True)[0].tolist())
    po_set: set[int] = set(is_po.nonzero(as_tuple=True)[0].tolist())
    removed: set[int] = pi_set | po_set

    n = x.size(0)
    kept: list[int] = [i for i in range(n) if i not in removed]
    old_to_new: dict[int, int] = {old: new for new, old in enumerate(kept)}

    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()
    attr_list = edge_attr.tolist()

    new_src: list[int] = []
    new_dst: list[int] = []
    new_attr: list[list[float]] = []
    self_loop_seen: set[tuple] = set()

    for u, v, ea in zip(src_list, dst_list, attr_list):
        u_pi = u in pi_set
        v_po = v in po_set

        if u_pi and v not in removed:
            v_new = old_to_new[v]
            key = (v_new, tuple(ea))
            if key not in self_loop_seen:
                new_src.append(v_new); new_dst.append(v_new); new_attr.append(ea)
                self_loop_seen.add(key)
        elif not u_pi and u not in removed and v_po:
            u_new = old_to_new[u]
            key = (u_new, tuple(ea))
            if key not in self_loop_seen:
                new_src.append(u_new); new_dst.append(u_new); new_attr.append(ea)
                self_loop_seen.add(key)
        elif u not in removed and v not in removed:
            new_src.append(old_to_new[u]); new_dst.append(old_to_new[v]); new_attr.append(ea)

    kept_t = torch.tensor(kept, dtype=torch.long)
    new_x = x[kept_t]

    if new_src:
        new_edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)
        new_edge_attr = torch.tensor(new_attr, dtype=edge_attr.dtype)
    else:
        new_edge_index = torch.empty((2, 0), dtype=torch.long)
        new_edge_attr = torch.empty((0, edge_attr.size(1)), dtype=edge_attr.dtype)

    out = Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr)
    out.num_nodes = len(kept)
    out.num_edges = new_edge_index.size(1)

    for key in data_obj.keys():
        if key in ("x", "edge_index", "edge_attr", "num_nodes", "num_edges", "edge_weight"):
            continue
        val = data_obj[key]
        if isinstance(val, torch.Tensor) and val.dim() > 0 and val.size(0) == n:
            setattr(out, key, val[kept_t])
        else:
            setattr(out, key, val)

    return out


# =====================================================================
# MASK INDEX CACHE
# =====================================================================

_SPARSE_INDEX_CACHE: dict[tuple[str, str], dict] = {}


def get_sparse_entry(
    cache_dir: Path,
    algo_name: str,
    basename: str,
) -> dict | None:
    cache_key = (str(cache_dir), algo_name)
    if cache_key not in _SPARSE_INDEX_CACHE:
        _SPARSE_INDEX_CACHE[cache_key] = {}
        for index_path in cache_dir.glob(f"{_SPARSE_PREFIX}{algo_name}*.pt"):
            try:
                chunk = torch.load(
                    index_path,
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
                _SPARSE_INDEX_CACHE[cache_key].update(chunk)
            except TypeError:
                chunk = torch.load(
                    index_path,
                    map_location="cpu",
                    weights_only=True,
                )
                _SPARSE_INDEX_CACHE[cache_key].update(chunk)
            except Exception as exc:
                print(f"[sparsification] WARNING: could not load chunk {index_path}: {exc}")

    return _SPARSE_INDEX_CACHE[cache_key].get(basename)


def clear_sparse_index_cache() -> None:
    _SPARSE_INDEX_CACHE.clear()


def precomputed_sparsification(
    data_obj: Data,
    algo_name: str,
    cache_path: str | Path | None = None,
) -> Data:
    if algo_name == "and_gate_only":
        if hasattr(data_obj, "and_gate_only_graph") or "and_gate_only_graph" in data_obj.keys():
            return data_obj.and_gate_only_graph
        return and_gate_only_sparsification(data_obj)

    mask_key = f"{algo_name}_sparsification_mask"
    mask: torch.Tensor | None = None

    if hasattr(data_obj, mask_key) or mask_key in data_obj.keys():
        mask = getattr(data_obj, mask_key)

    if mask is None and cache_path is not None:
        p = Path(cache_path)
        entry = get_sparse_entry(p.parent, algo_name, p.name)
        if entry is not None:
            mask = entry["mask"]

    if mask is None:
        raise AttributeError(
            f"Precomputed sparsification mask for algorithm '{algo_name}' not found.\n"
            f"  Checked embedded attribute '{mask_key}' on data_obj: not present.\n"
            f"  Checked index file '_sparse_{algo_name}*.pt' in cache directory"
            + (f" '{Path(cache_path).parent}': not present." if cache_path else ": cache_path not provided.")
        )

    device = data_obj.x.device
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.bool, device=device)
    else:
        mask = mask.to(dtype=torch.bool, device=device)

    if algo_name == "pagerank":
        data_obj = data_obj.subgraph(mask)
    else:
        data_obj.edge_index = data_obj.edge_index[:, mask]
        if hasattr(data_obj, "edge_attr") and data_obj.edge_attr is not None:
            data_obj.edge_attr = data_obj.edge_attr[mask]
        if hasattr(data_obj, "edge_weight") and data_obj.edge_weight is not None:
            data_obj.edge_weight = data_obj.edge_weight[mask]

    return data_obj


# =====================================================================
# WORKER TASK FOR PARALLEL EXECUTION
# =====================================================================

def _worker_initializer() -> None:
    import torch as _torch
    _torch.set_num_threads(1)
    _register_pyg_safe_globals()


def _process_single_cache_file(
    task_tuple: tuple[Path, str, str],
    algo_names: list[str],
    dropout_rate: float,
    stretch: float,
    keep_ratio: float,
    alpha: float,
    seed: int,
) -> tuple[str, str, dict] | None:
    """
    task_tuple contains: (graph_path, out_dir_str, basename)
    """
    graph_path, out_dir_str, basename = task_tuple
    
    if not graph_path.is_file():
        return None

    with open(graph_path, "rb") as fh:
        data_obj = torch.load(fh, map_location="cpu", weights_only=True)

    result: dict[str, dict] = {}
    for algo_name in algo_names:
        if algo_name == "random_edge_dropout":
            mask = random_edge_dropout(data_obj, dropout_rate=dropout_rate, seed=seed)
        elif algo_name == "spanner":
            mask = spanner_sparsification(data_obj, stretch=stretch, seed=seed)
        elif algo_name == "pagerank":
            mask = pagerank_sparsification(data_obj, keep_ratio=keep_ratio, alpha=alpha)
        else:
            raise ValueError(f"Unknown algorithm for precompute: '{algo_name}'")

        result[algo_name] = {
            "mask": mask.cpu().numpy().astype(np.bool_),
        }

    return out_dir_str, basename, result


# =====================================================================
# CORE UPDATE PIPELINE (MANIFEST BASED)
# =====================================================================

def update_from_manifests(
    manifest_dirs: list[str | Path],
    algo_names: list[str],
    dropout_rate: float = 0.5,
    stretch: float = 3.0,
    keep_ratio: float = 0.8,
    alpha: float = 0.85,
    seed: int = 42,
    replace_path: tuple[str, str] | None = None,
) -> None:
    """
    Computes sparsification masks based on JSON manifests rather than scanning dirs.
    """
    _register_pyg_safe_globals()

    precompute_algos = [a for a in algo_names if a != "and_gate_only"]
    if not precompute_algos:
        print("[Mask Precomputation] No precomputable algorithms requested. Nothing to do.")
        return

    print(f"[Mask Precomputation] Algorithms: {precompute_algos}")

    # 1. PARSE MANIFESTS & BUILD TASKS
    tasks: list[tuple[Path, str, str]] = []
    out_dir_cache_keys: dict[str, set[str]] = defaultdict(set)
    seen_cache_paths: set[str] = set()

    for m_dir in manifest_dirs:
        m_dir_path = Path(m_dir)
        if not m_dir_path.is_dir():
            print(f"[WARNING] Manifest directory not found: {m_dir_path}")
            continue
            
        # RESTRICT: Read only *_manifest.json files
        for manifest_file in m_dir_path.glob("*_manifest.json"):
            try:
                with open(manifest_file, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[WARNING] Skipping bad manifest {manifest_file}: {exc}")
                continue
                
            for entry in data.get("entries", []):
                g_path = Path(entry["graph_path"])
                c_path = Path(entry["cache_path"])
                
                # DEDUPLICATE: Track by absolute cache path
                abs_cache_path = str(c_path.absolute())
                if abs_cache_path in seen_cache_paths:
                    continue
                seen_cache_paths.add(abs_cache_path)
                
                out_dir_str = str(c_path.parent)
                if replace_path is not None:
                    out_dir_str = out_dir_str.replace(replace_path[0], replace_path[1])
                tasks.append((g_path, out_dir_str, c_path.name))
                out_dir_cache_keys[out_dir_str].add(c_path.name)

    if not tasks:
        print("[Mask Precomputation] No entries found in manifests.")
        return

    # 2. LOAD EXISTING INDICES TO SKIP ALREADY DONE FILES
    done_by_dir: dict[str, set[str]] = defaultdict(set)
    for out_d_str in out_dir_cache_keys.keys():
        all_done = []
        for a in precompute_algos:
            done = set()
            for index_path in Path(out_d_str).glob(f"{_SPARSE_PREFIX}{a}*.pt"):
                try:
                    chunk = torch.load(index_path, map_location="cpu", weights_only=True)
                    done.update(chunk.keys())
                except Exception:
                    pass
            all_done.append(done)
        if all_done:
            done_by_dir[out_d_str] = all_done[0].intersection(*all_done[1:]) if len(all_done) > 1 else all_done[0]

    filtered_tasks = []
    for t in tasks:
        if t[2] not in done_by_dir[t[1]]:
            filtered_tasks.append(t)

    print(f"Loaded {len(tasks)} unique tasks from manifests. Skipping {len(tasks) - len(filtered_tasks)} already done.")
    
    if not filtered_tasks:
        print("[Mask Precomputation] All tasks are already computed.")
        return

    # 3. PARALLEL COMPUTATION
    accumulated: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))

    try:
        all_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        all_cpus = os.cpu_count() or 1
    num_workers = max(1, all_cpus - 1)

    worker_fn = functools.partial(
        _process_single_cache_file,
        algo_names=precompute_algos,
        dropout_rate=dropout_rate,
        stretch=stretch,
        keep_ratio=keep_ratio,
        alpha=alpha,
        seed=seed,
    )

    success_count = 0
    error_count = 0

    def _flush_indices() -> None:
        chunk_id = int(time.time())
        for out_d_str, algo_map in list(accumulated.items()):
            for algo_name, index in list(algo_map.items()):
                if index:
                    Path(out_d_str).mkdir(parents=True, exist_ok=True)
                    index_path = Path(out_d_str) / f"{_SPARSE_PREFIX}{algo_name}_{chunk_id}_{uuid.uuid4().hex[:4]}.pt"
                    temp_file = index_path.with_suffix(".tmp")
                    torch.save(index, temp_file)
                    os.replace(temp_file, index_path)
                    index.clear()

    import torch.multiprocessing as mp
    mp_ctx = mp.get_context("spawn")

    with mp_ctx.Pool(processes=num_workers, initializer=_worker_initializer, maxtasksperchild=50) as pool:
        results_iter = pool.imap_unordered(worker_fn, filtered_tasks, chunksize=10)

        from tqdm import tqdm
        with tqdm(total=len(filtered_tasks), desc="Computing sparsification masks", unit=" files") as pbar:
            for result in results_iter:
                if result is not None:
                    out_d_str, basename, algo_results = result
                    for algo_name, entry in algo_results.items():
                        accumulated[out_d_str][algo_name][basename] = {
                            "mask": torch.from_numpy(entry["mask"]).clone(),
                        }
                    success_count += 1
                else:
                    error_count += 1

                pbar.update(1)

                if success_count > 0 and success_count % CHECKPOINT_EVERY == 0:
                    print(f"\n[Checkpoint] {success_count} done — flushing indices...")
                    _flush_indices()

    _flush_indices()

    print(f"\n[Mask Precomputation] Complete! Processed {success_count} files, {error_count} errors.")


# =====================================================================
# CORE UPDATE PIPELINE (FALLBACK TO DIRECTORIES)
# =====================================================================

def update_existing_cache_with_masks(directories: list[str | Path], algo_names: list[str], **kwargs) -> None:
    """Original directory-scanning implementation (kept for backward compatibility)."""
    _register_pyg_safe_globals()
    precompute_algos = [a for a in algo_names if a != "and_gate_only"]
    if not precompute_algos: return
    
    top_dirs = [Path(d).absolute() for d in directories]
    out_dirs_list = kwargs.pop("out_directories", None) or top_dirs
    dir_map = {str(d): str(o) for d, o in zip(top_dirs, out_dirs_list)}

    done_by_dir: dict[str, set[str]] = {}
    for top_dir in top_dirs:
        d_str = str(top_dir)
        out_d_str = dir_map[d_str]
        all_done = []
        for a in precompute_algos:
            done = set()
            for index_path in Path(out_d_str).glob(f"{_SPARSE_PREFIX}{a}*.pt"):
                try: done.update(torch.load(index_path, map_location="cpu", weights_only=True).keys())
                except Exception: pass
            all_done.append(done)
        done_by_dir[d_str] = all_done[0].intersection(*all_done[1:]) if len(all_done) > 1 else (all_done[0] if all_done else set())

    def _path_stream():
        for top_dir in top_dirs:
            d_str = str(top_dir)
            done_set = done_by_dir[d_str]
            try:
                with os.scandir(str(top_dir)) as scanner:
                    for entry in scanner:
                        if (entry.is_file(follow_symlinks=False)
                                and entry.name.endswith(".pt")
                                and not entry.name.startswith(_SPARSE_PREFIX)
                                and not entry.name.startswith(_MASKS_PREFIX) 
                                and entry.name not in done_set):
                            yield (Path(entry.path), dir_map[d_str], entry.name)
            except PermissionError as exc:
                print(f"[WARNING] Cannot scan {top_dir}: {exc}")

    tasks = list(_path_stream())
    if not tasks: return

    accumulated: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    
    worker_fn = functools.partial(_process_single_cache_file, algo_names=precompute_algos, **kwargs)
    
    import torch.multiprocessing as mp
    mp_ctx = mp.get_context("spawn")
    
    def _flush_indices() -> None:
        chunk_id = int(time.time())
        for out_d_str, algo_map in list(accumulated.items()):
            for algo_name, index in list(algo_map.items()):
                if index:
                    index_path = Path(out_d_str) / f"{_SPARSE_PREFIX}{algo_name}_{chunk_id}_{uuid.uuid4().hex[:4]}.pt"
                    temp_file = index_path.with_suffix(".tmp")
                    torch.save(index, temp_file)
                    os.replace(temp_file, index_path)
                    index.clear()
                    
    success_count = 0
    with mp_ctx.Pool(processes=max(1, (os.cpu_count() or 1) - 1), initializer=_worker_initializer, maxtasksperchild=50) as pool:
        from tqdm import tqdm
        for result in tqdm(pool.imap_unordered(worker_fn, tasks, chunksize=10), total=len(tasks)):
            if result:
                out_d_str, basename, algo_results = result
                for algo_name, entry in algo_results.items():
                    accumulated[out_d_str][algo_name][basename] = {"mask": torch.from_numpy(entry["mask"]).clone()}
                success_count += 1
                if success_count % CHECKPOINT_EVERY == 0: _flush_indices()

    _flush_indices()


if __name__ == "__main__":
    import argparse
    try:
        import config
    except ImportError:
        config = object()

    _seed         = getattr(config, "SPARSIFICATION_SEED", 42)
    _dropout_rate = getattr(config, "SPARSIFICATION_RANDOM_DROPOUT_RATE", 0.5)
    _stretch      = getattr(config, "SPARSIFICATION_SPANNER_STRETCH", 3.0)
    _keep_ratio   = getattr(config, "SPARSIFICATION_PAGERANK_KEEP_RATIO", 0.8)
    _alpha        = getattr(config, "SPARSIFICATION_PAGERANK_ALPHA", 0.85)

    parser = argparse.ArgumentParser(
        description="Precompute sparsification masks for cached graphs in parallel."
    )
    parser.add_argument(
        "algorithm",
        type=str,
        choices=["random_edge_dropout", "spanner", "pagerank", "and_gate_only", "all"],
        help="Sparsification algorithm to run, or 'all'.",
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dirs",
        nargs="+",
        help="One or more flat cache directories (or individual .pt files) to process.",
    )
    group.add_argument(
        "--manifest-dirs",
        nargs="+",
        help="Directories containing metadata .json manifests outlining 'graph_path' and 'cache_path'.",
    )
    
    parser.add_argument(
        "--out-dirs",
        nargs="+",
        required=False,
        help="Corresponding directories to save the index files (only valid with --dirs).",
    )
    parser.add_argument(
        "--replace-path",
        nargs=2,
        help="Redirect output paths by replacing a prefix string with a new one.",
    )
    args = parser.parse_args()

    algo_names = ["random_edge_dropout", "spanner", "pagerank", "and_gate_only"] if args.algorithm == "all" else [args.algorithm]

    print(
        f"[sparsification.py] Running for algorithm(s)={sorted(algo_names)}\n"
        f"  dropout_rate={_dropout_rate}\n  stretch={_stretch}\n  keep_ratio={_keep_ratio}\n  alpha={_alpha}\n  seed={_seed}\n"
    )

    if args.manifest_dirs:
        update_from_manifests(
            manifest_dirs=args.manifest_dirs,
            algo_names=algo_names,
            dropout_rate=_dropout_rate,
            stretch=_stretch,
            keep_ratio=_keep_ratio,
            alpha=_alpha,
            seed=_seed,
            replace_path=args.replace_path,
        )
    else:
        update_existing_cache_with_masks(
            directories=args.dirs,
            algo_names=algo_names,
            dropout_rate=_dropout_rate,
            stretch=_stretch,
            keep_ratio=_keep_ratio,
            alpha=_alpha,
            seed=_seed,
            out_directories=args.out_dirs,
        )
