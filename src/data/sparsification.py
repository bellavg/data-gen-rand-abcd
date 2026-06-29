from __future__ import annotations

import os
import time
import uuid
import functools
import torch
import numpy as np
import networkx as nx
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

# Prefix for index files — excluded from scanning so they are never treated
# as graph cache files.
_SPARSE_PREFIX = "_sparse_"
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
    """Returns a bool edge mask (True = keep edge).

    Uses a local Generator so the global RNG state is not disturbed.
    """
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
        if (u, v) in h_edges:
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
    """AIG-specific graph transformation: remove PI and PO nodes, replace with self-loops.

    Operates on a pre-loaded PyG ``Data`` object (from a cached ``.pt`` file).
    Uses the one-hot node-type feature stored in ``x``::

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

    .. note::
        This transform is applied **on-the-fly** in ``dataset.get()`` — there
        is no precomputation step.  The transform is fast (~1–5 ms per graph)
        and deterministic, so the overhead is hidden by DataLoader parallelism.
    """
    x = data_obj.x                          # shape [N, 4]
    edge_index = data_obj.edge_index        # shape [2, E]
    edge_attr = data_obj.edge_attr          # shape [E, 2]

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

    # Dynamically copy all other attributes (e.g. 'y', 'pos_enc', metadata)
    for key in data_obj.keys():
        if key in ("x", "edge_index", "edge_attr", "num_nodes", "num_edges", "edge_weight"):
            continue
        val = data_obj[key]
        if isinstance(val, torch.Tensor) and val.dim() > 0 and val.size(0) == n:
            # Node-level attribute (e.g. pos_enc, level) -> slice to kept nodes
            setattr(out, key, val[kept_t])
        else:
            # Graph-level attribute (e.g. y, num_pis, etc.) -> copy exactly
            setattr(out, key, val)

    return out


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
_SPARSE_INDEX_CACHE: dict[tuple[str, str], dict] = {}


def get_sparse_entry(
    cache_dir: Path,
    algo_name: str,
    basename: str,
) -> dict | None:
    """Return the index entry for *basename* in *cache_dir* for *algo_name*.

    The index is loaded from ``{cache_dir}/_sparse_{algo_name}*.pt`` chunk
    files on first access and then kept in the module-level
    ``_SPARSE_INDEX_CACHE``.  Entries have the form ``{"mask": torch.Tensor}``.

    Returns ``None`` if no index files exist or the basename is not present.
    """
    cache_key = (str(cache_dir), algo_name)
    if cache_key not in _SPARSE_INDEX_CACHE:
        _SPARSE_INDEX_CACHE[cache_key] = {}
        for index_path in cache_dir.glob(f"{_SPARSE_PREFIX}{algo_name}*.pt"):
            try:
                # mmap=True: tensor data is lazily paged from disk.
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
    """Drop all cached sparse indices (useful between trials in Optuna studies)."""
    _SPARSE_INDEX_CACHE.clear()


def precomputed_sparsification(
    data_obj: Data,
    algo_name: str,
    cache_path: str | Path | None = None,
) -> Data:
    """Apply a precomputed sparsification mask to a graph.

    Lookup order:

    1. **Embedded attributes** (backward-compatible): checks for
       ``{algo_name}_sparsification_mask`` directly on *data_obj*.

    2. **Directory index file**: looks up the mask in the per-directory
       ``_sparse_{algo_name}*.pt`` index chunks written by the current
       pipeline.  Requires *cache_path* to be provided so the directory
       can be derived.

    For ``and_gate_only``, the transform is applied on-the-fly (no
    precomputed mask needed).

    Args:
        data_obj:    A PyG ``Data`` object.
        algo_name:   The sparsification algorithm name (e.g. ``"random_edge_dropout"``).
        cache_path:  Absolute path to the ``.pt`` file that *data_obj* was
                     loaded from.  Required for the index-file lookup path.

    Returns:
        A ``Data`` object with the sparsification applied.

    Raises:
        AttributeError: If neither embedded attributes nor the index file
                        contain a mask for this graph + algorithm.
    """
    # --- and_gate_only: on-the-fly transform ---
    if algo_name == "and_gate_only":
        if hasattr(data_obj, "and_gate_only_graph") or "and_gate_only_graph" in data_obj.keys():
            return data_obj.and_gate_only_graph
        return and_gate_only_sparsification(data_obj)

    # --- Precomputed mask lookup ---
    mask_key = f"{algo_name}_sparsification_mask"
    mask: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # 1. Try embedded attributes (backward compat with old pipeline)
    # ------------------------------------------------------------------
    if hasattr(data_obj, mask_key) or mask_key in data_obj.keys():
        mask = getattr(data_obj, mask_key)

    # ------------------------------------------------------------------
    # 2. Try the per-directory index file
    # ------------------------------------------------------------------
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
            + f"\nPrecompute masks by running:\n"
            f"  python -m data.sparsification {algo_name} --dirs <cache_dir>"
        )

    # Cast to bool tensor on the correct device
    device = data_obj.x.device
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.bool, device=device)
    else:
        mask = mask.to(dtype=torch.bool, device=device)

    # --- Apply the mask (node mask for pagerank, edge mask for others) ---
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
    algo_names: list[str],
    dropout_rate: float,
    stretch: float,
    keep_ratio: float,
    alpha: float,
    seed: int,
) -> tuple[str, str, dict] | None:
    """Compute sparsification masks for one cache file and return them as tensors.

    **No disk writes are performed here.**  The caller (main process) is
    responsible for accumulating results and flushing the index to disk.

    Returns:
        ``(cache_dir_str, basename, mask_entry_dict)`` where *mask_entry_dict* maps
        ``algo_name`` → ``{"mask": numpy_array}``.
        Returns ``None`` if the file does not exist.
    """
    if not cache_path.is_file():
        return None

    with open(cache_path, "rb") as fh:
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
            raise ValueError(f"Unknown algorithm for precompute: '{algo_name}'. "
                             f"'and_gate_only' is applied on-the-fly — no precomputation needed.")

        result[algo_name] = {
            "mask": mask.cpu().numpy().astype(np.bool_),
        }

    return str(cache_path.parent), cache_path.name, result


# =====================================================================
# CORE UPDATE PIPELINE
# =====================================================================

def update_existing_cache_with_masks(
    directories: list[str | Path],
    algo_names: list[str],
    dropout_rate: float = 0.5,
    stretch: float = 3.0,
    keep_ratio: float = 0.8,
    alpha: float = 0.85,
    seed: int = 42,
    out_directories: list[str | Path] | None = None,
) -> None:
    """Compute sparsification masks for all cached graph files and save them in
    per-directory index files.

    Instead of embedding masks inside each individual graph ``.pt`` file
    (which requires loading and re-saving the full graph for every file),
    this pipeline writes chunked index files per *(cache directory,
    algorithm)* pair::

        {cache_dir}/_sparse_{algo_name}_{timestamp}_{uuid}.pt

    Each index file maps graph basenames to their precomputed mask tensors.

    **Resume support**: On startup the existing indices are loaded and files
    that already have all requested masks are skipped automatically.

    **Streaming**: Files are submitted to the worker pool immediately as
    ``os.scandir`` discovers them — workers start within seconds rather than
    waiting for a full directory scan to complete.

    **Checkpointing**: The indices are flushed to disk atomically every
    ``CHECKPOINT_EVERY`` completed files so that a SLURM time-limit kill
    preserves as much work as possible.

    Args:
        directories:     Flat cache directories to scan for graph ``.pt`` files.
        algo_names:      Sparsification algorithm names to compute.
        dropout_rate:    Edge keep threshold for ``random_edge_dropout`` (default 0.5).
        stretch:         Spanner stretch factor (default 3.0).
        keep_ratio:      Fraction of nodes to keep for ``pagerank`` (default 0.8).
        alpha:           PageRank damping factor (default 0.85).
        seed:            RNG seed for stochastic algorithms.
        out_directories: Corresponding directories to save the index files.
                         If None, index files are written to the source directories.
    """
    _register_pyg_safe_globals()

    # and_gate_only is on-the-fly — nothing to precompute.
    precompute_algos = [a for a in algo_names if a != "and_gate_only"]
    if not precompute_algos:
        print("[Mask Precomputation] No precomputable algorithms requested "
              "(and_gate_only is applied on-the-fly). Nothing to do.")
        return

    print(f"[Mask Precomputation] Algorithms: {precompute_algos}")

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

        # A basename is fully done only when it appears in ALL algo indices.
        if precompute_algos:
            all_done = []
            for a in precompute_algos:
                done = set()
                # Load keys from all existing chunks
                for index_path in Path(out_d_str).glob(f"{_SPARSE_PREFIX}{a}*.pt"):
                    try:
                        chunk = torch.load(index_path, map_location="cpu", weights_only=True)
                        done.update(chunk.keys())
                    except Exception as exc:
                        print(f"[WARNING] Could not load chunk {index_path}: {exc}")
                all_done.append(done)
            done_by_dir[d_str] = all_done[0].intersection(*all_done[1:]) if len(all_done) > 1 else all_done[0]
        else:
            done_by_dir[d_str] = set()

        print(f"  -> {out_d_str}: {len(done_by_dir[d_str])} entries already in index")

        # Initialize empty dict for NEW files
        accumulated[d_str] = {a: {} for a in precompute_algos}

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
                                and not entry.name.startswith(_SPARSE_PREFIX)
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
        algo_names=precompute_algos,
        dropout_rate=dropout_rate,
        stretch=stretch,
        keep_ratio=keep_ratio,
        alpha=alpha,
        seed=seed,
    )

    success_count = 0
    error_count   = 0

    def _flush_indices() -> None:
        """Atomically write all accumulated index dicts to disk as chunks and clear memory."""
        chunk_id = int(time.time())
        for d_str, algo_map in accumulated.items():
            out_d_str = dir_map[d_str]
            for algo_name, index in algo_map.items():
                if index:
                    index_path = Path(out_d_str) / f"{_SPARSE_PREFIX}{algo_name}_{chunk_id}_{uuid.uuid4().hex[:4]}.pt"
                    temp_file = index_path.with_suffix(".tmp")
                    torch.save(index, temp_file)
                    os.replace(temp_file, index_path)
                    index.clear()  # Clear memory!

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
        with tqdm(desc="Computing sparsification masks", unit=" files") as pbar:
            for result in results_iter:
                if result is not None:
                    d_str, basename, algo_results = result
                    for algo_name, entry in algo_results.items():
                        accumulated[d_str][algo_name][basename] = {
                            "mask": torch.from_numpy(entry["mask"]).clone(),
                        }
                    success_count += 1

                pbar.update(1)

                # Periodic checkpoint — preserve progress across SLURM kills.
                if success_count > 0 and success_count % CHECKPOINT_EVERY == 0:
                    print(f"\n[Checkpoint] {success_count} done — flushing indices...")
                    _flush_indices()

    # ------------------------------------------------------------------
    # 4. FINAL SAVE — flush any results accumulated since the last checkpoint
    # ------------------------------------------------------------------
    _flush_indices()

    # Print a quick summary of what we just did
    for d_str, algo_map in accumulated.items():
        print(f"  {dir_map[d_str]} chunks updated.")

    print(
        f"\n[Mask Precomputation] Complete! "
        f"Processed {success_count} files, {error_count} errors."
    )
    print("Graph .pt files were NOT modified — masks live in index files only.")


if __name__ == "__main__":
    import argparse
    import config

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
        help=(
            "Sparsification algorithm to run, or 'all' to run all precomputable algorithms.\n"
            "  and_gate_only is applied on-the-fly in dataset.get() — no precomputation needed."
        ),
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
        out_directories=args.out_dirs,
    )
