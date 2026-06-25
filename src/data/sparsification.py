from __future__ import annotations

import os
import uuid
import functools
import itertools
import concurrent.futures
from pathlib import Path

import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from tqdm import tqdm

# Prefix for index files so the scanner never treats them as graph files.
_SPARSE_PREFIX = "_sparse_"
CHECKPOINT_EVERY = 50_000   # how many completed files between atomic index saves


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

    for attr_name in ("level", "pi_paths", "local_sp_sum"):
        val = getattr(data_obj, attr_name, None)
        if val is not None and isinstance(val, torch.Tensor) and val.size(0) == n:
            setattr(out, attr_name, val[kept_t])

    for meta in ("num_pis", "num_pos"):
        val = getattr(data_obj, meta, None)
        if val is not None:
            setattr(out, meta, val)

    return out


# =====================================================================
# INDEX FILE HELPERS (write side — used by precompute pipeline)
# =====================================================================

def _get_index_path(cache_dir: Path, algo_name: str) -> Path:
    """``{cache_dir}/_sparse_{algo_name}.pt``"""
    return cache_dir / f"{_SPARSE_PREFIX}{algo_name}.pt"


def _load_sparse_index(cache_dir: Path, algo_name: str) -> dict:
    """Load existing sparse index, returning empty dict on miss/error.

    Index format::

        {
            "abc123.pt": {"mask": torch.Tensor},  # bool, shape [E] or [N]
            ...
        }
    """
    index_path = _get_index_path(cache_dir, algo_name)
    if not index_path.is_file():
        return {}
    try:
        return torch.load(index_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"[WARNING] Could not load sparse index {index_path}: {exc}. Starting fresh.")
        return {}


def _save_sparse_index(cache_dir: Path, algo_name: str, index: dict) -> None:
    """Atomically persist the sparse index for one directory + algorithm."""
    index_path = _get_index_path(cache_dir, algo_name)
    tmp = index_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
    torch.save(index, tmp)
    os.replace(tmp, index_path)


# =====================================================================
# INDEX FILE HELPERS (read side — used by dataset.py at training time)
# =====================================================================

# Per-worker lazy cache: loaded once on first access, reused for all subsequent get() calls.
_SPARSE_INDEX_CACHE: dict[tuple[str, str], dict] = {}


def get_sparse_entry(cache_dir: Path, algo_name: str, basename: str) -> dict | None:
    """Return the index entry ``{"mask": tensor}`` for one graph file.

    The index is loaded from ``{cache_dir}/_sparse_{algo_name}.pt`` on first
    access and then kept in the module-level ``_SPARSE_INDEX_CACHE``.
    With ``mmap=True``, tensor data is lazily paged from disk and the OS page
    cache is shared across DataLoader workers.

    Returns ``None`` if the index file does not exist or the basename is absent.
    """
    cache_key = (str(cache_dir), algo_name)
    if cache_key not in _SPARSE_INDEX_CACHE:
        index_path = _get_index_path(cache_dir, algo_name)
        if index_path.is_file():
            try:
                try:
                    _SPARSE_INDEX_CACHE[cache_key] = torch.load(
                        index_path, map_location="cpu", weights_only=True, mmap=True
                    )
                except TypeError:
                    # mmap kwarg not available in older PyTorch versions
                    _SPARSE_INDEX_CACHE[cache_key] = torch.load(
                        index_path, map_location="cpu", weights_only=True
                    )
            except Exception as exc:
                print(f"[sparsification] WARNING: could not load index {index_path}: {exc}")
                _SPARSE_INDEX_CACHE[cache_key] = {}
        else:
            _SPARSE_INDEX_CACHE[cache_key] = {}
    return _SPARSE_INDEX_CACHE[cache_key].get(basename)


def clear_sparse_index_cache() -> None:
    """Drop all cached sparse indices (e.g. between Optuna trials)."""
    _SPARSE_INDEX_CACHE.clear()


# =====================================================================
# WORKER
# =====================================================================

def _worker_initializer() -> None:
    """Called once per worker process at pool startup.

    Pins PyTorch to 1 intra-op thread (prevents thread-thrashing when many
    workers run concurrently) and registers PyG safe globals for
    ``weights_only=True`` loads.
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
) -> tuple[str, dict] | None:
    """Compute sparsification masks for one graph file.

    **No disk writes are performed here.**  The caller (main process) accumulates
    results and flushes index files to disk.

    Returns:
        ``(basename, {algo_name: {"mask": tensor}, ...})`` or ``None`` if file missing.
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
        result[algo_name] = {"mask": mask.to(dtype=torch.bool, device="cpu")}

    return cache_path.name, result


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
) -> None:
    """Compute sparsification masks for all cached graph files and save them in
    per-directory index files.

    Masks are stored in::

        {cache_dir}/_sparse_{algo_name}.pt

    rather than embedded inside each individual graph ``.pt`` file, which would
    require loading and re-saving every graph (slow + risky over NFS).

    **and_gate_only is not precomputed here** — it is applied on-the-fly in
    ``dataset.get()`` (fast deterministic transform, hidden by DataLoader parallelism).

    **Resume support**: existing indices are loaded upfront; already-computed
    files are skipped without touching the graph files.

    **Streaming**: files are submitted to the worker pool as ``os.scandir``
    discovers them — workers start within seconds rather than after a full scan.

    **Checkpointing**: indices are flushed atomically every ``CHECKPOINT_EVERY``
    completed files to preserve progress across SLURM time-limit kills.

    Args:
        directories:  Flat cache directories containing graph ``.pt`` files.
        algo_names:   Sparsification algorithm names to compute
                      (``"random_edge_dropout"``, ``"spanner"``, ``"pagerank"``).
                      ``"and_gate_only"`` is silently skipped.
        dropout_rate: Edge keep threshold for ``random_edge_dropout`` (default 0.5).
        stretch:      Spanner stretch factor (default 3.0).
        keep_ratio:   Fraction of nodes to keep for ``pagerank`` (default 0.8).
        alpha:        PageRank damping factor (default 0.85).
        seed:         RNG seed for stochastic algorithms.
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
    # ------------------------------------------------------------------
    top_dirs: list[Path] = []
    lone_files: list[Path] = []
    for d in directories:
        p = Path(d).absolute()
        (top_dirs if p.is_dir() else lone_files).append(p)

    accumulated: dict[str, dict[str, dict]] = {}
    done_by_dir: dict[str, set[str]] = {}

    for top_dir in top_dirs:
        d_str = str(top_dir)
        accumulated[d_str] = {a: _load_sparse_index(top_dir, a) for a in precompute_algos}
        if precompute_algos:
            all_done = [set(accumulated[d_str][a].keys()) for a in precompute_algos]
            done_by_dir[d_str] = (
                all_done[0].intersection(*all_done[1:]) if len(all_done) > 1 else all_done[0]
            )
        else:
            done_by_dir[d_str] = set()
        print(f"  -> {top_dir}: {len(done_by_dir[d_str])} entries already in index")

    # ------------------------------------------------------------------
    # 2. STREAMING PATH GENERATOR
    # ------------------------------------------------------------------
    def _path_stream():
        for p in lone_files:
            if p.suffix == ".pt" and not p.name.startswith(_SPARSE_PREFIX):
                d_str = str(p.parent)
                if d_str not in accumulated:
                    accumulated[d_str] = {a: {} for a in precompute_algos}
                    done_by_dir[d_str] = set()
                if p.name not in done_by_dir[d_str]:
                    yield d_str, p

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
    num_workers = max(1, int(all_cpus * 0.75))
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

    PENDING_LIMIT = num_workers * 8
    success_count = 0
    error_count   = 0

    def _flush_indices() -> None:
        for d_str, algo_map in accumulated.items():
            for algo_name, index in algo_map.items():
                if index:
                    _save_sparse_index(Path(d_str), algo_name, index)

    path_stream = _path_stream()

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_initializer,
    ) as executor:
        futures: dict[concurrent.futures.Future, tuple[str, Path]] = {}

        for d_str, path in itertools.islice(path_stream, PENDING_LIMIT):
            f = executor.submit(worker_fn, path)
            futures[f] = (d_str, path)

        print(
            f"[Mask Precomputation] {len(futures)} tasks submitted, "
            f"workers running. Streaming remaining files as slots open..."
        )

        with tqdm(desc="Computing sparsification masks", unit=" files") as pbar:
            while futures:
                done_futures, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done_futures:
                    d_str, path = futures.pop(future)
                    try:
                        result = future.result()
                        if result is not None:
                            basename, algo_results = result
                            for algo_name, entry in algo_results.items():
                                accumulated[d_str][algo_name][basename] = entry
                            success_count += 1
                    except Exception as exc:
                        error_count += 1
                        print(f"\n[ERROR] {path.name}: {exc}")

                    pbar.update(1)

                    try:
                        next_d_str, next_path = next(path_stream)
                        new_f = executor.submit(worker_fn, next_path)
                        futures[new_f] = (next_d_str, next_path)
                    except StopIteration:
                        pass

                if success_count > 0 and success_count % CHECKPOINT_EVERY == 0:
                    print(f"\n[Checkpoint] {success_count} done — flushing indices...")
                    _flush_indices()

    # ------------------------------------------------------------------
    # 4. FINAL SAVE
    # ------------------------------------------------------------------
    print(f"\n[Mask Precomputation] Saving final indices to disk...")
    _flush_indices()

    for d_str, algo_map in accumulated.items():
        for algo_name, index in algo_map.items():
            idx_path = _get_index_path(Path(d_str), algo_name)
            print(f"  {idx_path}  ({len(index)} entries)")

    print(
        f"\n[Mask Precomputation] Complete! "
        f"Processed {success_count} files, {error_count} errors."
    )
    print("Graph .pt files were NOT modified — masks live in index files only.")


if __name__ == "__main__":
    import config
    import argparse

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
