import os
import time
import uuid
from pathlib import Path

import torch
from tqdm import tqdm

_LEVELS_PREFIX = "_levels.pt"


def _register_pyg_safe_globals() -> None:
    import torch.serialization
    import torch_geometric.data.data as _pyg_data_mod
    import torch_geometric.data.storage as _pyg_storage
    from torch_geometric.data import Data

    try:
        from data.partition_utils import PartitionedData

        safe_globals = [Data, _pyg_storage.GlobalStorage, PartitionedData]
    except ImportError:
        safe_globals = [Data, _pyg_storage.GlobalStorage]

    for _name in ("DataTensorAttr", "DataEdgeAttr"):
        _cls = getattr(_pyg_data_mod, _name, None)
        if _cls is not None:
            safe_globals.append(_cls)
    torch.serialization.add_safe_globals(safe_globals)


def compute_node_levels(data_obj) -> torch.Tensor:
    num_nodes = data_obj.num_nodes
    x = data_obj.x
    edge_index = data_obj.edge_index

    levels = torch.zeros(num_nodes, dtype=torch.long)

    # Pre-group edges by destination
    dst = edge_index[1].tolist()
    src = edge_index[0].tolist()
    fanins = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        fanins[d].append(s)

    # Note: nodes are topologically sorted. Constants and PIs are first.
    for i in range(num_nodes):
        if x[i, 0] == 1.0 or x[i, 1] == 1.0:  # Const or PI
            levels[i] = 0
        elif x[i, 2] == 1.0 or x[i, 3] == 1.0:  # Gate or PO
            if fanins[i]:
                levels[i] = max(levels[f] for f in fanins[i]) + 1
            else:
                levels[i] = 0

    return levels


def _worker_initializer() -> None:
    import torch as _torch

    _torch.set_num_threads(1)
    _register_pyg_safe_globals()


def _process_single_cache_file(
    cache_path: Path,
) -> tuple[str, str, torch.Tensor] | None:
    if not cache_path.is_file():
        return None

    with open(cache_path, "rb") as fh:
        data_obj = torch.load(fh, map_location="cpu", weights_only=True)

    levels = compute_node_levels(data_obj)
    return str(cache_path.parent), cache_path.name, levels


def precompute_levels(
    directories: list[str | Path], out_directories: list[str | Path] | None = None
):
    _register_pyg_safe_globals()

    if out_directories is None:
        out_directories = directories

    top_dirs = [Path(d).absolute() for d in directories if Path(d).is_dir()]
    out_dirs_list = [
        Path(d).absolute()
        for d, orig_d in zip(out_directories, directories)
        if Path(orig_d).is_dir()
    ]

    dir_map = {str(d): str(o) for d, o in zip(top_dirs, out_dirs_list)}

    accumulated = {}
    done_by_dir = {}

    for top_dir in top_dirs:
        d_str = str(top_dir)
        out_d_str = dir_map[d_str]
        index_path = Path(out_d_str) / _LEVELS_PREFIX
        if index_path.is_file():
            try:
                accumulated[d_str] = torch.load(
                    index_path, map_location="cpu", weights_only=True
                )
                done_by_dir[d_str] = set(accumulated[d_str].keys())
            except Exception:
                accumulated[d_str] = {}
                done_by_dir[d_str] = set()
        else:
            accumulated[d_str] = {}
            done_by_dir[d_str] = set()

        print(f"  -> {out_d_str}: {len(done_by_dir[d_str])} entries already in index")

    def _path_stream():
        for top_dir in top_dirs:
            d_str = str(top_dir)
            done_set = done_by_dir[d_str]
            try:
                with os.scandir(str(top_dir)) as scanner:
                    for entry in scanner:
                        if (
                            entry.is_file(follow_symlinks=False)
                            and entry.name.endswith(".pt")
                            and not entry.name.startswith("_")
                            and entry.name not in done_set
                        ):
                            yield d_str, Path(entry.path)
            except PermissionError as exc:
                print(f"[WARNING] Cannot scan {top_dir}: {exc}")

    try:
        all_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        all_cpus = os.cpu_count() or 1
    num_workers = max(1, all_cpus - 1)

    print(f"[Levels Precomputation] Using {num_workers} parallel workers...")

    CHECKPOINT_EVERY = 50_000
    success_count = 0
    error_count = 0

    def _flush_indices() -> None:
        for d_str, index in accumulated.items():
            if index:
                out_d_str = dir_map[d_str]
                index_path = Path(out_d_str) / _LEVELS_PREFIX
                temp_file = index_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
                torch.save(index, temp_file)
                os.replace(temp_file, index_path)

    path_stream = (p for p in _path_stream())

    import torch.multiprocessing as mp

    mp_ctx = mp.get_context("spawn")

    with mp_ctx.Pool(
        processes=num_workers, initializer=_worker_initializer, maxtasksperchild=50
    ) as pool:
        results_iter = pool.imap_unordered(
            _process_single_cache_file,
            (path for d_str, path in path_stream),
            chunksize=10,
        )

        with tqdm(desc="Computing node levels", unit=" files") as pbar:
            for result in results_iter:
                if result is not None:
                    d_str, basename, levels = result
                    accumulated[d_str][basename] = levels
                    success_count += 1

                pbar.update(1)

                if success_count > 0 and success_count % CHECKPOINT_EVERY == 0:
                    _flush_indices()

    _flush_indices()
    print(
        f"\n[Levels Precomputation] Complete! Processed {success_count} files, {error_count} errors."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument(
        "--out-dirs",
        nargs="+",
        required=False,
        help="Corresponding directories to save the index files",
    )
    args = parser.parse_args()

    start_time = time.time()
    precompute_levels(args.dirs, out_directories=args.out_dirs)
    elapsed = time.time() - start_time
    print(f"[Levels Precomputation] Total elapsed time: {elapsed:.2f} seconds")
