import os
import uuid
import torch
import functools
import itertools
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

_LEVELS_PREFIX = "_levels.pt"

def _register_pyg_safe_globals() -> None:
    import torch.serialization
    from torch_geometric.data import Data
    import torch_geometric.data.data as _pyg_data_mod
    import torch_geometric.data.storage as _pyg_storage
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
        if x[i, 0] == 1.0 or x[i, 1] == 1.0: # Const or PI
            levels[i] = 0
        elif x[i, 2] == 1.0 or x[i, 3] == 1.0: # Gate or PO
            if fanins[i]:
                levels[i] = max(levels[f] for f in fanins[i]) + 1
            else:
                levels[i] = 0
                
    return levels


def _worker_initializer() -> None:
    import torch as _torch
    _torch.set_num_threads(1)
    _register_pyg_safe_globals()


def _process_single_cache_file(cache_path: Path) -> tuple[str, torch.Tensor] | None:
    if not cache_path.is_file():
        return None

    with open(cache_path, "rb") as fh:
        data_obj = torch.load(fh, map_location="cpu", weights_only=True)

    levels = compute_node_levels(data_obj)
    return cache_path.name, levels


def precompute_levels(directories: list[str | Path]):
    _register_pyg_safe_globals()
    
    top_dirs = [Path(d).absolute() for d in directories if Path(d).is_dir()]
    
    accumulated = {}
    done_by_dir = {}
    
    for top_dir in top_dirs:
        d_str = str(top_dir)
        index_path = top_dir / _LEVELS_PREFIX
        if index_path.is_file():
            try:
                accumulated[d_str] = torch.load(index_path, map_location="cpu", weights_only=True)
                done_by_dir[d_str] = set(accumulated[d_str].keys())
            except Exception:
                accumulated[d_str] = {}
                done_by_dir[d_str] = set()
        else:
            accumulated[d_str] = {}
            done_by_dir[d_str] = set()
            
        print(f"  -> {top_dir}: {len(done_by_dir[d_str])} entries already in index")
        
    def _path_stream():
        for top_dir in top_dirs:
            d_str = str(top_dir)
            done_set = done_by_dir[d_str]
            try:
                with os.scandir(str(top_dir)) as scanner:
                    for entry in scanner:
                        if (entry.is_file(follow_symlinks=False)
                                and entry.name.endswith(".pt")
                                and not entry.name.startswith("_")
                                and entry.name not in done_set):
                            yield d_str, Path(entry.path)
            except PermissionError as exc:
                print(f"[WARNING] Cannot scan {top_dir}: {exc}")

    try:
        all_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        all_cpus = os.cpu_count() or 1
    num_workers = min(32, max(1, int(all_cpus * 0.5)))
    
    print(f"[Levels Precomputation] Using {num_workers} parallel workers...")
    
    CHECKPOINT_EVERY = 50_000
    success_count = 0
    error_count = 0
    
    def _flush_indices() -> None:
        for d_str, index in accumulated.items():
            if index:
                index_path = Path(d_str) / _LEVELS_PREFIX
                temp_file = index_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
                torch.save(index, temp_file)
                os.replace(temp_file, index_path)

    path_stream = _path_stream()
    PENDING_LIMIT = num_workers * 8

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_initializer,
        mp_context=mp.get_context("spawn"),
        max_tasks_per_child=50,
    ) as executor:
        futures = {}
        for d_str, path in itertools.islice(path_stream, PENDING_LIMIT):
            f = executor.submit(_process_single_cache_file, path)
            futures[f] = (d_str, path)

        with tqdm(desc="Computing node levels", unit=" files") as pbar:
            while futures:
                done_futures, _ = concurrent.futures.wait(
                    futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done_futures:
                    d_str, path = futures.pop(future)
                    try:
                        result = future.result()
                        if result is not None:
                            basename, levels = result
                            accumulated[d_str][basename] = levels
                            success_count += 1
                    except Exception as exc:
                        error_count += 1
                        print(f"\n[ERROR] {path.name}: {exc}")

                    pbar.update(1)

                    try:
                        next_d_str, next_path = next(path_stream)
                        new_f = executor.submit(_process_single_cache_file, next_path)
                        futures[new_f] = (next_d_str, next_path)
                    except StopIteration:
                        pass

                if success_count > 0 and success_count % CHECKPOINT_EVERY == 0:
                    _flush_indices()

    _flush_indices()
    print(f"\n[Levels Precomputation] Complete! Processed {success_count} files, {error_count} errors.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", required=True)
    args = parser.parse_args()
    precompute_levels(args.dirs)
