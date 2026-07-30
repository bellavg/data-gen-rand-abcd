#!/usr/bin/env python
"""Precompute summarized (coarsened) graphs for a whole cache.

Reads the cached graphs listed in the dataset manifests, applies a
registered summarization method to each, and writes the materialized
coarsened graph to --out-dir under the *same basename*.  Because
``dataset._stable_graph_cache_name`` hashes the source graph path rather
than the cache directory, training can then point --tier0_cache_dir /
--tier1_cache_dir at a copy of that output and take its normal code path
with no summarization logic in the hot loop.

Run on the cluster (CPU only) via src/shell/precompute_summarization.sh,
which writes --out-dir to node-local scratch and packs the result into
tar.zst shards.  Never run this on a GPU node.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.sparsification import _register_pyg_safe_globals  # noqa: E402
from data.summarization import SUMMARIZATION_REGISTRY, summarize_graph  # noqa: E402

# Node counts let the dataset rebuild its manifest from stat() calls instead
# of loading every graph; the merged name and payload shape are dictated by
# dataset._rebuild_graph_cache.  Each shard writes its own file because the
# shards run on separate nodes and are later unpacked into one directory —
# a single fixed name would leave only the last shard's counts.
_NUM_NODES_GLOBAL = "_num_nodes_global.json"
_NUM_NODES_SHARD_GLOB = "_num_nodes_shard*.json"
CHECKPOINT_EVERY = 50_000   # atomic index save cadence (number of completed files)


def _shard_index_name(shard_id: int) -> str:
    return f"_num_nodes_shard{shard_id:03d}.json"


# =====================================================================
# WORKER TASK FOR PARALLEL EXECUTION
# =====================================================================

def _worker_initializer() -> None:
    import torch as _torch
    _torch.set_num_threads(1)
    _register_pyg_safe_globals()


def _summarize_single_file(
    task_tuple: tuple[str, str, str],
    method: str,
) -> dict | None:
    """task_tuple contains: (cache_path, out_path, graph_path).

    Returns None on any failure.  Letting an exception escape would abort the
    pool and lose the whole shard's unflushed index, so a single unreadable
    graph must not take the run down with it.
    """
    cache_path_str, out_path_str, graph_path = task_tuple
    cache_path = Path(cache_path_str)

    if not cache_path.is_file():
        print(f"[summarize] WARNING: missing cache file {cache_path}")
        return None

    try:
        with open(cache_path, "rb") as fh:
            data_obj = torch.load(fh, map_location="cpu", weights_only=True)

        nodes_before = int(data_obj.x.size(0))
        edges_before = int(data_obj.edge_index.size(1))

        started = time.perf_counter()
        summarized = summarize_graph(data_obj, method)
        elapsed = time.perf_counter() - started

        out_path = Path(out_path_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = out_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
        torch.save(summarized, temp_file)
        os.replace(temp_file, out_path)
    except Exception as exc:
        print(f"[summarize] WARNING: failed on {cache_path}: {exc}")
        return None

    return {
        "out_dir": str(out_path.parent),
        "graph_path": graph_path,
        "nodes_before": nodes_before,
        "nodes_after": int(summarized.x.size(0)),
        "edges_before": edges_before,
        "edges_after": int(summarized.edge_index.size(1)),
        "time_s": elapsed,
    }


# =====================================================================
# TASK BUILDING
# =====================================================================

def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
    temp_file.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temp_file, path)


def merge_shard_indexes(directory: str | Path) -> int:
    """Combine per-shard node-count files into ``_num_nodes_global.json``.

    Called after the shard archives have been unpacked into one directory,
    since that is the name and layout ``dataset._rebuild_graph_cache`` reads.
    Returns the number of entries written.
    """
    directory = Path(directory)
    merged: dict[str, int] = {}
    for shard_file in sorted(directory.glob(_NUM_NODES_SHARD_GLOB)):
        merged.update(_read_json(shard_file))
    if merged:
        _write_json_atomic(directory / _NUM_NODES_GLOBAL, merged)
    return len(merged)


def build_tasks(
    manifest_dirs: list[str | Path],
    out_root: Path,
    shard_id: int = 0,
    num_shards: int = 1,
) -> list[tuple[str, str, str]]:
    """Collect (cache_path, out_path, graph_path) triples from the manifests.

    Output mirrors the source layout one level deep, so a cache file in
    ``.../shared_tier0_cache/<hash>.pt`` lands in
    ``<out_root>/shared_tier0_cache/<hash>.pt``.  Only one algorithm's
    manifests should be passed at a time, since directory basenames are
    what keep the tiers apart.
    """
    tasks: list[tuple[str, str, str]] = []
    seen_cache_paths: set[str] = set()

    for m_dir in manifest_dirs:
        m_dir_path = Path(m_dir)
        if not m_dir_path.is_dir():
            print(f"[WARNING] Manifest directory not found: {m_dir_path}")
            continue

        for manifest_file in sorted(m_dir_path.glob("*_manifest.json")):
            try:
                with open(manifest_file) as f:
                    manifest = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[WARNING] Skipping bad manifest {manifest_file}: {exc}")
                continue

            # Manifests are named per cache signature, so a metadata directory
            # can hold several — including stale ones from earlier runs.  Log
            # what was consumed rather than silently summarizing dead entries.
            print(
                f"[summarize] {manifest_file.name}: "
                f"{len(manifest.get('entries', []))} entries"
            )

            for entry in manifest.get("entries", []):
                c_path = Path(entry["cache_path"])
                abs_cache_path = str(c_path.absolute())
                if abs_cache_path in seen_cache_paths:
                    continue
                seen_cache_paths.add(abs_cache_path)

                out_path = out_root / c_path.parent.name / c_path.name
                tasks.append((str(c_path), str(out_path), str(entry["graph_path"])))

    # Deterministic ordering so a shard covers the same slice on every rerun.
    tasks.sort()
    if num_shards > 1:
        tasks = tasks[shard_id::num_shards]
    return tasks


# =====================================================================
# CORE PIPELINE
# =====================================================================

def summarize_from_manifests(
    manifest_dirs: list[str | Path],
    method: str,
    out_dir: str | Path,
    shard_id: int = 0,
    num_shards: int = 1,
) -> None:
    _register_pyg_safe_globals()

    out_root = Path(out_dir)
    tasks = build_tasks(manifest_dirs, out_root, shard_id, num_shards)
    if not tasks:
        print("[summarize] No entries found in manifests.")
        return

    # 1. RESUME — a task is done only if its output exists *and* its node
    #    count is already recorded, otherwise the index would end up with
    #    holes after a restart.
    index_name = _shard_index_name(shard_id)
    num_nodes_by_dir: dict[str, dict[str, int]] = {}
    for _, out_path_str, _ in tasks:
        parent = str(Path(out_path_str).parent)
        if parent not in num_nodes_by_dir:
            num_nodes_by_dir[parent] = _read_json(Path(parent) / index_name)

    pending = [
        task
        for task in tasks
        if not (
            Path(task[1]).is_file()
            and task[2] in num_nodes_by_dir[str(Path(task[1]).parent)]
        )
    ]
    print(
        f"[summarize] shard {shard_id}/{num_shards}: {len(tasks)} graphs, "
        f"skipping {len(tasks) - len(pending)} already done."
    )
    if not pending:
        print("[summarize] Nothing to do.")
        return

    # 2. PARALLEL COMPUTATION
    try:
        all_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        all_cpus = os.cpu_count() or 1
    num_workers = max(1, min(all_cpus - 1, len(pending)))

    worker_fn = functools.partial(_summarize_single_file, method=method)

    totals: dict[str, float] = defaultdict(float)
    success_count = 0
    error_count = 0

    def _flush() -> None:
        for parent, mapping in num_nodes_by_dir.items():
            if mapping:
                _write_json_atomic(Path(parent) / index_name, mapping)

    import torch.multiprocessing as mp
    mp_ctx = mp.get_context("spawn")

    with mp_ctx.Pool(
        processes=num_workers, initializer=_worker_initializer, maxtasksperchild=50
    ) as pool:
        results_iter = pool.imap_unordered(worker_fn, pending, chunksize=10)

        from tqdm import tqdm
        with tqdm(total=len(pending), desc=f"Summarizing ({method})", unit=" graphs") as pbar:
            for result in results_iter:
                if result is not None:
                    num_nodes_by_dir[result["out_dir"]][result["graph_path"]] = result[
                        "nodes_after"
                    ]
                    for key in (
                        "nodes_before",
                        "nodes_after",
                        "edges_before",
                        "edges_after",
                        "time_s",
                    ):
                        totals[key] += result[key]
                    success_count += 1
                else:
                    error_count += 1

                pbar.update(1)

                if success_count > 0 and success_count % CHECKPOINT_EVERY == 0:
                    print(f"\n[Checkpoint] {success_count} done — flushing indices...")
                    _flush()

    _flush()

    # 3. STATS — the raw material for the compression/wall-clock table.
    stats = {
        "method": method,
        "shard_id": shard_id,
        "num_shards": num_shards,
        "graphs": success_count,
        "errors": error_count,
        "node_retention": (
            totals["nodes_after"] / totals["nodes_before"] if totals["nodes_before"] else 0.0
        ),
        "edge_retention": (
            totals["edges_after"] / totals["edges_before"] if totals["edges_before"] else 0.0
        ),
        "mean_time_s": totals["time_s"] / success_count if success_count else 0.0,
    }
    _write_json_atomic(
        out_root / f"_summary_stats_{method}_shard{shard_id:03d}.json", stats
    )

    print(
        f"\n[summarize] Complete! {success_count} graphs, {error_count} errors.\n"
        f"  node retention: {stats['node_retention']:.1%}\n"
        f"  edge retention: {stats['edge_retention']:.1%}\n"
        f"  mean time:      {stats['mean_time_s'] * 1000:.1f} ms/graph"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute summarized graphs for cached graphs in parallel."
    )
    parser.add_argument(
        "method",
        type=str,
        choices=sorted(SUMMARIZATION_REGISTRY),
        help="Summarization method to apply.",
    )
    parser.add_argument(
        "--manifest-dirs",
        nargs="+",
        required=True,
        help="Directories containing metadata .json manifests outlining "
             "'graph_path' and 'cache_path'.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Root directory for the summarized graphs (use node-local scratch).",
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    if not 0 <= args.shard_id < args.num_shards:
        parser.error(
            f"--shard-id must be in [0, {args.num_shards}), got {args.shard_id}"
        )

    summarize_from_manifests(
        manifest_dirs=args.manifest_dirs,
        method=args.method,
        out_dir=args.out_dir,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
