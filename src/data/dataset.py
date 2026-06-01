from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.serialization
from torch_geometric.data import Data as _PyGData
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.data import storage as _pyg_storage

from models.layers.positional_encodings import get_pe_transform

# Register PyG classes in the secure deserialization allowlist so torch.load
# can use weights_only=True (C++ deserializer) on cached .pt graph files.
# This bypasses the Python Unpickler entirely, eliminating the Unpickler memo
# dict leak that causes TypedStorage, UntypedStorage, cell, FileIO, and
# BufferedReader objects to accumulate linearly across training steps.
_pyg_safe_globals: list = [_PyGData, _pyg_storage.GlobalStorage]
for _name in ("DataTensorAttr", "DataEdgeAttr"):
    try:
        import torch_geometric.data.data as _pyg_data_mod

        _cls = getattr(_pyg_data_mod, _name, None)
        if _cls is not None:
            _pyg_safe_globals.append(_cls)
    except Exception:
        pass
torch.serialization.add_safe_globals(_pyg_safe_globals)
del _pyg_safe_globals, _pyg_data_mod, _name, _cls


@dataclass(frozen=True, slots=True)
class GraphSample:
    """Dataclass holding metadata for a single graph sample."""

    graph_path: str
    y_node_opt: float


# Module-level cache for raw CSV samples. The CSVs do not change during a run,
# so re-reading and re-parsing them on every trial (2x per trial for train+val)
# is pure waste. Key: tuple of resolved CSV path strings.
_CSV_SAMPLE_CACHE: dict[tuple[str, ...], list[GraphSample]] = {}

# Module-level cache for splits JSON (keyed by resolved cache_file path string).
# The splits file for a given algo+num_samples tag never changes once written.
_SPLITS_CACHE: dict[str, dict[str, list[str]]] = {}


def clear_dataset_global_caches() -> None:
    """Drop module-level sample/split caches.

    These caches are useful within a trial but can accumulate across long
    Optuna studies that instantiate many dataset objects in one process.
    """
    _CSV_SAMPLE_CACHE.clear()
    _SPLITS_CACHE.clear()


class AIGGraphRegressionDataset(PyGDataset):
    """Minimal graph-level regression dataset.

    Targets:
    - y[0] = node optimizability

    Required graph attributes loaded from .pt:
    - x, edge_index, edge_attr, level, pi_paths, local_sp_sum
    """

    _MANIFEST_VERSION = 1

    @property
    def raw_dir(self) -> str:
        if self._cache_meta_dir is not None:
            return str(self._cache_meta_dir / "raw")
        return super().raw_dir

    @property
    def processed_dir(self) -> str:
        if self._cache_meta_dir is not None:
            return str(self._cache_meta_dir / "processed")
        return super().processed_dir

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return []

    @property
    def has_process(self) -> bool:
        return self._cache_meta_dir is not None

    @property
    def has_download(self) -> bool:
        return False

    def download(self) -> None:
        return

    def __init__(
        self,
        csv_paths: str | Path | list[str | Path],
        *,
        positional_encoding: str | None = None,
        split: str | None = None,
        cache_dir: str | Path | None = None,
        tier0_cache_dir: str | Path | None = None,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        num_samples: int | None = None,
        num_workers: int = 0,
        hp_tuning_splits_path: str | Path | None = None,
    ) -> None:
        if isinstance(csv_paths, (str, Path)):
            self.csv_paths = [Path(csv_paths)]
        else:
            self.csv_paths = [Path(p) for p in csv_paths]

        self.positional_encoding = positional_encoding
        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._tier0_cache_dir = Path(tier0_cache_dir) if tier0_cache_dir is not None else None
        self.split_ratios = split_ratios
        self.seed = seed
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.hp_tuning_splits_path = hp_tuning_splits_path

        self._cache_graph_dir = (
            self.cache_dir / "processed_graphs" if self.cache_dir is not None else None
        )
        self._cache_meta_dir = (
            self.cache_dir / "metadata" if self.cache_dir is not None else None
        )
        self._cache_signature = self._build_cache_signature()
        self._manifest_path = (
            self._cache_meta_dir / f"dataset_{self._cache_signature}_manifest.json"
            if self._cache_meta_dir is not None
            else None
        )
        self._graph_cache_path_map: dict[str, Path] = {}
        self._node_sizes: list[int] | None = None

        self.pe_transform = get_pe_transform(
            pe_type=self.positional_encoding, attr_name="pos_enc"
        )

        self.samples = self._build_samples()
        pyg_root = (
            str(self._cache_meta_dir) if self._cache_meta_dir is not None else None
        )
        super().__init__(root=pyg_root, log=False)

        if self.cache_dir is not None:
            self.process()

    def _build_cache_signature(self) -> str:
        hasher = hashlib.sha1()
        hasher.update(str(self.seed).encode())
        hasher.update(str(self.split).encode())
        hasher.update(str(self.num_samples).encode())
        hasher.update("|".join(map(str, self.split_ratios)).encode())

        for csv_path in sorted(self.csv_paths):
            st = csv_path.stat()
            hasher.update(str(csv_path.absolute()).encode())
            hasher.update(str(st.st_size).encode())
            hasher.update(str(st.st_mtime_ns).encode())

        if self.hp_tuning_splits_path is not None:
            hp_path = Path(self.hp_tuning_splits_path)
            hp_st = hp_path.stat()
            hasher.update(str(hp_path).encode())
            hasher.update(str(hp_st.st_size).encode())
            hasher.update(str(hp_st.st_mtime_ns).encode())

        return hasher.hexdigest()[:16]

    def _read_candidate_samples(self) -> list[GraphSample]:
        cache_key = tuple(str(p) for p in self.csv_paths)
        if cache_key in _CSV_SAMPLE_CACHE:
            return _CSV_SAMPLE_CACHE[cache_key]

        required_cols = ["unoptimized_graph_path", "optimizability"]
        frames = [
            pd.read_csv(
                p,
                usecols=required_cols,
                dtype={"unoptimized_graph_path": str, "optimizability": float},
            )
            for p in self.csv_paths
        ]
        df = pd.concat(frames, ignore_index=True)

        samples = [
            GraphSample(
                graph_path=str(graph_path).replace(
                    "/gpfs/scratch1/shared", "/scratch-shared"
                ),
                y_node_opt=float(node_opt),
            )
            for graph_path, node_opt in zip(
                df["unoptimized_graph_path"].fillna(""),
                df["optimizability"],
                strict=False,
            )
        ]

        del df, frames
        _CSV_SAMPLE_CACHE[cache_key] = samples
        return samples

    def _load_or_create_split_keys(self, all_keys: list[str]) -> dict[str, list[str]]:
        if self.cache_dir is None or self.split is None:
            return self._create_split_keys(all_keys)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        algo_tag = "_".join(p.stem for p in self.csv_paths)
        sample_tag = f"_{self.num_samples}" if self.num_samples is not None else "_all"
        cache_file = self.cache_dir / f"{algo_tag}{sample_tag}_splits.json"
        cache_key = str(cache_file)

        if cache_key in _SPLITS_CACHE:
            return _SPLITS_CACHE[cache_key]

        try:
            with open(cache_file, encoding="utf-8") as fh:
                splits = json.load(fh)
            _SPLITS_CACHE[cache_key] = splits
            return splits
        except (json.JSONDecodeError, OSError):
            pass

        split_keys = self._create_split_keys(all_keys)
        temp_file = cache_file.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
        temp_file.write_text(
            json.dumps(split_keys, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temp_file, cache_file)
        _SPLITS_CACHE[cache_key] = split_keys
        return split_keys

    def _create_split_keys(self, all_keys: list[str]) -> dict[str, list[str]]:
        keys = list(all_keys)
        rng = random.Random(self.seed)
        rng.shuffle(keys)

        if self.num_samples is not None:
            keys = keys[: self.num_samples]

        total = sum(self.split_ratios)
        train_f = self.split_ratios[0] / total
        val_f = self.split_ratios[1] / total

        n = len(keys)
        n_train = int(n * train_f)
        n_val = int(n * val_f)

        return {
            "train": keys[:n_train],
            "val": keys[n_train : n_train + n_val],
            "test": keys[n_train + n_val :],
        }

    def _apply_split(self, samples: list[GraphSample]) -> list[GraphSample]:
        if self.hp_tuning_splits_path is not None:
            hp_path = Path(self.hp_tuning_splits_path)
            with open(hp_path, encoding="utf-8") as fh:
                hp_splits = json.load(fh)
            hp_keys = set(
                hp_splits.get("train", [])
                + hp_splits.get("val", [])
                + hp_splits.get("test", [])
            )
            samples = [s for s in samples if s.graph_path not in hp_keys]

        if self.split is None:
            if self.num_samples is not None:
                all_keys = [s.graph_path for s in samples]
                rng = random.Random(self.seed)
                rng.shuffle(all_keys)
                selected = set(all_keys[: self.num_samples])
                return [s for s in samples if s.graph_path in selected]
            return samples

        all_keys = [s.graph_path for s in samples]
        split_keys = self._load_or_create_split_keys(all_keys)
        selected = set(split_keys[self.split])
        return [s for s in samples if s.graph_path in selected]

    def _build_samples(self) -> list[GraphSample]:
        samples = self._read_candidate_samples()
        samples = self._apply_split(samples)
        if self._manifest_path is not None and self._manifest_path.is_file():
            return samples
        return [s for s in samples if Path(s.graph_path).is_file()]

    def _stable_graph_cache_name(self, graph_path: str) -> str:
        source = Path(graph_path)
        st = source.stat()
        token = f"{source.absolute()}|{st.st_size}|{st.st_mtime_ns}"
        digest = hashlib.sha1(token.encode()).hexdigest()
        return f"{digest}.pt"

    def _cached_graph_path(self, graph_path: str) -> Path:
        if self._cache_graph_dir is None:
            return Path(graph_path)
        if self._tier0_cache_dir is not None and "/tier0/" in graph_path:
            return self._tier0_cache_dir / self._stable_graph_cache_name(graph_path)
        return self._cache_graph_dir / self._stable_graph_cache_name(graph_path)

    def _torch_load_graph(self, graph_path: str | Path) -> _PyGData:
        with open(graph_path, "rb") as fh:
            return torch.load(fh, map_location="cpu", weights_only=True)

    def _cache_single_graph(self, graph_path: str) -> tuple[str, int]:
        cache_path = self._cached_graph_path(graph_path)
        if cache_path.is_file():
            obj = self._torch_load_graph(cache_path)
        else:
            obj = self._torch_load_graph(Path(graph_path))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
            torch.save(obj, tmp)
            tmp.replace(cache_path)

        num_nodes = int(obj.x.shape[0])
        return str(cache_path), num_nodes

    def _load_manifest(self) -> dict | None:
        if not self._manifest_path.is_file():
            return None
        try:
            with open(self._manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(manifest, dict) or not isinstance(
            manifest.get("entries"), list
        ):
            return None
        return manifest

    def _rebuild_graph_cache(self) -> dict:
        unique_paths = sorted({sample.graph_path for sample in self.samples})

        cpu_limit = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1)
        )
        n_threads = max(
            1,
            min(
                self.num_workers if self.num_workers > 0 else cpu_limit,
                cpu_limit,
            ),
        )
        print(
            f"[cache] Building graph cache: {len(unique_paths)} unique graphs "
            f"using {n_threads} threads -> {self._cache_graph_dir}",
            flush=True,
        )

        path_map: dict[str, str] = {}
        num_nodes_map: dict[str, int] = {}

        def _process_one(graph_path: str) -> tuple[str, str, int]:
            cached_path, num_nodes = self._cache_single_graph(graph_path)
            return graph_path, cached_path, num_nodes

        CHUNK_SIZE = max(n_threads * 4, 256)
        completed = 0
        total = len(unique_paths)
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            for chunk_start in range(0, total, CHUNK_SIZE):
                chunk = unique_paths[chunk_start : chunk_start + CHUNK_SIZE]
                for graph_path, cached_path, num_nodes in executor.map(
                    _process_one, chunk
                ):
                    path_map[graph_path] = cached_path
                    num_nodes_map[graph_path] = num_nodes
                    completed += 1
                    if completed % 1000 == 0 or completed == total:
                        print(f"[cache] {completed}/{total} graphs cached", flush=True)

        # Dump consolidated node sizes to a single JSON file to avoid per-graph .n files
        processed_dir_path = Path(self.processed_dir)
        processed_dir_path.mkdir(parents=True, exist_ok=True)
        node_sizes_path = processed_dir_path / "node_sizes.json"
        tmp_ns = node_sizes_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
        tmp_ns.write_text(
            json.dumps(num_nodes_map, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(tmp_ns, node_sizes_path)

        entries = [
            {
                "graph_path": sample.graph_path,
                "cache_path": path_map[sample.graph_path],
                "num_nodes": num_nodes_map[sample.graph_path],
            }
            for sample in self.samples
        ]

        return {
            "version": self._MANIFEST_VERSION,
            "num_samples": len(self.samples),
            "entries": entries,
        }

    def _apply_manifest(self, manifest: dict) -> None:
        self._graph_cache_path_map.clear()
        for entry in manifest["entries"]:
            if "cache_path" in entry:
                self._graph_cache_path_map[str(entry["graph_path"])] = Path(
                    entry["cache_path"]
                )
            elif "cache_name" in entry and self._cache_graph_dir is not None:
                # backward compat with v1 manifests
                self._graph_cache_path_map[str(entry["graph_path"])] = (
                    self._cache_graph_dir / entry["cache_name"]
                )
        self._node_sizes = [int(e["num_nodes"]) for e in manifest["entries"]]

    def process(self) -> bool:
        """Load or build the graph cache manifest.

        Returns:
            True if the manifest was freshly rebuilt, False if loaded from disk.
        """
        self._cache_meta_dir.mkdir(parents=True, exist_ok=True)

        manifest = self._load_manifest()
        if manifest is None:
            manifest = self._rebuild_graph_cache()
            tmp = self._manifest_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
            tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
            tmp.replace(self._manifest_path)
            self._apply_manifest(manifest)
            return True

        self._apply_manifest(manifest)
        return False

    def _load_graph_for_sample(self, sample: GraphSample) -> _PyGData:
        graph_path = sample.graph_path
        if self.cache_dir is not None:
            cached_path = self._graph_cache_path_map.get(graph_path)
            if cached_path is None:
                rebuilt, _ = self._cache_single_graph(graph_path)
                cached_path = Path(rebuilt)
                self._graph_cache_path_map[graph_path] = cached_path
            return self._torch_load_graph(cached_path)
        return self._torch_load_graph(graph_path)

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> _PyGData:
        sample = self.samples[idx]
        data_obj = self._load_graph_for_sample(sample)
        data_obj = self.pe_transform(data_obj)
        data_obj.y = torch.tensor([[sample.y_node_opt]], dtype=torch.float32)
        return data_obj

    def get_num_nodes_list(self) -> list[int]:
        if self._node_sizes is not None:
            return self._node_sizes

        if self._manifest_path is not None:
            manifest = self._load_manifest()
            if manifest is not None:
                self._node_sizes = [int(e["num_nodes"]) for e in manifest["entries"]]
                return self._node_sizes

        seen: dict[str, int] = {}
        sizes = []
        for s in self.samples:
            if s.graph_path not in seen:
                obj = self._torch_load_graph(s.graph_path)
                seen[s.graph_path] = int(obj.x.shape[0])
            sizes.append(seen[s.graph_path])
        self._node_sizes = sizes
        return sizes

    def release_runtime_caches(self) -> None:
        """Drop the node-sizes list after batch planning; keep path map intact for get()."""
        self._node_sizes = None


__all__ = ["AIGGraphRegressionDataset", "GraphSample", "clear_dataset_global_caches"]
