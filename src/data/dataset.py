from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.serialization
from torch_geometric.data import Data as _PyGData
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.data import storage as _pyg_storage
from torch_geometric.utils import degree

from data.sparsification import precomputed_sparsification
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
    design_key: str
    y_node_opt: float


# Module-level cache for raw CSV samples. The CSVs do not change during a run,
# so re-reading and re-parsing them on every trial (2x per trial for train+val)
# is pure waste. Key: tuple of resolved CSV path strings.
_CSV_SAMPLE_CACHE: dict[tuple[str, ...], list[GraphSample]] = {}

# Module-level cache for splits JSON (keyed by resolved cache_file path string).
# The splits file for a given algo+num_samples tag never changes once written.
_SPLITS_CACHE: dict[str, dict[str, object]] = {}
_SPLIT_CACHE_VERSION = 2
_GRAPH_CACHE_CONTENT_VERSION = 2


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

    _MANIFEST_VERSION = 2

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
        sparsification: str | None = None,
        sparsification_replace_path: tuple[str, str] | None = None,
        normalize_edges: bool = False,
        split: str | None = None,
        cache_dir: str | Path | None = None,
        tier0_cache_dir: str | Path | None = None,
        tier1_cache_dir: str | Path | None = None,
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
        self.sparsification = sparsification
        self.sparsification_replace_path = sparsification_replace_path
        self.normalize_edges = bool(normalize_edges)
        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._tier0_cache_dir = Path(tier0_cache_dir) if tier0_cache_dir is not None else None
        self._tier1_cache_dir = Path(tier1_cache_dir) if tier1_cache_dir is not None else None
        self.split_ratios = split_ratios
        self.seed = seed
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.hp_tuning_splits_path = hp_tuning_splits_path
        self._cache_precomputed_level_pe = (
            str(self.positional_encoding).lower() == "level"
            if self.positional_encoding is not None
            else False
        )

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
        hasher.update(f"split_cache_v{_SPLIT_CACHE_VERSION}".encode())
        hasher.update(f"graph_cache_v{_GRAPH_CACHE_CONTENT_VERSION}".encode())
        hasher.update(b"split_by=design")
        hasher.update(str(self.seed).encode())
        hasher.update(str(self.split).encode())
        hasher.update(str(self.num_samples).encode())
        hasher.update("|".join(map(str, self.split_ratios)).encode())
        hasher.update(str(self._tier0_cache_dir).encode())
        hasher.update(str(self._tier1_cache_dir).encode())
        hasher.update(str(self.positional_encoding).encode())

        for csv_path in sorted(self.csv_paths):
            hasher.update(str(csv_path.absolute()).encode())

        if self.hp_tuning_splits_path is not None:
            hp_path = Path(self.hp_tuning_splits_path)
            hasher.update(str(hp_path.absolute()).encode())

        return hasher.hexdigest()[:16]

    def _normalize_graph_path(self, graph_path: str) -> str:
        return str(graph_path).replace("/gpfs/scratch1/shared", "/scratch-shared")

    def _infer_design_key(self, graph_path: str) -> str:
        parts = Path(graph_path).parts
        for marker, offset in (("designs", 1), ("tier0", 1), ("tier1", 2), ("tier2", 2)):
            try:
                marker_idx = parts.index(marker)
            except ValueError:
                continue
            design_idx = marker_idx + offset
            if design_idx < len(parts) and parts[design_idx]:
                return parts[design_idx]
        return graph_path

    def _sample_rows(self, samples: list[GraphSample]) -> list[GraphSample]:
        if self.num_samples is None or self.num_samples >= len(samples):
            return samples
        rng = random.Random(self.seed)
        selected_idx = set(rng.sample(range(len(samples)), k=self.num_samples))
        return [sample for idx, sample in enumerate(samples) if idx in selected_idx]

    def _split_cache_meta(self) -> dict[str, object]:
        csv_files = []
        for csv_path in sorted(self.csv_paths):
            st = csv_path.stat()
            csv_files.append(
                {
                    "path": str(csv_path.absolute()),
                    "size": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                }
            )

        hp_splits_file = None
        if self.hp_tuning_splits_path is not None:
            hp_path = Path(self.hp_tuning_splits_path)
            hp_st = hp_path.stat()
            hp_splits_file = {
                "path": str(hp_path.absolute()),
                "size": hp_st.st_size,
                "mtime_ns": hp_st.st_mtime_ns,
            }

        return {
            "version": _SPLIT_CACHE_VERSION,
            "split_by": "design",
            "seed": self.seed,
            "split_ratios": list(self.split_ratios),
            "num_samples": self.num_samples,
            "csv_files": csv_files,
            "hp_tuning_splits_file": hp_splits_file,
        }

    def _is_compatible_split_payload(self, payload: object) -> bool:
        if not isinstance(payload, dict):
            return False
        for split_name in ("train", "val", "test"):
            split_values = payload.get(split_name)
            if not isinstance(split_values, list) or not all(
                isinstance(value, str) for value in split_values
            ):
                return False
        return payload.get("__meta__") == self._split_cache_meta()

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
                graph_path=self._normalize_graph_path(str(graph_path)),
                design_key=self._infer_design_key(
                    self._normalize_graph_path(str(graph_path))
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

    def _load_or_create_split_keys(
        self, samples: list[GraphSample]
    ) -> dict[str, list[str] | dict[str, object]]:
        if self.cache_dir is None or self.split is None:
            return self._create_split_keys(samples)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        algo_tag = "_".join(p.stem for p in self.csv_paths)
        sample_tag = f"_{self.num_samples}" if self.num_samples is not None else "_all"
        cache_file = self.cache_dir / f"{algo_tag}{sample_tag}_splits.json"
        cache_key = str(cache_file)

        cached_payload = _SPLITS_CACHE.get(cache_key)
        if self._is_compatible_split_payload(cached_payload):
            return cached_payload
        _SPLITS_CACHE.pop(cache_key, None)

        try:
            with open(cache_file, encoding="utf-8") as fh:
                splits = json.load(fh)
            if self._is_compatible_split_payload(splits):
                _SPLITS_CACHE[cache_key] = splits
                return splits
        except (json.JSONDecodeError, OSError):
            pass

        split_keys = self._create_split_keys(samples)
        temp_file = cache_file.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
        temp_file.write_text(
            json.dumps(split_keys, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temp_file, cache_file)
        _SPLITS_CACHE[cache_key] = split_keys
        return split_keys

    def _create_split_keys(
        self, samples: list[GraphSample]
    ) -> dict[str, list[str] | dict[str, object]]:
        samples = self._sample_rows(samples)
        design_keys = list(dict.fromkeys(sample.design_key for sample in samples))
        rng = random.Random(self.seed)
        rng.shuffle(design_keys)

        total = sum(self.split_ratios)
        train_f = self.split_ratios[0] / total
        val_f = self.split_ratios[1] / total

        n = len(design_keys)
        n_train = int(n * train_f)
        n_val = int(n * val_f)
        if n > 0 and train_f > 0.0 and n_train == 0:
            n_train = 1
        n_val = min(n_val, n - n_train)

        design_to_split = {
            design_key: "train" for design_key in design_keys[:n_train]
        }
        design_to_split.update(
            {
                design_key: "val"
                for design_key in design_keys[n_train : n_train + n_val]
            }
        )
        design_to_split.update(
            {design_key: "test" for design_key in design_keys[n_train + n_val :]}
        )

        split_keys: dict[str, list[str] | dict[str, object]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        for sample in samples:
            split_name = design_to_split[sample.design_key]
            split_keys[split_name].append(sample.graph_path)
        split_keys["__meta__"] = self._split_cache_meta()

        return split_keys

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
            return self._sample_rows(samples)

        split_keys = self._load_or_create_split_keys(samples)
        remaining = Counter(split_keys[self.split])
        selected_samples = []
        for sample in samples:
            if remaining[sample.graph_path] < 1:
                continue
            selected_samples.append(sample)
            remaining[sample.graph_path] -= 1
        return selected_samples

    def _build_samples(self) -> list[GraphSample]:
        samples = self._read_candidate_samples()
        samples = self._apply_split(samples)
        return [s for s in samples if Path(s.graph_path).is_file()]

    def _stable_graph_cache_name(self, graph_path: str) -> str:
        source = Path(graph_path)
        st = source.stat()
        token = (
            f"v{_GRAPH_CACHE_CONTENT_VERSION}|pe={self.positional_encoding}|"
            f"{source.absolute()}|{st.st_size}|{st.st_mtime_ns}"
        )
        digest = hashlib.sha1(token.encode()).hexdigest()
        return f"{digest}.pt"

    def _cache_root_for_graph(self, graph_path: str) -> Path:
        if self._tier0_cache_dir is not None and "/tier0/" in graph_path:
            return self._tier0_cache_dir
        if self._tier1_cache_dir is not None and "/tier1/" in graph_path:
            return self._tier1_cache_dir
        if self._cache_graph_dir is not None:
            return self._cache_graph_dir
        return Path(graph_path).parent

    def _cached_graph_path(self, graph_path: str) -> Path:
        if self._cache_graph_dir is None:
            return Path(graph_path)
        return self._cache_root_for_graph(graph_path) / self._stable_graph_cache_name(
            graph_path
        )

    def _torch_load_graph(self, graph_path: str | Path) -> _PyGData:
        with open(graph_path, "rb") as fh:
            return torch.load(fh, map_location="cpu", weights_only=True)

    def _prepare_cached_graph(self, data_obj: _PyGData) -> _PyGData:
        if self.normalize_edges:
            if (
                getattr(data_obj, "edge_weight", None) is None
                and getattr(data_obj, "edge_index", None) is not None
            ):
                edge_index = data_obj.edge_index
                if edge_index.numel() > 0:
                    row, col = edge_index
                    deg = degree(col, data_obj.num_nodes, dtype=data_obj.x.dtype)
                    deg_inv_sqrt = deg.pow(-0.5)
                    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
                    data_obj.edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                else:
                    data_obj.edge_weight = torch.empty((0,), dtype=data_obj.x.dtype)

        if self._cache_precomputed_level_pe and getattr(data_obj, "pos_enc", None) is None:
            data_obj = self.pe_transform(data_obj)
            # pe_transform (ExtractPrecomputedPE) already deleted 'level'; drop
            # the unused siblings so they are not persisted in the cache file.
            for _attr in ("pi_paths", "local_sp_sum"):
                if hasattr(data_obj, _attr):
                    delattr(data_obj, _attr)

        return data_obj

    def _cache_single_graph(self, graph_path: str) -> tuple[str, int]:
        cache_path = self._cached_graph_path(graph_path)
        if cache_path.is_file():
            obj = self._torch_load_graph(cache_path)
            needs_refresh = (
                (self.normalize_edges and getattr(obj, "edge_weight", None) is None)
                or (
                    self._cache_precomputed_level_pe
                    and getattr(obj, "pos_enc", None) is None
                )
            )
            if needs_refresh:
                obj = self._prepare_cached_graph(obj)
                tmp = cache_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
                torch.save(obj, tmp)
                tmp.replace(cache_path)
        else:
            obj = self._torch_load_graph(Path(graph_path))
            obj = self._prepare_cached_graph(obj)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
            torch.save(obj, tmp)
            tmp.replace(cache_path)

        num_nodes = int(obj.x.shape[0])
        return str(cache_path), num_nodes

    def _save_global_nn(self, path: Path, data: dict[str, int]) -> None:
        """Atomically persist the global num_nodes lookup for a cache directory."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
        tmp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

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
        manifest_paths = [str(entry.get("graph_path", "")) for entry in manifest["entries"]]
        sample_paths = [sample.graph_path for sample in self.samples]
        if Counter(manifest_paths) != Counter(sample_paths):
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

        # Stored one file per unique cache-directory so that reruns skip
        # torch.load for cached .pt files whose num_nodes is already known.
        cache_roots = {
            str(root): root
            for root in (
                self._cache_graph_dir,
                self._tier0_cache_dir,
                self._tier1_cache_dir,
            )
            if root is not None
        }
        global_nn_paths = {
            cache_key: root / "_num_nodes_global.json"
            for cache_key, root in cache_roots.items()
        }
        global_num_nodes = {cache_key: {} for cache_key in cache_roots}
        for cache_key, gpath in global_nn_paths.items():
            if gpath.is_file():
                try:
                    global_num_nodes[cache_key].update(
                        json.loads(gpath.read_text(encoding="utf-8"))
                    )
                except (json.JSONDecodeError, OSError):
                    pass

        path_map: dict[str, str] = {}
        num_nodes_map: dict[str, int] = {}

        def _process_one(graph_path: str) -> tuple[str, str, int]:
            cache_key = str(self._cache_root_for_graph(graph_path))
            gmap = global_num_nodes[cache_key]
            cache_path = self._cached_graph_path(graph_path)
            if cache_path.is_file() and graph_path in gmap:
                return graph_path, str(cache_path), gmap[graph_path]
            cached_path, num_nodes = self._cache_single_graph(graph_path)
            return graph_path, cached_path, num_nodes

        _SAVE_NN_EVERY = 10_000
        _last_saved = 0
        dirty_cache_keys: set[str] = set()
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
                    cache_key = str(self._cache_root_for_graph(graph_path))
                    gmap = global_num_nodes[cache_key]
                    if graph_path not in gmap:
                        gmap[graph_path] = num_nodes
                        dirty_cache_keys.add(cache_key)
                    if completed % 1000 == 0 or completed == total:
                        print(f"[cache] {completed}/{total} graphs cached", flush=True)
                # Persist global maps periodically so reruns can skip re-loading.
                if completed - _last_saved >= _SAVE_NN_EVERY or completed == total:
                    for cache_key in tuple(dirty_cache_keys):
                        self._save_global_nn(
                            global_nn_paths[cache_key],
                            global_num_nodes[cache_key],
                        )
                        dirty_cache_keys.remove(cache_key)
                    _last_saved = completed

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
        
        if self.sparsification is not None:
            cached_path = self._graph_cache_path_map.get(sample.graph_path)
            sparse_cache_path = cached_path
            if sparse_cache_path is not None and self.sparsification_replace_path is not None:
                old_p, new_p = self.sparsification_replace_path
                sparse_cache_path = Path(str(sparse_cache_path).replace(old_p, new_p))
            
            data_obj = precomputed_sparsification(
                data_obj, self.sparsification, cache_path=sparse_cache_path
            )

        if not self.normalize_edges and hasattr(data_obj, "edge_weight"):
            # Keep edge_weight in cache files, but drop it from runtime samples
            # to avoid unnecessary batching/device transfer when disabled.
            del data_obj.edge_weight
        if self.positional_encoding is not None and getattr(data_obj, "pos_enc", None) is None:
            data_obj = self.pe_transform(data_obj)
        # ExtractPrecomputedPE already deletes the one attr it consumed; mop up
        # any remaining siblings that weren't used as the PE source.
        if self.positional_encoding is not None:
            for _attr in ("level", "pi_paths", "local_sp_sum"):
                if hasattr(data_obj, _attr):
                    delattr(data_obj, _attr)
        data_obj.y = torch.tensor([[sample.y_node_opt]], dtype=torch.float32)

        # Clean up embedded masks from older cache versions just in case.
        for key in list(data_obj.keys()):
            if key.endswith("_dynamic_mask") or key.endswith("_dynamic_num_partitions"):
                delattr(data_obj, key)

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
