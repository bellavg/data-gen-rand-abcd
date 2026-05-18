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
from torch_geometric.data import Dataset as PyGDataset

from models.layers.positional_encodings import get_pe_transform


@dataclass(frozen=True)
class GraphSample:
    graph_path: str
    y_node_opt: float


# Module-level cache for raw CSV samples. The CSVs do not change during a run,
# so re-reading and re-parsing them on every trial (2× per trial for train+val)
# is pure waste.  Key: tuple of resolved CSV path strings.
_CSV_SAMPLE_CACHE: dict[tuple[str, ...], list[GraphSample]] = {}

# Module-level cache for splits JSON (keyed by resolved cache_file path string).
# The splits file for a given algo+num_samples tag never changes once written.
_SPLITS_CACHE: dict[str, dict[str, list[str]]] = {}


class AIGGraphRegressionDataset(PyGDataset):
    """
    Minimal graph-level regression dataset.

    Targets:
    - y[0] = node optimizability

    Required graph attributes loaded from .pt:
    - x, edge_index, edge_attr, level, pi_paths, local_sp_sum
    """

    _MANIFEST_VERSION = 1

    @property
    def raw_dir(self) -> str:
        # Keep PyG's raw_dir inside cache_dir when available so it never
        # escapes to an uncontrolled location.
        if self._cache_meta_dir is not None:
            return str(self._cache_meta_dir / "raw")
        return super().raw_dir

    @property
    def processed_dir(self) -> str:
        # Redirect PyG's processed_dir into cache_dir/metadata/processed so that
        # pre_transform.pt / pre_filter.pt are written there, not in ???/processed.
        if self._cache_meta_dir is not None:
            return str(self._cache_meta_dir / "processed")
        return super().processed_dir

    @property
    def raw_file_names(self) -> list[str]:
        # This dataset reads existing .pt paths from CSV and has no raw download step.
        return []

    @property
    def processed_file_names(self) -> list[str]:
        # Processing is handled by custom cache manifests, not PyG's processed_dir files.
        return []

    @property
    def has_process(self) -> bool:
        # Only allow PyG's _process() machinery (which calls makedirs) when we
        # actually have a cache_dir.  Without one there is nothing to persist and
        # the MISSING/"???" sentinel root must not materialise on disk.
        return self._cache_meta_dir is not None

    @property
    def has_download(self) -> bool:
        # Same rationale as has_process: suppress PyG's _download() makedirs
        # when there is no cache_dir to avoid the "???" sentinel directory.
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
        self._graph_num_nodes_map: dict[str, int] = {}

        # Initialize the PE transform factory
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
            # Stop resolving symlinks here too
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

        df = pd.concat(
            [pd.read_csv(p, dtype=str).fillna("") for p in self.csv_paths],
            ignore_index=True,
        )
        df["optimizability"] = df["optimizability"].astype(float)

        samples = [
            GraphSample(
                graph_path=row["unoptimized_graph_path"].replace(
                    "/gpfs/scratch1/shared", "/scratch-shared"
                ),
                y_node_opt=float(row["optimizability"]),
            )
            for row in df.to_dict("records")
        ]

        _CSV_SAMPLE_CACHE[cache_key] = samples
        return samples

    def _load_or_create_split_keys(self, all_keys: list[str]) -> dict[str, list[str]]:
        if self.cache_dir is None or self.split is None:
            return self._create_split_keys(all_keys)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        algo_tag = "_".join(p.stem for p in self.csv_paths)

        # Include num_samples in the cache filename so it doesn't collide with full datasets
        sample_tag = f"_{self.num_samples}" if self.num_samples is not None else "_all"
        cache_file = self.cache_dir / f"{algo_tag}{sample_tag}_splits.json"
        cache_key = str(cache_file)

        if cache_key in _SPLITS_CACHE:
            return _SPLITS_CACHE[cache_key]

        if cache_file.is_file():
            try:
                with open(cache_file, encoding="utf-8") as fh:
                    splits = json.load(fh)
                if all(name in splits for name in ("train", "val", "test")):
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

        # APPLY THE TOTAL LIMIT HERE, BEFORE SPLITTING
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
        # 1. Filter out HP tuning samples if specified
        if getattr(self, "hp_tuning_splits_path", None) is not None:
            hp_path = Path(self.hp_tuning_splits_path)
            if hp_path.is_file():
                try:
                    with open(hp_path, encoding="utf-8") as fh:
                        hp_splits = json.load(fh)
                    hp_keys = set(
                        hp_splits.get("train", [])
                        + hp_splits.get("val", [])
                        + hp_splits.get("test", [])
                    )
                    samples = [s for s in samples if s.graph_path not in hp_keys]
                except (json.JSONDecodeError, OSError) as exc:
                    raise RuntimeError(
                        f"Failed to load hp_tuning_splits_path '{hp_path}': {exc}"
                    ) from exc

        # 2. Handle when no specific split is requested
        if self.split is None:
            if self.num_samples is not None:
                # Shuffle and truncate consistently to match what _create_split_keys does
                all_keys = [s.graph_path for s in samples]
                rng = random.Random(self.seed)
                rng.shuffle(all_keys)
                selected = set(all_keys[: self.num_samples])
                return [s for s in samples if s.graph_path in selected]
            return samples

        # 3. Create or load split keys
        all_keys = [s.graph_path for s in samples]
        split_keys = self._load_or_create_split_keys(all_keys)
        selected = set(split_keys[self.split])
        return [s for s in samples if s.graph_path in selected]

    def _build_samples(self) -> list[GraphSample]:
        samples = self._read_candidate_samples()
        samples = self._apply_split(samples)
        valid_samples = []
        for s in samples:
            if Path(s.graph_path).is_file():
                valid_samples.append(s)
            else:
                print(f"[warning] Skipping missing graph on disk: {s.graph_path}")

        return valid_samples

    def _stable_graph_cache_name(self, graph_path: str) -> str:
        source = Path(graph_path)
        # Stop resolving symlinks to avoid compute node mount issues
        st = source.stat()
        token = f"{source.absolute()}|{st.st_size}|{st.st_mtime_ns}"

        digest = hashlib.sha1(token.encode()).hexdigest()
        return f"{digest}.pt"

    def _cached_graph_path(self, graph_path: str) -> Path:
        if self._cache_graph_dir is None:
            return Path(graph_path)
        return self._cache_graph_dir / self._stable_graph_cache_name(graph_path)

    def _cache_single_graph(self, graph_path: str) -> tuple[str, int]:
        cache_path = self._cached_graph_path(graph_path)
        meta_path = cache_path.with_suffix(".n")

        if cache_path.is_file():
            if meta_path.is_file():
                return str(cache_path), int(meta_path.read_text())
            # .pt exists but sidecar missing (graphs cached before this change):
            # load once to recover num_nodes and write the sidecar for next time.
            cached_obj = torch.load(cache_path, map_location="cpu", weights_only=False)
            num_nodes = int(cached_obj.x.shape[0])
            meta_path.write_text(str(num_nodes))
            return str(cache_path), num_nodes

        source_obj = torch.load(
            Path(graph_path), map_location="cpu", weights_only=False
        )
        num_nodes = int(source_obj.x.shape[0])

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
        torch.save(source_obj, tmp_path)
        tmp_path.replace(cache_path)
        meta_path.write_text(str(num_nodes))
        return str(cache_path), num_nodes

    def _load_manifest(self) -> dict | None:
        if self._manifest_path is None or not self._manifest_path.is_file():
            return None

        try:
            with open(self._manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None

        # Minimal shape guard: the signature-based filename already ensures this
        # manifest belongs to the right seed/split/num_samples/CSV combination.
        # Full per-entry validation was removed — it caused thousands of GPFS
        # stat() calls per trial.
        if not isinstance(manifest, dict) or not isinstance(
            manifest.get("entries"), list
        ):
            return None
        return manifest

    def _rebuild_graph_cache(self) -> dict:
        unique_paths = sorted({sample.graph_path for sample in self.samples})

        # Use threads for parallel I/O: torch.load/save releases the GIL so
        # multiple threads can saturate GPFS bandwidth concurrently.
        # Respect SLURM --cpus-per-task via sched_getaffinity; fall back to
        # cpu_count() on macOS/Windows where it is not present.
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

        def _process_one(graph_path: str) -> tuple[str, str, int]:
            cached_path, num_nodes = self._cache_single_graph(graph_path)
            return graph_path, cached_path, num_nodes

        completed = 0
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            for graph_path, cached_path, num_nodes in executor.map(
                _process_one, unique_paths
            ):
                self._graph_num_nodes_map[graph_path] = num_nodes
                completed += 1
                if completed % 1000 == 0 or completed == len(unique_paths):
                    print(
                        f"[cache] {completed}/{len(unique_paths)} graphs cached",
                        flush=True,
                    )

        entries: list[dict] = [
            {
                "graph_path": sample.graph_path,
                "num_nodes": self._graph_num_nodes_map[sample.graph_path],
            }
            for sample in self.samples
        ]

        return {
            "version": self._MANIFEST_VERSION,
            "num_samples": len(self.samples),
            "entries": entries,
        }

    def _apply_manifest(self, manifest: dict) -> None:
        self._graph_num_nodes_map.clear()
        for entry in manifest["entries"]:
            graph_path = str(entry["graph_path"])
            num_nodes = int(entry["num_nodes"])
            self._graph_num_nodes_map[graph_path] = num_nodes

    def process(self) -> bool:
        """Load or build the graph cache manifest.  Returns True if the manifest
        was freshly rebuilt (first run), False if loaded from disk cache."""
        if self.cache_dir is None:
            return True

        if self._cache_meta_dir is None or self._manifest_path is None:
            return True

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

    def _load_graph_for_sample(self, sample: GraphSample):
        graph_path = sample.graph_path
        if self.cache_dir is not None:
            cached_path = self._cached_graph_path(graph_path)
            if cached_path.is_file():
                graph_path = str(cached_path)
            else:
                rebuilt_cache_path, num_nodes = self._cache_single_graph(graph_path)
                self._graph_num_nodes_map[sample.graph_path] = num_nodes
                graph_path = rebuilt_cache_path

        return torch.load(graph_path, map_location="cpu", weights_only=False)

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int):
        sample = self.samples[idx]
        data_obj = self._load_graph_for_sample(sample)

        # Apply positional encoding transform (attaches to data_obj.pos_enc)
        data_obj = self.pe_transform(data_obj)

        # Keep targets on the Data object for graph-level regression.
        data_obj.y = torch.tensor([[sample.y_node_opt]], dtype=torch.float32)
        return data_obj

    def _sizes_cache_path(self) -> Path | None:
        """Disk cache path for node-size list, keyed by CSV set + split + sample count."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self._cache_signature}_node_sizes.json"

    def _num_nodes_for_sample(self, sample: GraphSample) -> int:
        cached = self._graph_num_nodes_map.get(sample.graph_path)
        if cached is not None:
            return cached

        data_obj = self._load_graph_for_sample(sample)
        num_nodes = int(data_obj.x.shape[0])
        self._graph_num_nodes_map[sample.graph_path] = num_nodes
        return num_nodes

    def _read_sizes_cache(self, sizes_cache_path: Path | None) -> list[int] | None:
        if sizes_cache_path is None or not sizes_cache_path.is_file():
            return None
        try:
            with open(sizes_cache_path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list) and len(data) == len(self.samples):
                return [int(v) for v in data]
        except (json.JSONDecodeError, OSError):
            return None
        return None

    def _write_sizes_cache(
        self, sizes_cache_path: Path | None, sizes: list[int]
    ) -> None:
        if sizes_cache_path is None:
            return
        sizes_cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = sizes_cache_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
        tmp.write_text(json.dumps(sizes), encoding="utf-8")
        os.replace(tmp, sizes_cache_path)

    def get_num_nodes_list(self) -> list[int]:
        """Return per-sample node counts, reusing a disk cache when available."""
        sizes_cache_path = self._sizes_cache_path()

        cached_sizes = self._read_sizes_cache(sizes_cache_path)
        if cached_sizes is not None:
            return cached_sizes

        sizes = [self._num_nodes_for_sample(sample) for sample in self.samples]
        self._write_sizes_cache(sizes_cache_path, sizes)
        return sizes

    def release_runtime_caches(self) -> None:
        """Drop in-memory per-graph metadata once batch planning is complete."""
        self._graph_num_nodes_map.clear()


__all__ = ["AIGGraphRegressionDataset", "GraphSample"]
