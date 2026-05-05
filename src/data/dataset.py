from __future__ import annotations

import hashlib
import json
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch_geometric.data import Dataset as PyGDataset

# Factory transform for handling various PE types (level, pi_paths, sinusoidal, etc.)
try:
    from models.layers.positional_encodings import get_pe_transform
except ImportError:
    # Fallback for different execution environments
    from src.models.layers.positional_encodings import get_pe_transform


@dataclass(frozen=True)
class GraphSample:
    graph_path: str
    y_node_opt: float


# Module-level cache for raw CSV samples. The CSVs do not change during a run,
# so re-reading and re-parsing them on every trial (2× per trial for train+val)
# is pure waste.  Key: tuple of resolved CSV path strings.
_CSV_SAMPLE_CACHE: Dict[Tuple[str, ...], List[GraphSample]] = {}

# Module-level cache for the cache signature (keyed by same CSV key + run params).
# Avoids 4 GPFS stat() calls per dataset instantiation after the first.
_SIGNATURE_CACHE: Dict[Tuple, str] = {}

# Module-level cache for splits JSON (keyed by resolved cache_file path string).
# The splits file for a given algo+num_samples tag never changes once written.
_SPLITS_CACHE: Dict[str, Dict[str, List[str]]] = {}

# Module-level cache for manifest dicts (keyed by resolved manifest path string).
# The manifest for a given cache_signature never changes once written.
_MANIFEST_CACHE: Dict[str, dict] = {}


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
    def raw_file_names(self) -> List[str]:
        # This dataset reads existing .pt paths from CSV and has no raw download step.
        return []

    @property
    def processed_file_names(self) -> List[str]:
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
        csv_paths: str | Path | List[str | Path],
        *,
        positional_encoding: Optional[str] = None,
        split: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        num_samples: Optional[int] = None,
        num_workers: int = 0,
        hp_tuning_splits_path: Optional[str | Path] = None,
    ) -> None:
        if isinstance(csv_paths, (str, Path)):
            self.csv_paths = [Path(csv_paths)]
        else:
            self.csv_paths = [Path(p) for p in csv_paths]

        self.positional_encoding = positional_encoding
        self.split = split
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
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
        self._graph_cache_map: Dict[str, str] = {}
        self._graph_num_nodes_map: Dict[str, int] = {}

        # Initialize the PE transform factory
        self.pe_transform = get_pe_transform(
            pe_type=self.positional_encoding, attr_name="pos_enc"
        )

        self.samples = self._build_samples()
        # Pass a real root so PyG never falls back to the "???" sentinel.  All
        # actual I/O still uses our own cache_dir paths; this only prevents the
        # stray "???" directory from being created in the working directory when
        # the dataset is used without a cache_dir.
        pyg_root = str(self._cache_meta_dir) if self._cache_meta_dir is not None else None
        super().__init__(root=pyg_root)

        # When cache_dir is present, process() returns True on first run (manifest
        # freshly built) and False on subsequent runs (loaded from disk cache —
        # graphs were already validated on the first run so no re-validation needed).
        # When there is no cache_dir we always validate to catch corrupt graphs early.
        manifest_was_rebuilt = self.cache_dir is None
        if self.cache_dir is not None:
            manifest_was_rebuilt = self.process()

        # Verify the first sample when running for the first time (fresh manifest
        # or no cache_dir). Skipped on reload from disk to avoid redundant I/O.
        if manifest_was_rebuilt:
            self._verify_first_sample(self.samples)

    def _build_cache_signature(self) -> str:
        sig_key = (
            self.seed,
            self.split,
            self.num_samples,
            tuple(self.split_ratios),
            tuple(str(p.resolve()) for p in sorted(self.csv_paths)),
            str(Path(self.hp_tuning_splits_path).resolve()) if self.hp_tuning_splits_path is not None else None,
        )
        if sig_key in _SIGNATURE_CACHE:
            return _SIGNATURE_CACHE[sig_key]

        hasher = hashlib.sha1()
        hasher.update(str(self.seed).encode())
        hasher.update(str(self.split).encode())
        hasher.update(str(self.num_samples).encode())
        hasher.update("|".join(map(str, self.split_ratios)).encode())

        for csv_path in sorted(self.csv_paths):
            resolved = csv_path.resolve()
            hasher.update(str(resolved).encode())
            try:
                st = resolved.stat()
                hasher.update(str(st.st_size).encode())
                hasher.update(str(st.st_mtime_ns).encode())
            except OSError:
                pass

        if self.hp_tuning_splits_path is not None:
            hp = Path(self.hp_tuning_splits_path).resolve()
            hasher.update(str(hp).encode())
            try:
                st = hp.stat()
                hasher.update(str(st.st_size).encode())
                hasher.update(str(st.st_mtime_ns).encode())
            except OSError:
                pass

        sig = hasher.hexdigest()[:16]
        _SIGNATURE_CACHE[sig_key] = sig
        return sig

    def _read_candidate_samples(self) -> List[GraphSample]:
        cache_key = tuple(str(p.resolve()) for p in self.csv_paths)
        if cache_key in _CSV_SAMPLE_CACHE:
            return _CSV_SAMPLE_CACHE[cache_key]

        df = pd.concat(
            [pd.read_csv(p, dtype=str).fillna("") for p in self.csv_paths],
            ignore_index=True,
        )
        df["optimizability"] = df["optimizability"].astype(float)

        samples = [
            GraphSample(
                graph_path=row["unoptimized_graph_path"],
                y_node_opt=row["optimizability"],
            )
            for row in df.to_dict("records")
        ]
        _CSV_SAMPLE_CACHE[cache_key] = samples
        return samples

    def _load_or_create_split_keys(self, all_keys: List[str]) -> Dict[str, List[str]]:
        if self.cache_dir is None or self.split is None:
            return self._create_split_keys(all_keys)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        algo_tag = "_".join(p.stem for p in self.csv_paths)

        # Include num_samples in the cache filename so it doesn't collide with full datasets
        sample_tag = f"_{self.num_samples}" if self.num_samples is not None else "_all"
        cache_file = self.cache_dir / f"{algo_tag}{sample_tag}_splits.json"
        cache_key = str(cache_file.resolve())

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
        temp_file.write_text(json.dumps(split_keys, indent=2, sort_keys=True))
        temp_file.rename(cache_file)
        _SPLITS_CACHE[cache_key] = split_keys
        return split_keys

    def _create_split_keys(self, all_keys: List[str]) -> Dict[str, List[str]]:
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

    def _apply_split(self, samples: List[GraphSample]) -> List[GraphSample]:
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
                except (json.JSONDecodeError, OSError):
                    pass

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

    def _build_samples(self) -> List[GraphSample]:
        samples = self._read_candidate_samples()
        samples = self._apply_split(samples)
        return samples

    def _stable_graph_cache_name(self, graph_path: str) -> str:
        source = Path(graph_path)
        resolved = source.resolve()
        token = str(resolved)
        try:
            st = resolved.stat()
            token = f"{token}|{st.st_size}|{st.st_mtime_ns}"
        except OSError:
            token = f"{token}|missing"

        digest = hashlib.sha1(token.encode()).hexdigest()
        return f"{digest}.pt"

    def _cached_graph_path(self, graph_path: str) -> Path:
        if self._cache_graph_dir is None:
            return Path(graph_path)
        return self._cache_graph_dir / self._stable_graph_cache_name(graph_path)

    def _validate_graph(self, data_obj, graph_path: str) -> int:
        if data_obj.x.dim() != 2:
            raise AssertionError(f"x should be 2D, got shape {data_obj.x.shape}")
        if data_obj.edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index should be [2, E], got {data_obj.edge_index.shape}"
            )

        edge_attr = getattr(data_obj, "edge_attr", None)
        if edge_attr is None:
            raise ValueError(f"edge_attr=None in {graph_path}")
        if edge_attr.dim() != 2:
            raise ValueError("edge_attr must be 2D")

        return int(data_obj.x.shape[0])

    def _cache_single_graph(self, graph_path: str) -> Tuple[str, int]:
        cache_path = self._cached_graph_path(graph_path)
        source_path = Path(graph_path)

        if cache_path.is_file():
            try:
                cached_obj = torch.load(
                    cache_path, map_location="cpu", weights_only=False
                )
                num_nodes = self._validate_graph(cached_obj, str(cache_path))
                return str(cache_path), num_nodes
            except Exception:
                cache_path.unlink(missing_ok=True)

        source_obj = torch.load(source_path, map_location="cpu", weights_only=False)
        num_nodes = self._validate_graph(source_obj, str(source_path))

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
        torch.save(source_obj, tmp_path)
        tmp_path.replace(cache_path)
        return str(cache_path), num_nodes

    def _load_manifest(self) -> Optional[dict]:
        if self._manifest_path is None or not self._manifest_path.is_file():
            return None

        cache_key = str(self._manifest_path.resolve())
        if cache_key in _MANIFEST_CACHE:
            return _MANIFEST_CACHE[cache_key]

        try:
            with open(self._manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None

        # Minimal shape guard: the signature-based filename already ensures this
        # manifest belongs to the right seed/split/num_samples/CSV combination.
        # Full per-entry validation was removed — it caused thousands of GPFS
        # stat() calls per trial.
        if not isinstance(manifest, dict) or not isinstance(manifest.get("entries"), list):
            return None

        _MANIFEST_CACHE[cache_key] = manifest
        return manifest

    def _rebuild_graph_cache(self) -> dict:
        unique_paths = sorted({sample.graph_path for sample in self.samples})
        by_graph: Dict[str, Dict[str, int | str]] = {}

        for graph_path in unique_paths:
            cached_path, num_nodes = self._cache_single_graph(graph_path)
            by_graph[graph_path] = {
                "cache_path": cached_path,
                "num_nodes": num_nodes,
            }

        entries: List[dict] = []
        for sample in self.samples:
            cached = by_graph[sample.graph_path]
            entries.append(
                {
                    "graph_path": sample.graph_path,
                    "cache_path": cached["cache_path"],
                    "num_nodes": cached["num_nodes"],
                }
            )

        return {
            "version": self._MANIFEST_VERSION,
            "num_samples": len(self.samples),
            "entries": entries,
        }

    def _apply_manifest(self, manifest: dict) -> None:
        self._graph_cache_map.clear()
        self._graph_num_nodes_map.clear()
        for entry in manifest["entries"]:
            graph_path = str(entry["graph_path"])
            cache_path = str(entry["cache_path"])
            num_nodes = int(entry["num_nodes"])
            self._graph_cache_map[graph_path] = cache_path
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
            cached_path = self._graph_cache_map.get(graph_path)
            if cached_path is None:
                rebuilt_cache_path, num_nodes = self._cache_single_graph(graph_path)
                self._graph_cache_map[graph_path] = rebuilt_cache_path
                self._graph_num_nodes_map[graph_path] = num_nodes
                cached_path = rebuilt_cache_path
            graph_path = cached_path

        try:
            return torch.load(graph_path, map_location="cpu", weights_only=False)
        except (FileNotFoundError, OSError):
            # Cached file was deleted from under us — rebuild and retry once.
            rebuilt_cache_path, num_nodes = self._cache_single_graph(sample.graph_path)
            self._graph_cache_map[sample.graph_path] = rebuilt_cache_path
            self._graph_num_nodes_map[sample.graph_path] = num_nodes
            return torch.load(rebuilt_cache_path, map_location="cpu", weights_only=False)

    def _verify_first_sample(self, samples: List[GraphSample]) -> None:
        if not samples:
            return
        data_obj = self._load_graph_for_sample(samples[0])
        self._validate_graph(data_obj, samples[0].graph_path)

        # Validate Positional Encoding attachment
        if (
            self.positional_encoding is not None
            and self.positional_encoding.lower() != "none"
        ):
            # Apply the transform to test if it correctly attaches 'pos_enc'
            data_obj = self.pe_transform(data_obj)
            pe = getattr(data_obj, "pos_enc", None)

            if pe is None:
                raise ValueError(
                    f"Transform failed to find/attach PE type '{self.positional_encoding}' "
                    f"for graph {samples[0].graph_path}"
                )

            if pe.dim() != 2 or pe.shape[0] != data_obj.x.shape[0]:
                raise ValueError(
                    "pos_enc should be 2D with N rows, got "
                    f"{pe.shape if pe is not None else None}"
                )

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int):
        sample = self.samples[idx]
        data_obj = self._load_graph_for_sample(sample)
        # Validation already occurred at startup via _verify_first_sample() and
        # process(); do not repeat it on every item fetch in the training hot path.

        # Apply positional encoding transform (attaches to data_obj.pos_enc)
        data_obj = self.pe_transform(data_obj)

        # Keep targets on the Data object for graph-level regression.
        data_obj.y = torch.tensor([[sample.y_node_opt]], dtype=torch.float32)
        return data_obj

    def _sizes_cache_path(self) -> Optional[Path]:
        """Disk cache path for node-size list, keyed by CSV set + sample count."""
        if self.cache_dir is None:
            return None
        algo_tag = "_".join(p.stem for p in self.csv_paths)
        sample_tag = f"_{self.num_samples}" if self.num_samples is not None else "_all"
        return self.cache_dir / f"{algo_tag}{sample_tag}_node_sizes.json"

    def _num_nodes_for_sample(self, sample: GraphSample) -> int:
        cached = self._graph_num_nodes_map.get(sample.graph_path)
        if cached is not None:
            return cached

        data_obj = self._load_graph_for_sample(sample)
        num_nodes = int(data_obj.x.shape[0])
        self._graph_num_nodes_map[sample.graph_path] = num_nodes
        return num_nodes

    def get_num_nodes_list(self) -> List[int]:
        """Return per-sample node counts.  First call computes and caches to disk;
        subsequent calls (even in different processes / workers / trials) load in
        < 1 s from a ~300 KB JSON file rather than scanning 50 K .pt files."""
        sizes_cache_path = self._sizes_cache_path()

        # --- fast path: disk cache hit ---
        if sizes_cache_path is not None and sizes_cache_path.is_file():
            try:
                with open(sizes_cache_path, encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list) and len(data) == len(self.samples):
                    return data
            except (json.JSONDecodeError, OSError):
                pass  # fall through to recompute

        # --- slow path: load every .pt file (uses in-process dict cache) ---
        sizes = [self._num_nodes_for_sample(sample) for sample in self.samples]

        # --- persist to disk for all future processes/workers/trials ---
        if sizes_cache_path is not None:
            sizes_cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = sizes_cache_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
            tmp.write_text(json.dumps(sizes))
            tmp.rename(sizes_cache_path)  # atomic on Linux

        return sizes


__all__ = ["AIGGraphRegressionDataset", "GraphSample"]
