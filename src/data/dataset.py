from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from data.dataset_utils import (
    graph_input_path_from_csv_row,
    parse_float,
    parse_int,
)
from src.constants import VALID_ALGORITHMS


@dataclass(frozen=True)
class GraphSample:
    key: str
    graph_path: str
    design: str
    algorithm: str
    tier_id: int
    y_node_opt: float
    y_depth_opt: float


_PE_FIELDS = (None, "level", "pi_paths", "local_sp_sum")


def _get_pe(
    data_obj: object, positional_encoding: Optional[str]
) -> Optional[torch.Tensor]:
    if positional_encoding is None:
        return None
    t = getattr(data_obj, positional_encoding, None)
    if t is None:
        return None
    return (t.unsqueeze(-1) if t.dim() == 1 else t).float()


class AIGGraphRegressionDataset(Dataset):
    """
    Minimal graph-level regression dataset.

    Targets:
    - y[0] = node optimizability
    - y[1] = depth optimizability

    Required graph attributes loaded from .pt:
    - x, edge_index, edge_attr
    - level, pi_paths, local_sp_sum (used to create pos_enc)
    """

    def __init__(
        self,
        csv_path: str | Path,
        graph_root: str | Path,
        *,
        algorithm: Optional[str] = None,
        positional_encoding: Optional[str] = None,
        split: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        num_samples: Optional[int] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.graph_root = Path(graph_root)
        self.algorithm = algorithm
        self.positional_encoding = positional_encoding
        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.split_ratios = split_ratios
        self.seed = int(seed)
        self.num_samples = num_samples

        self.samples = self._build_samples()

    def _read_candidate_samples(self) -> List[GraphSample]:
        df = pd.read_csv(self.csv_path, dtype=str).fillna("")
        if self.algorithm is not None:
            df = df[df["algorithm"].isin(["", self.algorithm])]

        samples = [
            GraphSample(
                key=row["file_path"],
                graph_path=str(graph_input_path_from_csv_row(self.graph_root, row)),
                design=row["design"],
                algorithm=row["algorithm"],
                tier_id=parse_int(row.get("tier_id", "0"), default=0),
                y_node_opt=parse_float(row.get("optimizability", "0"), 0.0),
                y_depth_opt=parse_float(row.get("depth_optimizability", "0"), 0.0),
            )
            for _, row in df.iterrows()
        ]
        return samples

    def _split_signature(self) -> str:
        algo_token = self.algorithm if self.algorithm is not None else "all"
        ratios_token = "-".join(f"{x:.6f}" for x in self.split_ratios)
        return f"algo={algo_token}|ratios={ratios_token}|seed={self.seed}"

    def _load_or_create_split_keys(self, all_keys: List[str]) -> Dict[str, List[str]]:
        if self.cache_dir is None or self.split is None:
            return self._create_split_keys(all_keys)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{self.split}.json"
        if cache_file.is_file():
            with open(cache_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("signature") == self._split_signature():
                splits = payload.get("splits", {})
                if all(name in splits for name in ("train", "val", "test")):
                    return {
                        "train": list(splits["train"]),
                        "val": list(splits["val"]),
                        "test": list(splits["test"]),
                    }

        split_keys = self._create_split_keys(all_keys)
        payload = {"signature": self._split_signature(), "splits": split_keys}
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return split_keys

    def _create_split_keys(self, all_keys: List[str]) -> Dict[str, List[str]]:
        train_ratio, val_ratio, test_ratio = self.split_ratios
        ratio_sum = train_ratio + val_ratio + test_ratio
        if ratio_sum <= 0:
            raise ValueError("split_ratios must sum to a positive value")

        keys = list(all_keys)
        rng = random.Random(self.seed)
        rng.shuffle(keys)

        train_ratio = train_ratio / ratio_sum
        val_ratio = val_ratio / ratio_sum
        # test takes the remainder for exact partitioning.

        n = len(keys)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = max(0, n - n_train - n_val)

        train_keys = keys[:n_train]
        val_keys = keys[n_train : n_train + n_val]
        test_keys = keys[n_train + n_val : n_train + n_val + n_test]
        return {"train": train_keys, "val": val_keys, "test": test_keys}

    def _apply_split(self, samples: List[GraphSample]) -> List[GraphSample]:
        if self.split is None:
            return samples

        all_keys = [s.key for s in samples]
        split_keys = self._load_or_create_split_keys(all_keys)
        selected = set(split_keys[self.split])
        return [s for s in samples if s.key in selected]

    def _build_samples(self) -> List[GraphSample]:
        samples = self._read_candidate_samples()
        samples = self._apply_split(samples)
        samples = self._apply_num_samples(samples)
        return samples

    def _apply_num_samples(self, samples: List[GraphSample]) -> List[GraphSample]:
        if self.num_samples is None:
            return samples
        rng = random.Random(self.seed)
        if self.algorithm is not None:
            k = min(self.num_samples, len(samples))
            return rng.sample(samples, k)
        # Mixed: sample num_samples // 4 from each algorithm.
        per_algo = self.num_samples // len(VALID_ALGORITHMS)
        grouped: Dict[str, List[GraphSample]] = {a: [] for a in VALID_ALGORITHMS}
        for s in samples:
            if s.algorithm in grouped:
                grouped[s.algorithm].append(s)
        selected: List[GraphSample] = []
        for algo, pool in grouped.items():
            selected.extend(rng.sample(pool, min(per_algo, len(pool))))
        rng.shuffle(selected)
        return selected

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        data_obj = torch.load(sample.graph_path, map_location="cpu", weights_only=False)

        # Edge features are required in this project.
        if getattr(data_obj, "edge_attr", None) is None:
            raise ValueError(f"Graph is missing edge_attr: {sample.graph_path}")

        data_obj.pos_enc = _get_pe(data_obj, self.positional_encoding)

        # Keep targets on the Data object for graph-level multi-target regression.
        data_obj.y = torch.tensor(
            [sample.y_node_opt, sample.y_depth_opt], dtype=torch.float32
        )

        # Keep metadata for filtering/debugging/analysis.
        data_obj.design = sample.design
        data_obj.algorithm = sample.algorithm
        data_obj.tier_id = sample.tier_id
        data_obj.sample_key = sample.key
        return data_obj


__all__ = ["AIGGraphRegressionDataset", "GraphSample"]
