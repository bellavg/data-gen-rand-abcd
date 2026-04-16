from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class GraphSample:
    graph_path: str
    y_node_opt: float


def _get_pe(
    data_obj: object, positional_encoding: Optional[str]
) -> Optional[torch.Tensor]:
    if positional_encoding is None:
        return None

    t = getattr(data_obj, positional_encoding, None)
    if t is None:
        return None

    t = t.unsqueeze(-1) if t.dim() == 1 else t

    # Keep discrete attributes as long/int for nn.Embedding
    if positional_encoding == "level":
        return t.long()

    # Otherwise continuous attributes become floats
    return t.float()


class AIGGraphRegressionDataset(Dataset):
    """
    Minimal graph-level regression dataset.

    Targets:
    - y[0] = node optimizability

    Required graph attributes loaded from .pt:
    - x, edge_index, edge_attr, level, pi_paths, local_sp_sum
    """

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

        self.samples = self._build_samples()

    def _read_candidate_samples(self) -> List[GraphSample]:
        df = pd.concat(
            [pd.read_csv(p, dtype=str).fillna("") for p in self.csv_paths],
            ignore_index=True,
        )
        df["optimizability"] = df["optimizability"].astype(float)

        return [
            GraphSample(
                graph_path=row["unoptimized_graph_path"],
                y_node_opt=row["optimizability"],
            )
            for row in df.to_dict("records")
        ]

    def _load_or_create_split_keys(self, all_keys: List[str]) -> Dict[str, List[str]]:
        if self.cache_dir is None or self.split is None:
            return self._create_split_keys(all_keys)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        algo_tag = "_".join(p.stem for p in self.csv_paths)
        cache_file = self.cache_dir / f"{algo_tag}_splits.json"
        if cache_file.is_file():
            splits = json.loads(cache_file.read_text())
            if all(name in splits for name in ("train", "val", "test")):
                return splits

        split_keys = self._create_split_keys(all_keys)
        cache_file.write_text(json.dumps(split_keys, indent=2, sort_keys=True))
        return split_keys

    def _create_split_keys(self, all_keys: List[str]) -> Dict[str, List[str]]:
        keys = list(all_keys)
        rng = random.Random(self.seed)
        rng.shuffle(keys)

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
        if self.split is None:
            return samples
        all_keys = [s.graph_path for s in samples]
        split_keys = self._load_or_create_split_keys(all_keys)
        selected = set(split_keys[self.split])
        return [s for s in samples if s.graph_path in selected]

    def _build_samples(self) -> List[GraphSample]:
        samples = self._read_candidate_samples()
        samples = self._apply_split(samples)
        samples = self._apply_num_samples(samples)
        self._verify_first_sample(samples)
        return samples

    def _verify_first_sample(self, samples: List[GraphSample]) -> None:
        if not samples:
            return
        data_obj = torch.load(
            samples[0].graph_path, map_location="cpu", weights_only=False
        )
        if data_obj.x.dim() != 2:
            raise AssertionError(f"x should be 2D, got shape {data_obj.x.shape}")
        if data_obj.edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index should be [2, E], got {data_obj.edge_index.shape}"
            )

        # Strict early validation: require edge_attr present and 2D at init time.
        edge_attr = getattr(data_obj, "edge_attr", None)
        if edge_attr is None:
            raise ValueError(f"edge_attr=None in {samples[0].graph_path}")
        if edge_attr.dim() != 2:
            raise ValueError("edge_attr must be 2D")

        if self.positional_encoding is not None:
            pe = _get_pe(data_obj, self.positional_encoding)
            if pe is None or pe.dim() != 2 or pe.shape[0] != data_obj.x.shape[0]:
                raise ValueError(
                    "pos_enc should be 2D with N rows, got "
                    f"{pe.shape if pe is not None else None}"
                )

    def _apply_num_samples(self, samples: List[GraphSample]) -> List[GraphSample]:
        if self.num_samples is None:
            return samples
        rng = random.Random(self.seed)
        return rng.sample(samples, min(self.num_samples, len(samples)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        data_obj = torch.load(sample.graph_path, map_location="cpu", weights_only=False)

        edge_attr = getattr(data_obj, "edge_attr", None)
        if edge_attr is None:
            raise ValueError(
                f"Loaded graph has edge_attr=None, which is not allowed: {sample.graph_path}"
            )
        if edge_attr.dim() != 2:
            raise ValueError(
                f"Loaded graph edge_attr must be 2D, got {tuple(edge_attr.shape)}: {sample.graph_path}"
            )

        data_obj.pos_enc = _get_pe(data_obj, self.positional_encoding)

        # Keep targets on the Data object for graph-level regression.
        data_obj.y = torch.tensor([[sample.y_node_opt]], dtype=torch.float32)
        return data_obj


__all__ = ["AIGGraphRegressionDataset", "GraphSample"]
