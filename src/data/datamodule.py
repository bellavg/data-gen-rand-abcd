from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from data.dataset import AIGGraphRegressionDataset


class BalancedDynamicBatchSampler:
    """Build fixed-cardinality batches that pair large and small graphs.

    Batch size stays equal to the tuned `batch_size`, while ordering is chosen
    to reduce collisions of multiple very large graphs in the same batch.
    """

    def __init__(
        self,
        sizes: List[int],
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.sizes = sizes
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        n = len(self.sizes)
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        indices = sorted(range(len(self.sizes)), key=lambda i: self.sizes[i])
        left = 0
        right = len(indices) - 1
        batches: List[List[int]] = []

        while left <= right:
            batch: List[int] = []
            take_large = True
            while left <= right and len(batch) < self.batch_size:
                if take_large:
                    batch.append(indices[right])
                    right -= 1
                else:
                    batch.append(indices[left])
                    left += 1
                take_large = not take_large
            batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        for batch in batches:
            yield batch


class AIGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_paths: str | Path | List[str | Path],
        *,
        positional_encoding: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        batch_size: int = 32,
        num_workers: int = 6,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        prefetch_factor: int = 1,
        dynamic_batching: bool = False,
        train_num_samples: Optional[int] = None,
        hp_tuning_splits_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        self.csv_paths = csv_paths
        self.positional_encoding = positional_encoding
        self.cache_dir = cache_dir
        self.split_ratios = split_ratios
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.dynamic_batching = dynamic_batching
        self.train_num_samples = train_num_samples
        self.hp_tuning_splits_path = hp_tuning_splits_path

    def _loader_kwargs(self, *, include_batch_size: bool = True) -> dict:
        kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if include_batch_size:
            kwargs["batch_size"] = self.batch_size
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def _make_dataset(self, split: str, num_samples: Optional[int] = None):
        return AIGGraphRegressionDataset(
            self.csv_paths,
            positional_encoding=self.positional_encoding,
            split=split,
            cache_dir=self.cache_dir,
            split_ratios=self.split_ratios,
            seed=self.seed,
            num_samples=num_samples,
            num_workers=self.num_workers,
            hp_tuning_splits_path=self.hp_tuning_splits_path,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.train_ds = self._make_dataset("train", self.train_num_samples)
            # Pass the same limit to val so it knows to use the 10k cache!
            self.val_ds = self._make_dataset("val", self.train_num_samples)
            # Pre-compute node sizes once per trial (result is disk-cached so
            # subsequent trials/workers pay < 1 s instead of scanning 50 K files).
            if self.dynamic_batching:
                self._train_sizes: List[int] = self.train_ds.get_num_nodes_list()

        if stage in ("test", None):
            # Pass the same limit to test
            self.test_ds = self._make_dataset("test", self.train_num_samples)

    def train_dataloader(self) -> DataLoader:
        if self.dynamic_batching:
            sizes = getattr(self, "_train_sizes", None)
            if sizes is None:
                sizes = self.train_ds.get_num_nodes_list()
            sampler = BalancedDynamicBatchSampler(
                sizes,
                batch_size=self.batch_size,
                shuffle=True,
                seed=self.seed,
            )
            return DataLoader(
                self.train_ds,
                batch_sampler=sampler,
                **self._loader_kwargs(include_batch_size=False),
            )

        return DataLoader(self.train_ds, shuffle=True, **self._loader_kwargs())

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, shuffle=False, **self._loader_kwargs())

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, shuffle=False, **self._loader_kwargs())


__all__ = ["AIGDataModule"]
