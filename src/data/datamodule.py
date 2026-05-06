from __future__ import annotations

import random
import warnings
from pathlib import Path

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
        sizes: list[int],
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
        batches: list[list[int]] = []

        while left <= right:
            batch: list[int] = []
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
        csv_paths: str | Path | list[str | Path],
        *,
        positional_encoding: str | None = None,
        cache_dir: str | Path | None = None,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        batch_size: int = 32,
        num_workers: int = 6,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        prefetch_factor: int = 1,
        dynamic_batching: bool = False,
        train_num_samples: int | None = None,
        test_num_samples: int | None = None,
        use_full_test_set: bool = False,
        hp_tuning_splits_path: str | Path | None = None,
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
        self.test_num_samples = test_num_samples
        self.use_full_test_set = use_full_test_set
        self.hp_tuning_splits_path = hp_tuning_splits_path

        if self.num_workers > 0 and self.prefetch_factor < 1:
            raise ValueError(
                f"prefetch_factor must be >= 1 when num_workers > 0, got {self.prefetch_factor}"
            )
        if self.num_workers <= 0 and self.prefetch_factor != 1:
            warnings.warn(
                "prefetch_factor is ignored when num_workers == 0; forcing it to 1.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.prefetch_factor = 1

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

    def _make_dataset(self, split: str, num_samples: int | None = None) -> AIGGraphRegressionDataset:
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

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_ds = self._make_dataset("train", self.train_num_samples)
            # Pass the same limit to val so it knows to use the 10k cache!
            self.val_ds = self._make_dataset("val", self.train_num_samples)
            # Pre-compute node sizes once per trial (result is disk-cached so
            # subsequent trials/workers pay < 1 s instead of scanning 50 K files).
            if self.dynamic_batching:
                self._train_sizes: list[int] = self.train_ds.get_num_nodes_list()

        elif stage == "validate":
            self.val_ds = self._make_dataset("val", self.train_num_samples)

        if stage in ("test", None):
            # Backward-compatible default: mirror train_num_samples when no
            # explicit test limit is provided. Set use_full_test_set=True to
            # evaluate the complete test split.
            if self.use_full_test_set:
                test_limit = None
            elif self.test_num_samples is not None:
                test_limit = self.test_num_samples
            else:
                test_limit = self.train_num_samples
            self.test_ds = self._make_dataset("test", test_limit)

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

        return DataLoader(
            self.train_ds,
            shuffle=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            shuffle=False,
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            shuffle=False,
            **self._loader_kwargs(),
        )


__all__ = ["AIGDataModule", "BalancedDynamicBatchSampler"]
