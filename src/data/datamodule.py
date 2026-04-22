from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from data.dataset import AIGGraphRegressionDataset


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
        self.train_num_samples = train_num_samples
        self.hp_tuning_splits_path = hp_tuning_splits_path

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

        if stage in ("test", None):
            # Pass the same limit to test
            self.test_ds = self._make_dataset("test", self.train_num_samples)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )


__all__ = ["AIGDataModule"]
