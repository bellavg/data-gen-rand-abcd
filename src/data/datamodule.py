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
        num_workers: int = 0,
        train_num_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.csv_paths = csv_paths
        self.positional_encoding = positional_encoding
        self.cache_dir = cache_dir
        self.split_ratios = split_ratios
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_num_samples = train_num_samples

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
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.train_ds = self._make_dataset("train", self.train_num_samples)
            self.val_ds = self._make_dataset("val")
        if stage in ("test", None):
            self.test_ds = self._make_dataset("test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


__all__ = ["AIGDataModule"]
