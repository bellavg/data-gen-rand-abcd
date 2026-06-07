from __future__ import annotations

import warnings
from pathlib import Path

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from data.dataset import AIGGraphRegressionDataset
from data.sampler import (
    BalancedDynamicBatchSampler,
    batch_plan_cache_path,
    load_or_build_batch_plan,
)


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
        dynamic_bucket_rules: list[tuple[int, int]] | None = None,
        train_num_samples: int | None = None,
        test_num_samples: int | None = None,
        hp_tuning_splits_path: str | Path | None = None,
        tier0_cache_dir: str | Path | None = None,
        tier1_cache_dir: str | Path | None = None,
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
        self.dynamic_bucket_rules = BalancedDynamicBatchSampler._normalize_bucket_rules(
            dynamic_bucket_rules,
            self.batch_size,
        )
        self.train_num_samples = train_num_samples
        self.test_num_samples = test_num_samples
        self.hp_tuning_splits_path = hp_tuning_splits_path
        self.tier0_cache_dir = tier0_cache_dir
        self.tier1_cache_dir = tier1_cache_dir

        if self.num_workers > 0 and self.prefetch_factor < 1:
            raise ValueError(
                f"prefetch_factor must be >= 1 when num_workers > 0, got {self.prefetch_factor}"
            )
        elif self.num_workers == 0 and self.prefetch_factor != 1:
            warnings.warn(
                "prefetch_factor is ignored when num_workers == 0; forcing it to 1.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.prefetch_factor = 1

    def _dynamic_batch_plan_cache_path(self) -> Path | None:
        if not self.dynamic_batching:
            return None
        sig = getattr(getattr(self, "train_ds", None), "_cache_signature", None)
        return batch_plan_cache_path(
            sig,
            cache_dir=self.cache_dir,
            batch_size=self.batch_size,
            bucket_rules=self.dynamic_bucket_rules,
        )

    def _loader_kwargs(
        self,
        *,
        include_batch_size: bool = True,
        is_train: bool = False,
    ) -> dict:
        kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if include_batch_size:
            kwargs["batch_size"] = self.batch_size
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers if is_train else False
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def _make_dataset(
        self, split: str, num_samples: int | None = None
    ) -> AIGGraphRegressionDataset:
        return AIGGraphRegressionDataset(
            self.csv_paths,
            positional_encoding=self.positional_encoding,
            split=split,
            cache_dir=self.cache_dir,
            tier0_cache_dir=self.tier0_cache_dir,
            tier1_cache_dir=self.tier1_cache_dir,
            split_ratios=self.split_ratios,
            seed=self.seed,
            num_samples=num_samples,
            num_workers=self.num_workers,
            hp_tuning_splits_path=self.hp_tuning_splits_path,
        )

    def _ensure_val_plan(self) -> None:
        if not (self.dynamic_batching and self.dynamic_bucket_rules):
            return
        val_sizes = self.val_ds.get_num_nodes_list()
        self._val_batch_plan: list[list[int]] = (
            BalancedDynamicBatchSampler.build_batch_plan(
                val_sizes,
                batch_size=self.batch_size,
                bucket_rules=self.dynamic_bucket_rules,
            )
        )
        self.val_ds.release_runtime_caches()

    def _make_bucketed_dataloader(
        self,
        ds,
        plan: list[list[int]],
        *,
        shuffle: bool,
        is_train: bool = False,
    ) -> DataLoader:
        sampler = BalancedDynamicBatchSampler(
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=self.seed,
            bucket_rules=self.dynamic_bucket_rules,
            precomputed_batches=plan,
        )
        return DataLoader(
            ds,
            batch_sampler=sampler,
            **self._loader_kwargs(include_batch_size=False, is_train=is_train),
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_ds = self._make_dataset("train", self.train_num_samples)
            self.val_ds = self._make_dataset("val", self.train_num_samples)
            if self.dynamic_batching and self.dynamic_bucket_rules:
                train_sizes = self.train_ds.get_num_nodes_list()
                self._train_batch_plan: list[list[int]] = load_or_build_batch_plan(
                    train_sizes,
                    batch_size=self.batch_size,
                    bucket_rules=self.dynamic_bucket_rules,
                    cache_path=self._dynamic_batch_plan_cache_path(),
                )
                self.train_ds.release_runtime_caches()
                self._ensure_val_plan()

        elif stage == "validate":
            self.val_ds = self._make_dataset("val", self.train_num_samples)
            self._ensure_val_plan()

        if stage in ("test", None):
            self.test_ds = self._make_dataset("test", self.test_num_samples)

    def train_dataloader(self) -> DataLoader:
        if self.dynamic_batching and self.dynamic_bucket_rules:
            plan = getattr(self, "_train_batch_plan", None)
            if plan is None:
                plan = load_or_build_batch_plan(
                    self.train_ds.get_num_nodes_list(),
                    batch_size=self.batch_size,
                    bucket_rules=self.dynamic_bucket_rules,
                    cache_path=self._dynamic_batch_plan_cache_path(),
                )
            self._train_batch_plan = None
            return self._make_bucketed_dataloader(
                self.train_ds,
                plan,
                shuffle=True,
                is_train=True,
            )
        return DataLoader(
            self.train_ds,
            shuffle=True,
            **self._loader_kwargs(is_train=True),
        )

    def val_dataloader(self) -> DataLoader:
        if self.dynamic_batching and self.dynamic_bucket_rules:
            precomputed = getattr(self, "_val_batch_plan", None)
            if precomputed is None:
                precomputed = BalancedDynamicBatchSampler.build_batch_plan(
                    self.val_ds.get_num_nodes_list(),
                    batch_size=self.batch_size,
                    bucket_rules=self.dynamic_bucket_rules,
                )
                self._val_batch_plan = precomputed
            return self._make_bucketed_dataloader(
                self.val_ds,
                precomputed,
                shuffle=False,
                is_train=False,
            )
        return DataLoader(
            self.val_ds,
            shuffle=False,
            **self._loader_kwargs(is_train=False),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            shuffle=False,
            **self._loader_kwargs(is_train=False),
        )


__all__ = ["AIGDataModule"]
