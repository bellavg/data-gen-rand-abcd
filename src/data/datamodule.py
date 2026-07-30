from __future__ import annotations

import warnings
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

import config
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
        sparsification: str | None = None,
        partition: str | None = None,
        normalize_edges: bool = config.NORMALIZE_EDGES,
        cache_dir: str | Path | None = None,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_by: str = config.SPLIT_BY,
        seed: int = 42,
        batch_size: int = 32,
        num_workers: int = config.NUM_WORKERS,
        persistent_workers: bool = config.PERSISTENT_WORKERS,
        pin_memory: bool = config.PIN_MEMORY,
        prefetch_factor: int = config.PREFETCH_FACTOR,
        dynamic_batching: bool = False,
        max_total_nodes: int = config.MAX_TOTAL_NODES_PER_BATCH,
        train_num_samples: int | None = None,
        test_num_samples: int | None = None,
        hp_tuning_splits_path: str | Path | None = None,
        tier0_cache_dir: str | Path | None = None,
        tier1_cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.csv_paths = csv_paths
        self.positional_encoding = positional_encoding
        self.sparsification = sparsification
        self.partition = partition
        self.normalize_edges = bool(normalize_edges)
        self.cache_dir = cache_dir
        self.split_ratios = split_ratios
        self.split_by = split_by
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.dynamic_batching = dynamic_batching
        self.max_total_nodes = max(1, int(max_total_nodes))
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
            max_total_nodes=self.max_total_nodes,
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
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def _make_dataset(
        self, split: str, num_samples: int | None = None
    ) -> AIGGraphRegressionDataset:
        return AIGGraphRegressionDataset(
            self.csv_paths,
            positional_encoding=self.positional_encoding,
            sparsification=self.sparsification,
            sparsification_replace_path=getattr(config, "SPARSIFICATION_REPLACE_PATH", None),
            partition=self.partition,
            normalize_edges=self.normalize_edges,
            split=split,
            cache_dir=self.cache_dir,
            tier0_cache_dir=self.tier0_cache_dir,
            tier1_cache_dir=self.tier1_cache_dir,
            split_ratios=self.split_ratios,
            split_by=self.split_by,
            seed=self.seed,
            num_samples=num_samples,
            num_workers=self.num_workers,
            hp_tuning_splits_path=self.hp_tuning_splits_path,
        )

    def _ensure_val_plan(self) -> None:
        if not self.dynamic_batching:
            return
        val_sizes = self.val_ds.get_num_nodes_list()
        self._val_batch_plan: list[list[int]] = (
            BalancedDynamicBatchSampler.build_batch_plan(
                val_sizes,
                max_total_nodes=self.max_total_nodes,
            )
        )
        self.val_ds.release_runtime_caches()

    def _make_budgeted_dataloader(
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
            max_total_nodes=self.max_total_nodes,
            precomputed_batches=plan,
        )
        return DataLoader(
            ds,
            batch_sampler=sampler,
            collate_fn=Batch.from_data_list,
            **self._loader_kwargs(include_batch_size=False, is_train=is_train),
        )

    def setup(self, stage: str | None = None) -> None:
        import time as _time

        if stage == "fit" and hasattr(self, "train_ds") and hasattr(self, "val_ds"):
            print(
                "[datamodule] setup() — datasets already loaded, skipping.", flush=True
            )
            return
        print(f"[datamodule] setup(stage={stage!r}) entered", flush=True)
        _t0 = _time.monotonic()
        if stage in ("fit", None):
            print("[datamodule] Creating train dataset ...", flush=True)
            self.train_ds = self._make_dataset("train", self.train_num_samples)
            print(
                f"[datamodule] Train dataset ready ({_time.monotonic() - _t0:.1f}s)",
                flush=True,
            )
            print("[datamodule] Creating val dataset ...", flush=True)
            self.val_ds = self._make_dataset("val", self.train_num_samples)
            print(
                f"[datamodule] Val dataset ready ({_time.monotonic() - _t0:.1f}s)",
                flush=True,
            )
            if self.dynamic_batching:
                train_sizes = self.train_ds.get_num_nodes_list()
                self._train_batch_plan: list[list[int]] = load_or_build_batch_plan(
                    train_sizes,
                    max_total_nodes=self.max_total_nodes,
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
        if self.dynamic_batching:
            plan = getattr(self, "_train_batch_plan", None)
            if plan is None:
                plan = load_or_build_batch_plan(
                    self.train_ds.get_num_nodes_list(),
                    max_total_nodes=self.max_total_nodes,
                    cache_path=self._dynamic_batch_plan_cache_path(),
                )
            self._train_batch_plan = None
            return self._make_budgeted_dataloader(
                self.train_ds,
                plan,
                shuffle=True,
                is_train=True,
            )
        return DataLoader(
            self.train_ds,
            shuffle=True,
            collate_fn=Batch.from_data_list,
            **self._loader_kwargs(is_train=True),
        )

    def val_dataloader(self) -> DataLoader:
        if self.dynamic_batching:
            precomputed = getattr(self, "_val_batch_plan", None)
            if precomputed is None:
                precomputed = BalancedDynamicBatchSampler.build_batch_plan(
                    self.val_ds.get_num_nodes_list(),
                    max_total_nodes=self.max_total_nodes,
                )
                self._val_batch_plan = precomputed
            return self._make_budgeted_dataloader(
                self.val_ds,
                precomputed,
                shuffle=False,
                is_train=False,
            )
        return DataLoader(
            self.val_ds,
            shuffle=False,
            collate_fn=Batch.from_data_list,
            **self._loader_kwargs(is_train=False),
        )

    def _ensure_test_plan(self) -> None:
        if not self.dynamic_batching:
            return
        self._test_batch_plan: list[list[int]] = (
            BalancedDynamicBatchSampler.build_batch_plan(
                self.test_ds.get_num_nodes_list(),
                max_total_nodes=self.max_total_nodes,
            )
        )
        self.test_ds.release_runtime_caches()

    def test_dataloader(self) -> DataLoader:
        if self.dynamic_batching:
            precomputed = getattr(self, "_test_batch_plan", None)
            if precomputed is None:
                self._ensure_test_plan()
                precomputed = self._test_batch_plan
            return self._make_budgeted_dataloader(
                self.test_ds,
                precomputed,
                shuffle=False,
                is_train=False,
            )
        return DataLoader(
            self.test_ds,
            shuffle=False,
            collate_fn=Batch.from_data_list,
            **self._loader_kwargs(is_train=False),
        )


__all__ = ["AIGDataModule"]
