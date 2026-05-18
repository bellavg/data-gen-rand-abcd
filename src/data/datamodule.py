from __future__ import annotations

import hashlib
import json
import random
import warnings
from collections import deque
from pathlib import Path

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from data.dataset import AIGGraphRegressionDataset


_DYNAMIC_BATCH_PLAN_CACHE: dict[str, list[list[int]]] = {}


class BalancedDynamicBatchSampler:
    """Build dynamic batches from graph sizes with optional bucket rules.

    Without bucket rules, batches are fixed-cardinality (`batch_size`) and
    pair large and small graphs to reduce collisions of many heavy samples.

    With bucket rules, the largest graph in a batch decides that batch's
    cardinality. This keeps all graphs in the epoch while forcing very large
    graphs to run in smaller batches (often singleton batches).
    """

    def __init__(
        self,
        sizes: list[int],
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        bucket_rules: list[tuple[int, int]] | None = None,
        precomputed_batches: list[list[int]] | None = None,
    ) -> None:
        self.sizes = sizes
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_rules = bucket_rules or []

        if precomputed_batches is None:
            self._base_batches = self.build_batch_plan(
                self.sizes,
                batch_size=self.batch_size,
                bucket_rules=self.bucket_rules,
            )
        else:
            self._base_batches = precomputed_batches

        self._epoch = 0

    @staticmethod
    def _normalize_bucket_rules(
        bucket_rules: list[tuple[int, int]] | None,
        batch_size: int,
    ) -> list[tuple[int, int]]:
        if not bucket_rules:
            return []

        return sorted(
            [
                (
                    max(1, int(min_nodes)),
                    max(1, min(batch_size, int(target_batch_size))),
                )
                for min_nodes, target_batch_size in bucket_rules
            ],
            key=lambda rule: rule[0],
            reverse=True,
        )

    @staticmethod
    def _build_fixed_batches(indices: list[int], *, batch_size: int) -> list[list[int]]:
        left = 0
        right = len(indices) - 1
        batches: list[list[int]] = []

        while left <= right:
            batch: list[int] = []
            take_large = True
            while left <= right and len(batch) < batch_size:
                if take_large:
                    batch.append(indices[right])
                    right -= 1
                else:
                    batch.append(indices[left])
                    left += 1
                take_large = not take_large
            batches.append(batch)

        return batches

    @staticmethod
    def _target_batch_size_for_largest(
        num_nodes: int,
        *,
        bucket_rules: list[tuple[int, int]],
        batch_size: int,
    ) -> int:
        for min_nodes, target_batch_size in bucket_rules:
            if num_nodes >= min_nodes:
                return target_batch_size
        return batch_size

    @staticmethod
    def _build_bucketed_batches(
        indices: list[int],
        *,
        sizes: list[int],
        bucket_rules: list[tuple[int, int]],
        batch_size: int,
    ) -> list[list[int]]:
        if not bucket_rules:
            return BalancedDynamicBatchSampler._build_fixed_batches(
                indices,
                batch_size=batch_size,
            )

        # Always pop the largest graph first, then fill from the smallest side.
        pool = deque(indices)
        batches: list[list[int]] = []

        while pool:
            largest_idx = pool.pop()
            largest_nodes = int(sizes[largest_idx])
            target_batch_size = BalancedDynamicBatchSampler._target_batch_size_for_largest(
                largest_nodes,
                bucket_rules=bucket_rules,
                batch_size=batch_size,
            )

            batch = [largest_idx]
            while pool and len(batch) < target_batch_size:
                batch.append(pool.popleft())

            batches.append(batch)

        return batches

    @classmethod
    def build_batch_plan(
        cls,
        sizes: list[int],
        *,
        batch_size: int,
        bucket_rules: list[tuple[int, int]] | None = None,
    ) -> list[list[int]]:
        batch_size = max(1, int(batch_size))
        normalized_rules = bucket_rules or []
        indices = sorted(range(len(sizes)), key=lambda i: sizes[i])

        if not normalized_rules:
            return cls._build_fixed_batches(indices, batch_size=batch_size)

        return cls._build_bucketed_batches(
            indices,
            sizes=sizes,
            bucket_rules=normalized_rules,
            batch_size=batch_size,
        )

    def __len__(self) -> int:
        return len(self._base_batches)

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        batches = list(self._base_batches)

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
        dynamic_bucket_rules: list[tuple[int, int]] | None = None,
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
        self.dynamic_bucket_rules = BalancedDynamicBatchSampler._normalize_bucket_rules(
            dynamic_bucket_rules,
            self.batch_size,
        )
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

    def _dynamic_batch_plan_cache_path(self) -> Path | None:
        

        dataset_signature = getattr(getattr(self, "train_ds", None), "_cache_signature", None)
        if not self.dynamic_batching or not self.dynamic_bucket_rules or self.cache_dir is None or not dataset_signature:
            return None
       

        rules_str = json.dumps(self.dynamic_bucket_rules, separators=(",", ":"))
        key = f"{dataset_signature}|bs={self.batch_size}|rules={rules_str}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / "metadata" / "dynamic_batches" / f"train_{digest}.json"

    def _normalize_cached_batch_plan(
        self,
        plan: object,
        *,
        sample_count: int,
    ) -> list[list[int]] | None:
        if sum(len(b) for b in plan) != sample_count or not isinstance(plan, list) or not plan:
            return None
        return plan

    def _load_or_build_train_batch_plan(self, sizes: list[int]) -> list[list[int]]:
        cache_path = self._dynamic_batch_plan_cache_path()
        if cache_path is not None:
            cache_key = str(cache_path)
            cached = _DYNAMIC_BATCH_PLAN_CACHE.get(cache_key)
            if cached is not None:
                return cached

            if cache_path.is_file():
                on_disk = json.loads(cache_path.read_text(encoding="utf-8"))
                normalized = self._normalize_cached_batch_plan(on_disk, sample_count=len(sizes))
                if normalized is not None:
                    _DYNAMIC_BATCH_PLAN_CACHE[cache_key] = normalized
                    return normalized

        built = BalancedDynamicBatchSampler.build_batch_plan(
            sizes,
            batch_size=self.batch_size,
            bucket_rules=self.dynamic_bucket_rules,
        )

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(f".tmp_{random.getrandbits(32):08x}")
            tmp_path.write_text(json.dumps(built), encoding="utf-8")
            tmp_path.replace(cache_path)
            _DYNAMIC_BATCH_PLAN_CACHE[str(cache_path)] = built

        return built

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
            if self.dynamic_batching and self.dynamic_bucket_rules:
                train_sizes = self.train_ds.get_num_nodes_list()
                self._train_batch_plan: list[list[int]] = self._load_or_build_train_batch_plan(
                    train_sizes
                )
                self._train_sizes = None
                self.train_ds.release_runtime_caches()

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
        if self.dynamic_batching and self.dynamic_bucket_rules:
            precomputed_batches = getattr(self, "_train_batch_plan", None)
            if precomputed_batches is None:
                sizes = self.train_ds.get_num_nodes_list()
                precomputed_batches = self._load_or_build_train_batch_plan(sizes)
            # Precomputed batch plans already encode ordering/grouping.
            # Keep runtime memory low by not retaining node-size vectors.
            sampler_sizes: list[int] = []
            sampler = BalancedDynamicBatchSampler(
                sampler_sizes,
                batch_size=self.batch_size,
                shuffle=True,
                seed=self.seed,
                bucket_rules=self.dynamic_bucket_rules,
                precomputed_batches=precomputed_batches,
            )
            self._train_batch_plan = None
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
