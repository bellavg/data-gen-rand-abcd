from __future__ import annotations

import hashlib
import json
import random
from collections import deque
from pathlib import Path

_DYNAMIC_BATCH_PLAN_CACHE: dict[str, list[list[int]]] = {}


def batch_plan_cache_path(
    dataset_signature: str | None,
    *,
    cache_dir: Path | str | None,
    batch_size: int,
    bucket_rules: list[tuple[int, int]],
) -> Path | None:
    if not dataset_signature or not bucket_rules or cache_dir is None:
        return None
    rules_str = json.dumps(bucket_rules, separators=(",", ":"))
    key = f"{dataset_signature}|bs={batch_size}|rules={rules_str}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / "metadata" / "dynamic_batches" / f"train_{digest}.json"


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
        sizes: list[int] | None = None,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        bucket_rules: list[tuple[int, int]] | None = None,
        precomputed_batches: list[list[int]] | None = None,
    ) -> None:
        self.shuffle = shuffle
        self.seed = seed
        if precomputed_batches is not None:
            self._base_batches = precomputed_batches
        else:
            if not sizes:
                raise ValueError(
                    "sizes required when precomputed_batches is not provided"
                )
            self._base_batches = self.build_batch_plan(
                sizes,
                batch_size=max(1, batch_size),
                bucket_rules=bucket_rules or [],
            )
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
        # Always pop the largest graph first, then fill from the smallest side.
        pool = deque(indices)
        batches: list[list[int]] = []

        while pool:
            largest_idx = pool.pop()
            largest_nodes = int(sizes[largest_idx])
            target_batch_size = (
                BalancedDynamicBatchSampler._target_batch_size_for_largest(
                    largest_nodes,
                    bucket_rules=bucket_rules,
                    batch_size=batch_size,
                )
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
        indices = sorted(range(len(sizes)), key=lambda i: sizes[i])
        if not bucket_rules:
            return cls._build_fixed_batches(indices, batch_size=batch_size)
        return cls._build_bucketed_batches(
            indices,
            sizes=sizes,
            bucket_rules=bucket_rules,
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


def _normalize_batch_plan(plan: object, *, sample_count: int) -> list[list[int]] | None:
    if (
        not isinstance(plan, list)
        or not plan
        or sum(len(b) for b in plan) != sample_count
    ):
        return None
    return plan


def load_or_build_batch_plan(
    sizes: list[int],
    *,
    batch_size: int,
    bucket_rules: list[tuple[int, int]],
    cache_path: Path | None = None,
) -> list[list[int]]:
    cache_key = str(cache_path) if cache_path is not None else None

    if cache_key:
        cached = _DYNAMIC_BATCH_PLAN_CACHE.get(cache_key)
        if cached is not None:
            return cached

        if cache_path.is_file():
            on_disk = json.loads(cache_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
            normalized = _normalize_batch_plan(on_disk, sample_count=len(sizes))
            if normalized is not None:
                _DYNAMIC_BATCH_PLAN_CACHE[cache_key] = normalized
                return normalized

    built = BalancedDynamicBatchSampler.build_batch_plan(
        sizes,
        batch_size=batch_size,
        bucket_rules=bucket_rules,
    )

    if cache_key:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(f".tmp_{random.getrandbits(32):08x}")
        tmp_path.write_text(json.dumps(built), encoding="utf-8")
        tmp_path.replace(cache_path)
        _DYNAMIC_BATCH_PLAN_CACHE[cache_key] = built

    return built


__all__ = [
    "BalancedDynamicBatchSampler",
    "batch_plan_cache_path",
    "load_or_build_batch_plan",
]
