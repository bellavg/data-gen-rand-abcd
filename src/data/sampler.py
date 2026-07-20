from __future__ import annotations

import hashlib
import json
import random
from collections import deque
from pathlib import Path

import config

_DYNAMIC_BATCH_PLAN_CACHE: dict[str, list[list[int]]] = {}
_NODE_BUDGET_PLAN_VERSION = "node_budget_v3"


def batch_plan_cache_path(
    dataset_signature: str | None,
    *,
    cache_dir: Path | str | None,
    max_total_nodes: int,
) -> Path | None:
    if not dataset_signature or cache_dir is None:
        return None
    key = f"{dataset_signature}|plan={_NODE_BUDGET_PLAN_VERSION}|max_nodes={int(max_total_nodes)}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / "metadata" / "dynamic_batches" / f"train_{digest}.json"


class BalancedDynamicBatchSampler:
    """Build dynamic batches from graph sizes under a total-node budget.

    ``batch_size`` is kept only for call-site compatibility; the node budget
    alone controls how many graphs are packed into a batch.
    """

    def __init__(
        self,
        sizes: list[int] | None = None,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        max_total_nodes: int = config.MAX_TOTAL_NODES_PER_BATCH,
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
                max_total_nodes=max(1, int(max_total_nodes)),
            )
        self._epoch = 0

    @staticmethod
    def _build_node_budgeted_batches(
        indices: list[int],
        *,
        sizes: list[int],
        max_total_nodes: int,
    ) -> list[list[int]]:
        pool = deque(indices)
        batches: list[list[int]] = []

        while pool:
            largest_idx = pool.pop()
            batch = [largest_idx]
            total_nodes = int(sizes[largest_idx])

            while pool:
                smallest_idx = pool[0]
                smallest_nodes = int(sizes[smallest_idx])
                if total_nodes + smallest_nodes > max_total_nodes:
                    break
                batch.append(pool.popleft())
                total_nodes += smallest_nodes

            batches.append(batch)

        return batches

    @classmethod
    def build_batch_plan(
        cls,
        sizes: list[int],
        *,
        max_total_nodes: int,
    ) -> list[list[int]]:
        indices = sorted(range(len(sizes)), key=lambda i: sizes[i])
        return cls._build_node_budgeted_batches(
            indices,
            sizes=sizes,
            max_total_nodes=max(1, int(max_total_nodes)),
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
    max_total_nodes: int,
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
        max_total_nodes=max_total_nodes,
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
