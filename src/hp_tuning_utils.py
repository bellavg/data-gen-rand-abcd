import gc
import logging
import time
from collections.abc import Callable

import optuna
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from data.datamodule import AIGDataModule
from data.sampler import BalancedDynamicBatchSampler, load_or_build_batch_plan


ATTENTION_ENCODERS = {"transformer_conv", "graphgps"}


def _set_trial_user_attr(
    trial: optuna.Trial, key: str, value: str | int | float | bool
) -> None:
    setter = getattr(trial, "set_user_attr", None)
    if not callable(setter):
        return

    # Some storages (SQLite) can intermittently raise locking/commit errors
    # under contention. Retry a few times with a short backoff and swallow
    # failures to avoid crashing the worker for a transient DB lock.
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            setter(key, value)
            return
        except Exception as exc:  # pragma: no cover - regression protection
            if attempt >= max_attempts:
                logging.warning(
                    "Failed to set trial user attr %s=%s for trial %s after %d attempts: %s",
                    key,
                    value,
                    getattr(trial, "number", "n/a"),
                    attempt,
                    exc,
                )
                return
            # backoff and retry
            time.sleep(0.5 * attempt)


def _mark_trial_outcome(
    trial: optuna.Trial,
    *,
    outcome: str,
    oom_like: bool,
    oom_kind: str | None = None,
    prune_reason: str | None = None,
    score: float | None = None,
    risk_score: float | None = None,
) -> None:
    _set_trial_user_attr(trial, "trial_outcome", outcome)
    _set_trial_user_attr(trial, "oom_like", bool(oom_like))
    _set_trial_user_attr(trial, "selection_eligible", outcome == "completed")
    if oom_kind is not None:
        _set_trial_user_attr(trial, "oom_kind", oom_kind)
    if prune_reason is not None:
        _set_trial_user_attr(trial, "prune_reason", prune_reason)
    if score is not None:
        _set_trial_user_attr(trial, "score", float(score))
    if risk_score is not None:
        _set_trial_user_attr(trial, "risk_score", float(risk_score))


class HPMemoryGuardError(RuntimeError):
    """Raised when an HP trial batch is estimated to exceed the memory budget."""


def _estimate_memory_tokens(
    *,
    num_nodes: int,
    num_edges: int,
    hidden_dim: int,
    num_layers: int,
    jk_mode: str,
    encoder_name: str,
    heads: int,
    expansion_factor: float,
) -> float:
    jk_multiplier = (num_layers + 1) if jk_mode == "cat" else 1
    attn_multiplier = heads if encoder_name in ATTENTION_ENCODERS else 1
    # GPS runs both a local MPNN pass and a global Performer pass per layer;
    # apply an extra 1.5x over plain TransformerConv to reflect dual-path cost.
    gps_multiplier = 1.5 if encoder_name == "graphgps" else 1.0
    layer_multiplier = max(1, num_layers)

    node_term = num_nodes * hidden_dim * layer_multiplier * jk_multiplier
    edge_term = num_edges * hidden_dim * attn_multiplier
    return (node_term + edge_term) * expansion_factor * gps_multiplier


def _parse_dynamic_bucket_rules(
    rule_text: str | None,
) -> list[tuple[int, int]]:
    """Parse dynamic bucket rules from CLI text.

    Format: "min_nodes:batch_size,min_nodes:batch_size,...".
    Example: "300000:1,180000:2,90000:4".
    """
    if not rule_text:
        return []

    parsed: list[tuple[int, int]] = []
    for chunk in (r.strip() for r in rule_text.split(",") if r.strip()):
        min_nodes, target = chunk.split(":", 1)
        parsed.append((int(min_nodes), int(target)))

    return parsed


def _build_guarded_collate(memory_guard: dict) -> Callable[[list], Batch]:
    max_tokens = float(memory_guard.get("max_tokens", float("inf")))
    hidden_dim = int(memory_guard.get("hidden_dim", 32))
    num_layers = int(memory_guard.get("num_layers", 2))
    jk_mode = str(memory_guard.get("jk_mode", "last"))
    encoder_name = str(memory_guard.get("encoder_name", "gine"))
    heads = int(memory_guard.get("heads", 1))
    expansion_factor = float(memory_guard.get("expansion_factor", 1.0))

    def guarded_collate(data_list: list) -> Batch:
        total_batch_tokens = 0.0
        for idx, data in enumerate(data_list):
            num_nodes = int(data.num_nodes)
            num_edges = int(data.edge_index.size(1))
            est_tokens = _estimate_memory_tokens(
                num_nodes=num_nodes,
                num_edges=num_edges,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                jk_mode=jk_mode,
                encoder_name=encoder_name,
                heads=heads,
                expansion_factor=expansion_factor,
            )
            total_batch_tokens += est_tokens
            if total_batch_tokens > max_tokens:
                raise HPMemoryGuardError(
                    "Memory guard pruned batch before forward allocation: "
                    f"batch_graphs={len(data_list)}, offending_index={idx}, "
                    f"nodes={num_nodes}, edges={num_edges}, "
                    f"graph_tokens={est_tokens:.2e}, "
                    f"batch_tokens={total_batch_tokens:.2e}, "
                    f"max_tokens={max_tokens:.2e}"
                )
        return Batch.from_data_list(data_list)

    return guarded_collate


def _install_hp_guarded_dataloaders(
    datamodule: AIGDataModule,
    memory_guard: dict,
    *,
    dynamic_batching: bool,
) -> None:
    guarded_collate = _build_guarded_collate(memory_guard)
    seed = int(getattr(datamodule, "seed", 42))
    bucket_rules = list(datamodule.dynamic_bucket_rules)
    use_buckets = dynamic_batching and bool(bucket_rules)

    if use_buckets:
        rules_text = ", ".join(f"{nodes}:{size}" for nodes, size in bucket_rules)
        print(
            "[memory_guard] "
            f"dynamic bucket rules (min_nodes:batch_size): {rules_text}"
        )

    def train_dataloader() -> DataLoader:
        if use_buckets:
            precomputed_batches = getattr(datamodule, "_train_batch_plan", None)
            if precomputed_batches is None:
                sizes = datamodule.train_ds.get_num_nodes_list()
                precomputed_batches = load_or_build_batch_plan(
                    sizes,
                    batch_size=datamodule.batch_size,
                    bucket_rules=bucket_rules,
                    cache_path=datamodule._dynamic_batch_plan_cache_path(),
                )
            sampler = BalancedDynamicBatchSampler(
                batch_size=datamodule.batch_size,
                shuffle=True,
                seed=seed,
                bucket_rules=bucket_rules,
                precomputed_batches=precomputed_batches,
            )
            datamodule._train_batch_plan = None
            return DataLoader(
                datamodule.train_ds,
                batch_sampler=sampler,
                collate_fn=guarded_collate,
                **datamodule._loader_kwargs(include_batch_size=False),
            )

        return DataLoader(
            datamodule.train_ds,
            shuffle=True,
            collate_fn=guarded_collate,
            **datamodule._loader_kwargs(),
        )

    def val_dataloader() -> DataLoader:
        if use_buckets:
            val_plan = getattr(datamodule, "_val_batch_plan", None)
            if val_plan is None:
                val_sizes = datamodule.val_ds.get_num_nodes_list()
                val_plan = BalancedDynamicBatchSampler.build_batch_plan(
                    val_sizes,
                    batch_size=datamodule.batch_size,
                    bucket_rules=bucket_rules,
                )
            datamodule._val_batch_plan = None
            sampler = BalancedDynamicBatchSampler(
                batch_size=datamodule.batch_size,
                shuffle=False,
                seed=seed,
                bucket_rules=bucket_rules,
                precomputed_batches=val_plan,
            )
            return DataLoader(
                datamodule.val_ds,
                batch_sampler=sampler,
                collate_fn=guarded_collate,
                **datamodule._loader_kwargs(include_batch_size=False),
            )
        return DataLoader(
            datamodule.val_ds,
            shuffle=False,
            collate_fn=guarded_collate,
            **datamodule._loader_kwargs(),
        )

    def test_dataloader() -> DataLoader:
        return DataLoader(
            datamodule.test_ds,
            shuffle=False,
            collate_fn=guarded_collate,
            **datamodule._loader_kwargs(),
        )

    datamodule.train_dataloader = train_dataloader
    datamodule.val_dataloader = val_dataloader
    datamodule.test_dataloader = test_dataloader


def _classify_oom_runtime_error(exc: RuntimeError) -> str | None:
    """Classify OOM-like RuntimeError as 'host' or 'cuda'."""
    text = str(exc).lower()

    if "memory guard" in text:
        return "guard"

    # Host-memory and mmap allocation failures (CPU RAM / VMA pressure).
    if "unable to mmap" in text or "cannot allocate memory" in text:
        return "host"
    if "dataloader worker" in text and "killed by signal" in text:
        return "host"
    if "dataloader worker" in text and "exited unexpectedly" in text:
        return "host"

    # CUDA allocator/device failures.
    if "cuda out of memory" in text:
        return "cuda"
    if "cuda error" in text or "illegal memory access" in text or "acceleratorerror" in text:
        return "cuda"

    # Fallback for generic OOM wording.
    if "out of memory" in text or "oom" in text:
        return "cuda" if "cuda" in text else "host"

    return None


def _runtime_oom_prune_payload(
    exc: RuntimeError,
    *,
    trial_number: int,
) -> tuple[str, str, bool, str] | None:
    """Build (oom_kind, prune_reason, oom_like, message) for OOM-like errors."""
    oom_kind = _classify_oom_runtime_error(exc)
    if oom_kind is None:
        return None

    if oom_kind == "cuda":
        return (
            "cuda",
            "runtime_cuda_oom",
            False,
            f"CUDA OOM-like RuntimeError. [Trial {trial_number}] Error: {exc}",
        )
    if oom_kind == "host":
        return (
            "host",
            "runtime_host_oom",
            True,
            f"Host-memory OOM-like RuntimeError. [Trial {trial_number}] Error: {exc}",
        )
    return (
        oom_kind,
        "runtime_oom",
        True,
        f"OOM-like RuntimeError. [Trial {trial_number}] Error: {exc}",
    )


def _estimate_trial_risk(
    *,
    batch_size: int,
    num_nodes: int,
    num_edges: int,
    hidden_dim: int,
    num_layers: int,
    jk_mode: str,
    encoder_name: str,
    heads: int,
    pe_type: str,
    pos_enc_dim: int,
) -> float:
    jk_multiplier = (num_layers + 1) if jk_mode == "cat" else 1
    attn_multiplier = heads if encoder_name in ATTENTION_ENCODERS else 1
    gps_multiplier = 1.5 if encoder_name == "graphgps" else 1.0
    pe_multiplier = 1.0 + (float(pos_enc_dim) / 128.0 if pe_type != "none" else 0.0)
    node_term = num_nodes * hidden_dim * num_layers * jk_multiplier * pe_multiplier
    edge_term = num_edges * hidden_dim * attn_multiplier
    return float(batch_size * (node_term + edge_term) * gps_multiplier)


def _select_trainer_precision() -> str:
    """Prefer BF16 mixed precision where available, else FP32."""
    try:
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except (AssertionError, RuntimeError):
        use_bf16 = False
    return "bf16-mixed" if use_bf16 else "32-true"


def _purge_trial_memory(
    *,
    trainer,
    datamodule,
    pruning_cb,
    early_stop_cb,
) -> None:
    # Sever callback-to-trainer back-references first so PL cannot keep the
    # trainer alive through a callback's self.trainer reference.
    if pruning_cb is not None:
        pruning_cb.trainer = None
    if early_stop_cb is not None:
        early_stop_cb.trainer = None

    optimizer = None
    if trainer is not None:
        optimizers = getattr(trainer, "optimizers", None)
        if optimizers:
            optimizer = optimizers[0]
        try:
            trainer._teardown()
        except Exception:
            pass
        try:
            if hasattr(trainer, "strategy") and trainer.strategy is not None:
                trainer.strategy.teardown()
        except Exception:
            pass
        try:
            if hasattr(trainer, "_accelerator_connector"):
                del trainer._accelerator_connector
        except Exception:
            pass
        try:
            trainer.fit_loop = None
        except Exception:
            pass
        try:
            trainer.validate_loop = None
        except Exception:
            pass
        try:
            trainer.test_loop = None
        except Exception:
            pass
        try:
            trainer.predict_loop = None
        except Exception:
            pass
        try:
            if hasattr(trainer, "_data_connector"):
                del trainer._data_connector
        except Exception:
            pass
        for lg in list(getattr(trainer, "loggers", []) or []):
            try:
                lg.finalize("failed")
            except Exception:
                pass
        trainer.callbacks = []
        trainer.loggers = []
        try:
            trainer.lightning_module = None
        except Exception:
            pass

    # Tear down DataModule and drop datasets so large PE tensors can be released
    # before the next trial starts.
    if datamodule is not None:
        try:
            datamodule.teardown("fit")
        except Exception:
            pass
        for attr in ("train_ds", "val_ds", "test_ds"):
            try:
                setattr(datamodule, attr, None)
            except Exception:
                pass

    # Zero-out optimizer grads before dropping references to reclaim memory faster.
    if optimizer is not None:
        try:
            for group in optimizer.param_groups:
                for p in group.get("params", []):
                    if p.grad is not None:
                        p.grad = None
        except Exception:
            pass

    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except RuntimeError:
            pass
