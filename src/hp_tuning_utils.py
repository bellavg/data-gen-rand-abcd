"""Utilities shared by the Optuna HP-tuning objective.

Root-cause of the previous memory leak (Python Unpickler memo accumulation
across multiprocessing workers) is fully resolved in ``src/data/dataset.py``
via ``weights_only=True`` + ``add_safe_globals``.

Module layout (trial lifecycle order)
--------------------------------------
1. Constants
2. Trial attribute helpers        -- bookkeeping at trial boundaries
3. Pre-allocation risk / prune    -- decide whether to start the trial
4. Memory-guard collate           -- arm the dataloader once trial starts
5. OOM classification             -- handle failures during training
6. Trainer helpers                -- misc training setup
7. Inter-trial cleanup            -- tear down after trial ends
"""

import ctypes
import gc
import logging
import time
import weakref
from collections.abc import Callable

import optuna
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from data.datamodule import AIGDataModule
from data.sampler import BalancedDynamicBatchSampler, load_or_build_batch_plan

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

ATTENTION_ENCODERS = {"transformer_conv", "graphgps"}


# ---------------------------------------------------------------------------
# 2. Trial attribute helpers
# ---------------------------------------------------------------------------


def _set_trial_user_attr(
    trial: optuna.Trial, key: str, value: str | int | float | bool
) -> None:
    """Set a trial user attribute, retrying on transient SQLite lock errors.

    Args:
        trial: Active Optuna trial.
        key: Attribute name.
        value: Attribute value (must be JSON-serialisable).
    """
    setter = getattr(trial, "set_user_attr", None)
    if not callable(setter):
        return

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            setter(key, value)
            return
        except Exception as exc:  # pragma: no cover
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
    """Write outcome metadata to trial user attributes.

    Args:
        trial: Active Optuna trial.
        outcome: One of ``"running"``, ``"completed"``, ``"pruned"``.
        oom_like: Whether the outcome was caused by an OOM-like error.
        oom_kind: ``"cuda"``, ``"host"``, ``"guard"``, ``"predicted"``, or
            ``None``.
        prune_reason: Short string describing why the trial was pruned.
        score: Best validation score (completed trials only).
        risk_score: Pre-allocation risk estimate (if computed).
    """
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


# ---------------------------------------------------------------------------
# 3. Pre-allocation risk / hard-prune
# ---------------------------------------------------------------------------


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
    """Estimate a dimensionless risk score for a trial configuration.

    Used by the optional ``--hard_prune`` gate to prune extreme configurations
    before any GPU memory is allocated.

    Args:
        batch_size: DataLoader batch size.
        num_nodes: Representative node count for the risk graph.
        num_edges: Representative edge count for the risk graph.
        hidden_dim: Model hidden dimension.
        num_layers: Number of message-passing layers.
        jk_mode: JumpingKnowledge mode.
        encoder_name: GNN encoder identifier.
        heads: Number of attention heads.
        pe_type: Positional encoding type (``"none"`` disables the PE term).
        pos_enc_dim: Positional encoding dimension.

    Returns:
        Risk score (higher means more likely to OOM).
    """
    jk_multiplier = (num_layers + 1) if jk_mode == "cat" else 1
    attn_multiplier = heads if encoder_name in ATTENTION_ENCODERS else 1
    gps_multiplier = 1.5 if encoder_name == "graphgps" else 1.0
    pe_multiplier = 1.0 + (float(pos_enc_dim) / 128.0 if pe_type != "none" else 0.0)
    node_term = num_nodes * hidden_dim * num_layers * jk_multiplier * pe_multiplier
    edge_term = num_edges * hidden_dim * attn_multiplier
    return float(batch_size * (node_term + edge_term) * gps_multiplier)


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
    """Heuristic token count for a single graph forward pass.

    Shared primitive used by both :func:`_estimate_trial_risk` and the
    guarded collate function.

    Args:
        num_nodes: Number of nodes in the graph.
        num_edges: Number of edges in the graph.
        hidden_dim: Model hidden dimension.
        num_layers: Number of message-passing layers.
        jk_mode: JumpingKnowledge aggregation mode (``"cat"`` multiplies by
            ``num_layers + 1``).
        encoder_name: GNN encoder identifier.
        heads: Number of attention heads (attention encoders only).
        expansion_factor: Safety multiplier applied to the total token count.

    Returns:
        Estimated token count (dimensionless heuristic).
    """
    jk_multiplier = (num_layers + 1) if jk_mode == "cat" else 1
    attn_multiplier = heads if encoder_name in ATTENTION_ENCODERS else 1
    # GPS runs both a local MPNN pass and a global Performer pass per layer.
    gps_multiplier = 1.5 if encoder_name == "graphgps" else 1.0
    layer_multiplier = max(1, num_layers)
    node_term = num_nodes * hidden_dim * layer_multiplier * jk_multiplier
    edge_term = num_edges * hidden_dim * attn_multiplier
    return (node_term + edge_term) * expansion_factor * gps_multiplier


# ---------------------------------------------------------------------------
# 4. Memory-guard collate (pre-allocation OOM prevention)
# ---------------------------------------------------------------------------


class HPMemoryGuardError(RuntimeError):
    """Raised when a batch is estimated to exceed the memory budget.

    Kept as a distinct subclass so the ``except HPMemoryGuardError`` branch
    in ``objective`` fires before the generic ``except RuntimeError`` OOM
    handler, making dispatch explicit rather than relying on string matching.
    """


def _build_guarded_collate(memory_guard: dict) -> Callable[[list], Batch]:
    """Return a collate function that raises before over-budget batches are built.

    Args:
        memory_guard: Dict with keys ``max_tokens``, ``hidden_dim``,
            ``num_layers``, ``jk_mode``, ``encoder_name``, ``heads``,
            ``expansion_factor``.

    Returns:
        A collate callable compatible with ``DataLoader(collate_fn=...)``.
    """
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
    """Monkey-patch datamodule dataloader methods with the guarded collate.

    Uses a ``weakref`` to ``datamodule`` so the closures do not form a strong
    reference cycle (datamodule → method → closure → datamodule).

    Dataset references (``train_ds`` / ``val_ds`` / ``test_ds``) are accessed
    lazily at call time because Lightning's ``setup()`` has not yet run when
    this function is called.

    Args:
        datamodule: The ``AIGDataModule`` instance to patch.
        memory_guard: Dict passed to :func:`_build_guarded_collate`.
        dynamic_batching: Whether to use node-budgeted dynamic batching.
    """
    guarded_collate = _build_guarded_collate(memory_guard)
    seed = int(getattr(datamodule, "seed", 42))
    max_total_nodes = int(datamodule.max_total_nodes)
    use_dynamic_batching = dynamic_batching

    batch_size = datamodule.batch_size
    train_kw = datamodule._loader_kwargs(is_train=True)
    train_kw_no_bs = datamodule._loader_kwargs(include_batch_size=False, is_train=True)
    val_kw = datamodule._loader_kwargs(is_train=False)
    val_kw_no_bs = datamodule._loader_kwargs(include_batch_size=False, is_train=False)
    _dm_ref = weakref.ref(datamodule)

    if use_dynamic_batching:
        print(
            f"[memory_guard] max_total_nodes_per_batch={max_total_nodes}"
        )

    def train_dataloader() -> DataLoader:
        dm = _dm_ref()
        if use_dynamic_batching:
            precomputed_batches = (
                getattr(dm, "_train_batch_plan", None) if dm is not None else None
            )
            if precomputed_batches is None:
                sizes = dm.train_ds.get_num_nodes_list()
                precomputed_batches = load_or_build_batch_plan(
                    sizes,
                    batch_size=batch_size,
                    max_total_nodes=max_total_nodes,
                    cache_path=dm._dynamic_batch_plan_cache_path(),
                )
            sampler = BalancedDynamicBatchSampler(
                batch_size=batch_size,
                shuffle=True,
                seed=seed,
                max_total_nodes=max_total_nodes,
                precomputed_batches=precomputed_batches,
            )
            if dm is not None:
                dm._train_batch_plan = None
            return DataLoader(
                dm.train_ds,
                batch_sampler=sampler,
                collate_fn=guarded_collate,
                **train_kw_no_bs,
            )
        return DataLoader(
            dm.train_ds, shuffle=True, collate_fn=guarded_collate, **train_kw
        )

    def val_dataloader() -> DataLoader:
        dm = _dm_ref()
        if use_dynamic_batching:
            val_plan = getattr(dm, "_val_batch_plan", None) if dm is not None else None
            if val_plan is None:
                val_sizes = dm.val_ds.get_num_nodes_list()
                val_plan = BalancedDynamicBatchSampler.build_batch_plan(
                    val_sizes,
                    batch_size=batch_size,
                    max_total_nodes=max_total_nodes,
                )
            if dm is not None:
                dm._val_batch_plan = None
            sampler = BalancedDynamicBatchSampler(
                batch_size=batch_size,
                shuffle=False,
                seed=seed,
                max_total_nodes=max_total_nodes,
                precomputed_batches=val_plan,
            )
            return DataLoader(
                dm.val_ds,
                batch_sampler=sampler,
                collate_fn=guarded_collate,
                **val_kw_no_bs,
            )
        return DataLoader(
            dm.val_ds, shuffle=False, collate_fn=guarded_collate, **val_kw
        )

    def test_dataloader() -> DataLoader:
        dm = _dm_ref()
        return DataLoader(
            dm.test_ds, shuffle=False, collate_fn=guarded_collate, **val_kw
        )

    datamodule.train_dataloader = train_dataloader
    datamodule.val_dataloader = val_dataloader
    datamodule.test_dataloader = test_dataloader


# ---------------------------------------------------------------------------
# 5. OOM classification
# ---------------------------------------------------------------------------


def _classify_oom_runtime_error(exc: RuntimeError) -> str | None:
    """Classify an OOM-like ``RuntimeError`` as ``'guard'``, ``'host'``, or
    ``'cuda'``.

    Args:
        exc: The caught ``RuntimeError``.

    Returns:
        ``"guard"`` / ``"host"`` / ``"cuda"`` on a match, ``None`` otherwise.
    """
    text = str(exc).lower()

    if "memory guard" in text:
        return "guard"
    if "unable to mmap" in text or "cannot allocate memory" in text:
        return "host"
    if "dataloader worker" in text and "killed by signal" in text:
        return "host"
    if "dataloader worker" in text and "exited unexpectedly" in text:
        return "host"
    if "cuda out of memory" in text:
        return "cuda"
    if (
        "cuda error" in text
        or "illegal memory access" in text
        or "acceleratorerror" in text
    ):
        return "cuda"
    if "out of memory" in text or "oom" in text:
        return "cuda" if "cuda" in text else "host"
    return None


def _runtime_oom_prune_payload(
    exc: RuntimeError,
    *,
    trial_number: int,
) -> tuple[str, str, bool, str] | None:
    """Build ``(oom_kind, prune_reason, oom_like, message)`` for OOM-like errors.

    Args:
        exc: The caught ``RuntimeError``.
        trial_number: Optuna trial number (used in the log message).

    Returns:
        A four-tuple on match, ``None`` if the error is not OOM-like.
    """
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


# ---------------------------------------------------------------------------
# 6. Trainer helpers
# ---------------------------------------------------------------------------


def _select_trainer_precision() -> str:
    """Return ``"bf16-mixed"`` when BF16 is supported, otherwise ``"32-true"``."""
    try:
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except (AssertionError, RuntimeError):
        use_bf16 = False
    return "bf16-mixed" if use_bf16 else "32-true"


# ---------------------------------------------------------------------------
# 7. Inter-trial cleanup
# ---------------------------------------------------------------------------


def _purge_trial_memory(
    *,
    trainer,
    datamodule,
    pruning_cb,
    early_stop_cb,
) -> None:
    """Release resources held by a completed or pruned trial.

    Handles normal inter-trial cleanup: optimizer grad tensors, Lightning loop
    and connector objects, dataset PE tensors, and batch-plan caches.  This is
    not a leak workaround — the leak is fixed at the deserialisation layer.

    Args:
        trainer: The ``pl.Trainer`` instance, or ``None``.
        datamodule: The ``AIGDataModule`` instance, or ``None``.
        pruning_cb: Pruning callback whose trainer back-reference is severed.
        early_stop_cb: Early-stop callback whose trainer back-reference is severed.
    """
    # Sever callback → trainer back-references first.
    if pruning_cb is not None:
        pruning_cb.trainer = None
    if early_stop_cb is not None:
        early_stop_cb.trainer = None

    optimizer = None
    if trainer is not None:
        optimizers = getattr(trainer, "optimizers", None)
        if optimizers:
            optimizer = optimizers[0]
        for teardown in (
            lambda: trainer._teardown(),
            lambda: (
                trainer.strategy.teardown() if trainer.strategy is not None else None
            ),
        ):
            try:
                teardown()
            except Exception:
                pass
        for attr in (
            "_accelerator_connector",
            "fit_loop",
            "validate_loop",
            "test_loop",
            "predict_loop",
            "_data_connector",
            "lightning_module",
        ):
            try:
                setattr(trainer, attr, None)
            except Exception:
                pass
        for lg in list(getattr(trainer, "loggers", []) or []):
            try:
                lg.finalize("failed")
            except Exception:
                pass
        trainer.callbacks = []
        trainer.loggers = []

    if datamodule is not None:
        try:
            datamodule.teardown("fit")
        except Exception:
            pass
        # Clear batch-plan refs and dataset refs so PE tensors are released.
        for attr in (
            "_train_batch_plan",
            "_val_batch_plan",
            "train_ds",
            "val_ds",
            "test_ds",
        ):
            try:
                setattr(datamodule, attr, None)
            except Exception:
                pass
        # Break closure cycles introduced by _install_hp_guarded_dataloaders.
        for attr in ("train_dataloader", "val_dataloader", "test_dataloader"):
            try:
                setattr(datamodule, attr, None)
            except Exception:
                pass

    # Zero optimizer grads before dropping references.
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

    # Return glibc/tcmalloc pages to the OS.
    try:
        libc = ctypes.CDLL(None)
        release_fn = getattr(libc, "MallocExtension_ReleaseFreeMemory", None)
        if callable(release_fn):
            release_fn()
        else:
            trim = getattr(libc, "malloc_trim", None)
            if callable(trim):
                trim(0)
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except RuntimeError:
            pass

    # Clear module-level caches so index lists don't accumulate across trials.
    from data.sampler import _DYNAMIC_BATCH_PLAN_CACHE  # noqa: PLC0415

    _DYNAMIC_BATCH_PLAN_CACHE.clear()

    try:
        from data.dataset import clear_dataset_global_caches  # noqa: PLC0415

        clear_dataset_global_caches()
    except Exception:
        pass
