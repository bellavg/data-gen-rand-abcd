import argparse
import gc
import logging
import os
import resource
import sys
import warnings
from collections.abc import Callable

import optuna
from optuna.storages import RDBStorage
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from data.datamodule import AIGDataModule, BalancedDynamicBatchSampler
from models.lightning_model import AIGRegressionLightningModule
from optuna.integration import PyTorchLightningPruningCallback



warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="optuna")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.allow_tf32 = True


ATTENTION_ENCODERS = {"transformer_conv", "graphgps"}


def _set_trial_user_attr(trial: optuna.Trial, key: str, value: str | int | float | bool) -> None:
    setter = getattr(trial, "set_user_attr", None)
    if callable(setter):
        setter(key, value)


def _mark_trial_outcome(
    trial: optuna.Trial,
    *,
    outcome: str,
    oom_like: bool,
    prune_reason: str | None = None,
    score: float | None = None,
    risk_score: float | None = None,
) -> None:
    _set_trial_user_attr(trial, "trial_outcome", outcome)
    _set_trial_user_attr(trial, "oom_like", bool(oom_like))
    _set_trial_user_attr(trial, "selection_eligible", outcome == "completed")
    if prune_reason is not None:
        _set_trial_user_attr(trial, "prune_reason", prune_reason)
    if score is not None:
        _set_trial_user_attr(trial, "score", float(score))
    if risk_score is not None:
        _set_trial_user_attr(trial, "risk_score", float(risk_score))


def _peak_rss_bytes() -> int:
    # On Linux ru_maxrss is KiB; on macOS it is bytes.
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024


def _current_rss_bytes() -> int | None:
    # Prefer /proc on Linux clusters; fallback to None if unavailable.
    statm = "/proc/self/statm"
    if not os.path.exists(statm):
        return None
    with open(statm, "r", encoding="utf-8") as f:
        fields = f.read().strip().split()
    if len(fields) < 2:
        return None
    resident_pages = int(fields[1])
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    return resident_pages * page_size


def _format_gib(value_bytes: int | None) -> str:
    if value_bytes is None:
        return "n/a"
    return f"{value_bytes / (1024**3):.3f} GiB"


def _should_log_memory_telemetry(trial: optuna.Trial, args) -> bool:
    max_trials = int(getattr(args, "memory_telemetry_trials", 0) or 0)
    return max_trials > 0 and trial.number < max_trials


def _log_trial_memory(stage: str, trial: optuna.Trial, args) -> None:
    if not _should_log_memory_telemetry(trial, args):
        return

    rss_current = _current_rss_bytes()
    rss_peak = _peak_rss_bytes()

    cuda_alloc = None
    cuda_reserved = None
    if torch.cuda.is_available():
        cuda_alloc = int(torch.cuda.memory_allocated())
        cuda_reserved = int(torch.cuda.memory_reserved())

    print(
        "[memory] "
        f"trial={trial.number} stage={stage} "
        f"rss_current={_format_gib(rss_current)} "
        f"rss_peak={_format_gib(rss_peak)} "
        f"cuda_allocated={_format_gib(cuda_alloc)} "
        f"cuda_reserved={_format_gib(cuda_reserved)}"
    )


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

    def train_dataloader() -> DataLoader:
        if dynamic_batching:
            sizes = getattr(datamodule, "_train_sizes", None)
            if sizes is None:
                sizes = datamodule.train_ds.get_num_nodes_list()
            sampler = BalancedDynamicBatchSampler(
                sizes,
                batch_size=datamodule.batch_size,
                shuffle=True,
                seed=seed,
            )
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


def _is_oom_like_runtime_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    if "out of memory" in text or "oom" in text:
        return True
    if "dataloader worker" in text and "killed by signal" in text:
        return True
    if "dataloader worker" in text and "exited unexpectedly" in text:
        return True
    if "memory guard" in text:
        return True
    return False


def _estimate_trial_risk(
    *,
    batch_size: int,
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
    # GPS has local MPNN + global Performer per layer; extra 1.5x vs TransformerConv.
    gps_multiplier = 1.5 if encoder_name == "graphgps" else 1.0
    pe_multiplier = 1.0 + (float(pos_enc_dim) / 128.0 if pe_type != "none" else 0.0)
    return float(
        batch_size
        * hidden_dim
        * num_layers
        * jk_multiplier
        * attn_multiplier
        * pe_multiplier
        * gps_multiplier
    )


def _purge_trial_memory(
    *,
    trainer,
    datamodule,
    model,
    optimizer,
    batch,
    dataloader,
    val_dataloader,
    pruning_cb,
    early_stop_cb,
):
    # 1. Sever callback-to-trainer back-references first so PL can't keep the
    #    trainer alive through a callback's self.trainer reference.
    if pruning_cb is not None:
        pruning_cb.trainer = None
    if early_stop_cb is not None:
        early_stop_cb.trainer = None

    if trainer is not None:
        # 2. Explicit PL teardown sequence: strategy → accelerator → loggers
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
        # 3. Clear all circular PL state containers
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
        # 4. Clear loggers and callbacks
        for lg in list(getattr(trainer, "loggers", []) or []):
            try:
                lg.finalize("failed")
            except Exception:
                pass
        trainer.callbacks = []
        trainer.loggers = []
        # 5. Detach model from trainer to break any final cycle
        try:
            trainer.lightning_module = None
        except Exception:
            pass

    # 6. Tear down DataModule
    if datamodule is not None:
        try:
            datamodule.teardown("fit")
        except Exception:
            pass

    # 7. Zero-out optimiser state tensors before del to reclaim GPU memory faster
    if optimizer is not None:
        try:
            for group in optimizer.param_groups:
                for p in group.get("params", []):
                    if p.grad is not None:
                        p.grad = None
        except Exception:
            pass

    # 8. Delete all references — individual dels + None assignments here are
    # redundant with the return-None tuple, but gc.collect below needs the
    # refcounts to drop first.  The return tuple unpack in the caller is what
    # actually zeros out the outer names.
    gc.collect()
    gc.collect()  # Two passes to break complex reference cycles
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure GPU ops drained before next trial
        except RuntimeError:
            # Never let cleanup exceptions mask the real trial outcome.
            pass

    return (None, None, None, None, None, None, None, None, None)


def _trial_outcome_callback(study: optuna.Study, frozen_trial) -> None:
    outcome = str(
        frozen_trial.user_attrs.get("trial_outcome", frozen_trial.state.name.lower())
    )
    oom_like = bool(frozen_trial.user_attrs.get("oom_like", False))
    selection_eligible = bool(frozen_trial.user_attrs.get("selection_eligible", False))
    reason = str(frozen_trial.user_attrs.get("prune_reason", ""))

    if frozen_trial.value is None:
        value_text = "n/a"
    else:
        value_text = f"{float(frozen_trial.value):.6f}"

    reason_text = f" reason={reason}" if reason else ""
    print(
        "[selection] "
        f"trial={frozen_trial.number} "
        f"state={frozen_trial.state.name} "
        f"outcome={outcome} "
        f"oom_like={oom_like} "
        f"eligible={selection_eligible} "
        f"value={value_text}"
        f"{reason_text}"
    )


def objective(trial: optuna.Trial, args):
    _mark_trial_outcome(trial, outcome="running", oom_like=False)

    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    huber_delta = trial.suggest_float("huber_delta", 0.5, 2.0)

    encoder_name = trial.suggest_categorical(
        "encoder_name",
        ["gine", "transformer_conv", "graphgps", "egin", "gcn", "vanilla_mpnn"],
    )

    # Use a single, fixed choice set for every parameter so that the Optuna
    # RDB distribution stays identical across all trials regardless of which
    # encoder was sampled.  Dynamic per-encoder subsets caused a
    # "CategoricalDistribution does not support dynamic value space" error
    # when Trial N registered a different set than Trial 0.
    # The risk-score system (_estimate_trial_risk) and hard_prune_risk threshold
    # already prune attention + jk_cat + large hidden_dim combos before training.
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])

    pe_type = trial.suggest_categorical(
        "pe_type",
        ["none", "level", "pi_paths", "local_sp_sum"],
    )
    pos_enc_dim = (
        trial.suggest_categorical("pos_enc_dim", [16, 32, 64, 128])
        if pe_type != "none"
        else 0
    )
    pooling_type = trial.suggest_categorical("pooling_type", ["mean", "max", "sum"])

    encoder_kwargs = {
        "num_layers": trial.suggest_int("num_layers", 2, 8),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "norm_type": trial.suggest_categorical(
            "norm_type", ["batch", "layer", "graph"]
        ),
        "jk_mode": trial.suggest_categorical("jk_mode", ["last", "max", "sum", "cat"]),
        "hid_dim": hidden_dim,
    }

    if encoder_name in ATTENTION_ENCODERS:
        encoder_kwargs["heads"] = trial.suggest_categorical("heads", [1, 2, 4])

    if encoder_name == "egin":
        encoder_kwargs["num_mlp_layers"] = trial.suggest_int("num_mlp_layers", 2, 4)
        encoder_kwargs["dot_update"] = trial.suggest_categorical(
            "egin_dot_update", [True, False]
        )
        encoder_kwargs["edge_mlp"] = trial.suggest_categorical(
            "egin_edge_mlp", [True, False]
        )
        encoder_kwargs["edge_hidden_dim"] = trial.suggest_categorical(
            "edge_hidden_dim", [32, 64, 128]
        )

    print(f"\n{'=' * 60}")
    print(f"TRIAL {trial.number} STARTED")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    print(f"{'=' * 60}\n")
    _log_trial_memory("trial_start", trial, args)

    model = None
    trainer = None
    datamodule = None
    optimizer = None
    batch = None
    dataloader = None
    val_dataloader = None
    pruning_cb = None
    early_stop_cb = None
    risk_score = _estimate_trial_risk(
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        num_layers=encoder_kwargs["num_layers"],
        jk_mode=encoder_kwargs["jk_mode"],
        encoder_name=encoder_name,
        heads=int(encoder_kwargs.get("heads", 1)),
        pe_type=pe_type,
        pos_enc_dim=pos_enc_dim,
    )

    hard_prune_risk = float(getattr(args, "hard_prune_risk", 200000.0))
    _set_trial_user_attr(trial, "hard_prune_risk", hard_prune_risk)
    _set_trial_user_attr(trial, "risk_score", float(risk_score))
    if risk_score > hard_prune_risk:
        msg = (
            f"Pre-allocation prune for high-risk trial. "
            f"estimated_risk={risk_score:.2f} threshold={hard_prune_risk:.2f} "
            f"params={trial.params}"
        )
        print(f"\n{msg}")
        _mark_trial_outcome(
            trial,
            outcome="pruned",
            oom_like=True,
            prune_reason="hard_prune_risk",
            risk_score=risk_score,
        )
        raise optuna.TrialPruned(msg)

    try:
        workers = getattr(args, "num_workers", 2)
        persistent = getattr(args, "persistent_workers", False)
        pin_memory = getattr(args, "pin_memory", False)
        prefetch_factor = int(getattr(args, "prefetch_factor", 1))
        dynamic_batching = getattr(args, "dynamic_batching", False)
        dataset_seed = int(getattr(args, "dataset_seed", 42))

        guard_expansion = 3.0 if encoder_name in ATTENTION_ENCODERS else 2.0
        memory_guard = {
            "hidden_dim": hidden_dim,
            "num_layers": encoder_kwargs["num_layers"],
            "jk_mode": encoder_kwargs["jk_mode"],
            "encoder_name": encoder_name,
            "heads": int(encoder_kwargs.get("heads", 1)),
            "expansion_factor": guard_expansion,
            "max_tokens": float(getattr(args, "memory_guard_max_tokens", 3.5e8)),
        }

        datamodule = AIGDataModule(
            csv_paths=args.csv_paths,
            positional_encoding=pe_type if pe_type != "none" else None,
            batch_size=batch_size,
            split_ratios=(0.8, 0.2, 0.0),
            seed=dataset_seed,
            cache_dir=args.cache_dir,
            train_num_samples=args.train_samples,
            num_workers=workers,
            persistent_workers=persistent,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            dynamic_batching=dynamic_batching,
        )
        _install_hp_guarded_dataloaders(
            datamodule,
            memory_guard,
            dynamic_batching=dynamic_batching,
        )
        dataloader = getattr(datamodule, "train_dataloader", None)
        val_dataloader = getattr(datamodule, "val_dataloader", None)

        model = AIGRegressionLightningModule(
            encoder_name=encoder_name,
            hidden_dim=hidden_dim,
            pe_type=pe_type,
            pos_enc_dim=pos_enc_dim,
            pooling_type=pooling_type,
            encoder_kwargs=encoder_kwargs,
            lr=lr,
            weight_decay=weight_decay,
            huber_delta=huber_delta,
        )

        pruning_cb = PyTorchLightningPruningCallback(trial, monitor="val/mae_node")
        early_stop_cb = EarlyStopping(monitor="val/mae_node", patience=3, mode="min")

        csv_logger = CSVLogger(
            save_dir=args.log_dir,
            name="optuna_metrics",
            version=f"trial_{trial.number}",
        )

        # H100 supports BF16 tensor cores natively; fall back to FP32 elsewhere.
        try:
            _use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        except (AssertionError, RuntimeError):
            _use_bf16 = False
        _precision = "bf16-mixed" if _use_bf16 else "32-true"

        trainer = pl.Trainer(
            max_epochs=15,
            max_time={"hours": float(getattr(args, "max_trial_hours", 2.0))},
            accelerator="auto",
            devices=1,
            precision=_precision,
            gradient_clip_val=1.0,
            log_every_n_steps=10,
            callbacks=[pruning_cb, early_stop_cb],
            logger=csv_logger,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
        )

        trainer.fit(model, datamodule=datamodule)
        _log_trial_memory("post_fit", trial, args)

        optimizers = getattr(trainer, "optimizers", None)
        if optimizers:
            optimizer = optimizers[0]

        score = (
            early_stop_cb.best_score.item()
            if early_stop_cb.best_score is not None
            else float("inf")
        )
        _mark_trial_outcome(
            trial,
            outcome="completed",
            oom_like=False,
            score=score,
            risk_score=risk_score,
        )
        print(f"\n[Trial {trial.number}] COMPLETED. Score: {score:.6f}")
        return score

    except HPMemoryGuardError as e:
        msg = f"Memory guard pruned trial before allocation. [Trial {trial.number}] {e}"
        print(f"\n{msg}")
        _mark_trial_outcome(
            trial,
            outcome="pruned",
            oom_like=True,
            prune_reason="memory_guard",
            risk_score=risk_score,
        )
        _log_trial_memory("guard_pruned", trial, args)
        (
            model,
            trainer,
            datamodule,
            pruning_cb,
            early_stop_cb,
            optimizer,
            batch,
            dataloader,
            val_dataloader,
        ) = _purge_trial_memory(
            trainer=trainer,
            datamodule=datamodule,
            model=model,
            optimizer=optimizer,
            batch=batch,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            pruning_cb=pruning_cb,
            early_stop_cb=early_stop_cb,
        )
        raise optuna.TrialPruned(msg) from None

    except RuntimeError as e:
        if _is_oom_like_runtime_error(e):
            msg = (
                f"OOM-like RuntimeError. [Trial {trial.number}] "
                f"estimated_risk={risk_score:.2f} Params: {trial.params}. Error: {e}"
            )
            print(f"\n{msg}")
            _mark_trial_outcome(
                trial,
                outcome="pruned",
                oom_like=True,
                prune_reason="runtime_oom",
                risk_score=risk_score,
            )
            _log_trial_memory("oom_caught", trial, args)
            if optimizer is None and trainer is not None:
                optimizers = getattr(trainer, "optimizers", None)
                if optimizers:
                    optimizer = optimizers[0]

            (
                model,
                trainer,
                datamodule,
                pruning_cb,
                early_stop_cb,
                optimizer,
                batch,
                dataloader,
                val_dataloader,
            ) = _purge_trial_memory(
                trainer=trainer,
                datamodule=datamodule,
                model=model,
                optimizer=optimizer,
                batch=batch,
                dataloader=dataloader,
                val_dataloader=val_dataloader,
                pruning_cb=pruning_cb,
                early_stop_cb=early_stop_cb,
            )
            raise optuna.TrialPruned(msg) from None
        raise

    finally:
        (
            model,
            trainer,
            datamodule,
            pruning_cb,
            early_stop_cb,
            optimizer,
            batch,
            dataloader,
            val_dataloader,
        ) = _purge_trial_memory(
            trainer=trainer,
            datamodule=datamodule,
            model=model,
            optimizer=optimizer,
            batch=batch,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            pruning_cb=pruning_cb,
            early_stop_cb=early_stop_cb,
        )
        _log_trial_memory("post_cleanup", trial, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter Tuning for AIG Regression"
    )
    parser.add_argument("--db_url", type=str, required=True, help="SQLite DB URL, e.g. sqlite:///path/to/study.db")
    parser.add_argument(
        "--study_name", type=str, required=True, help="Optuna study name"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=".",
        help="Checkpoint directory (currently unused).",
    )
    parser.add_argument("--csv_paths", nargs="+", required=True, help="Paths to CSVs")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory in DataLoader (speeds transfer to GPU).",
    )
    parser.add_argument(
        "--persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive between batches.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=1,
        help="Batches prefetched per worker. Lower values reduce host-memory pressure.",
    )
    parser.add_argument(
        "--dynamic_batching",
        action="store_true",
        help="Enable dynamic batch construction based on graph size.",
    )
    parser.add_argument(
        "--memory_guard_max_tokens",
        type=float,
        default=3.5e8,
        help="Memory guard threshold in heuristic activation tokens.",
    )
    parser.add_argument(
        "--hard_prune_risk",
        type=float,
        default=200000.0,
        help="Prune trial before trainer start if risk score exceeds this.",
    )
    parser.add_argument(
        "--dataset_seed",
        type=int,
        default=42,
        help="Deterministic dataset split seed shared by all trials.",
    )
    parser.add_argument(
        "--memory_telemetry_trials",
        type=int,
        default=0,
        help=(
            "Log RSS/CUDA memory telemetry only for the first N trials "
            "(0 disables telemetry)."
        ),
    )
    parser.add_argument(
        "--cache_dir", type=str, help="Directory to save dataset splits"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=".",
        help="Directory to save lightning logs (default: current working directory).",
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=50000,
        help="Number of graphs for HP training",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="Maximum number of Optuna trials (None = run until wall-time limit).",
    )
    parser.add_argument(
        "--max_trial_hours",
        type=float,
        default=2.0,
        help=(
            "Per-trial wall-time cap passed to the Lightning Trainer. "
            "Set lower for Stage-1 exploration (e.g. 1.0) to avoid long trials "
            "consuming the budget that could run two shorter ones."
        ),
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=42,
        help=(
            "Random seed for the Optuna TPE sampler. Use a different value per "
            "array-job worker so each worker explores a distinct HP region."
        ),
    )
    args = parser.parse_args()

    storage = RDBStorage(url=args.db_url)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=args.sampler_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        callbacks=[_trial_outcome_callback],
    )
