import argparse
import logging
import os
import sqlite3
import warnings

import optuna
import pytorch_lightning as pl
import torch
from optuna.storages import RDBStorage
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from data.datamodule import AIGDataModule
from hp_tuning_utils import (
    ATTENTION_ENCODERS,
    HPMemoryGuardError,
    _estimate_trial_risk,
    _install_hp_guarded_dataloaders,
    _mark_trial_outcome,
    _purge_trial_memory,
    _runtime_oom_prune_payload,
    _select_trainer_precision,
    _set_trial_user_attr,
)
from models.lightning_model import AIGRegressionLightningModule

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class PyTorchLightningPruningCallback(pl.Callback):
    """Minimal Optuna pruning callback for ``pytorch_lightning``.

    Replaces ``optuna.integration.PyTorchLightningPruningCallback``, which
    requires the unified ``lightning`` package.  This version depends only on
    ``pytorch_lightning`` (standalone) and ``optuna``.
    """

    def __init__(self, trial: optuna.Trial, monitor: str) -> None:
        super().__init__()
        self._trial = trial
        self._monitor = monitor

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        logs = trainer.callback_metrics
        if self._monitor not in logs:
            return
        epoch = int(trainer.current_epoch)
        value = float(logs[self._monitor])
        self._trial.report(value, step=epoch)
        if self._trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial pruned at epoch {epoch} ({self._monitor}={value:.6f})"
            )


class _MemSnapshotCallback(pl.Callback):
    """Emit ``MemoryTraceSession`` snapshots at steps 0/1000/2000/3000.

    Active only when ``SLURM_ARRAY_TASK_ID == 1`` and ``trial_number == 0``.
    All other calls return after a single boolean check — effectively free.

    Args:
        session: A ``MemoryTraceSession`` instance (already constructed).
    """

    def __init__(self, session) -> None:
        super().__init__()
        self._session = session

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Capture step-0 baseline after the initial sanity-check validation.
        if trainer.global_step == 0:
            self._session.capture_baseline(step=0)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        self._session.maybe_capture_step(trainer.global_step)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._session.clear()


# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="optuna")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.allow_tf32 = True


BATCH_SIZE_CHOICES = [4, 8, 16, 32]


# ---------------------------------------------------------------------------
# Study callbacks
# ---------------------------------------------------------------------------


def _trial_outcome_callback(study: optuna.Study, frozen_trial) -> None:
    outcome = str(
        frozen_trial.user_attrs.get("trial_outcome", frozen_trial.state.name.lower())
    )
    oom_like = bool(frozen_trial.user_attrs.get("oom_like", False))
    oom_kind = str(frozen_trial.user_attrs.get("oom_kind", "none"))
    selection_eligible = bool(frozen_trial.user_attrs.get("selection_eligible", False))
    reason = str(frozen_trial.user_attrs.get("prune_reason", ""))

    value_text = (
        "n/a" if frozen_trial.value is None else f"{float(frozen_trial.value):.6f}"
    )
    reason_text = f" reason={reason}" if reason else ""
    print(
        "[selection] "
        f"trial={frozen_trial.number} "
        f"state={frozen_trial.state.name} "
        f"outcome={outcome} "
        f"oom_like={oom_like} "
        f"oom_kind={oom_kind} "
        f"eligible={selection_eligible} "
        f"value={value_text}"
        f"{reason_text}"
    )


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Optuna objective: sample a config, train, return best val MAE.

    Args:
        trial: Active Optuna trial.
        args: Parsed CLI arguments.

    Returns:
        Best validation MAE (lower is better).

    Raises:
        optuna.TrialPruned: On OOM-like errors or pre-allocation risk exceeded.
        RuntimeError: On non-OOM runtime errors (re-raised as-is).
    """
    _mark_trial_outcome(trial, outcome="running", oom_like=False, oom_kind="none")

    batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    huber_delta = trial.suggest_float("huber_delta", 0.5, 2.0)

    encoder_name = trial.suggest_categorical(
        "encoder_name",
        ["gine", "transformer_conv", "graphgps", "egin", "gcn", "vanilla_mpnn"],
    )
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

    trainer = None
    datamodule = None
    pruning_cb = None
    early_stop_cb = None
    risk_score = None

    # Optional pre-allocation prune for clearly oversized configs.
    # Pure Python arithmetic — runs once before any GPU memory is touched.
    if args.hard_prune:
        risk_score = _estimate_trial_risk(
            batch_size=batch_size,
            num_nodes=360_000,
            num_edges=750_000,
            hidden_dim=hidden_dim,
            num_layers=encoder_kwargs["num_layers"],
            jk_mode=encoder_kwargs["jk_mode"],
            encoder_name=encoder_name,
            heads=int(encoder_kwargs.get("heads", 1)),
            pe_type=pe_type,
            pos_enc_dim=pos_enc_dim,
        )
        _set_trial_user_attr(trial, "risk_score", float(risk_score))
        if risk_score > args.hard_prune_risk:
            msg = (
                f"Pre-allocation prune for high-risk trial. "
                f"batch_size={batch_size} "
                f"estimated_risk={risk_score:.2e} threshold={args.hard_prune_risk:.2e}"
            )
            print(f"\n{msg}")
            _mark_trial_outcome(
                trial,
                outcome="pruned",
                oom_like=True,
                oom_kind="predicted",
                prune_reason="hard_prune_risk",
                risk_score=risk_score,
            )
            raise optuna.TrialPruned(msg)

    # Warn if residual GPU memory is unexpectedly high before this trial starts.
    if torch.cuda.is_available():
        pre_alloc_gib = torch.cuda.memory_allocated() / (1024**3)
        if pre_alloc_gib > 10.0:
            print(
                f"[Trial {trial.number}] WARNING: {pre_alloc_gib:.1f} GiB GPU memory "
                f"already allocated before trial start — possible residual from "
                f"previous trial cleanup."
            )

    try:
        guard_expansion = 3.0 if encoder_name in ATTENTION_ENCODERS else 2.0
        memory_guard = {
            "hidden_dim": hidden_dim,
            "num_layers": encoder_kwargs["num_layers"],
            "jk_mode": encoder_kwargs["jk_mode"],
            "encoder_name": encoder_name,
            "heads": int(encoder_kwargs.get("heads", 1)),
            "expansion_factor": guard_expansion,
            "max_tokens": float(args.memory_guard_max_tokens),
        }

        datamodule = AIGDataModule(
            csv_paths=args.csv_paths,
            positional_encoding=pe_type if pe_type != "none" else None,
            batch_size=batch_size,
            split_ratios=(0.8, 0.2, 0.0),
            seed=args.dataset_seed,
            cache_dir=args.cache_dir,
            train_num_samples=args.train_samples,
            num_workers=args.num_workers,
            persistent_workers=args.persistent_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            dynamic_batching=args.dynamic_batching,
            max_total_nodes=args.max_total_nodes_per_batch,
        )
        _install_hp_guarded_dataloaders(
            datamodule,
            memory_guard,
            dynamic_batching=args.dynamic_batching,
        )

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

        pruning_cb = PyTorchLightningPruningCallback(trial, monitor="val_mae_epoch")
        early_stop_cb = EarlyStopping(monitor="val_mae_epoch", patience=3, mode="min")
        callbacks = [pruning_cb, early_stop_cb]

        # Lightweight post-fix sanity snapshots: job 1, trial 0 only.
        # MemoryTraceSession respects MEM_TRACE_STEP_INTERVAL / MEM_TRACE_MAX_STEP
        # env vars; defaults fire at steps 1000, 2000, 3000 with a baseline at 0.
        array_job_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
        if trial.number == 0 and array_job_id == 1:
            from mem_trace import MemoryTraceSession  # noqa: PLC0415

            callbacks.append(_MemSnapshotCallback(MemoryTraceSession.from_env()))

        csv_logger = CSVLogger(
            save_dir=args.log_dir,
            name="optuna_metrics",
            version=f"trial_{trial.number}",
        )

        trainer = pl.Trainer(
            max_epochs=15,
            max_time={"hours": args.max_trial_hours},
            accelerator="auto",
            devices=1,
            precision=_select_trainer_precision(),
            gradient_clip_val=1.0,
            log_every_n_steps=max(1, args.log_every_n_steps),
            # val_check_interval=max(1, args.val_check_interval),
            callbacks=callbacks,
            logger=csv_logger,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
        )

        trainer.fit(model, datamodule=datamodule)

        score = (
            early_stop_cb.best_score.item()
            if early_stop_cb.best_score is not None
            else float("inf")
        )
        _mark_trial_outcome(
            trial,
            outcome="completed",
            oom_like=False,
            oom_kind="none",
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
            oom_kind="guard",
            prune_reason="memory_guard",
            risk_score=risk_score,
        )
        raise optuna.TrialPruned(msg) from None

    except (ConnectionResetError, BrokenPipeError) as e:
        msg = f"OOM-like IPC failure (worker killed). [Trial {trial.number}] Error: {e}"
        print(f"\n{msg}")
        _mark_trial_outcome(
            trial,
            outcome="pruned",
            oom_like=True,
            oom_kind="host",
            prune_reason="runtime_host_oom",
            risk_score=risk_score,
        )
        raise optuna.TrialPruned(msg) from None

    except RuntimeError as e:
        payload = _runtime_oom_prune_payload(e, trial_number=trial.number)
        if payload is None:
            raise

        oom_kind, prune_reason, oom_like, msg = payload
        print(f"\n{msg}")
        if oom_kind == "host" and args.num_workers > 0:
            print(
                "[oom_hint] DataLoader worker host-OOM detected while num_workers>0. "
                "Use --num_workers 0 for Stage-2 stability on heavy-tail graphs."
            )
        _mark_trial_outcome(
            trial,
            outcome="pruned",
            oom_like=oom_like,
            oom_kind=oom_kind,
            prune_reason=prune_reason,
            risk_score=risk_score,
        )
        raise optuna.TrialPruned(msg) from None

    finally:
        _purge_trial_memory(
            trainer=trainer,
            datamodule=datamodule,
            pruning_cb=pruning_cb,
            early_stop_cb=early_stop_cb,
        )


# ---------------------------------------------------------------------------
# Stage-2 seeding from Stage-1 results
# ---------------------------------------------------------------------------


def _seed_study_from_best(
    study: optuna.Study,
    *,
    source_db_url: str,
    source_study_name: str,
    top_n: int,
    seed_mode: str = "import",
) -> None:
    """Seed a Stage-2 study from the top-N completed trials of a Stage-1 study.

    Args:
        study: Target Stage-2 Optuna study.
        source_db_url: SQLite URL for the Stage-1 database.
        source_study_name: Study name inside ``source_db_url``.
        top_n: Number of top trials to seed from.
        seed_mode: ``"enqueue"`` re-runs the configs; ``"import"`` adds them
            as already-completed observations without re-running.
    """
    if top_n <= 0:
        return

    source_storage = RDBStorage(
        url=source_db_url,
        engine_kwargs={"connect_args": {"timeout": 60, "check_same_thread": False}},
    )
    source_study = optuna.load_study(
        study_name=source_study_name,
        storage=source_storage,
    )

    eligible = [
        t
        for t in source_study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.user_attrs.get("selection_eligible", False)
        and t.value is not None
    ]
    eligible.sort(key=lambda t: t.value)
    top_trials = eligible[:top_n]

    if not top_trials:
        print("[seed] No eligible Stage-1 trials found to seed from.")
        return

    action = "Seeding" if seed_mode == "enqueue" else "Importing"
    verb = "enqueued" if seed_mode == "enqueue" else "imported"
    print(f"[seed] {action} Stage-2 study with top-{len(top_trials)} Stage-1 trials.")

    for t in top_trials:
        seed_attrs = {
            **t.user_attrs,
            "seeded_from_stage1": True,
            "seed_source_trial_number": int(t.number),
            "seed_mode": seed_mode,
        }
        if seed_mode == "enqueue":
            study.enqueue_trial(t.params, user_attrs=seed_attrs, skip_if_exists=True)
        else:
            study.add_trial(
                optuna.trial.create_trial(
                    params=t.params,
                    distributions=t.distributions,
                    value=t.value,
                    user_attrs=seed_attrs,
                )
            )
        print(f"[seed]   {verb} trial #{t.number} value={t.value:.6f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Prefer filesystem-backed tensor sharing for large graph batches to avoid
    # file-descriptor pressure in long multi-worker DataLoader runs.
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except (AttributeError, RuntimeError, ValueError) as exc:
        print(f"[ipc] WARNING: failed to set sharing strategy to file_system: {exc}")

    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter Tuning for AIG Regression"
    )
    parser.add_argument(
        "--db_url",
        type=str,
        default="",
        help="SQLite DB URL, e.g. sqlite:///path/to/study.db (unused when --in_memory_storage is set)",
    )
    parser.add_argument(
        "--in_memory_storage",
        action="store_true",
        help=(
            "Use optuna.storages.InMemoryStorage instead of SQLite RDBStorage. "
            "Eliminates WAL mmap growth. Suitable for single-worker runs where "
            "cross-process study sharing is not needed."
        ),
    )
    parser.add_argument(
        "--study_name", type=str, required=True, help="Optuna study name"
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
        "--max_total_nodes_per_batch",
        type=int,
        default=1_000_000,
        help="Maximum total node budget per dynamically constructed batch.",
    )
    parser.add_argument(
        "--memory_guard_max_tokens",
        type=float,
        default=3.5e8,
        help="Memory guard threshold in heuristic activation tokens.",
    )
    parser.add_argument(
        "--hard_prune",
        action="store_true",
        help="Enable pre-allocation pruning of high-risk trials (default: off).",
    )
    parser.add_argument(
        "--hard_prune_risk",
        type=float,
        default=1e10,
        help=(
            "Risk threshold used when --hard_prune is set. Evaluated against a "
            "360K-node graph at each sampled trial batch_size."
        ),
    )
    parser.add_argument(
        "--dataset_seed",
        type=int,
        default=42,
        help="Deterministic dataset split seed shared by all trials.",
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
            "Set lower for Stage-1 exploration (e.g. 1.0)."
        ),
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=1000,
        help="Trainer metric logging interval in steps (default: 1000).",
    )
    # parser.add_argument(
    #     "--val_check_interval",
    #     type=int,
    #     default=1.0,
    #     help=(
    #         "Run validation every N train steps so pruning/early-stop decisions "
    #         "happen before epoch end on long epochs."
    #     ),
    # )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=42,
        help=(
            "Random seed for the Optuna TPE sampler. Use a different value per "
            "array-job worker so each worker explores a distinct HP region."
        ),
    )
    parser.add_argument(
        "--seed_from_db_url",
        type=str,
        default=None,
        help=(
            "Stage-1 SQLite DB URL to seed Stage-2 from (e.g. sqlite:///stage1.db). "
            "If set, the top --seed_top_n trials from that study are enqueued first."
        ),
    )
    parser.add_argument(
        "--seed_study_name",
        type=str,
        default=None,
        help="Study name inside --seed_from_db_url to read Stage-1 results from.",
    )
    parser.add_argument(
        "--seed_top_n",
        type=int,
        default=20,
        help="Number of top Stage-1 trials to enqueue at the start of Stage-2 (default: 20).",
    )
    parser.add_argument(
        "--seed_mode",
        type=str,
        choices=["enqueue", "import"],
        default="import",
        help=(
            "How to apply Stage-1 seeds in Stage-2: 'enqueue' re-runs seeded configs "
            "first; 'import' adds them as completed observations without re-running."
        ),
    )
    args = parser.parse_args()

    # SQLite WAL setup — only when using RDB storage.
    if not args.in_memory_storage and args.db_url.startswith("sqlite:///"):
        db_path = args.db_url[len("sqlite:///") :]
        _wal_con = sqlite3.connect(db_path)
        _wal_con.execute("PRAGMA journal_mode=WAL;")
        _wal_con.close()

    if args.in_memory_storage:
        storage = optuna.storages.InMemoryStorage()
        print("[storage] Using InMemoryStorage (no SQLite WAL)", flush=True)
    else:
        storage = RDBStorage(
            url=args.db_url,
            engine_kwargs={"connect_args": {"timeout": 60, "check_same_thread": False}},
        )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            multivariate=True, seed=args.sampler_seed, warn_independent_sampling=False
        ),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    if args.seed_from_db_url and args.seed_study_name:
        _seed_study_from_best(
            study,
            source_db_url=args.seed_from_db_url,
            source_study_name=args.seed_study_name,
            top_n=args.seed_top_n,
            seed_mode=args.seed_mode,
        )

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        callbacks=[_trial_outcome_callback],
    )

    try:
        best = study.best_trial
        print(f"\n{'=' * 60}")
        print(f"BEST TRIAL: #{best.number}  score={best.value:.6f}")
        for k, v in best.params.items():
            print(f"  {k}: {v}")
        print(f"{'=' * 60}\n")
    except ValueError:
        print("No completed trials found.")
