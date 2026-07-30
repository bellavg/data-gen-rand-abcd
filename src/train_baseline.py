"""Training entrypoint for baseline models (SynthNet, HOGA) on this project's AIG dataset.

Mirrors train.py's structure (argparse -> AIGDataModule -> Lightning module ->
Trainer) but swaps in a baseline model + baselines.common.lightning_wrapper
instead of the primary UnifiedGraphBaseModel/AIGRegressionLightningModule, so
train.py and the primary model stay completely untouched.

Baseline model hyperparameters and training config (optimizer, loss, LR,
scheduler) default to each paper's own published values (SynthNet:
models/qor/SynthNetV3/train.py; HOGA: Deng et al. DAC'24 Section 3.3/4.1 --
see baselines/hoga/regressor.py's module docstring for exactly which values
are published vs. assumed, since a couple of HOGA's knobs -- heads, dropout --
still have no published QoR-task source), not this project's own config.py
defaults -- those are two separate baseline papers with their own training
setups. Only data
loading/splitting/caching (AIGDataModule / AIGGraphRegressionDataset) is
reused unchanged, since identical splits are required for a fair comparison
against the primary model, and that part isn't "baseline config".

Both baselines use a plain fixed batch size rather than this project's
node-budget dynamic batching -- dynamic batching is this project's own
OOM-avoidance mechanism for the primary model, not part of either paper's
published training config.
"""

from __future__ import annotations

import argparse
import os
import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

import config
from baselines.common.lightning_wrapper import BaselineRegressionLightningModule
from baselines.hoga.hop_features import HopFeatureCache, collate_hoga_batch, num_hop_slots
from baselines.hoga.regressor import DEFAULT_HEADS as HOGA_DEFAULT_HEADS
from baselines.hoga.regressor import DEFAULT_HIDDEN_DIM as HOGA_DEFAULT_HIDDEN_DIM
from baselines.hoga.regressor import DEFAULT_LR as HOGA_DEFAULT_LR
from baselines.hoga.regressor import DEFAULT_NUM_HOPS as HOGA_DEFAULT_NUM_HOPS
from baselines.hoga.regressor import DEFAULT_NUM_LAYERS as HOGA_DEFAULT_NUM_LAYERS
from baselines.hoga.regressor import HOGAGraphRegressor
from baselines.openabc_synthnet.regressor import (
    DEFAULT_DROP_RATIO,
    DEFAULT_FC_HIDDEN_DIM,
    DEFAULT_GNN_HIDDEN_DIM,
    DEFAULT_NODE_EMB_DIM,
    DEFAULT_NUM_FC_LAYER,
    SynthNetGraphRegressor,
)
from data.datamodule import AIGDataModule
from train_utils import PreciseEarlyStopping, TrainingStartupCallback

torch.set_num_threads(1)

# Published defaults for each baseline paper's own training setup -- see the
# regressor modules for exactly where each of these comes from.
SYNTHNET_DEFAULTS = {"batch_size": 64, "lr": 0.001, "weight_decay": 0.0}
HOGA_DEFAULTS = {
    "batch_size": config.BATCH_SIZE,  # no published QoR-task batch size; see baselines/hoga/regressor.py
    "lr": HOGA_DEFAULT_LR,  # 0.0001, published (Deng et al. DAC'24, Sec 3.3/4.1)
    "weight_decay": 0.0,
}


def _select_precision() -> str:
    try:
        return (
            "bf16-mixed"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "32-true"
        )
    except (AssertionError, RuntimeError):
        return "32-true"


def _select_accelerator_and_devices() -> tuple[str, int]:
    require_gpu = str(os.environ.get("AIG_REQUIRE_GPU", "")).lower() in {
        "1",
        "true",
        "yes",
    }
    try:
        if torch.cuda.is_available():
            torch.cuda.get_device_properties(0)
            return "gpu", 1
    except (AssertionError, RuntimeError) as exc:
        if require_gpu:
            raise RuntimeError(
                "GPU was requested but CUDA could not be initialized. "
                "On SLURM this usually means the Python process was not launched "
                "inside a GPU job step, CUDA_VISIBLE_DEVICES is wrong, or the node "
                "driver is unhealthy. Try launching with srun and check nvidia-smi."
            ) from exc
        return "cpu", 1
    if require_gpu:
        raise RuntimeError(
            "GPU was requested but torch.cuda.is_available() is False. "
            "Check the SLURM GPU allocation, CUDA_VISIBLE_DEVICES, and nvidia-smi output."
        )
    return "cpu", 1


def _build_model(args: argparse.Namespace) -> nn.Module:
    if args.baseline == "synthnet":
        return SynthNetGraphRegressor(
            node_emb_dim=args.synthnet_node_emb_dim,
            gnn_hidden_dim=args.synthnet_gnn_hidden_dim,
            num_fc_layer=args.synthnet_num_fc_layer,
            fc_hidden_dim=args.synthnet_fc_hidden_dim,
            drop_ratio=args.synthnet_drop_ratio,
            task_out_dim=config.TASK_OUT_DIM,
        )
    if args.baseline == "hoga":
        return HOGAGraphRegressor(
            in_channels=config.NODE_INPUT_DIM,
            hidden_channels=args.hoga_hidden_dim,
            num_layers=args.hoga_num_layers,
            dropout=args.hoga_dropout,
            num_hops=num_hop_slots(args.hoga_num_hops, directed=args.hoga_directed),
            heads=args.hoga_heads,
            head_dropout=args.hoga_head_dropout,
            task_out_dim=config.TASK_OUT_DIM,
        )
    raise ValueError(f"Unknown baseline: {args.baseline!r}")


def _loader_kwargs(args: argparse.Namespace) -> dict:
    kwargs: dict = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def _plain_loader(ds, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds, shuffle=shuffle, collate_fn=Batch.from_data_list, **_loader_kwargs(args)
    )


def _hoga_loader(ds, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    wrapped = HopFeatureCache(
        ds,
        num_hops=args.hoga_num_hops,
        cache_dir=args.hoga_hop_cache_dir,
        directed=args.hoga_directed,
    )
    return DataLoader(
        wrapped, shuffle=shuffle, collate_fn=collate_hoga_batch, **_loader_kwargs(args)
    )


def main(args: argparse.Namespace) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.multiprocessing.set_sharing_strategy("file_system")

    if args.algorithm not in config.VALID_ALGORITHMS:
        raise ValueError(
            f"Algorithm '{args.algorithm}' must be one of {config.VALID_ALGORITHMS}"
        )
    if args.baseline == "hoga" and not args.hoga_hop_cache_dir:
        raise ValueError("--hoga_hop_cache_dir is required when --baseline hoga")

    defaults = SYNTHNET_DEFAULTS if args.baseline == "synthnet" else HOGA_DEFAULTS
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.weight_decay is None:
        args.weight_decay = defaults["weight_decay"]

    print(f"--- Starting Baseline Training: {args.baseline} / {args.algorithm} ---")

    datamodule = AIGDataModule(
        csv_paths=args.csv_paths,
        # Neither baseline reads .pos_enc -- both derive their own features
        # from .x/.edge_index/.edge_attr directly -- but the per-graph cache
        # filename AND content in dataset.py both key on positional_encoding
        # (see _stable_graph_cache_name/_prepare_cached_graph). Passing None
        # here would silently miss the primary model's existing shared
        # tier0_cache_dir/tier1_cache_dir cache entirely (different hash,
        # different file) and rebuild a full second copy of the same ~700k
        # graphs from scratch. Matching config.PE_TYPE makes cache lookups
        # hit the already-built shared cache; the resulting unused pos_enc
        # attribute is harmless (the baseline models never read it).
        positional_encoding=config.PE_TYPE if config.PE_TYPE != "none" else None,
        sparsification=None,
        partition=None,
        batch_size=args.batch_size,
        split_ratios=(0.8, 0.1, 0.1),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        cache_dir=args.cache_dir if args.cache_dir else None,
        hp_tuning_splits_path=args.hp_tuning_splits_path,
        tier0_cache_dir=args.tier0_cache_dir,
        tier1_cache_dir=args.tier1_cache_dir,
        dynamic_batching=False,  # both baselines use a fixed batch size; see module docstring
    )

    print("[main] Loading datasets before Trainer/WandB init ...", flush=True)
    ds_start = time.monotonic()
    # "fit" only, matching train.py. The test split is evaluated separately,
    # and setting up the test stage here would build a graph cache for ~96k
    # test graphs during the GPU job (warmup_train_cache.sh warms train + val
    # only, by design).
    datamodule.setup("fit")
    print(
        f"[main] Datasets loaded in {time.monotonic() - ds_start:.1f}s", flush=True
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.baseline == "hoga":
        train_loader = _hoga_loader(datamodule.train_ds, args, shuffle=True)
        val_loader = _hoga_loader(datamodule.val_ds, args, shuffle=False)
    else:
        train_loader = _plain_loader(datamodule.train_ds, args, shuffle=True)
        val_loader = _plain_loader(datamodule.val_ds, args, shuffle=False)

    base_model = _build_model(args)
    model = BaselineRegressionLightningModule(
        base_model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name="adam",
        loss_fn=nn.MSELoss(),
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
    )

    run_label = f"{args.baseline}_{args.algorithm}"
    algo_checkpoint_dir = os.path.join(args.checkpoint_dir, run_label)
    os.makedirs(algo_checkpoint_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=algo_checkpoint_dir,
        save_top_k=3,
        save_last=True,
        monitor="val_loss",
        mode="min",
        filename="{epoch:02d}-val_loss={val_loss:.4f}",
        save_on_train_epoch_end=True,
    )
    early_stop_cb = PreciseEarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min",
        verbose=True,
        check_on_train_epoch_end=True,
    )

    log_dir = f"{args.log_dir}_{run_label}"
    os.makedirs(log_dir, exist_ok=True)
    print("[main] Initialising WandB logger ...", flush=True)
    wandb_start = time.monotonic()
    logger = WandbLogger(
        project="AIG-SUMMARIZE",
        entity="isabella-v-gardner-university-of-amsterdam",
        name=f"train_baseline_{run_label}",
        save_dir=log_dir,
    )
    print(f"[main] WandB ready in {time.monotonic() - wandb_start:.1f}s", flush=True)

    callbacks = [
        checkpoint_cb,
        early_stop_cb,
        LearningRateMonitor(logging_interval="epoch"),
        TrainingStartupCallback(
            report_every_n_steps=args.log_steps,
            max_batch_compute_reports=args.max_batch_compute_reports,
        ),
    ]

    precision = _select_precision()
    accelerator, devices = _select_accelerator_and_devices()
    print(f"Using {precision} Automatic Mixed Precision (AMP)", flush=True)
    print(f"Using accelerator={accelerator}, devices={devices}", flush=True)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        enable_progress_bar=False,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=args.gradient_clip_val,
        # Match train.py's explicit 0 (Lightning's own default is 2) so the
        # baseline and the primary model start training from the same point.
        num_sanity_val_steps=0,
        log_every_n_steps=args.log_steps,
    )

    print(
        f"--- Running Training for baseline={args.baseline} algorithm={args.algorithm} ---",
        flush=True,
    )
    fit_start = time.monotonic()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(
        f"--- trainer.fit completed in {time.monotonic() - fit_start:.1f}s ---",
        flush=True,
    )

    # No trainer.test() here: the test split is evaluated separately, from the
    # saved checkpoints. Matches train.py, which also fits only.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a baseline model (SynthNet or HOGA) on this project's AIG dataset"
    )

    parser.add_argument(
        "--baseline", type=str, required=True, choices=["synthnet", "hoga"]
    )
    parser.add_argument("--algorithm", type=str, default="Orchestrate")
    parser.add_argument("--csv_paths", nargs="+", required=True)

    # Training config: default to None, resolved per-baseline from
    # SYNTHNET_DEFAULTS / HOGA_DEFAULTS in main() (see module docstring).
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    # Unlike lr/batch_size/loss/optimizer above, the ReduceLROnPlateau settings
    # are NOT published: neither baseline paper uses an LR scheduler at all (see
    # baselines/common/lightning_wrapper.py). Since the values are ours either
    # way, take config.py's -- the same schedule the primary model trains under,
    # so the comparison differs by architecture rather than LR schedule. The
    # previous 0.1/10 also left the scheduler inert: patience 10 epochs can
    # never fire under --patience 4 early stopping.
    parser.add_argument(
        "--scheduler_factor", type=float, default=config.SCHEDULER_FACTOR
    )
    parser.add_argument(
        "--scheduler_patience", type=int, default=config.SCHEDULER_PATIENCE
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_epochs", type=int, default=80
    )  # models/qor/SynthNetV3/train.py default
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--log_steps", type=int, default=config.LOG_EVERY_N_STEPS)
    parser.add_argument(
        "--max_batch_compute_reports",
        type=int,
        default=config.MAX_BATCH_COMPUTE_REPORTS,
    )

    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--prefetch_factor", type=int, default=config.PREFETCH_FACTOR)
    parser.add_argument(
        "--pin_memory",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=config.PIN_MEMORY,
    )
    parser.add_argument(
        "--persistent_workers",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=config.PERSISTENT_WORKERS,
    )

    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--tier0_cache_dir", type=str, default=None)
    parser.add_argument("--tier1_cache_dir", type=str, default=None)
    parser.add_argument("--hp_tuning_splits_path", type=str, default=None)
    # No --split_by flag: this branch's AIGGraphRegressionDataset hardcodes
    # design-level splitting (see data/dataset.py, "split_by": "design" baked
    # into its cache signature) -- there's no alternative to select on this
    # branch. A configurable split_by (design/recipe/random) was added on
    # main after this branch diverged; it isn't available here.

    # SynthNet hyperparameters (defaults: models/qor/SynthNetV3/train.py).
    parser.add_argument(
        "--synthnet_node_emb_dim", type=int, default=DEFAULT_NODE_EMB_DIM
    )
    parser.add_argument(
        "--synthnet_gnn_hidden_dim", type=int, default=DEFAULT_GNN_HIDDEN_DIM
    )
    parser.add_argument(
        "--synthnet_num_fc_layer", type=int, default=DEFAULT_NUM_FC_LAYER
    )
    parser.add_argument(
        "--synthnet_fc_hidden_dim", type=int, default=DEFAULT_FC_HIDDEN_DIM
    )
    parser.add_argument("--synthnet_drop_ratio", type=float, default=DEFAULT_DROP_RATIO)

    # HOGA hyperparameters. hidden_dim/num_layers/num_hops/lr are published
    # (Deng et al. DAC'24, Sec 3.3/4.1); heads carries over from the Gamora
    # task's run.sh; dropout has no published source for either task -- see
    # baselines/hoga/regressor.py's module docstring for the full breakdown.
    parser.add_argument(
        "--hoga_hidden_dim", type=int, default=HOGA_DEFAULT_HIDDEN_DIM
    )
    parser.add_argument(
        "--hoga_num_layers", type=int, default=HOGA_DEFAULT_NUM_LAYERS
    )
    parser.add_argument("--hoga_dropout", type=float, default=config.DROPOUT)
    parser.add_argument(
        "--hoga_num_hops",
        type=int,
        default=HOGA_DEFAULT_NUM_HOPS,
        help="Propagation depth per direction, i.e. K (see baselines/hoga/hop_features.py).",
    )
    parser.add_argument("--hoga_heads", type=int, default=HOGA_DEFAULT_HEADS)
    parser.add_argument("--hoga_head_dropout", type=float, default=0.3)
    parser.add_argument(
        "--hoga_directed",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=True,
    )
    parser.add_argument(
        "--hoga_hop_cache_dir",
        type=str,
        default=None,
        help="Required when --baseline hoga; see baselines/hoga/hop_features.py.",
    )

    args = parser.parse_args()
    main(args)
