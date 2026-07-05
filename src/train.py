import argparse
import os
import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

import config
from config import HEAD_DROPOUT, NORMALIZE_EDGES
from data.datamodule import AIGDataModule
from models.lightning_model import AIGRegressionLightningModule
from train_utils import PreciseEarlyStopping, TrainingStartupCallback

torch.set_num_threads(1)

ENCODER_KWARGS_DEFAULTS = config.ENCODER_KWARGS_DEFAULTS


def _select_precision() -> str:
    try:
        return (
            "bf16-mixed"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "32-true"
        )
    except (AssertionError, RuntimeError):
        return "32-true"


def main(args):
    if getattr(args, "enable_hardware_profiler", False):
        print(
            "[train] --enable_hardware_profiler is deprecated and ignored; "
            "epoch timing is logged automatically and WandB captures hardware telemetry.",
            flush=True,
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision("high")

    torch.multiprocessing.set_sharing_strategy("file_system")

    # 1. Validate Algorithm
    if args.algorithm not in config.VALID_ALGORITHMS:
        raise ValueError(
            f"Algorithm '{args.algorithm}' must be one of {config.VALID_ALGORITHMS}"
        )

    # 2. Setup DataModule & load datasets EARLY (CPU work — no GPU needed)
    #    This runs before WandB/Trainer so that slow network calls or
    #    Lightning overhead don't burn GPU wall-clock time.
    datamodule = AIGDataModule(
        csv_paths=args.csv_paths,
        positional_encoding=args.pe_type if args.pe_type != "none" else None,
        sparsification=args.sparsification,
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
        dynamic_batching=getattr(args, "dynamic_batching", False),
        max_total_nodes=args.max_total_nodes_per_batch,
    )

    print("[main] Loading datasets before Trainer/WandB init ...", flush=True)
    ds_start = time.monotonic()
    datamodule.setup("fit")
    print(
        f"[main] Datasets loaded in {time.monotonic() - ds_start:.1f}s",
        flush=True,
    )

    # Seed torch RNG for reproducible weight init, dropout, and DataLoader
    # shuffle.  Lighter than pl.seed_everything (which also seeds numpy,
    # stdlib random, and prints to stdout).
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 3. Configure Encoder Kwargs
    encoder_kwargs = ENCODER_KWARGS_DEFAULTS.copy()
    encoder_kwargs.update(
        {
            "num_layers": args.num_layers,
            "hid_dim": args.hidden_dim,
            "dropout": args.dropout,
            "norm_type": args.norm_type,
            "jk_mode": args.jk_mode,
            "normalize_edges": NORMALIZE_EDGES,
        }
    )

    # 4. Initialize the Lightning Module
    model = AIGRegressionLightningModule(
        encoder_name=args.encoder_name,
        hidden_dim=args.hidden_dim,
        pe_type=args.pe_type,
        pos_enc_dim=args.pos_enc_dim if args.pe_type != "none" else 0,
        pooling_type=args.pooling_type,
        encoder_kwargs=encoder_kwargs,
        head_dropout=HEAD_DROPOUT,
        lr=args.lr,
        weight_decay=args.weight_decay,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        warmup_start_lr=args.warmup_start_lr,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        loss_fn=nn.SmoothL1Loss(beta=0.01),
    )

    # 5. Define Callbacks and Logger
    sparsification_name = args.sparsification or "none"
    algo_checkpoint_dir = os.path.join(args.checkpoint_dir, f"{args.algorithm}_{sparsification_name}")
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

    # Use WandbLogger
    # WandB — init AFTER datasets are loaded so network delays don't waste
    # GPU allocation time.  WANDB_INIT_TIMEOUT caps the API handshake.
    log_dir = f"{args.log_dir}_{sparsification_name}"
    os.makedirs(log_dir, exist_ok=True)
    print("[main] Initialising WandB logger ...", flush=True)
    wandb_start = time.monotonic()
    logger = WandbLogger(
        project="AIG-SUMMARIZE",
        entity="isabella-v-gardner-university-of-amsterdam",
        name=f"train_{args.algorithm}_sparsification_{sparsification_name}",
        save_dir=log_dir,
    )
    print(f"[main] WandB ready in {time.monotonic() - wandb_start:.1f}s", flush=True)

    # 6. Initialize Trainer with Improvements
    callbacks = [
        checkpoint_cb,
        early_stop_cb,
        LearningRateMonitor(logging_interval="epoch"),
        TrainingStartupCallback(
            report_every_n_steps=args.log_steps,
            max_batch_compute_reports=args.max_batch_compute_reports,
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        enable_progress_bar=False,
        devices=1,
        precision=_select_precision(),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=args.gradient_clip_val,
        val_check_interval=args.val_check_interval,
        num_sanity_val_steps=args.num_sanity_val_steps,
        log_every_n_steps=args.log_steps,
    )

    # 7. Run Training & Testing
    print(f"--- Running Training for {args.algorithm} ---")
    fit_start = time.monotonic()
    trainer.fit(model, datamodule=datamodule)
    print(
        f"--- trainer.fit completed for {args.algorithm} in {time.monotonic() - fit_start:.1f}s ---",
        flush=True,
    )

    # print(f"--- Running Test Set for {args.algorithm} ---")
    # trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Final AIG Regression Model per Algorithm"
    )

    # Hyperparameters
    parser.add_argument("--encoder_name", type=str, default=config.ENCODER_NAME)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--hidden_dim", type=int, default=config.HIDDEN_DIM)
    parser.add_argument("--pe_type", type=str, default=config.PE_TYPE)
    parser.add_argument("--pos_enc_dim", type=int, default=config.POS_ENC_DIM)
    parser.add_argument("--pooling_type", type=str, default=config.POOLING_TYPE)
    parser.add_argument("--num_layers", type=int, default=config.NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=config.DROPOUT)
    parser.add_argument("--norm_type", type=str, default=config.NORM_TYPE)
    parser.add_argument("--jk_mode", type=str, default=config.JK_MODE)
    parser.add_argument(
        "--dynamic_batching",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=config.DYNAMIC_BATCHING,
    )
    parser.add_argument(
        "--max_total_nodes_per_batch",
        type=int,
        default=config.MAX_TOTAL_NODES_PER_BATCH,
    )

    # Training Loop Parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--min_lr", type=float, default=config.MIN_LR)
    parser.add_argument("--warmup_steps", type=int, default=config.WARMUP_STEPS)
    parser.add_argument(
        "--warmup_start_lr", type=float, default=config.WARMUP_START_LR
    )
    parser.add_argument(
        "--scheduler_patience", type=int, default=config.SCHEDULER_PATIENCE
    )
    parser.add_argument(
        "--scheduler_factor", type=float, default=config.SCHEDULER_FACTOR
    )
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--check_val_every_n", type=float, default=0.5)
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=1.0,
        help=(
            "Validation frequency: if <1.0, fraction of training epoch; "
            "if >=1.0, number of training batches between validations."
        ),
    )
    parser.add_argument(
        "--num_sanity_val_steps",
        type=int,
        default=0,
        help="Lightning sanity-validation batches before training. Default 0 for final-train startup.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=8)
    parser.add_argument(
        "--enable_hardware_profiler",
        action="store_true",
        help="Deprecated no-op kept for CLI compatibility; WandB already captures hardware telemetry.",
    )
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
    parser.add_argument("--log_steps", type=int, default=config.LOG_EVERY_N_STEPS)
    parser.add_argument(
        "--max_batch_compute_reports",
        type=int,
        default=config.MAX_BATCH_COMPUTE_REPORTS,
        help=(
            "Maximum number of '[train] Batch compute' lines to print "
            "during the full run."
        ),
    )

    # Algorithm & Data Arguments
    parser.add_argument(
        "--heads",
        type=int,
        default=getattr(config, "HEADS", 4),
        help="Attention heads (transformer_conv / graphgps only)",
    )
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--csv_paths", nargs="+", required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--tier0_cache_dir", type=str, default=None)
    parser.add_argument("--tier1_cache_dir", type=str, default=None)
    parser.add_argument("--hp_tuning_splits_path", type=str, default=None)
    parser.add_argument(
        "--sparsification",
        type=lambda x: x.lower() if x.lower() != "none" else None,
        default=None,
        help="Sparsification algorithm to apply (e.g. 'random_edge_dropout'). Pass 'none' to disable.",
    )

    args = parser.parse_args()
    main(args)
