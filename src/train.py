import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

import config
from data.datamodule import AIGDataModule
from models.lightning_model import AIGRegressionLightningModule
from train_utils import HardwareProfilerCallback

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
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True

    # 1. Validate Algorithm
    if args.algorithm not in config.VALID_ALGORITHMS:
        raise ValueError(
            f"Algorithm '{args.algorithm}' must be one of {config.VALID_ALGORITHMS}"
        )

    # Set seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    print(f"--- Starting Final Training for Algorithm: {args.algorithm} ---")

    # 2. Setup DataModule
    # Parse dynamic_bucket_rules string into list of (int, int) tuples
    parsed_rules = []
    if getattr(args, "dynamic_bucket_rules", None):
        try:
            parsed_rules = [
                tuple(map(int, item.split(":")))
                for item in args.dynamic_bucket_rules.split(",")
                if item.strip()
            ]
        except Exception as e:
            raise ValueError(
                f"Failed to parse dynamic_bucket_rules: {args.dynamic_bucket_rules}"
            ) from e

    datamodule = AIGDataModule(
        csv_paths=args.csv_paths,
        positional_encoding=args.pe_type if args.pe_type != "none" else None,
        batch_size=args.batch_size,
        split_ratios=(0.8, 0.1, 0.1),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor,
        cache_dir=args.cache_dir if args.cache_dir else None,
        hp_tuning_splits_path=args.hp_tuning_splits_path,
        dynamic_batching=getattr(args, "dynamic_batching", False),
        dynamic_bucket_rules=parsed_rules,
    )

    # 3. Configure Encoder Kwargs
    encoder_kwargs = ENCODER_KWARGS_DEFAULTS.copy()
    encoder_kwargs.update(
        {
            "num_layers": args.num_layers,
            "hid_dim": args.hidden_dim,
            "dropout": args.dropout,
            "norm_type": args.norm_type,
            "jk_mode": args.jk_mode,
        }
    )

    if args.encoder_name in ["transformer_conv", "graphgps"]:
        encoder_kwargs["heads"] = args.heads
    
    # 4. Initialize the Lightning Module
    model = AIGRegressionLightningModule(
        encoder_name=args.encoder_name,
        hidden_dim=args.hidden_dim,
        pe_type=args.pe_type,
        pos_enc_dim=args.pos_enc_dim if args.pe_type != "none" else 0,
        pooling_type=args.pooling_type,
        encoder_kwargs=encoder_kwargs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        huber_delta=args.huber_delta,
        scheduler_patience=args.scheduler_patience,
    )

    # 5. Define Callbacks and Logger
    algo_checkpoint_dir = os.path.join(args.checkpoint_dir, args.algorithm)
    os.makedirs(algo_checkpoint_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=algo_checkpoint_dir,
        save_top_k=3,
        save_last=True,
        monitor="val/loss",
        mode="min",
        filename="{epoch:02d}-val_loss={val/loss:.4f}", 
    )

    early_stop_cb = EarlyStopping(
        monitor="val/loss", patience=args.patience, mode="min", verbose=True
    )

    # Use WandbLogger
    logger = WandbLogger(
        project="AIG-SUMMARIZE",
        entity="isabella-v-gardner-university-of-amsterdam",
        name=f"train_{args.algorithm}",
        save_dir=args.log_dir,
    )

    # 6. Initialize Trainer with Improvements
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        precision=_select_precision(),
        callbacks=[
            checkpoint_cb,
            early_stop_cb,
            LearningRateMonitor(logging_interval="epoch"),
            HardwareProfilerCallback(),
        ],
        logger=logger,
        gradient_clip_val=args.gradient_clip_val,
        check_val_every_n_epoch=args.check_val_every_n,
    )

    # 7. Run Training & Testing
    trainer.fit(model, datamodule=datamodule)

    print(f"--- Running Test Set for {args.algorithm} ---")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Final AIG Regression Model per Algorithm"
    )

    # Hyperparameters
    parser.add_argument("--encoder_name", type=str, default=config.ENCODER_NAME)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--huber_delta", type=float, default=config.HUBER_DELTA)
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
        "--dynamic_bucket_rules", type=str, default=config.DYNAMIC_BUCKET_RULES
    )

    # Training Loop Parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--scheduler_patience", type=int, default=10)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--check_val_every_n", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)

    # Algorithm & Data Arguments
    parser.add_argument(
        "--heads",
        type=int,
        default=config.HEADS,
        help="Attention heads (transformer_conv / graphgps only)",
    )
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--csv_paths", nargs="+", required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--hp_tuning_splits_path", type=str, default=None)

    args = parser.parse_args()
    main(args)
