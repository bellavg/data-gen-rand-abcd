import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger  #

from constants import ENCODER_KWARGS_DEFAULTS, VALID_ALGORITHMS

# Project Imports
from data.datamodule import AIGDataModule
from models.lightning_model import AIGRegressionLightningModule


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True

    # 1. Validate Algorithm
    if args.algorithm not in VALID_ALGORITHMS:
        raise ValueError(
            f"Algorithm '{args.algorithm}' must be one of {VALID_ALGORITHMS}"
        )

    # Set seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    print(f"--- Starting Final Training for Algorithm: {args.algorithm} ---")

    # 2. Setup DataModule
    datamodule = AIGDataModule(
        csv_paths=args.csv_paths,
        positional_encoding=args.pe_type if args.pe_type != "none" else None,
        batch_size=args.batch_size,
        split_ratios=(0.8, 0.1, 0.1),
        num_workers=args.num_workers,
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
        encoder_kwargs["heads"] = getattr(args, "heads", 4)

    if args.encoder_name == "egin":
        encoder_kwargs["egin_kwargs"].update(
            {
                "num_mlp_layers": 2,
                "dot_update": False,
                "edge_mlp": True,
                "edge_hidden_dim": args.hidden_dim,
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
        lr=args.lr,
        huber_delta=args.huber_delta,
        scheduler_patience=args.scheduler_patience,  #
    )

    # 5. Define Callbacks and Logger
    algo_checkpoint_dir = os.path.join(args.checkpoint_dir, args.algorithm)
    os.makedirs(algo_checkpoint_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=algo_checkpoint_dir,
        filename=f"{args.algorithm}-{{epoch:02d}}-loss={{val/loss:.4f}}-mae={{val/mae_node:.4f}}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )

    early_stop_cb = EarlyStopping(
        monitor="val/loss", patience=args.patience, mode="min", verbose=True
    )

    # Use WandbLogger
    logger = WandbLogger(
        project="aig_regression", name=f"train_{args.algorithm}", save_dir=args.log_dir
    )

    # 6. Initialize Trainer with Improvements
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        gradient_clip_val=args.gradient_clip_val,  #
        check_val_every_n_epoch=args.check_val_every_n,  #
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
    parser.add_argument("--encoder_name", type=str, default="gine")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--pe_type", type=str, default="none")
    parser.add_argument("--pos_enc_dim", type=int, default=16)
    parser.add_argument("--pooling_type", type=str, default="mean")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--norm_type", type=str, default="batch")
    parser.add_argument("--jk_mode", type=str, default="last")

    # Training Loop Parameters
    parser.add_argument("--seed", type=int, default=42)  #
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)  #
    parser.add_argument("--scheduler_patience", type=int, default=10)  #
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)  #
    parser.add_argument("--check_val_every_n", type=int, default=1)  #
    parser.add_argument("--num_workers", type=int, default=4)

    # Algorithm & Data Arguments
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--csv_paths", nargs="+", required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--hp_tuning_splits_path", type=str, default=None)

    args = parser.parse_args()
    main(args)
