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
from data.summarize_graphs import assert_exact_depth_supports_model
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


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.multiprocessing.set_sharing_strategy("file_system")

    if args.algorithm not in config.VALID_ALGORITHMS:
        raise ValueError(
            f"Algorithm '{args.algorithm}' must be one of {config.VALID_ALGORITHMS}"
        )

    if args.sparsification is not None and args.partition is not None:
        raise ValueError(
            "--sparsification and --partition are mutually exclusive; set only one."
        )

    exact = args.model == "exact"
    # The exact model consumes no positional encoding — but --pe_type also
    # decides the graph cache *filename* (dataset._stable_graph_cache_name
    # hashes it), and the reducts inherited their filenames from the
    # production pe=level cache the precompute read.  So the dataset keeps
    # --pe_type and only the model drops it.  Passing --pe_type none instead
    # would make every reduct lookup miss, silently re-cache the raw
    # uncoarsened graphs into the staging directory, and only fail hours
    # later inside the model.
    model_pe_type = "none" if exact else args.pe_type
    if exact:
        # The reducts were built at a fixed refinement depth and then cached;
        # --num_layers is tuned long afterwards, and a deeper model quietly
        # stops being exact.  Check before anything expensive starts.
        assert_exact_depth_supports_model(
            [d for d in (args.tier0_cache_dir, args.tier1_cache_dir) if d],
            args.num_layers,
        )

    print(f"--- Starting Final Training for Algorithm: {args.algorithm} ---")

    # 2. Setup DataModule & load datasets EARLY (CPU work — no GPU needed)
    #    This runs before WandB/Trainer so that slow network calls or
    #    Lightning overhead don't burn GPU wall-clock time.
    datamodule = AIGDataModule(
        csv_paths=args.csv_paths,
        positional_encoding=args.pe_type if args.pe_type != "none" else None,
        sparsification=args.sparsification,
        partition=args.partition,
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
        exact_schema=exact,
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
    if exact:
        # The exact encoder has no normalization and no edge features, so
        # leaving these in would have WandB report norm_type="layer" for a
        # model that has no norm at all — the exact class of silently-wrong
        # reporting the model selector exists to prevent.
        for key in ("norm_type", "edge_attr_dim", "normalize_edges"):
            encoder_kwargs.pop(key, None)

    # 4. Initialize the Lightning Module
    model = AIGRegressionLightningModule(
        encoder_name=args.encoder_name,
        hidden_dim=args.hidden_dim,
        node_input_dim=(
            config.EXACT_NODE_INPUT_DIM if exact else config.NODE_INPUT_DIM
        ),
        pe_type=model_pe_type,
        pos_enc_dim=args.pos_enc_dim if model_pe_type != "none" else 0,
        pooling_type=args.pooling_type,
        model_type=args.model,
        encoder_kwargs=encoder_kwargs,
        head_dropout=HEAD_DROPOUT,
        lr=args.lr,
        weight_decay=args.weight_decay,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        warmup_start_lr=args.warmup_start_lr,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        compile_model=args.torch_compile,
        loss_fn=nn.SmoothL1Loss(beta=0.01),
    )

    # 5. Define Callbacks and Logger
    # When neither sparsification nor partitioning is active, use just the
    # algorithm name so checkpoints, logs, and WandB runs are named
    # "Orchestrate" etc. Otherwise append whichever one is active as a
    # suffix (they are mutually exclusive, enforced above).
    if args.sparsification is not None:
        run_label = f"{args.algorithm}_{args.sparsification}"
        wandb_run_name = f"train_{args.algorithm}_sparsification_{args.sparsification}"
    elif args.partition is not None:
        run_label = f"{args.algorithm}_{args.partition}"
        wandb_run_name = f"train_{args.algorithm}_partition_{args.partition}"
    else:
        run_label = args.algorithm
        wandb_run_name = f"train_{args.algorithm}"

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

    # WandB — init AFTER datasets are loaded so network delays don't waste
    # GPU allocation time.  WANDB_INIT_TIMEOUT caps the API handshake.
    log_dir = f"{args.log_dir}_{run_label}"
    os.makedirs(log_dir, exist_ok=True)
    print("[main] Initialising WandB logger ...", flush=True)
    wandb_start = time.monotonic()
    logger = WandbLogger(
        project="AIG-SUMMARIZE",
        entity="isabella-v-gardner-university-of-amsterdam",
        name=wandb_run_name,
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
        val_check_interval=args.val_check_interval,
        num_sanity_val_steps=args.num_sanity_val_steps,
        log_every_n_steps=args.log_steps,
    )

    # 7. Run Training & Testing
    print(f"--- Running Training for {args.algorithm} ---", flush=True)
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
    parser.add_argument(
        "--model",
        type=str,
        choices=("production", "exact"),
        default="production",
        help=(
            "Base model. 'exact' selects the exact-compression track "
            "(ExactGraphBaseModel): no normalization, size-weighted pooling, "
            "edge_weight multiplicity, no edge_attr, pe_type must be 'none'. "
            "Use it only with a cache built by an exact summarization method "
            "(wl_exact). Deliberately explicit rather than inferred from the "
            "data — see models.lightning_model._build_base_model."
        ),
    )
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
    parser.add_argument("--warmup_start_lr", type=float, default=config.WARMUP_START_LR)
    parser.add_argument(
        "--scheduler_patience", type=int, default=config.SCHEDULER_PATIENCE
    )
    parser.add_argument(
        "--scheduler_factor", type=float, default=config.SCHEDULER_FACTOR
    )
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
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
    parser.add_argument(
        "--torch_compile",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=config.TORCH_COMPILE,
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
    parser.add_argument(
        "--partition",
        type=lambda x: x.lower() if x.lower() != "none" else None,
        default=None,
        help="Graph partitioning algorithm to apply (e.g. 'metis'). Pass 'none' to disable. "
        "Mutually exclusive with --sparsification.",
    )

    args = parser.parse_args()
    main(args)
