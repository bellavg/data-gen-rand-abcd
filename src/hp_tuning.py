import argparse

import optuna
import pytorch_lightning as pl

# Project Imports
try:
    from data.datamodule import AIGDataModule
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.data.datamodule import AIGDataModule

try:
    from optuna.integration import PyTorchLightningPruningCallback
except ModuleNotFoundError:  # pragma: no cover - optional dependency

    class PyTorchLightningPruningCallback:  # type: ignore[no-redef]
        """No-op fallback when optuna-integration is unavailable."""

        def __init__(self, *args, **kwargs):
            pass


from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from models.lightning_model import AIGRegressionLightningModule


def objective(trial: optuna.Trial, args):
    # 1. Hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    huber_delta = trial.suggest_float("huber_delta", 0.5, 2.0)

    encoder_name = trial.suggest_categorical(
        "encoder_name", ["gine", "transformer_conv", "graphgps", "egin"]
    )
    embed_dim = trial.suggest_categorical("embed_dim", [128, 256, 384, 512, 768])
    pe_type = trial.suggest_categorical(
        "pe_type", ["none", "level", "edge_rel_dist", "pi_paths", "local_sp_sum"]
    )

    pos_enc_dim = (
        trial.suggest_categorical("pos_enc_dim", [16, 32, 64, 128])
        if pe_type != "none"
        else 0
    )

    encoder_kwargs = {
        "num_layers": trial.suggest_int("num_layers", 3, 10),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "norm_type": trial.suggest_categorical(
            "norm_type", ["batch", "layer", "graph", "none"]
        ),
    }
    if encoder_name in ["transformer_conv", "graphgps"]:
        encoder_kwargs["heads"] = trial.suggest_categorical("heads", [4, 8, 16])

    # 2. Data Module using args.csv_paths
    workers = getattr(args, "num_workers", 4)
    datamodule = AIGDataModule(
        csv_paths=args.csv_paths,
        positional_encoding=pe_type if pe_type != "none" else None,
        batch_size=batch_size,
        split_ratios=(0.7, 0.2, 0.1),
        train_num_samples=10000,
        num_workers=workers,
    )

    # 3. Model Setup
    model = AIGRegressionLightningModule(
        encoder_name=encoder_name,
        embed_dim=embed_dim,
        pe_type=pe_type,
        pos_enc_dim=pos_enc_dim,
        encoder_kwargs=encoder_kwargs,
        lr=lr,
        huber_delta=huber_delta,
    )

    # 4. Callbacks (Saving checkpoints to args.checkpoint_dir)
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f"trial_{trial.number}_best",
        monitor="val/mae_node",
        mode="min",
        save_top_k=1,
    )

    # 5. Trainer with timeouts
    trainer = pl.Trainer(
        max_epochs=100,
        max_time={"minutes": 60},
        accelerator="auto",
        devices=1,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="val/mae_node"),
            EarlyStopping(monitor="val/loss", patience=10, mode="min"),
            checkpoint_cb,
        ],
        logger=False,
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule=datamodule)

    return (
        checkpoint_cb.best_model_score.item() if checkpoint_cb.best_model_score else 1.0
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter Tuning for AIG Regression"
    )
    parser.add_argument(
        "--db_url",
        type=str,
        required=True,
        help="Database URL (e.g., sqlite:///optuna.db)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory to save best model checkpoints",
    )
    parser.add_argument(
        "--csv_paths", nargs="+", required=True, help="List of paths to algorithm CSVs"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes for dataset loading",
    )
    args = parser.parse_args()

    # Ensure Optuna handles potential SQLite locking gracefully
    # by increasing the timeout when trying to write to the DB.
    storage = optuna.storages.RDBStorage(
        url=args.db_url, engine_kwargs={"connect_args": {"timeout": 60}}
    )

    study = optuna.create_study(
        study_name="aig_optimization_5days",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    # Pass args to the objective function
    study.optimize(lambda trial: objective(trial, args), n_trials=None)
