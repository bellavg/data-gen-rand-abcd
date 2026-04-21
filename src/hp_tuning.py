import argparse
import gc
import logging
import warnings

import optuna
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from optuna.storages import JournalFileStorage, JournalStorage
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

# Project Imports
try:
    from data.datamodule import AIGDataModule
except ImportError:
    from src.data.datamodule import AIGDataModule

try:
    from optuna.integration import PyTorchLightningPruningCallback
except ModuleNotFoundError:

    class PyTorchLightningPruningCallback(Callback):
        def __init__(self, *args, **kwargs):
            super().__init__()


from models.lightning_model import AIGRegressionLightningModule

torch.multiprocessing.set_sharing_strategy("file_system")
# 1. Suppress standard Python DeprecationWarnings and UserWarnings
warnings.filterwarnings("ignore")

# 2. Set PyTorch Lightning's logger to only show errors (hides the SLURM/TPU prints)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def objective(trial: optuna.Trial, args):
    # 1. Global Hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    huber_delta = trial.suggest_float("huber_delta", 0.5, 2.0)

    encoder_name = trial.suggest_categorical(
        "encoder_name",
        ["gine", "transformer_conv", "graphgps", "egin", "gcn", "vanilla_mpnn"],
    )

    # Suggest a single unified hidden dimension
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])

    # 2. Positional Encoding
    pe_type = trial.suggest_categorical(
        "pe_type",
        ["none", "level", "pi_paths", "local_sp_sum"],
    )
    pos_enc_dim = (
        trial.suggest_categorical("pos_enc_dim", [16, 32, 64, 128, 256])
        if pe_type != "none"
        else 0
    )

    # 3. Encoder Specific Hyperparameters
    encoder_kwargs = {
        "num_layers": trial.suggest_int(
            "num_layers", 3, 10
        ),  # GNN Message Passing depth
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "norm_type": trial.suggest_categorical(
            "norm_type",
            [
                "batch",
                "layer",
                "graph",
            ],
        ),
        "jk_mode": trial.suggest_categorical("jk_mode", ["last", "max", "sum", "cat"]),
    }

    # Ensure the chosen hidden dimension is passed through to the encoder
    encoder_kwargs["hid_dim"] = hidden_dim
    if encoder_name in ["transformer_conv", "graphgps"]:
        encoder_kwargs["heads"] = trial.suggest_categorical("heads", [4, 8])

    # --- EGIN SPECIFIC TUNING ---
    if encoder_name == "egin":
        # Internal MLP depth (keep shallow to avoid vanishing gradients) [cite: 1, 12]
        encoder_kwargs["num_mlp_layers"] = trial.suggest_int("num_mlp_layers", 2, 4)

        # FIX: Change suggest_bool to suggest_categorical
        encoder_kwargs["dot_update"] = trial.suggest_categorical(
            "egin_dot_update", [True, False]
        )
        encoder_kwargs["edge_mlp"] = trial.suggest_categorical(
            "egin_edge_mlp", [True, False]
        )

        # Tune the edge projection dim (8-64 range recommended for low-diversity edges) [cite: 290]
        encoder_kwargs["edge_hidden_dim"] = trial.suggest_categorical(
            "edge_hidden_dim", [8, 16, 32, 64, 128]
        )
    # ----------------------------

    # 4. Data Module
    workers = getattr(args, "num_workers", 4)
    datamodule = AIGDataModule(
        csv_paths=args.csv_paths,
        positional_encoding=pe_type if pe_type != "none" else None,
        batch_size=batch_size,
        split_ratios=(0.8, 0.2, 0.0),
        train_num_samples=args.train_samples,
        num_workers=workers,
    )

    # 5. Model Setup
    model = AIGRegressionLightningModule(
        encoder_name=encoder_name,
        hidden_dim=hidden_dim,
        pe_type=pe_type,
        pos_enc_dim=pos_enc_dim,
        encoder_kwargs=encoder_kwargs,
        lr=lr,
        huber_delta=huber_delta,
    )

    pruning_cb = PyTorchLightningPruningCallback(trial, monitor="val/mae_node")
    early_stop_cb = EarlyStopping(monitor="val/mae_node", patience=5, mode="min")

    callbacks = [
        pruning_cb,
        early_stop_cb,
    ]

    csv_logger = CSVLogger(
        save_dir=args.log_dir, name="optuna_metrics", version=f"trial_{trial.number}"
    )

    # 7. Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        max_time={"hours": 3},
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=csv_logger,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    try:
        trainer.fit(model, datamodule=datamodule)
    except torch.OutOfMemoryError:
        # Force teardown of dataloaders
        del trainer
        del datamodule
        gc.collect()
        torch.cuda.empty_cache()
        raise optuna.TrialPruned(
            f"CUDA Out of Memory. [Trial {trial.number}] with Failed Params: {trial.params}"
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Force teardown of dataloaders
            del trainer
            del datamodule
            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.TrialPruned(
                f"\n[Trial {trial.number}] Pruned due to CUDA Out of Memory (RuntimeError)."
            )
        else:
            raise e

    return (
        early_stop_cb.best_score.item()
        if early_stop_cb.best_score is not None
        else float("inf")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter Tuning for AIG Regression"
    )
    parser.add_argument("--db_url", type=str, required=True, help="Database URL")
    parser.add_argument(
        "--study_name", type=str, required=True, help="Optuna study name"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True, help="Checkpoint directory"
    )
    parser.add_argument("--csv_paths", nargs="+", required=True, help="Paths to CSVs")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--cache_dir", type=str, help="Directory to save dataset splits"
    )
    parser.add_argument("--log_dir", type=str, help="Directory to save lightning logs")
    parser.add_argument(
        "--train_samples",
        type=int,
        default=25000,
        help="Number of graphs for HP training",
    )
    args = parser.parse_args()

    storage = JournalStorage(JournalFileStorage(args.db_url))

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    study.optimize(lambda trial: objective(trial, args), n_trials=None)
