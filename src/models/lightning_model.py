from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from constants import EDGE_ATTR_DIM, NODE_INPUT_DIM, TASK_OUT_DIM

# Import your unified base model (adjust the path as needed)
try:
    from models.base_model import UnifiedGraphBaseModel
except ImportError:
    from base_model import UnifiedGraphBaseModel


class AIGRegressionLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for the UnifiedGraphBaseModel.
    Specifically designed for node optimizability AIG regression.
    """

    def __init__(
        self,
        encoder_name: str,
        embed_dim: int,
        encoder_kwargs: Dict[str, Any],
        node_input_dim: int = NODE_INPUT_DIM,
        edge_attr_dim: int = EDGE_ATTR_DIM,
        task_out_dim: int = TASK_OUT_DIM,
        pe_type: str = "none",
        pos_enc_dim: int = 0,
        project_with_pos_enc: bool = True,  # <--- ADD THIS
        huber_delta: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = UnifiedGraphBaseModel(
            encoder_name=self.hparams.encoder_name,
            embed_dim=self.hparams.embed_dim,
            node_input_dim=self.hparams.node_input_dim,
            edge_attr_dim=self.hparams.edge_attr_dim,
            task_out_dim=self.hparams.task_out_dim,
            pe_type=self.hparams.pe_type,
            pos_enc_dim=self.hparams.pos_enc_dim,
            encoder_kwargs=self.hparams.encoder_kwargs,
            project_with_pos_enc=self.hparams.project_with_pos_enc,  # <--- ADD THIS
        )

    def forward(self, batch):
        """Passes the PyG batch through the base model."""
        return self.model.forward_batch(batch)

    def _compute_loss_and_metrics(self, batch, batch_idx, prefix: str):
        """
        Helper function to compute loss and log metrics.
        batch.y is expected to be shape [BatchSize, 1] (Node Opt)
        """
        preds = self.forward(batch)
        targets = batch.y

        # Make sure target shape matches predictions (safeguard against PyG squeezing)
        if targets.dim() == 1:
            targets = targets.view(-1, self.hparams.task_out_dim)

        # Huber Loss prevents zero-collapse but handles outliers gracefully.
        # delta=1.0 is standard, but you can tune it (e.g., delta=2.0) if needed.
        loss = F.huber_loss(preds, targets, delta=self.hparams.huber_delta)

        # Calculate MAE for human-readable logging (keep as L1)
        # Use squeeze(-1) to convert [N, 1] -> [N] (more robust than explicit column indexing)
        mae_node_opt = F.l1_loss(preds.squeeze(-1), targets.squeeze(-1))

        # Log metrics only when attached to a Trainer to avoid warnings in
        # unit tests that call `training_step`/`validation_step` directly.
        # Access the internal `_trainer` attribute to avoid invoking the
        # `trainer` property (which raises when not attached).
        if getattr(self, "_trainer", None) is not None:
            # Use batch.num_graphs when available for correct averaging.
            batch_size = getattr(batch, "num_graphs", None)
            self.log(
                f"{prefix}/loss",
                loss,
                batch_size=batch_size,
                sync_dist=True,
                prog_bar=True,
            )
            self.log(
                f"{prefix}/mae_node",
                mae_node_opt,
                batch_size=batch_size,
                sync_dist=True,
            )

        return loss

    def training_step(self, batch, batch_idx):
        # Trigger Performer projection redrawing (if using GraphGPS with Performer)
        if hasattr(self.model.encoder, "redraw_projection"):
            self.model.encoder.redraw_projection.redraw_projections()

        return self._compute_loss_and_metrics(batch, batch_idx, prefix="train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss_and_metrics(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx):
        return self._compute_loss_and_metrics(batch, batch_idx, prefix="test")

    def configure_optimizers(self):
        # 1. Get your initial LR
        initial_lr = self.hparams.lr

        # 2. Dynamically set min_lr to be 1/1000th of the initial LR
        min_lr_value = initial_lr * 1e-3

        # 3. Define your optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=initial_lr)

        # 4. Define the scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=20,
            min_lr=min_lr_value,  # <--- Safely bounded!
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
                "interval": "epoch",
            },
        }
