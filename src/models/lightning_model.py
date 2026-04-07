from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import your unified base model (adjust the path as needed)
try:
    from models.base_model import UnifiedGraphBaseModel
except ImportError:
    from base_model import UnifiedGraphBaseModel


class AIGRegressionLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for the UnifiedGraphBaseModel.
    Specifically designed for dual-target AIG regression (Node & Depth Optimizability).
    """

    def __init__(
        self,
        encoder_name: str,
        embed_dim: int,
        node_input_dim: int = 4,
        num_edge_types: int = 2,
        task_out_dim: int = 2,
        pe_type: str = "none",
        pos_enc_dim: int = 0,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        # Save hyperparameters so they are logged automatically and accessible via self.hparams
        self.save_hyperparameters()

        # Instantiate the unified base model
        self.model = UnifiedGraphBaseModel(
            encoder_name=self.hparams.encoder_name,
            embed_dim=self.hparams.embed_dim,
            node_input_dim=self.hparams.node_input_dim,
            edge_attr_dim=self.hparams.num_edge_types,
            task_out_dim=self.hparams.task_out_dim,
            pe_type=self.hparams.pe_type,
            pos_enc_dim=self.hparams.pos_enc_dim,
            encoder_kwargs=self.hparams.encoder_kwargs,
        )

    def forward(self, batch):
        """Passes the PyG batch through the base model."""
        return self.model.forward_batch(batch)

    def _compute_loss_and_metrics(self, batch, batch_idx, prefix: str):
        """
        Helper function to compute loss and log metrics.
        batch.y is expected to be shape [BatchSize, 2] (Node Opt, Depth Opt)
        """
        preds = self.forward(batch)
        targets = batch.y

        # Make sure target shape matches predictions (safeguard against PyG squeezing)
        if targets.dim() == 1:
            targets = targets.view(-1, self.hparams.task_out_dim)

        # Huber Loss prevents zero-collapse but handles outliers gracefully.
        # delta=1.0 is standard, but you can tune it (e.g., delta=2.0) if needed.
        loss = F.huber_loss(preds, targets, delta=1.0)

        # Calculate individual MAEs for human-readable logging (keep as L1)
        mae_node_opt = F.l1_loss(preds[:, 0], targets[:, 0])
        mae_depth_opt = F.l1_loss(preds[:, 1], targets[:, 1])

        # Log metrics (on_step=False, on_epoch=True averages over the epoch)
        self.log(
            f"{prefix}/loss",
            loss,
            batch_size=batch.num_graphs,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            f"{prefix}/mae_node",
            mae_node_opt,
            batch_size=batch.num_graphs,
            sync_dist=True,
        )
        self.log(
            f"{prefix}/mae_depth",
            mae_depth_opt,
            batch_size=batch.num_graphs,
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
        """Sets up the optimizer and learning rate scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Reduce LR on Plateau monitors the validation loss
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=20,
            min_lr=1e-5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",  # Must match the logged name exactly
                "frequency": 1,
            },
        }
