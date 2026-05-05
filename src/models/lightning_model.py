from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from constants import EDGE_ATTR_DIM, NODE_INPUT_DIM, TASK_OUT_DIM

# Import your unified base model (adjust the path as needed)
from models.base_model import UnifiedGraphBaseModel


class AIGRegressionLightningModule(pl.LightningModule):
    def __init__(
        self,
        encoder_name: str,
        hidden_dim: int,
        encoder_kwargs: Dict[str, Any],
        node_input_dim: int = NODE_INPUT_DIM,
        edge_attr_dim: int = EDGE_ATTR_DIM,
        task_out_dim: int = TASK_OUT_DIM,
        pe_type: str = "none",
        pos_enc_dim: int = 0,
        pooling_type: str = "mean",
        huber_delta: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler_patience: int = 5,  # <-- NEW: Added scheduler patience
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = UnifiedGraphBaseModel(
            encoder_name=self.hparams.encoder_name,
            hidden_dim=self.hparams.hidden_dim,
            node_input_dim=self.hparams.node_input_dim,
            edge_attr_dim=self.hparams.edge_attr_dim,
            task_out_dim=self.hparams.task_out_dim,
            pe_type=self.hparams.pe_type,
            pos_enc_dim=self.hparams.pos_enc_dim,
            pooling_type=self.hparams.pooling_type,
            encoder_kwargs=self.hparams.encoder_kwargs,
        )

    def forward(self, batch):
        return self.model.forward_batch(batch)

    def _compute_loss_and_metrics(
        self, batch, batch_idx, prefix: str
    ) -> torch.Tensor | None:
        preds = self.forward(batch)
        targets = batch.y

        if targets.dim() == 1:
            targets = targets.view(-1, self.hparams.task_out_dim)

        loss = F.huber_loss(preds, targets, delta=self.hparams.huber_delta)
        mae_node_opt = F.l1_loss(preds.squeeze(-1), targets.squeeze(-1))

        if getattr(self, "_trainer", None) is not None:
            batch_size = getattr(batch, "num_graphs", 1)

            self.log(
                f"{prefix}/loss",
                loss.detach(),
                batch_size=batch_size,
                sync_dist=False,
                prog_bar=True,
            )
            self.log(
                f"{prefix}/mae_node",
                mae_node_opt.detach(),
                batch_size=batch_size,
                sync_dist=False,
            )

        # Return the live loss only for training so the backward pass works.
        # Returning a tensor from val/test causes PL to hold the entire
        # computation graph in memory across the epoch for metric aggregation.
        return loss if prefix == "train" else None

    def training_step(self, batch, batch_idx):
        if hasattr(self.model.encoder, "redraw_projection"):
            self.model.encoder.redraw_projection.redraw_projections()
        return self._compute_loss_and_metrics(batch, batch_idx, prefix="train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss_and_metrics(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx):
        return self._compute_loss_and_metrics(batch, batch_idx, prefix="test")

    def configure_optimizers(self):
        initial_lr = self.hparams.lr
        min_lr_value = initial_lr * 1e-3

        optimizer = torch.optim.Adam(
            self.parameters(), lr=initial_lr, weight_decay=self.hparams.weight_decay
        )

        # Use the scheduler_patience arg passed from train.py
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=self.hparams.scheduler_patience,  # Linked to CLI arg
            min_lr=min_lr_value,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "frequency": 1,
                "interval": "epoch",
            },
        }
