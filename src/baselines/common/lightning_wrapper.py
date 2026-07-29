"""Generic PyTorch Lightning wrapper for baseline graph-regression models.

Deliberately kept separate from models.lightning_model.AIGRegressionLightningModule
rather than sharing a refactored base class with it. That module is already
tested and drives the project's primary model; the baselines' own training
configs (plain Adam vs. AdamW+warmup, MSELoss vs. SmoothL1Loss, no LR warmup)
differ from it enough that forcing a shared base class would mean editing
working code for marginal reuse. The loss/metric/optimizer bookkeeping below
is intentionally a separate (if similar) copy: leave what isn't broken alone.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError, R2Score

_STAGES: tuple[str, ...] = ("train", "val", "test")


class BaselineRegressionLightningModule(pl.LightningModule):
    """Wraps an arbitrary `nn.Module` (`forward(batch) -> Tensor[num_graphs, 1]`)
    with training/validation/test loops, RMSE/R² metrics, and an optimizer +
    ReduceLROnPlateau scheduler -- no LR warmup, matching the plain-Adam
    training setups both vendored baseline papers use.

    Args:
        model: The baseline model, e.g. `SynthNetGraphRegressor` or
            `HOGAGraphRegressor`. Called as `model(batch)`.
        lr: Optimizer learning rate.
        weight_decay: Optimizer L2 regularization coefficient.
        optimizer_name: `"adam"` or `"adamw"`.
        loss_fn: Loss module applied to `(preds, targets)`. Defaults to
            `nn.MSELoss()` to match both vendored papers' own training setups.
        scheduler_factor: `ReduceLROnPlateau` multiplicative LR decay factor.
        scheduler_patience: `ReduceLROnPlateau` patience in epochs.
        monitor: Metric name the scheduler watches.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float,
        weight_decay: float = 0.0,
        optimizer_name: str = "adam",
        loss_fn: nn.Module | None = None,
        scheduler_factor: float = 0.1,
        scheduler_patience: int = 10,
        monitor: str = "val_loss",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn"])
        self.model = model
        self.loss_fn: nn.Module = loss_fn if loss_fn is not None else nn.MSELoss()

        self.rmse_metrics: nn.ModuleDict = nn.ModuleDict(
            {f"s_{stage}": MeanSquaredError(squared=False) for stage in _STAGES}
        )
        self.r2_metrics: nn.ModuleDict = nn.ModuleDict(
            {f"s_{stage}": R2Score() for stage in ("val", "test")}
        )

    def forward(self, batch: Any) -> torch.Tensor:
        return self.model(batch)

    @staticmethod
    def _batch_size(batch: Any) -> int:
        num_graphs: int | None = getattr(batch, "num_graphs", None)
        if num_graphs is not None:
            return int(num_graphs)
        y: torch.Tensor | None = getattr(batch, "y", None)
        if y is not None and y.dim() > 0:
            return int(y.size(0))
        return 1

    def _compute_loss_and_metrics(self, batch: Any, prefix: str) -> torch.Tensor:
        if isinstance(batch, (list, tuple)):
            from torch_geometric.data import Batch

            batch = Batch.from_data_list(batch)

        preds: torch.Tensor = self.forward(batch).squeeze(-1)

        targets: torch.Tensor = batch.y
        if targets.dim() == 1:
            targets = targets.view(-1, 1)
        targets = targets.squeeze(-1)

        loss: torch.Tensor = self.loss_fn(preds, targets)
        b_size: int = self._batch_size(batch)

        on_step: bool = prefix == "train"
        log_kwargs: dict[str, Any] = {
            "batch_size": b_size,
            "sync_dist": False,
            "prog_bar": False,
            "on_step": on_step,
            "on_epoch": True,
        }
        self.log(f"{prefix}_loss", loss, **log_kwargs)

        is_sanity: bool = getattr(
            getattr(self, "trainer", None), "sanity_checking", False
        )
        if is_sanity:
            return loss

        rmse: MeanSquaredError = self.rmse_metrics[f"s_{prefix}"]  # type: ignore[assignment]
        rmse(preds, targets)
        self.log(f"{prefix}_rmse", rmse, **log_kwargs)

        if prefix in ("val", "test"):
            r2: R2Score = self.r2_metrics[f"s_{prefix}"]  # type: ignore[assignment]
            r2(preds, targets)
            self.log(
                f"{prefix}_r2",
                r2,
                batch_size=b_size,
                sync_dist=False,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._compute_loss_and_metrics(batch, prefix="train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._compute_loss_and_metrics(batch, prefix="val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._compute_loss_and_metrics(batch, prefix="test")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer_name = str(self.hparams.optimizer_name).lower()
        if optimizer_name == "adam":
            optimizer_cls = torch.optim.Adam
        elif optimizer_name == "adamw":
            optimizer_cls = torch.optim.AdamW
        else:
            raise ValueError(f"Unknown optimizer_name: {self.hparams.optimizer_name!r}")

        optimizer = optimizer_cls(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(self.hparams.scheduler_factor),
            patience=int(self.hparams.scheduler_patience),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": self.hparams.monitor,
                "strict": True,
            },
        }
