"""PyTorch Lightning module for AIG graph-level regression tasks.

Targets are assumed to lie in the range [0, 1]. The module supports:
    - Linear LR warm-up followed by ReduceLROnPlateau scheduling.
    - Per-stage RMSE and R² metrics in ``nn.ModuleDict`` to prevent state
      leakage between train / val / test stages.
    - Asymmetric logging granularity per the project spec:

        +---------+--------------+---------------+-------------+
        | Metric  | Train (step) | Train (epoch) | Val (epoch) |
        +=========+==============+===============+=============+
        | Loss    |     Yes      |      Yes      |     Yes     |
        +---------+--------------+---------------+-------------+
        | RMSE    |     Yes      |      Yes      |     Yes     |
        +---------+--------------+---------------+-------------+
        | R²      |      No      |       No      |     Yes     |
        +---------+--------------+---------------+-------------+
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError, R2Score

from config import MIN_LR, SCHEDULER_FACTOR, SCHEDULER_PATIENCE, WARMUP_START_LR, WARMUP_STEPS
from constants import EDGE_ATTR_DIM, NODE_INPUT_DIM, TASK_OUT_DIM
from models.base_model import UnifiedGraphBaseModel

# Ordered tuple of all stages; used to build metric ModuleDicts.
_STAGES: tuple[str, ...] = ("train", "val", "test")


class AIGRegressionLightningModule(pl.LightningModule):
    """LightningModule for AIG graph-level regression.

    Wraps ``UnifiedGraphBaseModel`` with training, validation, and test loops,
    metric tracking, and an AdamW optimiser with linear warmup followed by
    ReduceLROnPlateau scheduling.

    Args:
        encoder_name: Name of the GNN encoder backbone.
        hidden_dim: Hidden dimensionality for the encoder and prediction head.
        encoder_kwargs: Extra keyword arguments forwarded verbatim to the
            encoder constructor.
        node_input_dim: Dimensionality of raw node features.
        edge_attr_dim: Dimensionality of raw edge attributes.
        task_out_dim: Number of scalar regression targets per graph.
        pe_type: Positional encoding type. Pass ``"none"`` to disable.
        pos_enc_dim: Dimensionality of the positional encoding vectors.
        pooling_type: Graph-level pooling strategy, e.g. ``"mean"`` or
            ``"sum"``.
        head_dropout: Dropout probability in the prediction head. ``None``
            disables dropout.
        lr: Peak learning rate for the optimizer.
        weight_decay: AdamW L2 regularisation coefficient.
        min_lr: Floor learning rate enforced by ReduceLROnPlateau.
        warmup_steps: Number of optimizer steps used for linear warmup.
        warmup_start_lr: Learning rate at warmup step 0.
        scheduler_patience: Plateau patience in epochs before LR reduction.
        scheduler_factor: Multiplicative LR decay factor on plateau.
        loss_fn: Loss module applied to ``(preds, targets)``. Defaults to
            ``nn.L1Loss()`` (MAE). Must use ``reduction="mean"`` so that
            per-step losses are already graph-averaged; this keeps
            Lightning's epoch aggregation correct.
    """

    def __init__(
        self,
        encoder_name: str,
        hidden_dim: int,
        encoder_kwargs: dict[str, Any],
        node_input_dim: int = NODE_INPUT_DIM,
        edge_attr_dim: int = EDGE_ATTR_DIM,
        task_out_dim: int = TASK_OUT_DIM,
        pe_type: str = "none",
        pos_enc_dim: int = 0,
        pooling_type: str = "mean",
        head_dropout: float | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        min_lr: float = MIN_LR,
        warmup_steps: int = WARMUP_STEPS,
        warmup_start_lr: float = WARMUP_START_LR,
        scheduler_patience: int = SCHEDULER_PATIENCE,
        scheduler_factor: float = SCHEDULER_FACTOR,
        loss_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()
        # loss_fn is excluded because arbitrary nn.Module instances are not
        # always safely serialisable by Lightning's hyperparameter snapshot.
        self.save_hyperparameters(ignore=["loss_fn"])
        self.loss_fn: nn.Module = loss_fn if loss_fn is not None else nn.L1Loss()

        # ------------------------------------------------------------------ #
        # Core model                                                           #
        # ------------------------------------------------------------------ #
        self.model = torch.compile(
            UnifiedGraphBaseModel(
                encoder_name=self.hparams.encoder_name,
                hidden_dim=self.hparams.hidden_dim,
                node_input_dim=self.hparams.node_input_dim,
                edge_attr_dim=self.hparams.edge_attr_dim,
                task_out_dim=self.hparams.task_out_dim,
                pe_type=self.hparams.pe_type,
                pos_enc_dim=self.hparams.pos_enc_dim,
                pooling_type=self.hparams.pooling_type,
                head_dropout=self.hparams.head_dropout,
                encoder_kwargs=self.hparams.encoder_kwargs,
            ),
            dynamic=True,
        )

        # ------------------------------------------------------------------ #
        # Metrics                                                              #
        # ------------------------------------------------------------------ #
        # nn.ModuleDict is required (not a plain dict) so that Lightning
        # automatically moves metrics to the correct device and resets their
        # internal state between epochs. R² is only tracked for val/test
        # because per-step R² on a single mini-batch is statistically
        # meaningless and can be highly misleading.
        self.rmse_metrics: nn.ModuleDict = nn.ModuleDict(
            {f"s_{stage}": MeanSquaredError(squared=False) for stage in _STAGES}
        )
        self.r2_metrics: nn.ModuleDict = nn.ModuleDict(
            {f"s_{stage}": R2Score() for stage in ("val", "test")}
        )

    # ---------------------------------------------------------------------- #
    # Forward                                                                  #
    # ---------------------------------------------------------------------- #

    def forward(self, batch: Any) -> torch.Tensor:
        """Run a forward pass on a PyG Batch.

        Args:
            batch: A ``torch_geometric.data.Batch`` object containing the
                batched graph data.

        Returns:
            Prediction tensor of shape ``(num_graphs, task_out_dim)``.
        """
        return self.model.forward_batch(batch)

    # ---------------------------------------------------------------------- #
    # Private helpers                                                           #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _batch_size(batch: Any) -> int:
        """Return the number of graphs in ``batch`` for Lightning's loss weighting.

        ``self.log(..., batch_size=N, on_epoch=True)`` uses ``N`` as a weight
        when reducing per-step scalars to an epoch mean:

            epoch_loss = sum(loss_i * N_i) / sum(N_i)

        Because ``loss_fn`` uses ``reduction="mean"`` over graph-level targets,
        each step's loss is already a graph-averaged scalar. Passing
        ``num_graphs`` as the weight therefore yields the correct
        graph-weighted epoch mean — larger batches contribute proportionally
        more, but each *graph* within a batch contributes equally regardless
        of its node count.

        Args:
            batch: The current PyG ``Batch`` object.

        Returns:
            Number of graphs in the batch. Falls back to the first dimension
            of ``batch.y`` if ``num_graphs`` is absent, and to ``1`` as a
            last resort to avoid division-by-zero inside Lightning.
        """
        num_graphs: int | None = getattr(batch, "num_graphs", None)
        if num_graphs is not None:
            return int(num_graphs)

        # Infer from targets as a secondary fallback.
        y: torch.Tensor | None = getattr(batch, "y", None)
        if y is not None and y.dim() > 0:
            return int(y.size(0))

        return 1

    def _compute_loss_and_metrics(
        self,
        batch: Any,
        prefix: str,
    ) -> torch.Tensor:
        """Compute the loss, update torchmetrics, and log everything.

        This is the single shared implementation called by ``training_step``,
        ``validation_step``, and ``test_step``. The ``prefix`` argument drives
        all branching so there is no duplicated logic between stages.

        Logging behaviour per stage:

        - **train**: loss and RMSE logged at both step and epoch level.
          R² is intentionally omitted — per-step R² on a mini-batch is
          numerically unstable and not meaningful.
        - **val / test**: loss, RMSE, and R² logged at epoch level only,
          giving clean aggregate evaluation metrics.

        Torchmetric state updates are skipped during Lightning's sanity-check
        pass (the few validation batches run before training begins) to prevent
        the small, unrepresentative sanity batches from polluting the metric
        accumulators before real training starts. The loss is still returned
        so the sanity check can complete normally.

        Args:
            batch: The current PyG ``Batch`` object.
            prefix: Stage identifier — one of ``"train"``, ``"val"``, or
                ``"test"``.

        Returns:
            Scalar loss tensor. Returned for all stages so that Lightning's
            training loop can call ``.backward()`` on it; ``validation_step``
            and ``test_step`` simply discard the return value.
        """
        preds: torch.Tensor = self.forward(batch).squeeze(-1)

        targets: torch.Tensor = batch.y
        if targets.dim() == 1:
            targets = targets.view(-1, self.hparams.task_out_dim)
        targets = targets.squeeze(-1)

        loss: torch.Tensor = self.loss_fn(preds, targets)
        b_size: int = self._batch_size(batch)

        # Shared kwargs eliminate repeated arguments across every self.log call.
        on_step: bool = prefix == "train"
        log_kwargs: dict[str, Any] = {
            "batch_size": b_size,
            "sync_dist": False,
            "prog_bar": False,
            "on_step": on_step,
            "on_epoch": True,
        }

        self.log(f"{prefix}_loss", loss, **log_kwargs)

        # Skip torchmetrics updates during the sanity-check pass only.
        # Loss is still logged above so Lightning can proceed with its checks.
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

    # ---------------------------------------------------------------------- #
    # Lightning step hooks                                                      #
    # ---------------------------------------------------------------------- #

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Execute a single training step.

        Args:
            batch: The current PyG ``Batch`` object.
            batch_idx: Index of this batch within the current epoch.

        Returns:
            Scalar loss tensor passed to ``.backward()`` by Lightning.
        """
        return self._compute_loss_and_metrics(batch, prefix="train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Execute a single validation step.

        Args:
            batch: The current PyG ``Batch`` object.
            batch_idx: Index of this batch within the current epoch.
        """
        self._compute_loss_and_metrics(batch, prefix="val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Execute a single test step.

        Args:
            batch: The current PyG ``Batch`` object.
            batch_idx: Index of this batch within the current epoch.
        """
        self._compute_loss_and_metrics(batch, prefix="test")

    # ---------------------------------------------------------------------- #
    # Optimiser & scheduler                                                     #
    # ---------------------------------------------------------------------- #

    def configure_optimizers(self) -> dict[str, Any]:
        """Build AdamW with linear warmup + ReduceLROnPlateau.

        Warmup is applied per optimizer step via ``optimizer_step`` and ramps LR
        from ``warmup_start_lr`` to ``lr`` across ``warmup_steps`` updates.
        After warmup, LR adaptation is handled by ``ReduceLROnPlateau`` on
        ``val_loss`` with factor/patience from hyperparameters.

        Returns:
            A Lightning-compatible configuration dict containing
            ``"optimizer"`` and ``"lr_scheduler"`` keys.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        warmup_steps = int(self.hparams.warmup_steps)
        if warmup_steps > 0:
            warmup_start_lr = float(self.hparams.warmup_start_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_start_lr

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(self.hparams.scheduler_factor),
            patience=int(self.hparams.scheduler_patience),
            min_lr=float(self.hparams.min_lr),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
            },
        }

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Any | None = None,
    ) -> None:
        """Apply optimizer update and perform per-step linear warmup.

        Args:
            epoch: Current epoch index, provided by Lightning.
            batch_idx: Current batch index within the epoch.
            optimizer: The optimizer instance to step.
            optimizer_closure: Optional closure for optimizers that require it.
        """
        warmup_steps = int(self.hparams.warmup_steps)
        if warmup_steps > 0:
            current_step = int(getattr(self, "global_step", 0))
            if current_step < warmup_steps:
                warmup_start_lr = float(self.hparams.warmup_start_lr)
                peak_lr = float(self.hparams.lr)
                progress = float(current_step + 1) / float(warmup_steps)
                step_lr = warmup_start_lr + progress * (peak_lr - warmup_start_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = step_lr

        optimizer.step(closure=optimizer_closure)
