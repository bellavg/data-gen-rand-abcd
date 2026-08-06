"""Generic PyTorch Lightning wrapper for baseline graph-regression models.

Deliberately kept separate from models.lightning_model.AIGRegressionLightningModule
rather than sharing a refactored base class with it. That module is already
tested and drives the project's primary model; the baselines' own training
configs (plain Adam vs. AdamW+warmup, MSELoss vs. SmoothL1Loss, no LR warmup)
differ from it enough that forcing a shared base class would mean editing
working code for marginal reuse. The loss/metric/optimizer bookkeeping below
is intentionally a separate (if similar) copy: leave what isn't broken alone.

COMPILE. `compile_model` mirrors models/lightning_model.py's own pattern
(`torch.compile(base_model, dynamic=True)`) exactly. `dynamic=True` is not
optional given every batch here has a different shape (dynamic node-budget
batching, or plain variable-size graphs) -- but be precise about what it buys
on this torch version (2.11): it is NOT "avoids a recompile every batch".
Measured directly on `GamoraGraphRegressor` across 6 distinct batch shapes,
per-call compile time in seconds:
    dynamic=True:  [3.44, 0.00, 0.00, 0.00, 0.00, 0.00]
    dynamic=None:  [0.95, 6.23, 0.00, 0.00, 0.00, 0.00]  (torch's own default)
Torch's "automatic dynamic shapes" already adapts after ONE recompile even
without `dynamic=True` -- the difference `dynamic=True` buys is skipping that
one extra recompile (here, ~6.2s) by tracing shape-polymorphic the first time,
not preventing an unbounded per-batch cost that was never actually there on
this torch version. Still worth setting explicitly rather than relying on
automatic detection to trigger correctly on the first real training step.

Default is FALSE here, unlike the primary model's default TRUE. This wrapper
is shared by all four baselines, and enabling compile changes what actually
executes on the GPU for whichever one uses it. Only Gamora has real
verification behind it -- see `src/unittests/baselines/test_gamora.py`'s
`TestGamoraTorchCompile` (eager-vs-compiled forward on the actual
`GamoraGraphRegressor`, gradients, four batch shapes) and
`src/unittests/baselines/test_lightning_wrapper.py` (the generic wrapping/
checkpoint-key behavior below, on a toy model). HOGA's custom
`MultiheadAttention` and DeepGate4's gradient-checkpointed sparse transformer
are exactly the kind of code that has, in other codebases, interacted badly
with `torch.compile` (checkpointing especially -- recompute-in-backward and
Dynamo's graph capture have known rough edges together), and neither has been
tested here. Turn this on per-baseline only after doing the same verification
Gamora got, not by flipping the default.

KNOWN LIMITATION, not a bug: compiling `GamoraGraphRegressor` produces a graph
break at `global_mean_pool` -- PyG's `scatter` calls `int(index.max())`
internally, which forces a Python-level sync Dynamo cannot trace through. The
model still compiles and runs correctly (verified numerically, see above),
just as two graph segments rather than one fully fused kernel, so the realized
speedup is bounded by what's compiled BEFORE the break (the SAGEConv stack,
activations, dropout, BatchNorm) rather than the whole forward pass.

NOT VERIFIED, and worth knowing before trusting a compiled-vs-uncompiled
comparison: all of the numerical verification above ran fp32 on CPU. The
actual H100 job runs bf16-mixed autocast (`_select_precision()` in
train_baseline.py), and Inductor's fusion does not necessarily insert the same
precision-casting boundaries eager-mode autocast does. "Compile only changes
speed, not results" is the expectation, not a demonstrated fact, under
bf16-mixed specifically -- it has not been checked there, and Gamora also has
a BatchNorm layer that the fp32/CPU verification did not stress under
autocast. If a compiled and an uncompiled Gamora run are ever compared
directly, treat that axis the way this project already treats SynthNet's
`upstream_edge_direction` -- a real configuration difference to report, not an
implementation detail to assume away.

CHECKPOINT KEYS. `torch.compile` wraps the model in `OptimizedModule`, which
prefixes every parameter/buffer key with `_orig_mod.` (verified: `compiled.
state_dict().keys()` come back as `_orig_mod.lin.weight`, not `lin.weight` --
`compiled._orig_mod is model`, so the underlying tensors are identical, only
the key names differ). Left alone, that breaks the strip-a-fixed-prefix-then-
`load_state_dict(strict=True)` pattern every baseline-checkpoint consumer in
this repo already uses (e.g. diagnose_synthnet_baseline.py:108-113) --
Gamora's checkpoints would silently be the only ones in a different key
format. `state_dict()`/`load_state_dict()` below correct for this, so a
checkpoint saved under `compile_model=True` has the exact same key set as one
saved under `False`, and either can be loaded into either.
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
        compile_model: Wrap `model` in `torch.compile(model, dynamic=True)`.
            Default False -- see this module's docstring for why, and which
            baseline has actually been verified under it.
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
        compile_model: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn"])
        self.model = torch.compile(model, dynamic=True) if compile_model else model
        self.loss_fn: nn.Module = loss_fn if loss_fn is not None else nn.MSELoss()

        self.rmse_metrics: nn.ModuleDict = nn.ModuleDict(
            {f"s_{stage}": MeanSquaredError(squared=False) for stage in _STAGES}
        )
        self.r2_metrics: nn.ModuleDict = nn.ModuleDict(
            {f"s_{stage}": R2Score() for stage in ("val", "test")}
        )

    def forward(self, batch: Any) -> torch.Tensor:
        return self.model(batch)

    # -- torch.compile checkpoint-key normalisation -- see module docstring. --
    _COMPILE_KEY_PREFIX = "model._orig_mod."
    _CLEAN_KEY_PREFIX = "model."

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        sd = super().state_dict(*args, **kwargs)
        if not self.hparams.compile_model:
            return sd
        return type(sd)(
            (
                self._CLEAN_KEY_PREFIX + key[len(self._COMPILE_KEY_PREFIX) :]
                if key.startswith(self._COMPILE_KEY_PREFIX)
                else key,
                value,
            )
            for key, value in sd.items()
        )

    def load_state_dict(self, state_dict: Any, *args: Any, **kwargs: Any) -> Any:
        if self.hparams.compile_model:
            state_dict = type(state_dict)(
                (
                    self._COMPILE_KEY_PREFIX + key[len(self._CLEAN_KEY_PREFIX) :]
                    if key.startswith(self._CLEAN_KEY_PREFIX)
                    and not key.startswith(self._COMPILE_KEY_PREFIX)
                    else key,
                    value,
                )
                for key, value in state_dict.items()
            )
        return super().load_state_dict(state_dict, *args, **kwargs)

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
