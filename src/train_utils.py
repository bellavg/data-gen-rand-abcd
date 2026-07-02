import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping


class PreciseEarlyStopping(EarlyStopping):
    """EarlyStopping with non-rounded improvement logging for small metrics."""

    @staticmethod
    def _fmt(value) -> str:
        return f"{float(value):.6g}"

    def _improvement_message(self, current: torch.Tensor) -> str:
        if torch.isfinite(self.best_score):
            improvement = abs(float(self.best_score) - float(current))
            return (
                f"Metric {self.monitor} improved by {self._fmt(improvement)} >= "
                f"min_delta = {self._fmt(abs(self.min_delta))}. "
                f"New best score: {self._fmt(current)}"
            )
        return f"Metric {self.monitor} improved. New best score: {self._fmt(current)}"


class TrainingStartupCallback(pl.Callback):
    def __init__(
        self,
        report_every_n_steps: int = 1000,
        max_batch_compute_reports: int = 4,
    ):
        self._fit_start_time: float | None = None
        self._epoch_start_time: float | None = None
        self._first_batch_reported = False
        self._batch_start_time: float | None = None
        self._report_every_n_steps = max(1, int(report_every_n_steps))
        self._max_batch_compute_reports = max(0, int(max_batch_compute_reports))
        self._batch_compute_reports_emitted = 0

    @staticmethod
    def _batch_size(batch) -> int:
        num_graphs = getattr(batch, "num_graphs", None)
        if num_graphs is not None:
            return int(num_graphs)

        if isinstance(batch, (list, tuple)):
            return len(batch)

        y = getattr(batch, "y", None)
        if y is not None and getattr(y, "dim", lambda: 0)() > 0:
            return int(y.size(0))

        return 1

    def _elapsed(self) -> float:
        if self._fit_start_time is None:
            return 0.0
        return time.monotonic() - self._fit_start_time

    def on_fit_start(self, trainer, pl_module):
        self._fit_start_time = time.monotonic()
        print(
            "[train] Fit entered; preparing dataloaders and training loop...",
            flush=True,
        )

    def on_sanity_check_start(self, trainer, pl_module):
        print("[train] Running sanity validation before training...", flush=True)

    def on_sanity_check_end(self, trainer, pl_module):
        print(
            f"[train] Sanity validation finished after {self._elapsed():.1f}s.",
            flush=True,
        )

    def on_train_start(self, trainer, pl_module):
        print(
            f"[train] Training loop started after {self._elapsed():.1f}s.",
            flush=True,
        )
        print(
            "[train] Batches per epoch: "
            f"train={int(getattr(trainer, 'num_training_batches', -1))}",
            flush=True,
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._batch_start_time = time.monotonic()
        if not self._first_batch_reported and batch_idx == 0:
            self._first_batch_reported = True
            print(
                f"[train] First training batch started after {self._elapsed():.1f}s.",
                flush=True,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        end_time = time.monotonic()
        if self._batch_start_time is not None:
            step_time = end_time - self._batch_start_time
            pl_module.log(
                "train_step_time_s",
                step_time,
                batch_size=self._batch_size(batch),
                on_step=True,
                on_epoch=True,
            )
            if (
                (batch_idx + 1) % self._report_every_n_steps == 0
                and self._batch_compute_reports_emitted
                < self._max_batch_compute_reports
            ):
                print(
                    f"[train] Batch compute: idx={batch_idx} step_s={step_time:.3f}",
                    flush=True,
                )
                self._batch_compute_reports_emitted += 1

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start_time = time.monotonic()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._epoch_start_time is not None:
            pl_module.log(
                "epoch_time_seconds",
                time.monotonic() - self._epoch_start_time,
                batch_size=1,
                on_step=False,
                on_epoch=True,
            )
