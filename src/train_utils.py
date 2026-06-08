import time
import pytorch_lightning as pl


class TrainingStartupCallback(pl.Callback):
    def __init__(self):
        self._fit_start_time: float | None = None
        self._epoch_start_time: float | None = None
        self._first_batch_reported = False

    def _elapsed(self) -> float:
        if self._fit_start_time is None:
            return 0.0
        return time.monotonic() - self._fit_start_time

    def on_fit_start(self, trainer, pl_module):
        self._fit_start_time = time.monotonic()
        print("[train] Fit entered; preparing dataloaders and training loop...", flush=True)

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

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._first_batch_reported or batch_idx != 0:
            return
        self._first_batch_reported = True
        print(
            f"[train] First training batch started after {self._elapsed():.1f}s.",
            flush=True,
        )

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start_time = time.monotonic()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._epoch_start_time is None:
            return
        epoch_duration = time.monotonic() - self._epoch_start_time
        pl_module.log("epoch_time_seconds", epoch_duration)
