import time
import torch
import pytorch_lightning as pl


class TrainingStartupCallback(pl.Callback):
    def __init__(self):
        self._fit_start_time: float | None = None
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


# Hardware profiler callback: logs epoch wall time and peak VRAM to WandB
class HardwareProfilerCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_duration = time.time() - getattr(self, "_epoch_start_time", time.time())
        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if torch.cuda.is_available()
            else 0.0
        )
        pl_module.log("epoch_time_seconds", epoch_duration)
        pl_module.log("peak_vram_mb", peak_vram_mb)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
