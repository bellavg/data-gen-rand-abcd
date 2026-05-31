import time
import torch
import pytorch_lightning as pl

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
