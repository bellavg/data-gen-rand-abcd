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
        self._train_loop_start_time: float | None = None
        self._batch_start_time: float | None = None
        self._last_batch_end_time: float | None = None
        self._train_wait_time_sum = 0.0
        self._train_step_time_sum = 0.0
        self._train_batch_count = 0
        self._train_graph_count = 0
        self._train_node_count = 0
        self._train_edge_count = 0
        self._report_every_n_steps = max(1, int(report_every_n_steps))
        self._max_batch_compute_reports = max(0, int(max_batch_compute_reports))
        self._batch_compute_reports_emitted = 0

    def _should_report_batch(self, trainer, batch_idx: int) -> bool:
        return trainer.current_epoch == 0 and batch_idx < 3

    def _can_emit_batch_compute_report(self) -> bool:
        return self._batch_compute_reports_emitted < self._max_batch_compute_reports

    def _format_val_batches(self, trainer) -> str:
        num_val_batches = getattr(trainer, "num_val_batches", None)
        if isinstance(num_val_batches, (list, tuple)):
            return ",".join(str(int(v)) for v in num_val_batches)
        if num_val_batches is None:
            return "unknown"
        return str(int(num_val_batches))

    def _batch_stats(self, batch) -> tuple[int, int, int]:
        if isinstance(batch, (list, tuple)):
            stats = [self._batch_stats(item) for item in batch]
            return tuple(map(sum, zip(*stats))) if stats else (0, 0, 0)

        num_graphs = getattr(batch, "num_graphs", None)
        if num_graphs is None:
            targets = getattr(batch, "y", None)
            if getattr(targets, "dim", None) is not None and targets.dim() > 0:
                num_graphs = int(targets.size(0))
            else:
                num_graphs = 1
        else:
            num_graphs = int(num_graphs)
        num_nodes = int(batch.x.size(0)) if getattr(batch, "x", None) is not None else 0
        edge_index = getattr(batch, "edge_index", None)
        num_edges = int(edge_index.size(1)) if edge_index is not None else 0
        return num_graphs, num_nodes, num_edges

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
        self._train_loop_start_time = time.monotonic()
        self._last_batch_end_time = self._train_loop_start_time
        print(
            f"[train] Training loop started after {self._elapsed():.1f}s.",
            flush=True,
        )
        print(
            "[train] Batches per epoch: "
            f"train={int(getattr(trainer, 'num_training_batches', -1))}, "
            f"val={self._format_val_batches(trainer)}",
            flush=True,
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.monotonic()
        self._batch_start_time = now
        wait_time = 0.0
        if self._last_batch_end_time is not None:
            wait_time = now - self._last_batch_end_time
            self._train_wait_time_sum += wait_time

        num_graphs, num_nodes, num_edges = self._batch_stats(batch)
        self._train_batch_count += 1
        self._train_graph_count += num_graphs
        self._train_node_count += num_nodes
        self._train_edge_count += num_edges

        if self._first_batch_reported or batch_idx != 0:
            if self._should_report_batch(trainer, batch_idx):
                print(
                    "[train] Batch stats: "
                    f"idx={batch_idx} graphs={num_graphs} nodes={num_nodes} "
                    f"edges={num_edges} data_wait_s={wait_time:.3f}",
                    flush=True,
                )
            return
        self._first_batch_reported = True
        print(
            f"[train] First training batch started after {self._elapsed():.1f}s.",
            flush=True,
        )
        print(
            "[train] Batch stats: "
            f"idx={batch_idx} graphs={num_graphs} nodes={num_nodes} "
            f"edges={num_edges} data_wait_s={wait_time:.3f}",
            flush=True,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        end_time = time.monotonic()
        if self._batch_start_time is not None:
            step_time = end_time - self._batch_start_time
            self._train_step_time_sum += step_time
            num_graphs, _, _ = self._batch_stats(batch)

            # Log step time to W&B - logs on every step and averages at end of epoch
            pl_module.log(
                "train_step_time_s",
                step_time,
                batch_size=num_graphs,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

            if self._should_report_batch(trainer, batch_idx) and self._can_emit_batch_compute_report():
                print(
                    f"[train] Batch compute: idx={batch_idx} step_s={step_time:.3f}",
                    flush=True,
                )
                self._batch_compute_reports_emitted += 1
        self._last_batch_end_time = end_time

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start_time = time.monotonic()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._epoch_start_time is None:
            return
        epoch_duration = time.monotonic() - self._epoch_start_time
        pl_module.log("epoch_time_seconds", epoch_duration)
        if self._train_batch_count > 0:
            avg_graphs = self._train_graph_count / self._train_batch_count
            avg_nodes = self._train_node_count / self._train_batch_count
            avg_edges = self._train_edge_count / self._train_batch_count
            avg_wait = self._train_wait_time_sum / self._train_batch_count
            avg_step = self._train_step_time_sum / self._train_batch_count
            print(
                "[train] Epoch summary: "
                f"avg_graphs_per_batch={avg_graphs:.2f} "
                f"avg_nodes_per_batch={avg_nodes:.0f} "
                f"avg_edges_per_batch={avg_edges:.0f} "
                f"avg_data_wait_s={avg_wait:.3f} "
                f"avg_step_s={avg_step:.3f} "
                f"epoch_s={epoch_duration:.1f}",
                flush=True,
            )
