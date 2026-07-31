"""Training entrypoint for baseline models (SynthNet, HOGA) on this project's AIG dataset.

Mirrors train.py's structure (argparse -> AIGDataModule -> Lightning module ->
Trainer) but swaps in a baseline model + baselines.common.lightning_wrapper
instead of the primary UnifiedGraphBaseModel/AIGRegressionLightningModule, so
train.py and the primary model stay completely untouched.

Baseline model hyperparameters and training config (optimizer, loss, LR,
scheduler) default to each paper's own published values (SynthNet:
models/qor/SynthNetV3/train.py; HOGA: Deng et al. DAC'24 Section 3.3/4.1 --
see baselines/hoga/regressor.py's module docstring for exactly which values
are published vs. assumed, since a couple of HOGA's knobs -- heads, dropout --
still have no published QoR-task source), not this project's own config.py
defaults -- those are two separate baseline papers with their own training
setups. Only data
loading/splitting/caching (AIGDataModule / AIGGraphRegressionDataset) is
reused unchanged, since identical splits are required for a fair comparison
against the primary model, and that part isn't "baseline config".

SynthNet uses a plain fixed batch size (its published 64), matching its paper.

HOGA does NOT, and this is a deliberate, documented deviation. Upstream HOGA
minibatches over *nodes*, not graphs: main_gamora.py builds
`Data.TensorDataset(data.x[train_idx], ...)` with `batch_size=1024`, i.e. 1024
nodes sampled from a single preprocessed graph, and its `HOGA.forward(x)` takes
only `[num_nodes, num_hop_slots, feat]` -- no edge_index, no batch vector,
because the node axis *is* the attention batch axis. That works upstream
because the task is per-node classification on one graph. This project's task
is one scalar per whole graph, so global_mean_pool needs every node of a graph
present in the same forward pass, which forces graph-level batching.

Batching a fixed *graph* count then makes memory unbounded: this dataset's
graphs average ~40k nodes (max 366k), so batch_size=32 is ~1.29M nodes, and
HOGA's trunk holds [num_nodes, 6, 256] activations = ~3.0 KB/node under the
bf16-mixed AMP selected on H100, i.e. ~3.9 GB for a single activation tensor
before attention and backward -- an immediate OOM. Bounding *nodes* per batch
(--hoga_max_nodes_per_batch) is therefore closer to upstream's own batching
unit than a fixed graph count is, not further from it. There is no published
QoR-task batch size to match regardless: HOGA never released QoR training
code (see baselines/hoga/regressor.py).

A node budget alone leaves each optimizer step averaging only ~7.5 graph-level
losses (300k / ~40k), versus ~75 for the primary model, because one graph is
one label regardless of its node count -- so a 40k-node graph reduces gradient
variance no more than a 400-node one does. HOGA therefore also accumulates
(--accumulate_grad_batches, 10 in train_baseline_hoga.sh) so the two models
are optimized under comparable gradient noise. This is approximate, not exact:
Lightning divides each micro-batch loss by the constant accumulate_grad_batches
rather than by its graph count, so micro-batches weigh equally however many
graphs they hold -- and since node-budget packing puts fewer graphs in batches
containing large graphs, large graphs draw more per-graph weight. Effective
sample size over the window is the harmonic mean, somewhat below 10 x 7.5.

The node budget and the accumulation count MUST be retuned together: their
product is the effective batch, and the whole point of the pairing is to hold
it near the primary model's ~75 graphs/update. Changing one alone silently
changes the optimization regime this file exists to keep comparable.

Everything the paper DOES publish for the QoR task is kept verbatim --
hidden_dim=256, num_layers=1, lr=0.0001, num_hops=5. Note num_layers=1 is not
an oversight: it counts gated self-attention layers over the 6-slot hop
dimension, not message-passing layers. Depth of field comes from num_hops=5,
already baked into the precomputed features, which is why HOGA.forward() never
sees a graph. Batching and accumulation are adapted because no published value
exists and the task differs (graph-level pooling vs upstream's per-node
prediction), not because the published config was inconvenient. If memory
still forces a change, cap graph size at the dataset level rather than
altering those four values.
"""

from __future__ import annotations

import argparse
import os
import random
import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

import config
from baselines.common.lightning_wrapper import BaselineRegressionLightningModule
from baselines.hoga.hop_features import HopFeatureCache, collate_hoga_batch, num_hop_slots
from baselines.hoga.regressor import DEFAULT_HEADS as HOGA_DEFAULT_HEADS
from baselines.hoga.regressor import DEFAULT_HIDDEN_DIM as HOGA_DEFAULT_HIDDEN_DIM
from baselines.hoga.regressor import DEFAULT_LR as HOGA_DEFAULT_LR
from baselines.hoga.regressor import DEFAULT_NUM_HOPS as HOGA_DEFAULT_NUM_HOPS
from baselines.hoga.regressor import DEFAULT_NUM_LAYERS as HOGA_DEFAULT_NUM_LAYERS
from baselines.hoga.regressor import HOGAGraphRegressor
from baselines.openabc_synthnet.regressor import (
    DEFAULT_DROP_RATIO,
    DEFAULT_FC_HIDDEN_DIM,
    DEFAULT_GNN_HIDDEN_DIM,
    DEFAULT_NODE_EMB_DIM,
    DEFAULT_NUM_FC_LAYER,
    SynthNetGraphRegressor,
)
from data.datamodule import AIGDataModule
from data.sampler import BalancedDynamicBatchSampler
from train_utils import PreciseEarlyStopping, TrainingStartupCallback

torch.set_num_threads(1)

# Published defaults for each baseline paper's own training setup -- see the
# regressor modules for exactly where each of these comes from.
SYNTHNET_DEFAULTS = {"batch_size": 64, "lr": 0.001, "weight_decay": 0.0}
HOGA_DEFAULTS = {
    "batch_size": config.BATCH_SIZE,  # no published QoR-task batch size; see baselines/hoga/regressor.py
    "lr": HOGA_DEFAULT_LR,  # 0.0001, published (Deng et al. DAC'24, Sec 3.3/4.1)
    "weight_decay": 0.0,
}


def _batch_limit(value: str) -> int | float:
    """Parse a Trainer batch limit, preserving Lightning's int/float distinction.

    Lightning reads an `int` as an absolute batch count and a `float` as a
    fraction of the epoch. Its `_determine_batch_limits` coerces any float > 1
    with no fractional part back to an int, so `type=float` would in fact
    handle `15000` fine. The distinction bites at exactly one value: `1` means
    "a single batch", `1.0` means "the whole epoch". Under `type=float`,
    `--limit_val_batches 1` would silently become `1.0` and run the full
    32k-batch validation instead of one batch; this converter keeps the two
    apart, so the flag means what it says at every value.
    """
    return float(value) if "." in value else int(value)


def _select_precision() -> str:
    try:
        return (
            "bf16-mixed"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "32-true"
        )
    except (AssertionError, RuntimeError):
        return "32-true"


def _select_accelerator_and_devices() -> tuple[str, int]:
    require_gpu = str(os.environ.get("AIG_REQUIRE_GPU", "")).lower() in {
        "1",
        "true",
        "yes",
    }
    try:
        if torch.cuda.is_available():
            torch.cuda.get_device_properties(0)
            return "gpu", 1
    except (AssertionError, RuntimeError) as exc:
        if require_gpu:
            raise RuntimeError(
                "GPU was requested but CUDA could not be initialized. "
                "On SLURM this usually means the Python process was not launched "
                "inside a GPU job step, CUDA_VISIBLE_DEVICES is wrong, or the node "
                "driver is unhealthy. Try launching with srun and check nvidia-smi."
            ) from exc
        return "cpu", 1
    if require_gpu:
        raise RuntimeError(
            "GPU was requested but torch.cuda.is_available() is False. "
            "Check the SLURM GPU allocation, CUDA_VISIBLE_DEVICES, and nvidia-smi output."
        )
    return "cpu", 1


def _build_model(args: argparse.Namespace) -> nn.Module:
    if args.baseline == "synthnet":
        return SynthNetGraphRegressor(
            node_emb_dim=args.synthnet_node_emb_dim,
            gnn_hidden_dim=args.synthnet_gnn_hidden_dim,
            num_fc_layer=args.synthnet_num_fc_layer,
            fc_hidden_dim=args.synthnet_fc_hidden_dim,
            drop_ratio=args.synthnet_drop_ratio,
            task_out_dim=config.TASK_OUT_DIM,
            upstream_edge_direction=args.synthnet_upstream_edge_direction,
        )
    if args.baseline == "hoga":
        return HOGAGraphRegressor(
            in_channels=config.NODE_INPUT_DIM,
            hidden_channels=args.hoga_hidden_dim,
            num_layers=args.hoga_num_layers,
            dropout=args.hoga_dropout,
            num_hops=num_hop_slots(args.hoga_num_hops, directed=args.hoga_directed),
            heads=args.hoga_heads,
            head_dropout=args.hoga_head_dropout,
            task_out_dim=config.TASK_OUT_DIM,
        )
    raise ValueError(f"Unknown baseline: {args.baseline!r}")


def _loader_kwargs(args: argparse.Namespace) -> dict:
    kwargs: dict = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def _plain_loader(ds, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds, shuffle=shuffle, collate_fn=Batch.from_data_list, **_loader_kwargs(args)
    )


def _hoga_loader(ds, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    wrapped = HopFeatureCache(
        ds,
        num_hops=args.hoga_num_hops,
        cache_dir=args.hoga_hop_cache_dir,
        directed=args.hoga_directed,
    )

    if not args.hoga_max_nodes_per_batch:
        return DataLoader(
            wrapped, shuffle=shuffle, collate_fn=collate_hoga_batch, **_loader_kwargs(args)
        )

    # Node-budget batching (see this module's docstring for why HOGA needs it
    # and SynthNet does not). HopFeatureCache is index-aligned with `ds` --
    # same length, __getitem__ delegates straight through -- so a plan built
    # from ds's node counts indexes `wrapped` correctly.
    # list(...) so the in-place shuffle below can never reach a shared object.
    # build_batch_plan returns a fresh list today, but load_or_build_batch_plan
    # (data/sampler.py) hands back the process-wide cache entry, and swapping
    # to it here is the obvious next optimization -- at which point shuffling
    # in place would corrupt that cache for every other sampler holding it.
    plan = list(
        BalancedDynamicBatchSampler.build_batch_plan(
            ds.get_num_nodes_list(),
            max_total_nodes=args.hoga_max_nodes_per_batch,
        )
    )
    # build_batch_plan sorts indices by node count, then repeatedly anchors a
    # batch on the largest remaining graph and backfills with the smallest
    # ones that fit (data/sampler.py). The plan therefore comes out ordered by
    # descending anchor size, so ANY PREFIX of it is "the biggest graphs plus
    # the smallest", containing nothing from the middle of the distribution.
    # That is harmless while a full epoch runs, but --limit_val_batches takes
    # exactly such a prefix and the val loader is built with shuffle=False, so
    # a capped val pass would score a bimodal extremes-only sample and hand it
    # to ModelCheckpoint, PreciseEarlyStopping and ReduceLROnPlateau -- and it
    # would no longer be comparable to train.py's val_loss, which is the whole
    # point of the baseline. Shuffling the plan once, off a fixed seed, makes
    # a truncated pass a representative sample while keeping it byte-identical
    # from epoch to epoch. (The train loader reshuffles per epoch on top of
    # this; the one-time shuffle costs it nothing.)
    #
    # Offset from `seed`, because the sampler's own per-epoch shuffle uses
    # Random(seed + epoch): an unoffset `seed` here would replay the identical
    # Fisher-Yates draw at epoch 0, making that epoch's order the square of one
    # permutation rather than two independent ones. Note +1 only RELOCATES that
    # collision to epoch 1 rather than removing it -- a large offset would be
    # needed to avoid it entirely. Left as is because the consequence is
    # cosmetic: one epoch out of ~60 sees a composed-with-itself batch order,
    # which is still a valid permutation of the same batches.
    random.Random(args.seed + 1).shuffle(plan)

    sampler = BalancedDynamicBatchSampler(
        batch_size=args.batch_size,
        shuffle=shuffle,
        seed=args.seed,
        max_total_nodes=args.hoga_max_nodes_per_batch,
        precomputed_batches=plan,
    )
    ds.release_runtime_caches()

    kwargs = _loader_kwargs(args)
    kwargs.pop("batch_size")
    return DataLoader(
        wrapped, batch_sampler=sampler, collate_fn=collate_hoga_batch, **kwargs
    )


def main(args: argparse.Namespace) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.multiprocessing.set_sharing_strategy("file_system")

    if args.algorithm not in config.VALID_ALGORITHMS:
        raise ValueError(
            f"Algorithm '{args.algorithm}' must be one of {config.VALID_ALGORITHMS}"
        )
    defaults = SYNTHNET_DEFAULTS if args.baseline == "synthnet" else HOGA_DEFAULTS
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.weight_decay is None:
        args.weight_decay = defaults["weight_decay"]

    print(f"--- Starting Baseline Training: {args.baseline} / {args.algorithm} ---")

    datamodule = AIGDataModule(
        csv_paths=args.csv_paths,
        # Neither baseline reads .pos_enc -- both derive their own features
        # from .x/.edge_index/.edge_attr directly -- but the per-graph cache
        # filename AND content in dataset.py both key on positional_encoding
        # (see _stable_graph_cache_name/_prepare_cached_graph). Passing None
        # here would silently miss the primary model's existing shared
        # tier0_cache_dir/tier1_cache_dir cache entirely (different hash,
        # different file) and rebuild a full second copy of the same ~700k
        # graphs from scratch. Matching config.PE_TYPE makes cache lookups
        # hit the already-built shared cache; the resulting unused pos_enc
        # attribute is harmless (the baseline models never read it).
        positional_encoding=config.PE_TYPE if config.PE_TYPE != "none" else None,
        # Baselines train on UNREDUCED graphs, deliberately and always -- there
        # is no --sparsification flag on this entrypoint. Both baseline papers
        # define their model over the full netlist, and sparsification is this
        # project's own contribution, not part of either baseline.
        #
        # The consequence for reporting: the like-for-like counterpart is
        # train.py run via shell/train_no_sparsification.sh (--sparsification
        # none), NOT the shell/train.sh array, whose four tasks each apply one
        # of and_gate_only / random_edge_dropout / spanning_forest / pagerank.
        # Comparing a baseline against a sparsified primary run confounds
        # architecture with sparsification and cannot support a claim about
        # either. It also means every node-count figure quoted in this file and
        # in train_baseline_hoga.sh (~40k mean, 366k max, ~32.4e9 per epoch) is
        # the unreduced count, which is the regime these jobs actually run in.
        sparsification=None,
        partition=None,
        batch_size=args.batch_size,
        split_ratios=(0.8, 0.1, 0.1),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        cache_dir=args.cache_dir if args.cache_dir else None,
        hp_tuning_splits_path=args.hp_tuning_splits_path,
        tier0_cache_dir=args.tier0_cache_dir,
        tier1_cache_dir=args.tier1_cache_dir,
        # HOGA's node budget is applied in _hoga_loader, which builds its own
        # sampler over these datasets; SynthNet keeps its published fixed batch
        # size. Neither uses the datamodule's own loaders. See module docstring.
        dynamic_batching=False,
    )

    print("[main] Loading datasets before Trainer/WandB init ...", flush=True)
    ds_start = time.monotonic()
    # "fit" only, matching train.py. The test split is evaluated separately,
    # and setting up the test stage here would build a graph cache for ~96k
    # test graphs during the GPU job (warmup_train_cache.sh warms train + val
    # only, by design).
    datamodule.setup("fit")
    print(
        f"[main] Datasets loaded in {time.monotonic() - ds_start:.1f}s", flush=True
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.baseline == "hoga":
        train_loader = _hoga_loader(datamodule.train_ds, args, shuffle=True)
        val_loader = _hoga_loader(datamodule.val_ds, args, shuffle=False)
    else:
        train_loader = _plain_loader(datamodule.train_ds, args, shuffle=True)
        val_loader = _plain_loader(datamodule.val_ds, args, shuffle=False)

    base_model = _build_model(args)
    model = BaselineRegressionLightningModule(
        base_model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name="adam",
        loss_fn=nn.MSELoss(),
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
    )

    run_label = f"{args.baseline}_{args.algorithm}"
    algo_checkpoint_dir = os.path.join(args.checkpoint_dir, run_label)
    os.makedirs(algo_checkpoint_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=algo_checkpoint_dir,
        save_top_k=3,
        save_last=True,
        monitor="val_loss",
        mode="min",
        filename="{epoch:02d}-val_loss={val_loss:.4f}",
        save_on_train_epoch_end=True,
    )
    early_stop_cb = PreciseEarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min",
        verbose=True,
        check_on_train_epoch_end=True,
    )

    log_dir = f"{args.log_dir}_{run_label}"
    os.makedirs(log_dir, exist_ok=True)
    print("[main] Initialising WandB logger ...", flush=True)
    wandb_start = time.monotonic()
    logger = WandbLogger(
        project="AIG-SUMMARIZE",
        entity="isabella-v-gardner-university-of-amsterdam",
        name=f"train_baseline_{run_label}",
        save_dir=log_dir,
    )
    print(f"[main] WandB ready in {time.monotonic() - wandb_start:.1f}s", flush=True)

    callbacks = [
        checkpoint_cb,
        early_stop_cb,
        LearningRateMonitor(logging_interval="epoch"),
        TrainingStartupCallback(
            report_every_n_steps=args.log_steps,
            max_batch_compute_reports=args.max_batch_compute_reports,
        ),
    ]

    precision = _select_precision()
    accelerator, devices = _select_accelerator_and_devices()
    print(f"Using {precision} Automatic Mixed Precision (AMP)", flush=True)
    print(f"Using accelerator={accelerator}, devices={devices}", flush=True)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        enable_progress_bar=False,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        # Match train.py's explicit 0 (Lightning's own default is 2) so the
        # baseline and the primary model start training from the same point.
        num_sanity_val_steps=0,
        log_every_n_steps=args.log_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )

    print(
        f"--- Running Training for baseline={args.baseline} algorithm={args.algorithm} ---",
        flush=True,
    )
    fit_start = time.monotonic()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(
        f"--- trainer.fit completed in {time.monotonic() - fit_start:.1f}s ---",
        flush=True,
    )

    # No trainer.test() here: the test split is evaluated separately, from the
    # saved checkpoints. Matches train.py, which also fits only.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a baseline model (SynthNet or HOGA) on this project's AIG dataset"
    )

    parser.add_argument(
        "--baseline", type=str, required=True, choices=["synthnet", "hoga"]
    )
    parser.add_argument("--algorithm", type=str, default="Orchestrate")
    parser.add_argument("--csv_paths", nargs="+", required=True)

    # Training config: default to None, resolved per-baseline from
    # SYNTHNET_DEFAULTS / HOGA_DEFAULTS in main() (see module docstring).
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    # Unlike lr/batch_size/loss/optimizer above, the ReduceLROnPlateau settings
    # are NOT published: neither baseline paper uses an LR scheduler at all (see
    # baselines/common/lightning_wrapper.py). Since the values are ours either
    # way, take config.py's -- the same schedule the primary model trains under,
    # so the comparison differs by architecture rather than LR schedule. The
    # previous 0.1/10 also left the scheduler inert: patience 10 epochs can
    # never fire under --patience 4 early stopping.
    parser.add_argument(
        "--scheduler_factor", type=float, default=config.SCHEDULER_FACTOR
    )
    parser.add_argument(
        "--scheduler_patience", type=int, default=config.SCHEDULER_PATIENCE
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_epochs", type=int, default=80
    )  # models/qor/SynthNetV3/train.py default
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    # Defaults to 1 (no accumulation) so SynthNet keeps its published
    # batch_size=64 effective batch untouched. train_baseline_hoga.sh sets 10:
    # HOGA's node budget yields ~7.5 graphs per micro-batch (300k / ~40k
    # nodes) against the primary model's ~75 (3M / ~40k), an optimization
    # confound rather than an architectural difference. Node count buys no
    # variance reduction -- one graph is one label is one loss term, whether
    # it has 400 nodes or 400k. Caveat: Lightning divides by the CONSTANT
    # accumulate_grad_batches, so micro-batches are weighted equally however
    # many graphs they hold, and the effective sample size is the harmonic
    # mean, somewhat below 10 x 7.5. Approximates the primary model's regime;
    # does not match it exactly. Must be retuned whenever
    # --hoga_max_nodes_per_batch changes. See this module's docstring.
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    # Epoch subsampling. A full epoch is ~75k micro-batches plus ~16k val
    # batches at the 300k node budget train_baseline_hoga.sh sets (it was
    # 149,485 / 32,211 at the old 150k budget, measured at ~12h/epoch before
    # the hop-slot and attention fixes), so a single epoch outruns any useful
    # checkpoint/early-stop cadence. Both default to 1.0 (full epoch, no
    # behaviour change) and are set explicitly in train_baseline_hoga.sh.
    #
    # The train sampler reshuffles batch ORDER per epoch (seed + epoch, see
    # data/sampler.py), so a fractional/absolute train limit draws a different
    # subset every epoch rather than replaying the same head of the dataset.
    # The val loader is built with shuffle=False, so its limit always takes
    # the same batches -- deliberately, so val_loss stays a stable signal for
    # ModelCheckpoint and PreciseEarlyStopping instead of moving each epoch.
    parser.add_argument("--limit_train_batches", type=_batch_limit, default=1.0)
    parser.add_argument("--limit_val_batches", type=_batch_limit, default=1.0)
    parser.add_argument("--log_steps", type=int, default=config.LOG_EVERY_N_STEPS)
    parser.add_argument(
        "--max_batch_compute_reports",
        type=int,
        default=config.MAX_BATCH_COMPUTE_REPORTS,
    )

    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--prefetch_factor", type=int, default=config.PREFETCH_FACTOR)
    parser.add_argument(
        "--pin_memory",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=config.PIN_MEMORY,
    )
    parser.add_argument(
        "--persistent_workers",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=config.PERSISTENT_WORKERS,
    )

    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--tier0_cache_dir", type=str, default=None)
    parser.add_argument("--tier1_cache_dir", type=str, default=None)
    parser.add_argument("--hp_tuning_splits_path", type=str, default=None)
    # No --split_by flag: this branch's AIGGraphRegressionDataset hardcodes
    # design-level splitting (see data/dataset.py, "split_by": "design" baked
    # into its cache signature) -- there's no alternative to select on this
    # branch. A configurable split_by (design/recipe/random) was added on
    # main after this branch diverged; it isn't available here.

    # SynthNet hyperparameters (defaults: models/qor/SynthNetV3/train.py).
    parser.add_argument(
        "--synthnet_node_emb_dim", type=int, default=DEFAULT_NODE_EMB_DIM
    )
    parser.add_argument(
        "--synthnet_gnn_hidden_dim", type=int, default=DEFAULT_GNN_HIDDEN_DIM
    )
    parser.add_argument(
        "--synthnet_num_fc_layer", type=int, default=DEFAULT_NUM_FC_LAYER
    )
    parser.add_argument(
        "--synthnet_fc_hidden_dim", type=int, default=DEFAULT_FC_HIDDEN_DIM
    )
    parser.add_argument("--synthnet_drop_ratio", type=float, default=DEFAULT_DROP_RATIO)
    # True (default) reverses this project's fanin -> node edges to upstream's
    # node -> fanin convention before the GCN, so messages flow toward the
    # primary inputs exactly as in OpenABC-D. False keeps this project's native
    # direction. See baselines/openabc_synthnet/regressor.py's docstring.
    parser.add_argument(
        "--synthnet_upstream_edge_direction",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=True,
    )

    # HOGA hyperparameters. hidden_dim/num_layers/num_hops/lr are published
    # (Deng et al. DAC'24, Sec 3.3/4.1); heads carries over from the Gamora
    # task's run.sh; dropout has no published source for either task -- see
    # baselines/hoga/regressor.py's module docstring for the full breakdown.
    parser.add_argument(
        "--hoga_hidden_dim", type=int, default=HOGA_DEFAULT_HIDDEN_DIM
    )
    parser.add_argument(
        "--hoga_num_layers", type=int, default=HOGA_DEFAULT_NUM_LAYERS
    )
    parser.add_argument("--hoga_dropout", type=float, default=config.DROPOUT)
    parser.add_argument(
        "--hoga_num_hops",
        type=int,
        default=HOGA_DEFAULT_NUM_HOPS,
        help="Propagation depth per direction, i.e. K (see baselines/hoga/hop_features.py).",
    )
    parser.add_argument("--hoga_heads", type=int, default=HOGA_DEFAULT_HEADS)
    parser.add_argument("--hoga_head_dropout", type=float, default=0.3)
    # Default False, matching both the paper (Section 3.1 defines a single
    # symmetric-normalized adjacency and stacks K+1 slots) and upstream, whose
    # --directed is action='store_true' and whose published run.sh never
    # passes it. True selects this project's own fanin/fanout extension, which
    # doubles the slot width to 1 + 2K.
    parser.add_argument(
        "--hoga_directed",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=False,
    )
    parser.add_argument(
        "--hoga_max_nodes_per_batch",
        type=int,
        default=300_000,
        help=(
            "Total-node budget per HOGA batch, replacing --batch_size for the "
            "hoga baseline (0 disables, restoring fixed graph-count batching). "
            "At ~3.0 KB/node per [N, 6, hidden] activation under bf16-mixed "
            "(undirected hop features, the default), 300k nodes is ~920 MB per "
            "activation tensor. Retune --accumulate_grad_batches alongside "
            "this: their product is the effective batch. NOTE this bounds "
            "typical batches only: a graph larger than the budget still forms "
            "a singleton batch (it cannot be split -- graph-level pooling "
            "needs all its nodes at once), so the largest graph sets an "
            "irreducible peak, ~1.1 GB/activation at config.MAX_NUM_GATES. "
            "See this module's docstring."
        ),
    )
    parser.add_argument(
        "--hoga_hop_cache_dir",
        type=str,
        default=None,
        help=(
            "Optional on-disk hop-feature cache. Leave unset (the default) to "
            "compute hop features in the dataloader workers instead -- at full "
            "dataset scale the cache needs ~3.1 TB / ~788k files and does not "
            "fit scratch quota. See baselines/hoga/hop_features.py."
        ),
    )

    args = parser.parse_args()
    main(args)
