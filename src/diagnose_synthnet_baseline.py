"""Diagnose a collapsed SynthNet baseline run (flat val_loss, R2 ~ 0 or negative).

Answers three questions from a trained checkpoint, without retraining:

  1. Are the predictions actually constant?  std(preds) vs std(targets).
  2. If so, where did it collapse -- the GNN trunk or the FC head?  Reports the
     between-graph std of the graph embedding (concat of global_max_pool and
     global_mean_pool).  A near-zero std means every graph produces the same
     embedding and the head can only learn a constant; a healthy std with
     constant predictions instead points at the head.
  3. Do the train and val splits even share a target distribution?  Splits are
     design-level (data/dataset.py bakes in split_by=design), so held-out
     designs can have a visibly different optimizability spread -- which by
     itself drives val R2 negative for any constant predictor.

Val graphs are sampled with a seeded permutation, not taken from the front of
the split: `_read_candidate_samples` preserves CSV row order and the CSVs are
per-design concatenations, so the first N rows all belong to one design. A
contiguous slice would measure within-design spread and "confirm" collapse
regardless of the model.

`--upstream_edge_direction` is REQUIRED and has no default. It is a plain
attribute, so it appears in neither `state_dict()` nor `ckpt["hyper_parameters"]`
(`lightning_wrapper.py` calls `save_hyperparameters(ignore=["model", ...])`),
and a mismatch would silently run the trunk with `edge_index` reversed relative
to training -- applying BatchNorm running stats collected under the opposite
message direction, which makes even a healthy trunk look collapsed. Checkpoints
trained before that flag existed used this project's native direction, i.e.
`false`.

Paths default to the ones train_baseline_synthnet.sh uses, so the splits this
loads are the splits the checkpoint was trained on. Run on a login or
interactive node with that script's environment:

    export PYTHONPATH=$HOME/data-gen-rand-abcd/src
    python -m diagnose_synthnet_baseline --ckpt /path/to/best.ckpt \
        --upstream_edge_direction false
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch_geometric.data import Batch

import config
from baselines.openabc_synthnet.regressor import SynthNetGraphRegressor
from data.datamodule import AIGDataModule


def _default_paths(algorithm: str) -> dict[str, str]:
    user = os.environ.get("USER")
    if not user:
        raise RuntimeError(
            "USER is unset, so the /scratch-shared paths cannot be derived. "
            "Pass --cache_dir/--tier0_cache_dir/--tier1_cache_dir/"
            "--hp_tuning_splits_path explicitly."
        )
    home = Path.home()
    scratch = f"/scratch-shared/{user}"
    return {
        "csv": str(
            home
            / "data-gen-rand-abcd/data/designs/design_metadata"
            / f"algo_{algorithm}_ml.csv"
        ),
        "cache_dir": f"{scratch}/aig_train_run/{algorithm}/cache",
        "tier0": f"{scratch}/aig_train_run/shared_tier0_cache",
        "tier1": f"{scratch}/aig_train_run/shared_tier1_cache",
        "hp_splits": (
            f"{scratch}/big_optuna_run/shared_dataset_cache/"
            "algo_Orchestrate_ml_algo_Deepsyn_ml_algo_Syn4_ml_algo_C2RS_ml_50000_splits.json"
        ),
    }


def _report_target_distributions(datamodule: AIGDataModule) -> None:
    print("\n=== Q3: target distribution per split (no model involved) ===")
    for name, dataset in (("train", datamodule.train_ds), ("val", datamodule.val_ds)):
        y = torch.tensor([s.y_node_opt for s in dataset.samples])
        designs = {s.design_key for s in dataset.samples}
        print(
            f"  {name:5s} n={len(y):>7d}  designs={len(designs):>3d}  "
            f"mean={y.mean():.5f}  std={y.std():.5f}  "
            f"min={y.min():.5f}  max={y.max():.5f}"
        )
    print(
        "  A train/val std ratio far from 1 means the design split shifted the\n"
        "  target distribution; a constant predictor fitted on train is then\n"
        "  miscalibrated on val, which alone produces a negative val R2."
    )


def _load_model(ckpt_path: str, upstream_edge_direction: bool) -> SynthNetGraphRegressor:
    model = SynthNetGraphRegressor(
        task_out_dim=config.TASK_OUT_DIM,
        upstream_edge_direction=upstream_edge_direction,
    )
    # weights_only=False: a Lightning .ckpt carries callback/scheduler/hparam
    # state, not just tensors, and torch >= 2.11 defaults weights_only=True.
    # This is a locally produced file.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # BaselineRegressionLightningModule holds the baseline under `self.model`;
    # its rmse_metrics/r2_metrics ModuleDicts use different prefixes.
    state_dict = {
        key[len("model.") :]: value
        for key, value in ckpt["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def main(args: argparse.Namespace) -> None:
    defaults = _default_paths(args.algorithm)

    datamodule = AIGDataModule(
        csv_paths=[args.csv_path or defaults["csv"]],
        # Must match train_baseline.py exactly, or the cache signature changes
        # and this loads a different split than the checkpoint trained on.
        positional_encoding=config.PE_TYPE if config.PE_TYPE != "none" else None,
        sparsification=None,
        partition=None,
        batch_size=args.batch_size,
        split_ratios=(0.8, 0.1, 0.1),
        num_workers=0,
        prefetch_factor=1,
        cache_dir=args.cache_dir or defaults["cache_dir"],
        tier0_cache_dir=args.tier0_cache_dir or defaults["tier0"],
        tier1_cache_dir=args.tier1_cache_dir or defaults["tier1"],
        hp_tuning_splits_path=args.hp_tuning_splits_path or defaults["hp_splits"],
        dynamic_batching=False,
    )
    datamodule.setup("fit")

    _report_target_distributions(datamodule)

    model = _load_model(args.ckpt, args.upstream_edge_direction)

    val_ds = datamodule.val_ds
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(val_ds), generator=generator)[: args.num_graphs]
    sampled_designs = {val_ds.samples[int(i)].design_key for i in order}

    embeddings, predictions, targets = [], [], []
    with torch.no_grad():
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            batch = Batch.from_data_list([val_ds[int(i)] for i in indices])
            emb = model.encode(batch)
            embeddings.append(emb)
            predictions.append(model.head(emb).squeeze(-1))
            targets.append(batch.y.view(-1))

    pred = torch.cat(predictions)
    targ = torch.cat(targets)
    emb = torch.cat(embeddings)

    ss_res = ((pred - targ) ** 2).sum()
    ss_tot = ((targ - targ.mean()) ** 2).sum()

    print(
        f"\n=== Q1: are predictions constant? "
        f"({len(order)} val graphs, {len(sampled_designs)} designs) ==="
    )
    print(f"  preds    mean={pred.mean():.6f}  std={pred.std():.6f}")
    print(f"  targets  mean={targ.mean():.6f}  std={targ.std():.6f}")
    print(f"  R2       {1 - ss_res / ss_tot:.4f}")
    print(
        "  std(preds) several orders of magnitude below std(targets) means the\n"
        "  model has collapsed to predicting a single value."
    )

    channel_std = emb.std(dim=0)
    half = channel_std.numel() // 2
    print("\n=== Q2: trunk or head? (between-graph std of the graph embedding) ===")
    print(
        f"  all {channel_std.numel()} channels  mean={channel_std.mean():.6f}  "
        f"max={channel_std.max():.6f}  min={channel_std.min():.6f}"
    )
    print(f"  max-pool half   mean std={channel_std[:half].mean():.6f}")
    print(f"  mean-pool half  mean std={channel_std[half:].mean():.6f}")
    print(
        f"  channels with std < 1e-3: {int((channel_std < 1e-3).sum())}"
        f"/{channel_std.numel()}"
    )
    print(
        "  Near-zero here means the trunk collapsed: batch_norm2 standardises\n"
        "  node features over every node in the batch, then pooling averages\n"
        "  ~40k nodes per graph, so each graph's pooled vector concentrates on\n"
        "  the same value. Healthy spread here instead implicates the head."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose a collapsed SynthNet baseline checkpoint"
    )
    parser.add_argument("--ckpt", required=True, help="Lightning .ckpt to inspect")
    parser.add_argument(
        "--upstream_edge_direction",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        required=True,
        help=(
            "Must match the --synthnet_upstream_edge_direction the checkpoint "
            "trained under. No default on purpose: the flag is not recorded in "
            "the checkpoint, and a silent mismatch invalidates every number "
            "below. Checkpoints predating the flag used 'false'."
        ),
    )
    parser.add_argument("--algorithm", type=str, default="Orchestrate")
    parser.add_argument(
        "--num_graphs",
        type=int,
        default=512,
        help="How many val graphs to score, sampled at random (default: 512)",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--seed", type=int, default=42, help="Seeds the val-graph sample."
    )
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--tier0_cache_dir", type=str, default=None)
    parser.add_argument("--tier1_cache_dir", type=str, default=None)
    parser.add_argument("--hp_tuning_splits_path", type=str, default=None)
    main(parser.parse_args())
