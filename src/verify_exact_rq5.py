#!/usr/bin/env python
"""RQ5 for the exact track: train on reducts, infer on full graphs.

For every other summarization method RQ5 is a genuine experiment — the
reduct is an approximation, so predicting from a full graph with a model
trained on reducts may or may not transfer.  For ``wl_exact`` it is a
*verification*: the reduct and the original produce the same graph embedding
by construction, so the two predictions are the same number and
train-on-reduced already is train-on-full.  There is nothing to train; the
deliverable is one forward pass per state and the residual between them,
reported explicitly rather than hidden behind an assert.

The full-graph input is free: ``fold_inversions_into_x`` with no merge
applied *is* the exact-schema uncoarsened graph, so this points one dataset
at the reduct cache and one at the ordinary unsummarized cache and folds the
latter on the way out.

Run on the cluster; needs a checkpoint from a ``--model exact`` training run.

  python -m verify_exact_rq5 \
      --algorithm Orchestrate --csv_paths <csv> \
      --checkpoint <ckpt.ckpt> \
      --reduct_cache_dir <stage>   --reduct_tier0_cache_dir ... \
      --full_cache_dir   <raw>     --full_tier0_cache_dir   ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import config
from data.datamodule import AIGDataModule
from data.exact_graph import fold_inversions_into_x
from data.summarize_graphs import assert_exact_depth_supports_model
from models.lightning_model import AIGRegressionLightningModule


def _fold_preserving_y(data):
    """fold_inversions_into_x, but keep the regression target.

    Used as a PyG dataset ``transform``.  ``fold_inversions_into_x`` returns
    a fresh Data holding only what the exact schema needs, so ``y`` (attached
    by ``AIGGraphRegressionDataset.get``) has to be carried over by hand.
    """
    out = fold_inversions_into_x(data)
    out.y = data.y
    return out


def _make_test_dataset(args, cache_dir, tier0, tier1, transform=None):
    datamodule = AIGDataModule(
        csv_paths=args.csv_paths,
        # Names the cache files (dataset._stable_graph_cache_name hashes it),
        # so it has to match whatever the caches were built with — not the
        # model's pe_type, which for the exact track is always "none".
        positional_encoding=args.pe_type if args.pe_type != "none" else None,
        exact_schema=True,
        batch_size=args.batch_size,
        split_ratios=(0.8, 0.1, 0.1),
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=1 if args.num_workers == 0 else config.PREFETCH_FACTOR,
        cache_dir=cache_dir,
        tier0_cache_dir=tier0,
        tier1_cache_dir=tier1,
        hp_tuning_splits_path=args.hp_tuning_splits_path,
        # Node counts differ between the two states, so a node-budgeted plan
        # would group the samples differently and break the pairing below.
        dynamic_batching=False,
        test_num_samples=args.num_samples,
    )
    datamodule.setup("test")
    datamodule.test_ds.transform = transform
    return datamodule


@torch.no_grad()
def _predict(model, loader, device) -> torch.Tensor:
    preds = []
    for batch in loader:
        preds.append(model(batch.to(device)).detach().float().cpu().reshape(-1))
    return torch.cat(preds) if preds else torch.empty(0)


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = AIGRegressionLightningModule.load_from_checkpoint(
        args.checkpoint, map_location=device
    )
    if model.hparams.model_type != "exact":
        raise ValueError(
            f"{args.checkpoint} was trained with model_type="
            f"{model.hparams.model_type!r}; this verification only applies to "
            "the exact track."
        )
    # Same guard training uses.  Without it, pointing this at a shallower
    # reduct cache than the model was trained on produces a real nonzero
    # residual, which the closing message below would tell the operator to
    # treat as a pipeline bug.
    assert_exact_depth_supports_model(
        [
            d
            for d in (args.reduct_tier0_cache_dir, args.reduct_tier1_cache_dir)
            if d
        ],
        int(model.hparams.encoder_kwargs["num_layers"]),
    )

    model.eval().to(device)

    reduct_dm = _make_test_dataset(
        args, args.reduct_cache_dir, args.reduct_tier0_cache_dir,
        args.reduct_tier1_cache_dir,
    )
    full_dm = _make_test_dataset(
        args, args.full_cache_dir, args.full_tier0_cache_dir,
        args.full_tier1_cache_dir, transform=_fold_preserving_y,
    )

    # Pairing the two prediction vectors by position is only valid if both
    # datasets enumerate the same graphs in the same order.  Same seed, same
    # ratios and same CSVs should guarantee it; check rather than assume,
    # because a mismatch would show up as a plausible-looking small residual.
    reduct_paths = [s.graph_path for s in reduct_dm.test_ds.samples]
    full_paths = [s.graph_path for s in full_dm.test_ds.samples]
    if reduct_paths != full_paths:
        raise ValueError(
            f"Test splits differ: {len(reduct_paths)} reduct samples vs "
            f"{len(full_paths)} full samples, or a different order. The two "
            "caches must be built from the same CSVs, seed and split ratios."
        )

    reduct_nodes = sum(reduct_dm.test_ds.get_num_nodes_list())
    full_nodes = sum(full_dm.test_ds.get_num_nodes_list())

    reduct_preds = _predict(model, reduct_dm.test_dataloader(), device)
    full_preds = _predict(model, full_dm.test_dataloader(), device)

    residual = (reduct_preds - full_preds).abs()
    scale = full_preds.abs().clamp_min(1e-12)
    report = {
        "checkpoint": str(args.checkpoint),
        "algorithm": args.algorithm,
        "graphs": int(reduct_preds.numel()),
        "node_retention": (reduct_nodes / full_nodes) if full_nodes else 0.0,
        "max_abs_residual": float(residual.max()) if residual.numel() else 0.0,
        "mean_abs_residual": float(residual.mean()) if residual.numel() else 0.0,
        "max_rel_residual": (
            float((residual / scale).max()) if residual.numel() else 0.0
        ),
        "mean_pred_reduct": float(reduct_preds.mean()) if residual.numel() else 0.0,
        "mean_pred_full": float(full_preds.mean()) if residual.numel() else 0.0,
    }

    print("\n=== RQ5 (exact track): reduct vs full-graph inference ===")
    for key, value in report.items():
        print(f"  {key:20s} {value}")
    print(
        "\nThese are float32 rounding residuals, not an approximation error: "
        "the exact quotient makes the two graph embeddings equal, so a "
        "residual above ~1e-5 relative means something in the pipeline is "
        "not exact and should be investigated, not reported."
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--csv_paths", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--reduct_cache_dir", type=str, required=True)
    parser.add_argument("--reduct_tier0_cache_dir", type=str, default=None)
    parser.add_argument("--reduct_tier1_cache_dir", type=str, default=None)

    parser.add_argument("--full_cache_dir", type=str, required=True)
    parser.add_argument("--full_tier0_cache_dir", type=str, default=None)
    parser.add_argument("--full_tier1_cache_dir", type=str, default=None)

    parser.add_argument("--hp_tuning_splits_path", type=str, default=None)
    parser.add_argument(
        "--pe_type",
        type=str,
        default=config.PE_TYPE,
        help=(
            "The pe setting the CACHES were built with, not the model's. It "
            "goes into the cache filename hash, so it must match the training "
            "run that produced the source cache — the exact model itself "
            "never reads a positional encoding."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None)
    parser.add_argument("--out", type=str, default=None)

    parsed = parser.parse_args()
    if parsed.algorithm not in config.VALID_ALGORITHMS:
        parser.error(
            f"Algorithm '{parsed.algorithm}' must be one of {config.VALID_ALGORITHMS}"
        )
    main(parsed)
