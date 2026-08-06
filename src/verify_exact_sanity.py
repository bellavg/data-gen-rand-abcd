#!/usr/bin/env python
"""Sanity check: does coarsening preserve the GNN's graph-level output?

No training needed. The property under test — that a colour-refinement
quotient produces the same graph embedding as the original graph — is a
statement about the *architecture* (message passing + pooling +
normalization), not about learned weights: the exact track's forward pass
on a coarsened graph is the same arithmetic expression as on the original,
just re-associated (see models/layers/gcn_exact.py and
models/base_model_exact.py for why it is built that way), so a single seed
of frozen, random weights is enough to demonstrate it. This script checks
one seed on one synthetic fixture; ``TestExactnessThroughRealModel`` in
``unittests/models/test_gcn_exact.py`` covers multiple seeds, jk_modes and
a real AIG.

Reports both tracks on the same fixture graph, side by side:
  - exact    ExactGraphBaseModel   (mandatory edge_weight, size-weighted
             pooling, no normalization) -> residual should be ~0.
  - general  UnifiedGraphBaseModel (edge_attr fused pre-activation, plain
             mean pooling, GraphNorm on node features) -> residual should
             NOT be ~0; this quantifies why the exact track exists.

  python -m verify_exact_sanity [--out report.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from data.exact_graph import apply_exact_merge_map, fold_inversions_into_x
from data.summarization import apply_merge_map, color_refinement
from models.base_model import UnifiedGraphBaseModel
from models.base_model_exact import ExactGraphBaseModel


def _symmetric_graph() -> Data:
    """Two structurally identical PI-pair->AND cones feeding one PO."""
    x = torch.zeros(8, 4, dtype=torch.float32)
    x[0, 0] = 1.0
    x[[1, 2, 4, 5], 1] = 1.0
    x[[3, 6], 2] = 1.0
    x[7, 3] = 1.0
    edge_index = torch.tensor(
        [[1, 2, 4, 5, 3, 6], [3, 3, 6, 6, 7, 7]], dtype=torch.long
    )
    edge_attr = torch.tensor([[1, 0]] * 6, dtype=torch.float32)
    level = torch.tensor([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [1.0], [2.0]])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, level=level)
    data.num_nodes = 8
    return data


def _run(model, data: Data) -> torch.Tensor:
    batch = Batch.from_data_list([data])
    with torch.no_grad():
        return model.forward_batch(batch)


def _exact_report(seed: int) -> dict:
    torch.manual_seed(seed)
    model = ExactGraphBaseModel(
        hidden_dim=16, num_layers=2, node_input_dim=5, jk_mode="cat"
    ).eval()

    folded = fold_inversions_into_x(_symmetric_graph())
    cluster = color_refinement(folded, depth=2, pe_aware=False)
    coarse = apply_exact_merge_map(folded, cluster, int(cluster.max()) + 1)

    full_out = _run(model, folded)
    coarse_out = _run(model, coarse)
    residual = (full_out - coarse_out).abs()
    return {
        "track": "exact",
        "nodes_full": int(folded.x.size(0)),
        "nodes_coarse": int(coarse.x.size(0)),
        "output_full": full_out.reshape(-1).tolist(),
        "output_coarse": coarse_out.reshape(-1).tolist(),
        "max_abs_residual": float(residual.max()),
    }


def _general_report(seed: int) -> dict:
    torch.manual_seed(seed)
    model = UnifiedGraphBaseModel(
        encoder_name="gcn",
        hidden_dim=16,
        encoder_kwargs={"num_layers": 2, "hid_dim": 16},
        pe_type="none",
        task_out_dim=1,
        pooling_type="mean",
    ).eval()

    full = _symmetric_graph()
    cluster = color_refinement(full, depth=2, count_cap=None)
    coarse = apply_merge_map(full, cluster, int(cluster.max()) + 1)

    full_out = _run(model, full)
    coarse_out = _run(model, coarse)
    residual = (full_out - coarse_out).abs()
    return {
        "track": "general",
        "nodes_full": int(full.x.size(0)),
        "nodes_coarse": int(coarse.x.size(0)),
        "output_full": full_out.reshape(-1).tolist(),
        "output_coarse": coarse_out.reshape(-1).tolist(),
        "max_abs_residual": float(residual.max()),
    }


def main(args: argparse.Namespace) -> None:
    reports = [_exact_report(args.seed), _general_report(args.seed)]

    print(
        "\n=== Coarsening-invariance sanity check "
        "(frozen random weights, no training) ==="
    )
    for report in reports:
        print(
            f"\n[{report['track']}] nodes "
            f"{report['nodes_full']} -> {report['nodes_coarse']}"
        )
        print(f"  output (full)   {report['output_full']}")
        print(f"  output (coarse) {report['output_coarse']}")
        print(f"  max |residual|  {report['max_abs_residual']:.3e}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    main(parser.parse_args())
