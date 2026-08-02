"""Every fabricated number in the figure set, in one file.

Nothing here is measured. Each table is a placeholder that exists so the
Results chapter has a figure slot to argue against while the run that would
fill it is still outstanding, and every one of them carries the command that
would replace it.

Invariants, so that a placeholder can never be mistaken for a result:

* every frame has a ``measured`` column, and it is False on every fabricated
  row;
* the numbers are deliberately round and implausibly regular (0.900, 0.850,
  0.800 ...) rather than realistic-looking;
* the plotting side reads ``measured`` and hatches, reddens and watermarks
  accordingly (:func:`analysis.style.mark_fake`).

Delete a block from this file as soon as the corresponding run lands.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# RQ1 — baseline models, tiers 2 and 3
# ---------------------------------------------------------------------------
TODO_BASELINE_MODELS = (
    "Train GCN / GraphSAGE / GIN under the headline pooling, head and budget "
    "(src/shell/train.sh with --encoder_name), and finish the DeepGate4 and "
    "HOGA ports on the baselines/openabc-synthnet-hoga branch. SynthNet is the "
    "only published baseline that has produced a number "
    "(train_baseline_synthnet_Orchestrate, val R2 = -0.168); the HOGA run "
    "crashed at epoch 0."
)

BASELINE_MODELS = pd.DataFrame(
    [
        # tier 2 — standard encoders, identical pooling/head/training
        ("GCN (mean pool)", "2: standard encoders", 0.050, 0.030, 0.100, 0.150, False),
        ("GraphSAGE", "2: standard encoders", 0.048, 0.029, 0.150, 0.200, False),
        ("GIN", "2: standard encoders", 0.046, 0.028, 0.200, 0.250, False),
        # tier 3 — published circuit models
        ("SynthNet", "3: published models", 0.021, 0.012, -0.168, 0.000, False),
        ("HOGA", "3: published models", 0.050, 0.030, 0.100, 0.100, False),
        ("DeepGate4", "3: published models", 0.045, 0.027, 0.250, 0.300, False),
    ],
    columns=["model", "tier", "rmse", "mae", "r2", "spearman", "measured"],
)
# SynthNet's val R2 is real; its RMSE/MAE/Spearman on the test split are not,
# so the whole row stays flagged.


# ---------------------------------------------------------------------------
# RQ1a — protocol sensitivity (split strategy)
# ---------------------------------------------------------------------------
TODO_SPLIT_PROTOCOL = (
    "IN FLIGHT: the --split_by random and --split_by recipe training runs are "
    "on the cluster now (the design-disjoint run already exists). Identical "
    "encoder, budget and seed; only the split changes. When they land, run "
    "src/test.py for each, delete this block and rerun analysis.make_all."
)

SPLIT_PROTOCOL = pd.DataFrame(
    [
        ("Random (per-row)", 0.020, 0.900, 0.900, False),
        ("Recipe-disjoint", 0.030, 0.700, 0.750, False),
        ("Design-disjoint (ours)", 0.042921, 0.342838, 0.185113, True),
    ],
    columns=["protocol", "rmse", "r2", "spearman", "measured"],
)


# ---------------------------------------------------------------------------
# RQ2/RQ3/RQ4 — the summarization family (no method has been run)
# ---------------------------------------------------------------------------
TODO_SUMMARIZATION = (
    "No summarization configuration has been trained or measured. Needs: "
    "sbatch src/shell/precompute_summarization.sh for the offline pass, then "
    "one training run and one src/test.py pass per method. Random within-type "
    "merging is not implemented at all."
)

SUMMARIZATION = pd.DataFrame(
    [
        # method, node_ret, edge_ret, offline_s, rmse, r2, spearman,
        # vram_mb, step_time_s, cross_state_r2
        ("random_merge", 0.700, 0.700, 0.500, 0.060, 0.100, 0.100, 800.0, 0.030, 0.000),
        ("convmatch", 0.500, 0.600, 20.000, 0.050, 0.200, 0.200, 700.0, 0.028, 0.100),
        ("cone", 0.600, 0.650, 2.000, 0.045, 0.300, 0.300, 750.0, 0.026, 0.200),
        ("mffc", 0.400, 0.500, 3.000, 0.040, 0.350, 0.350, 600.0, 0.024, 0.250),
        ("wl", 0.800, 0.850, 5.000, 0.042921, 0.342838, 0.185113, 900.0, 0.032, 0.342838),
    ],
    columns=[
        "reduction_method",
        "node_retention",
        "edge_retention",
        "offline_s",
        "rmse",
        "r2",
        "spearman",
        "peak_vram_mb",
        "step_time_s",
        "cross_state_r2",
    ],
).assign(measured=False, reduction_type="summarization")
# The wl row deliberately repeats the full-graph baseline's numbers: colour
# refinement at count-cap infinity is lossless for this encoder, so that is the
# value the positive control MUST reproduce. It is a prediction, not a result.


# ---------------------------------------------------------------------------
# RQ2/RQ3 — colour-refinement depth probe (residual redundancy after strash)
# ---------------------------------------------------------------------------
TODO_WL_DEPTH = (
    "Run the residual-redundancy probe at refinement depth d = 1..4 "
    "(count_cap = None, direction = backward). Depth 1 should find almost "
    "nothing precisely because every corpus graph is strashed; that is a "
    "reportable result either way."
)

WL_DEPTH = pd.DataFrame(
    [
        (1, 0.990, 1.0, False),
        (2, 0.900, 1.0, False),
        (3, 0.850, 1.0, False),
        (4, 0.800, 1.0, False),
    ],
    columns=["depth", "node_retention", "accuracy_retained", "measured"],
)


# ---------------------------------------------------------------------------
# RQ3 — H1, effective receptive field
# ---------------------------------------------------------------------------
TODO_RECEPTIVE_FIELD = (
    "The receptive-field metric (mean k-hop fanin-cone size, k = 4 encoder "
    "layers, measured before and after reduction) is specified in the "
    "methodology and NOT implemented. Until it exists, H1 (path contraction) "
    "is asserted rather than tested and must not be reported as evidenced."
)

RECEPTIVE_FIELD = pd.DataFrame(
    [
        ("none", 100.0, 100.0, True),
        ("random_edge_dropout", 100.0, 70.0, False),
        ("spanning_forest", 100.0, 50.0, False),
        ("pagerank", 100.0, 60.0, False),
        ("and_gate_only", 100.0, 65.0, False),
        ("metis", 100.0, 40.0, False),
        ("span_weighted_metis", 100.0, 45.0, False),
        ("mffc", 100.0, 250.0, False),
        ("cone", 100.0, 200.0, False),
        ("wl", 100.0, 150.0, False),
    ],
    columns=["reduction_method", "cone_before", "cone_after", "measured"],
)
# Only the "none" row is trivially true (a graph is its own unreduced self).


# ---------------------------------------------------------------------------
# RQ4 — seed variance
# ---------------------------------------------------------------------------
TODO_SEED_VARIANCE = (
    "Every configuration is trained exactly once, so no RQ4 gap can be "
    "distinguished from run-to-run noise. Cheapest sufficient fix: 3 seeds on "
    "the four RQ4 pairings only (8 configurations x 3 seeds), not on the whole "
    "sweep."
)


# ---------------------------------------------------------------------------
# RQ5 — CPU inference
# ---------------------------------------------------------------------------
TODO_CPU_INFERENCE = (
    "No CPU evaluation pass survives in the exported results: every "
    "inference_results CSV is device=cuda. Re-run src/shell/test_cpu.sh to get "
    "the 'train cheap, infer full on modest hardware' half of RQ5."
)

CPU_INFERENCE = pd.DataFrame(
    [
        ("none", "cuda", 74.65, 48159.9, True),
        ("none", "cpu", 5.00, 0.0, False),
        ("spanning_forest", "cpu", 8.00, 0.0, False),
        ("pagerank", "cpu", 9.00, 0.0, False),
        ("and_gate_only", "cpu", 10.00, 0.0, False),
        ("metis", "cpu", 7.00, 0.0, False),
    ],
    columns=["reduction_method", "device", "throughput_graphs_per_s", "peak_vram_mb", "measured"],
)


# ---------------------------------------------------------------------------
# Corpus statistics not derivable from the exported results
# ---------------------------------------------------------------------------
TODO_CORPUS_STATS = (
    "The exported predictions cover the evaluation splits only. Whole-corpus "
    "statistics (3.9M AIGs across tiers 0/1/2, per-tier label and size "
    "distributions, AND-gate fraction, depth distribution) need a stats pass "
    "over the graph cache."
)

CORPUS_STATS = pd.DataFrame(
    [
        ("Tier 0 (base graphs)", 231055, 40000, 0.100, False),
        ("Tier 1 (single-algorithm)", 924220, 35000, 0.050, False),
    ],
    columns=["tier", "graphs", "median_nodes", "mean_optimizability", "measured"],
)
