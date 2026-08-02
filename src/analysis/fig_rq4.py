"""RQ4 figures: did AIG-specific knowledge buy anything?

RQ2 and RQ3 rank every method on efficiency and accuracy. This module asks the
narrower, paired question, and the pairing is stated before the gap is
reported -- an unstated compression mismatch is what turns "method A is better"
into "method A reduced less".

Three kinds of pairing exist, and only the first two have data:

1. matched by construction -- the partitioners, which keep every node and
   differ only in which edges they cut, at the same $k$;
2. matched by calibration -- PageRank at keep-ratio 0.8 against the
   parameter-free AND-gate-only, which lands at 0.821 node retention;
3. matched against a random control -- summarization, which has never been run.

The spanning-forest / random-edge-dropout pair is deliberately drawn as
*unmatched*: the configured drop rate of 0.3 leaves 69.7% of edges against
spanning forest's 58.1%, so the arms are not comparable and the figure says so.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from analysis.style import (
    FAKE_COLOR,
    FAMILY_COLORS,
    INK_MUTED,
    INK_SECONDARY,
    WIDE,
    color_for,
    label_for,
    savefig,
    style_axes,
)

# (generic arm, domain-informed arm, pairing kind, tolerance note)
PAIRINGS = [
    ("metis", "span_weighted_metis", "by construction",
     "same $k$, all nodes kept; only the cut objective differs"),
    ("random", "level_slicing", "by construction",
     "same $k$, all nodes kept; only the assignment rule differs"),
    ("pagerank", "and_gate_only", "by calibration",
     "0.800 vs 0.821 node retention"),
    ("random_edge_dropout", "spanning_forest", "NOT MATCHED",
     "0.697 vs 0.581 edge retention; raise the drop rate to 0.419 to match"),
]

METRICS = [("r2", "$R^2$"), ("rmse", "RMSE"), ("spearman", "Spearman $\\rho$")]


def build_pairings(matched: pd.DataFrame, offline: pd.DataFrame) -> pd.DataFrame:
    """One row per pairing, with both arms' metrics and achieved compression."""
    by_method = matched.set_index("reduction_method")
    off = offline.set_index("reduction_method")
    rows = []
    for generic, domain, kind, note in PAIRINGS:
        if generic not in by_method.index or domain not in by_method.index:
            continue
        row = {"generic": generic, "domain": domain, "kind": kind, "note": note}
        for arm, key in [("generic", generic), ("domain", domain)]:
            for col, _ in METRICS:
                row[f"{arm}_{col}"] = by_method.loc[key, col]
            for col in ["node_retention", "edge_retention"]:
                row[f"{arm}_{col}"] = off.loc[key, col] if key in off.index else np.nan
        for col, _ in METRICS:
            row[f"delta_{col}"] = row[f"domain_{col}"] - row[f"generic_{col}"]
        # Which axis a pairing is matched ON, and therefore the only axis its
        # mismatch may be judged on. Both partitioner pairs keep every node at
        # the same k, so they are matched on nodes and their edge cut is an
        # OUTCOME of the comparison, not a confound; PageRank was calibrated to
        # AND-gate-only's node retention. Only the edge-mask sparsification pair
        # is matched on edges — and it is the one that is not matched.
        row["matched_on"] = "edge" if generic == "random_edge_dropout" else "node"
        row["retention_mismatch"] = abs(
            row[f"domain_{row['matched_on']}_retention"]
            - row[f"generic_{row['matched_on']}_retention"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_gaps(pairings: pd.DataFrame, out: Path) -> None:
    """Domain-informed minus generic, within each pairing.

    Read the bars against the mismatch column, not on their own: the fourth
    pairing is not matched, so its gap conflates the heuristic with the fact
    that the two arms removed different amounts.
    """
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.9), sharey=True)
    y = np.arange(len(pairings))

    for ax, (col, name) in zip(axes, METRICS):
        values = pairings[f"delta_{col}"]
        # For RMSE lower is better, so a domain win is a negative delta.
        wins = values < 0 if col == "rmse" else values > 0
        colors = [
            FAMILY_COLORS["partition"] if w else FAKE_COLOR for w in wins
        ]
        for i, unmatched in enumerate(pairings["kind"] == "NOT MATCHED"):
            if unmatched:
                colors[i] = INK_MUTED
        bars = ax.barh(y, values, height=0.6, color=colors)
        for i, unmatched in enumerate(pairings["kind"] == "NOT MATCHED"):
            if unmatched:
                bars[i].set_hatch("xxx")
                bars[i].set_edgecolor("white")
        ax.axvline(0, color=INK_SECONDARY, lw=1.0)
        ax.set_title(f"$\\Delta$ {name}\n(domain $-$ generic)")
        style_axes(ax, grid_axis="x")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [
            f"{label_for(d)}\nvs. {label_for(g)}"
            for g, d in zip(pairings["generic"], pairings["domain"])
        ],
        fontsize=6.5,
    )
    axes[0].invert_yaxis()

    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(facecolor=FAMILY_COLORS["partition"], label="domain-informed wins"),
            Patch(facecolor=FAKE_COLOR, label="generic wins"),
            Patch(facecolor=INK_MUTED, hatch="xxx", edgecolor="white", label="arms not matched"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncols=3,
        fontsize=6.5,
    )
    fig.suptitle(
        "RQ4: the paired gap, at matched compression where matching was possible",
        y=1.05,
    )
    savefig(fig, out, "rq4_paired_gaps")


def pairing_compression(pairings: pd.DataFrame, out: Path) -> None:
    """How close each pairing actually is.

    Every arm's achieved node and edge retention, drawn as a paired dumbbell.
    A pair whose arms sit far apart on this axis cannot support a claim about
    the heuristic.
    """
    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    y = np.arange(len(pairings))

    for ax, axis in zip(axes, ["node", "edge"]):
        for yi, (_, row) in enumerate(pairings.iterrows()):
            g, d = row[f"generic_{axis}_retention"], row[f"domain_{axis}_retention"]
            # Only the axis a pairing was matched on can be "not matched".
            relevant = row["matched_on"] == axis
            unmatched = relevant and row["kind"] == "NOT MATCHED"
            ax.plot(
                [g, d], [yi, yi],
                color=FAKE_COLOR if unmatched else INK_MUTED,
                lw=2.2 if unmatched else 1.2,
                ls="--" if unmatched else "-",
            )
            ax.plot(g, yi, "o", ms=6, color=color_for(row["generic"]),
                    markerfacecolor="white", markeredgewidth=1.5)
            ax.plot(d, yi, "o", ms=6, color=color_for(row["domain"]))
            label = f"$\\Delta$ {abs(d - g):.3f}"
            ax.text(
                max(g, d) + 0.03, yi,
                label + ("  (matched here)" if relevant else ""),
                fontsize=6, va="center",
                color=FAKE_COLOR if unmatched else
                (INK_SECONDARY if relevant else INK_MUTED),
                fontweight="bold" if relevant else "normal",
            )
        ax.set_xlim(0.3, 1.55)
        ax.set_xlabel(f"{axis.capitalize()} retention (open = generic, filled = domain)")
        ax.set_title(
            f"{axis.capitalize()} retention"
            + ("\n(matching axis for the partitioners and PageRank pair)"
               if axis == "node" else
               "\n(matching axis for the edge-mask pair; an outcome elsewhere)")
        )
        style_axes(ax, grid_axis="x")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [f"{label_for(d, True)} / {label_for(g, True)}"
         for g, d in zip(pairings["generic"], pairings["domain"])],
        fontsize=6.5,
    )
    axes[0].invert_yaxis()
    fig.suptitle(
        "How well matched is each pairing? Judge each on the axis it was matched on.",
        y=1.04,
    )
    savefig(fig, out, "rq4_pairing_compression")


def domain_vs_generic_overview(matched: pd.DataFrame, out: Path) -> None:
    """The unpaired view: every domain-informed method against every generic
    one, on the same axes. Weaker evidence than the pairings, but it answers
    the question a reader asks first."""
    from analysis.style import meta

    df = matched[matched["reduction_method"] != "none"].copy()
    df["domain"] = df["reduction_method"].map(lambda m: meta(m)["domain"])

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
    for ax, (col, name) in zip(axes, METRICS):
        groups = [
            df.loc[~df["domain"], col].to_numpy(),
            df.loc[df["domain"], col].to_numpy(),
        ]
        for i, (values, label) in enumerate(zip(groups, ["Generic", "Domain-informed"])):
            ax.plot(
                np.full(len(values), i) + np.linspace(-0.08, 0.08, len(values)),
                values,
                "o", ms=6, color=FAMILY_COLORS["partition"] if i else INK_MUTED,
                markerfacecolor=FAMILY_COLORS["partition"] if i else "white",
                markeredgewidth=1.3,
            )
            ax.plot([i - 0.2, i + 0.2], [np.mean(values)] * 2, color=INK_SECONDARY, lw=1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Generic", "Domain-\ninformed"], fontsize=6.5)
        ax.set_xlim(-0.45, 1.45)
        ax.set_title(name)
        style_axes(ax)
    fig.suptitle(
        f"Unpaired comparison over all {len(df)} measured reductions "
        "(bar = group mean)",
        y=1.05,
    )
    savefig(fig, out, "rq4_domain_vs_generic")


def build(matched, offline, out) -> pd.DataFrame:
    pairings = build_pairings(matched, offline)
    paired_gaps(pairings, out)
    pairing_compression(pairings, out)
    domain_vs_generic_overview(matched, out)
    return pairings
