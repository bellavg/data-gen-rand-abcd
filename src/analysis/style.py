"""Shared figure style, palette and method metadata for the thesis figures.

Everything the Results chapter draws goes through here so that colour, method
naming and ordering are decided in exactly one place. Two rules drive the
design:

1. Colour encodes the reduction *family*, never the individual method. There
   are nine measured configurations and more to come; a nine-hue categorical
   palette is unreadable and colourblind-unsafe. Family (four values) is the
   organising concept anyway, and individual methods are identified by their
   axis position and direct labels.
2. Domain-informed methods are hatched rather than given their own colour, so
   "generic vs domain-informed" reads as an attribute of a bar rather than as a
   second, competing colour axis.

Hues are the validated categorical palette from the dataviz reference (light
mode, fixed slot order).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — figures are written, never shown

import matplotlib.pyplot as plt

# --- Palette -----------------------------------------------------------------
# Categorical slots 1/2/3/8 of the validated reference palette, plus its
# secondary-ink grey for the baseline (a reference level, not a series).
FAMILY_COLORS = {
    "baseline": "#52514e",
    "partition": "#2a78d6",
    "sparsification": "#eb6834",
    "summarization": "#1baf7a",
}
FAKE_COLOR = "#e34948"  # slot 8, reserved for fabricated placeholder data

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#d8d7d2"

DOMAIN_HATCH = "///"  # domain-informed methods
FAKE_HATCH = "xxx"

# --- Dataset-chapter accents --------------------------------------------------
# fig_dataset.py describes the corpus itself (tiers, source scripts, splits),
# axes the family scheme above has no opinion on. Named here anyway, so no hex
# literal in fig_dataset.py stands for a colour nothing else can look up.
TIER_COLORS = {"tier0": "#2a78d6", "tier1": "#eb6834"}
SOURCE_COLORS = {
    "tier0 base": "#52514e",
    "Syn4": "#eb6834",
    "Deepsyn": "#2a78d6",
    "C2RS": "#1baf7a",
}
SPLIT_COLORS = {"val": "#2a78d6", "test": "#eb6834"}
CORPUS_SAMPLE_COLOR = "#eb6834"
DATASET_FLAG_COLOR = "#e34948"  # reference-line / flagged-value accent; shares
# FAKE_COLOR's hue by coincidence of the validated palette, not by meaning --
# nothing marked with this in fig_dataset.py is fabricated.

# --- Figure geometry ---------------------------------------------------------
# msc_thesis.tex is \twocolumn. A column is ~3.35in; the full text block is
# ~6.9in. Figures are drawn at their final printed size so that fonts land at
# the intended point size without \includegraphics rescaling them.
COL = (3.35, 2.5)
COL_TALL = (3.35, 3.4)
WIDE = (6.9, 3.0)
WIDE_TALL = (6.9, 4.4)
WIDE_XTALL = (6.9, 6.2)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.titlepad": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
            "hatch.linewidth": 0.6,
            "figure.titlesize": 10,
            "figure.titleweight": "bold",
            "figure.constrained_layout.use": True,
        }
    )


# --- Method registry ---------------------------------------------------------
# ``measured`` is False for every configuration that has not been run. Those
# rows are drawn from fabricated numbers and must carry a TODO in the figure,
# the caption and the table.
METHODS: dict[str, dict] = {
    "none": {
        "label": "Full graph",
        "short": "full",
        "family": "baseline",
        "domain": False,
        "measured": True,
    },
    # --- partitioning ---
    "random": {
        "label": "Random hashing",
        "short": "rand-hash",
        "family": "partition",
        "domain": False,
        "measured": True,
    },
    "metis": {
        "label": "METIS",
        "short": "metis",
        "family": "partition",
        "domain": False,
        "measured": True,
    },
    "level_slicing": {
        "label": "Level slicing",
        "short": "level-slice",
        "family": "partition",
        "domain": True,
        "measured": True,
    },
    "span_weighted_metis": {
        "label": "Span-weighted METIS",
        "short": "span-metis",
        "family": "partition",
        "domain": True,
        "measured": True,
    },
    # --- sparsification ---
    "random_edge_dropout": {
        "label": "Random edge dropout",
        "short": "edge-drop",
        "family": "sparsification",
        "domain": False,
        "measured": True,
    },
    "spanning_forest": {
        "label": "Spanning forest",
        "short": "span-forest",
        "family": "sparsification",
        "domain": False,
        "measured": True,
    },
    "pagerank": {
        "label": "PageRank pruning",
        "short": "pagerank",
        "family": "sparsification",
        "domain": False,
        "measured": True,
    },
    "and_gate_only": {
        "label": "AND-gate only",
        "short": "and-only",
        "family": "sparsification",
        "domain": True,
        "measured": True,
    },
    # --- summarization: implemented or specified, NEVER RUN ---
    "random_merge": {
        "label": "Random within-type",
        "short": "rand-merge",
        "family": "summarization",
        "domain": False,
        "measured": False,
    },
    "convmatch": {
        "label": "ConvMatch",
        "short": "convmatch",
        "family": "summarization",
        "domain": False,
        "measured": False,
    },
    "wl": {
        "label": "Colour refinement (exact)",
        "short": "wl-exact",
        "family": "summarization",
        "domain": True,
        "measured": False,
    },
    "mffc": {
        "label": "MFFC coarsening",
        "short": "mffc",
        "family": "summarization",
        "domain": True,
        "measured": False,
    },
    "cone": {
        "label": "Cone coarsening",
        "short": "cone",
        "family": "summarization",
        "domain": True,
        "measured": False,
    },
}

# Display order: baseline, then each family generic-first.
METHOD_ORDER = [
    "none",
    "random",
    "metis",
    "level_slicing",
    "span_weighted_metis",
    "random_edge_dropout",
    "spanning_forest",
    "pagerank",
    "and_gate_only",
    "random_merge",
    "convmatch",
    "cone",
    "mffc",
    "wl",
]

MEASURED_ORDER = [m for m in METHOD_ORDER if METHODS[m]["measured"]]

FAMILY_MARKERS = {
    "baseline": "*",
    "partition": "o",
    "sparsification": "s",
    "summarization": "^",
}


def meta(method) -> dict:
    """Method metadata, NaN-safe: the baseline config writes an empty
    ``reduction_method`` cell, which pandas reads back as NaN.

    An unregistered method falls back to ``measured=False``, i.e. it is drawn
    and tagged as fabricated. This has to fail closed: the whole
    fabricated-data marking runs through this function, so a typo in a method
    name or a new entry in ``fake_data`` that was never added to ``METHODS``
    would otherwise render as an ordinary measured result.
    """
    key = method if isinstance(method, str) and method else "none"
    return METHODS.get(key, {
        "label": str(key),
        "short": str(key),
        "family": "summarization",
        "domain": False,
        "measured": False,
    })


def color_for(method) -> str:
    return FAMILY_COLORS[meta(method)["family"]]


def hatch_for(method) -> str | None:
    m = meta(method)
    if not m["measured"]:
        return FAKE_HATCH
    return DOMAIN_HATCH if m["domain"] else None


def label_for(method, short: bool = False) -> str:
    m = meta(method)
    return m["short"] if short else m["label"]


def sort_key(method) -> int:
    key = method if isinstance(method, str) and method else "none"
    return METHOD_ORDER.index(key) if key in METHOD_ORDER else len(METHOD_ORDER)


# --- Legends -----------------------------------------------------------------
def family_legend(
    ax,
    families: list[str],
    *,
    include_domain: bool = True,
    include_fake: bool = False,
    **kw,
) -> None:
    """Colour-by-family legend, always drawn when more than one family is on the
    axes: identity must never rest on colour alone."""
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=FAMILY_COLORS[f], edgecolor="none", label=f.capitalize())
        for f in families
    ]
    if include_domain:
        handles.append(
            Patch(
                facecolor="white",
                edgecolor=INK_SECONDARY,
                hatch=DOMAIN_HATCH,
                label="Domain-informed",
            )
        )
    if include_fake:
        handles.append(
            Patch(
                facecolor=FAKE_COLOR,
                edgecolor="white",
                hatch=FAKE_HATCH,
                label="Fabricated (never run)",
            )
        )
    ax.legend(handles=handles, **kw)


def style_axes(ax, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def row_height(n_rows: int, *, per_row: float = 0.24, base: float = 1.3) -> float:
    """Figure height for a horizontal-bar chart with ``n_rows`` categories.

    A fixed height silently squashes the bars once the method list grows, which
    is exactly what happens when the summarization rows are appended.
    """
    return max(2.5, base + per_row * n_rows)


def clip_bar_axis(ax, values, floor: float) -> list[tuple[int, float]]:
    """Clip a horizontal bar axis at ``floor`` and return the off-scale bars.

    METIS reaches $R^2 = -3$ on the matched-state pass and one design reaches
    $-748$. Drawn to scale they compress every other bar into a single pixel
    column, so the axis is clipped and each clipped bar is labelled with its
    real value.

    Both limits are set, not just the left one: setting only ``left`` leaves
    matplotlib to autoscale the right edge from the *unclipped* data range, so
    a bar at $-748$ pushes the right edge out to $+37$ and the clipping buys
    nothing.

    A bar truncated at the floor and a bar that genuinely ends there are
    otherwise indistinguishable, which reads as the axis itself stopping at
    the floor rather than continuing past it. Every off-scale bar therefore
    gets a break marker: a left-pointing triangle sitting on the boundary,
    the same "value continues past this point" glyph used for clipped points
    elsewhere in the RQ3 scatter figures. Callers still label the real value
    themselves, next to this marker.
    """
    on_scale = [v for v in values if v == v and v >= floor]
    top = max(on_scale) if on_scale else floor + 1.0
    ax.set_xlim(floor, top + 0.12 * max(abs(top - floor), 1e-9))
    off_scale = [(i, v) for i, v in enumerate(values) if v == v and v < floor]
    if off_scale:
        ax.plot(
            [floor] * len(off_scale),
            [i for i, _ in off_scale],
            marker="<",
            ms=6,
            linestyle="none",
            color="white",
            markeredgecolor=INK_SECONDARY,
            markeredgewidth=0.9,
            clip_on=False,
            zorder=6,
        )
    return off_scale


# --- Fabricated-data marking -------------------------------------------------
FAKE_BANNER = "FABRICATED PLACEHOLDER: NOT MEASURED"


def mark_fake(fig, ax=None, note: str = "", *, watermark: bool = True) -> None:
    """Make a placeholder figure impossible to mistake for a result.

    Three independent signals, because any one of them can be lost when a
    figure is cropped, printed in greyscale, or viewed as a thumbnail: a red
    frame around the whole figure, a diagonal watermark across the axes, and a
    footer naming what has to be run.

    Pass ``watermark=False`` when only *some* rows are fabricated. Those rows
    are already red and cross-hatched individually, and a watermark across the
    whole axes would wrongly condemn the measured rows beside them.
    """
    fig.patch.set_edgecolor(FAKE_COLOR)
    fig.patch.set_linewidth(3)

    axes = [ax] if ax is not None else fig.get_axes()
    for a in axes:
        if watermark:
            a.text(
                0.5,
                0.5,
                "TODO\nFAKE DATA",
                transform=a.transAxes,
                fontsize=20,
                fontweight="bold",
                color=FAKE_COLOR,
                alpha=0.28,
                ha="center",
                va="center",
                rotation=24,
                zorder=10,
            )
        for spine in a.spines.values():
            spine.set_edgecolor(FAKE_COLOR)
            spine.set_linestyle((0, (4, 2)))

    # Wrapped, because an unwrapped note becomes one very long line and
    # bbox_inches="tight" then stretches the saved figure to fit it.
    width = max(60, int(fig.get_figwidth() * 17))
    lines = [FAKE_BANNER]
    if note:
        lines += textwrap.wrap(note, width=width)
    fig.text(
        0.5,
        -0.015,
        "\n".join(lines),
        ha="center",
        va="top",
        fontsize=6.5,
        fontweight="bold",
        color=FAKE_COLOR,
    )


def savefig(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"[figures] {path}")
    return path
