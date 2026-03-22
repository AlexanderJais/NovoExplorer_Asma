"""UpSet plot visualizations for comparing DEG overlaps across comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

from novoview.plotting.theme import (
    WONG_PALETTE,
    VOLCANO_COLORS,
    DIVERGING_CMAP,
    SEQUENTIAL_CMAP,
    apply_plotly_theme,
    apply_matplotlib_theme,
    get_plotly_template,
    format_axis_label,
)


def create_upset_data(
    deg_results: dict[str, pd.DataFrame],
    padj_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
) -> pd.DataFrame:
    """Convert DEG results to a binary membership matrix.

    Parameters
    ----------
    deg_results : dict[str, pd.DataFrame]
        Mapping of comparison name to DEG DataFrame.  Each DataFrame must
        contain columns ``gene_name``, ``padj``, and ``log2fc``.
    padj_threshold : float
        Adjusted p-value cutoff for calling a gene significant.
    log2fc_threshold : float
        Absolute log2 fold-change cutoff.

    Returns
    -------
    pd.DataFrame
        Binary matrix with genes as rows and comparisons as columns.
        A value of 1 indicates the gene is significant in that comparison.
    """
    sig_sets: dict[str, set[str]] = {}
    for name, df in deg_results.items():
        if "gene_name" not in df.columns:
            df = df.copy()
            df["gene_name"] = df.index.astype(str)
        mask = (df["padj"] < padj_threshold) & (df["log2fc"].abs() >= log2fc_threshold)
        sig_sets[name] = set(df.loc[mask, "gene_name"])

    all_genes = sorted(set().union(*sig_sets.values()))
    binary = pd.DataFrame(0, index=all_genes, columns=list(deg_results.keys()))
    for name, genes in sig_sets.items():
        binary.loc[binary.index.isin(genes), name] = 1

    return binary


def create_upset_plot(
    binary_matrix: pd.DataFrame,
    title: str = "",
):
    """Build an UpSet-style plot using matplotlib.

    The plot consists of three aligned panels:

    * **Left** -- horizontal bars showing the size of each individual set.
    * **Bottom** -- a dot-matrix indicating which sets participate in each
      intersection.
    * **Top** -- vertical bars showing the cardinality of each intersection.

    Intersections are sorted by descending cardinality.

    Parameters
    ----------
    binary_matrix : pd.DataFrame
        Binary membership matrix (genes x comparisons) as returned by
        :func:`create_upset_data`.
    title : str
        Plot title.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and the main (intersection bar) axes.
    """
    apply_matplotlib_theme()

    sets = binary_matrix.columns.tolist()
    n_sets = len(sets)

    # Compute intersections --------------------------------------------------
    # Each unique row pattern defines an intersection.
    pattern_strings = binary_matrix.apply(lambda row: tuple(row.values), axis=1)
    intersection_counts = pattern_strings.value_counts()
    # Keep only non-empty intersections
    intersection_counts = intersection_counts[
        intersection_counts.index.map(lambda t: sum(t) > 0)
    ]
    # Sort by cardinality (descending)
    intersection_counts = intersection_counts.sort_values(ascending=False)

    n_intersections = len(intersection_counts)

    # Set sizes (horizontal bars on the left)
    set_sizes = binary_matrix.sum(axis=0)

    # Layout -----------------------------------------------------------------
    # We use a gridspec with:
    #   top row:  intersection bars  (spanning right columns)
    #   bottom row: dot matrix        (spanning right columns)
    #   left column: set size bars

    fig = plt.figure(figsize=(max(6, n_intersections * 0.45 + 2), max(4, n_sets * 0.5 + 2)))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1, max(2.5, n_intersections * 0.35)],
        height_ratios=[2, 1],
        hspace=0.05,
        wspace=0.05,
    )

    ax_setsize = fig.add_subplot(gs[1, 0])   # bottom-left: set sizes
    ax_inter = fig.add_subplot(gs[0, 1])      # top-right: intersection bars
    ax_matrix = fig.add_subplot(gs[1, 1])     # bottom-right: dot matrix
    ax_empty = fig.add_subplot(gs[0, 0])      # top-left: blank
    ax_empty.axis("off")

    bar_color = WONG_PALETTE[0]
    dot_active_color = "#333333"
    dot_inactive_color = "#DDDDDD"

    # -- Intersection bars (top) ---------------------------------------------
    x_positions = np.arange(n_intersections)
    ax_inter.bar(
        x_positions,
        intersection_counts.values,
        color=bar_color,
        width=0.6,
        edgecolor="none",
    )
    ax_inter.set_xlim(-0.5, n_intersections - 0.5)
    ax_inter.set_ylabel("Intersection Size")
    ax_inter.set_xticks([])
    ax_inter.set_xticklabels([])
    if title:
        ax_inter.set_title(title, fontweight="semibold")

    # Annotate bar values
    for i, val in enumerate(intersection_counts.values):
        ax_inter.text(i, val, str(val), ha="center", va="bottom", fontsize=6)

    # -- Dot matrix (bottom-right) -------------------------------------------
    for i, pattern in enumerate(intersection_counts.index):
        for j in range(n_sets):
            color = dot_active_color if pattern[j] == 1 else dot_inactive_color
            ax_matrix.scatter(i, j, color=color, s=40, zorder=3)
        # Connect active dots with a vertical line
        active = [j for j in range(n_sets) if pattern[j] == 1]
        if len(active) > 1:
            ax_matrix.vlines(
                i, min(active), max(active),
                colors=dot_active_color, linewidth=1.5, zorder=2,
            )

    ax_matrix.set_xlim(-0.5, n_intersections - 0.5)
    ax_matrix.set_ylim(-0.5, n_sets - 0.5)
    ax_matrix.set_yticks(range(n_sets))
    ax_matrix.set_yticklabels(sets, fontsize=7)
    ax_matrix.set_xticks([])
    ax_matrix.invert_yaxis()
    ax_matrix.set_frame_on(False)
    ax_matrix.tick_params(left=False, bottom=False)

    # -- Set size bars (bottom-left) -----------------------------------------
    y_positions = np.arange(n_sets)
    ax_setsize.barh(
        y_positions,
        set_sizes.values,
        color=bar_color,
        height=0.6,
        edgecolor="none",
    )
    ax_setsize.set_ylim(-0.5, n_sets - 0.5)
    ax_setsize.set_yticks(range(n_sets))
    ax_setsize.set_yticklabels(sets, fontsize=7)
    ax_setsize.invert_xaxis()
    ax_setsize.invert_yaxis()
    ax_setsize.set_xlabel("Set Size", fontsize=7)

    fig.align_ylabels([ax_setsize, ax_matrix])
    fig.tight_layout()

    return fig, ax_inter
