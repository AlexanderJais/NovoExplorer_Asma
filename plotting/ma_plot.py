"""MA plot visualizations for differential expression results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from plotting.theme import (
    WONG_PALETTE,
    VOLCANO_COLORS,
    DIVERGING_CMAP,
    SEQUENTIAL_CMAP,
    apply_plotly_theme,
    apply_matplotlib_theme,
    get_plotly_template,
    format_axis_label,
)


def _classify_genes(
    deg_df: pd.DataFrame,
    padj_threshold: float,
    log2fc_threshold: float,
) -> pd.Series:
    """Return a Series of 'up', 'down', or 'ns' for each gene."""
    sig = deg_df["padj"] < padj_threshold
    up = sig & (deg_df["log2fc"] >= log2fc_threshold)
    down = sig & (deg_df["log2fc"] <= -log2fc_threshold)
    category = pd.Series("ns", index=deg_df.index)
    category[up] = "up"
    category[down] = "down"
    return category


# ---------------------------------------------------------------------------
# Plotly MA plot
# ---------------------------------------------------------------------------


def create_ma_plot_plotly(
    deg_df: pd.DataFrame,
    padj_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
    title: str = "",
) -> go.Figure:
    """Create an interactive Plotly MA plot.

    Parameters
    ----------
    deg_df : pd.DataFrame
        Must contain columns ``gene_name``, ``log2fc``, ``padj``, and
        ``basemean``.
    padj_threshold : float
        Adjusted p-value significance cutoff.
    log2fc_threshold : float
        Absolute log2 fold-change cutoff.
    title : str
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    required = {"log2fc", "padj", "basemean"}
    missing = required - set(deg_df.columns)
    if missing:
        raise ValueError(f"DEG DataFrame missing required columns: {missing}")

    df = deg_df.dropna(subset=["log2fc", "padj", "basemean"]).copy()
    df["log2_basemean"] = np.log2(df["basemean"].clip(lower=1e-10))
    category = _classify_genes(df, padj_threshold, log2fc_threshold)

    fig = go.Figure()

    for cat, label in [("ns", "NS"), ("up", "Up"), ("down", "Down")]:
        mask = category == cat
        sub = df[mask]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=sub["log2_basemean"],
                y=sub["log2fc"],
                mode="markers",
                marker=dict(
                    color=VOLCANO_COLORS[cat],
                    size=4,
                    opacity=0.7,
                ),
                name=label,
                text=sub["gene_name"] if "gene_name" in sub.columns else sub.index.astype(str),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "log2(baseMean): %{x:.2f}<br>"
                    "log2FC: %{y:.2f}<br>"
                    "padj: %{customdata:.2e}<extra></extra>"
                ),
                customdata=sub["padj"],
            )
        )

    # Horizontal reference line at y=0
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        line_width=1,
    )

    fig.update_layout(
        title=title,
        xaxis_title=format_axis_label("log2(baseMean)"),
        yaxis_title=format_axis_label("log2 Fold Change"),
    )

    apply_plotly_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Matplotlib MA plot
# ---------------------------------------------------------------------------


def create_ma_plot_matplotlib(
    deg_df: pd.DataFrame,
    padj_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
    title: str = "",
):
    """Create a static matplotlib MA plot.

    Parameters
    ----------
    deg_df : pd.DataFrame
        Must contain columns ``gene_name``, ``log2fc``, ``padj``, and
        ``basemean``.
    padj_threshold : float
        Adjusted p-value significance cutoff.
    log2fc_threshold : float
        Absolute log2 fold-change cutoff.
    title : str
        Plot title.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    apply_matplotlib_theme()

    required = {"log2fc", "padj", "basemean"}
    missing = required - set(deg_df.columns)
    if missing:
        raise ValueError(f"DEG DataFrame missing required columns: {missing}")

    df = deg_df.dropna(subset=["log2fc", "padj", "basemean"]).copy()
    df["log2_basemean"] = np.log2(df["basemean"].clip(lower=1e-10))
    category = _classify_genes(df, padj_threshold, log2fc_threshold)

    fig, ax = plt.subplots(figsize=(5, 4))

    for cat, label in [("ns", "NS"), ("up", "Up"), ("down", "Down")]:
        mask = category == cat
        sub = df[mask]
        if sub.empty:
            continue
        ax.scatter(
            sub["log2_basemean"],
            sub["log2fc"],
            c=VOLCANO_COLORS[cat],
            s=6,
            alpha=0.7,
            edgecolors="none",
            label=label,
        )

    # Horizontal reference line at y=0
    ax.axhline(0, linestyle="--", color="gray", linewidth=0.5)

    ax.set_xlabel(format_axis_label("log2(baseMean)"))
    ax.set_ylabel(format_axis_label("log2 Fold Change"))
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=6, markerscale=1.5)
    fig.tight_layout()

    return fig, ax
