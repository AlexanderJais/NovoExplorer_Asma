"""Volcano plot visualizations for differential expression results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from novoview.plotting.theme import (
    WONG_PALETTE,
    VOLCANO_COLORS,
    apply_plotly_theme,
    apply_matplotlib_theme,
    format_axis_label,
)


def _classify_genes(
    deg_df: pd.DataFrame,
    padj_threshold: float,
    log2fc_threshold: float,
) -> pd.Series:
    """Return a Series of 'up', 'down', or 'ns' for each gene."""
    sig = deg_df["padj"] < padj_threshold
    up = sig & (deg_df["log2fc"] > log2fc_threshold)
    down = sig & (deg_df["log2fc"] < -log2fc_threshold)
    category = pd.Series("ns", index=deg_df.index)
    category[up] = "up"
    category[down] = "down"
    return category


def _top_significant(
    deg_df: pd.DataFrame,
    category: pd.Series,
    top_n: int,
) -> pd.DataFrame:
    """Return the top N most significant non-NS genes for labelling."""
    sig_mask = category != "ns"
    sig_genes = deg_df.loc[sig_mask].copy()
    sig_genes = sig_genes.sort_values("padj", ascending=True)
    return sig_genes.head(top_n)


# ---------------------------------------------------------------------------
# Plotly volcano
# ---------------------------------------------------------------------------


def create_volcano_plotly(
    deg_df: pd.DataFrame,
    padj_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
    title: str = "",
    top_n_labels: int = 10,
) -> go.Figure:
    """Create an interactive Plotly volcano plot.

    Parameters
    ----------
    deg_df : pd.DataFrame
        Must contain columns ``gene_name``, ``log2fc``, and ``padj``.
    padj_threshold : float
        Adjusted p-value significance cutoff.
    log2fc_threshold : float
        Absolute log2 fold-change cutoff.
    title : str
        Plot title.
    top_n_labels : int
        Number of top significant genes to annotate.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # Validate required columns
    required = {"log2fc", "padj"}
    missing = required - set(deg_df.columns)
    if missing:
        raise ValueError(f"DEG DataFrame missing required columns: {missing}")

    df = deg_df.dropna(subset=["padj", "log2fc"]).copy()
    # Ensure gene_name column exists for labeling
    if "gene_name" not in df.columns:
        df["gene_name"] = df.index.astype(str)
    df["neg_log10_padj"] = -np.log10(df["padj"].clip(lower=1e-300))
    category = _classify_genes(df, padj_threshold, log2fc_threshold)

    fig = go.Figure()

    # Plot each category as a separate trace for legend control
    _MAX_NS_POINTS = 2000  # downsample NS genes for rendering performance
    for cat, label in [("ns", "NS"), ("up", "Up"), ("down", "Down")]:
        mask = category == cat
        sub = df.loc[mask]
        if sub.empty:
            continue
        # Downsample non-significant genes to avoid slow JSON serialisation
        if cat == "ns" and len(sub) > _MAX_NS_POINTS:
            sub = sub.sample(n=_MAX_NS_POINTS, random_state=42)
        fig.add_trace(
            go.Scattergl(
                x=sub["log2fc"],
                y=sub["neg_log10_padj"],
                mode="markers",
                marker=dict(
                    color=VOLCANO_COLORS[cat],
                    size=4,
                    opacity=0.7,
                ),
                name=label,
                text=sub["gene_name"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "log2FC: %{x:.2f}<br>"
                    "padj: %{customdata:.2e}<extra></extra>"
                ),
                customdata=sub["padj"],
            )
        )

    # Threshold lines
    neg_log10_cutoff = -np.log10(padj_threshold)
    fig.add_hline(
        y=neg_log10_cutoff,
        line_dash="dash",
        line_color="gray",
        line_width=1,
    )
    fig.add_vline(
        x=log2fc_threshold,
        line_dash="dash",
        line_color="gray",
        line_width=1,
    )
    fig.add_vline(
        x=-log2fc_threshold,
        line_dash="dash",
        line_color="gray",
        line_width=1,
    )

    # Annotate top significant genes
    top_genes = _top_significant(df, category, top_n_labels)
    for _, row in top_genes.iterrows():
        fig.add_annotation(
            x=row["log2fc"],
            y=row["neg_log10_padj"],
            text=row["gene_name"],
            showarrow=True,
            arrowhead=0,
            arrowwidth=0.5,
            ax=0,
            ay=-12,
            font=dict(size=8),
        )

    fig.update_layout(
        title=title,
        xaxis_title=format_axis_label("log2 Fold Change"),
        yaxis_title=format_axis_label("-log10(padj)"),
    )

    apply_plotly_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Matplotlib volcano
# ---------------------------------------------------------------------------


def create_volcano_matplotlib(
    deg_df: pd.DataFrame,
    padj_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
    title: str = "",
    top_n_labels: int = 10,
):
    """Create a static matplotlib volcano plot.

    Parameters
    ----------
    deg_df : pd.DataFrame
        Must contain columns ``gene_name``, ``log2fc``, and ``padj``.
    padj_threshold : float
        Adjusted p-value significance cutoff.
    log2fc_threshold : float
        Absolute log2 fold-change cutoff.
    title : str
        Plot title.
    top_n_labels : int
        Number of top significant genes to annotate.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    from adjustText import adjust_text

    apply_matplotlib_theme()

    required = {"log2fc", "padj"}
    missing = required - set(deg_df.columns)
    if missing:
        raise ValueError(f"DEG DataFrame missing required columns: {missing}")

    df = deg_df.dropna(subset=["padj", "log2fc"]).copy()
    if "gene_name" not in df.columns:
        df["gene_name"] = df.index.astype(str)
    df["neg_log10_padj"] = -np.log10(df["padj"].clip(lower=1e-300))
    category = _classify_genes(df, padj_threshold, log2fc_threshold)

    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot each category
    for cat, label in [("ns", "NS"), ("up", "Up"), ("down", "Down")]:
        mask = category == cat
        sub = df.loc[mask]
        if sub.empty:
            continue
        ax.scatter(
            sub["log2fc"],
            sub["neg_log10_padj"],
            c=VOLCANO_COLORS[cat],
            s=6,
            alpha=0.7,
            edgecolors="none",
            label=label,
        )

    # Threshold lines
    neg_log10_cutoff = -np.log10(padj_threshold)
    ax.axhline(neg_log10_cutoff, linestyle="--", color="gray", linewidth=0.5)
    ax.axvline(log2fc_threshold, linestyle="--", color="gray", linewidth=0.5)
    ax.axvline(-log2fc_threshold, linestyle="--", color="gray", linewidth=0.5)

    # Label top significant genes with adjustText
    top_genes = _top_significant(df, category, top_n_labels)
    texts = []
    for _, row in top_genes.iterrows():
        texts.append(
            ax.text(
                row["log2fc"],
                row["neg_log10_padj"],
                row["gene_name"],
                fontsize=5,
            )
        )
    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.4))

    ax.set_xlabel(format_axis_label("log2 Fold Change"))
    ax.set_ylabel(format_axis_label("-log10(padj)"))
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=6, markerscale=1.5)
    fig.tight_layout()

    return fig, ax
