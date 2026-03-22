"""Enrichment analysis visualizations (dot plots and bar charts)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from novoview.plotting.theme import (
    WONG_PALETTE,
    SEQUENTIAL_CMAP,
    apply_plotly_theme,
    format_axis_label,
)


def _prepare_enrichment(
    enrichment_df: pd.DataFrame,
    max_terms: int,
) -> pd.DataFrame:
    """Sort by significance and keep top terms."""
    df = enrichment_df.dropna(subset=["padj"]).copy()
    df["neg_log10_padj"] = -np.log10(df["padj"].clip(lower=1e-300))
    df = df.sort_values("padj", ascending=True).head(max_terms)
    # Reverse so most significant is at top when plotted on y-axis
    df = df.iloc[::-1]
    return df


# ---------------------------------------------------------------------------
# Dot plot
# ---------------------------------------------------------------------------


def create_enrichment_dotplot(
    enrichment_df: pd.DataFrame,
    title: str = "",
    max_terms: int = 15,
) -> go.Figure:
    """Create an enrichment dot plot.

    Parameters
    ----------
    enrichment_df : pd.DataFrame
        Must contain columns ``term_name``, ``padj``, and either
        ``gene_ratio`` or ``gene_count``.
    title : str
        Plot title.
    max_terms : int
        Maximum number of terms to display.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = _prepare_enrichment(enrichment_df, max_terms)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title or "Enrichment",
            annotations=[dict(
                text="No significant enrichment terms to display",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14, color="#7A7A7A"),
            )],
        )
        return fig

    # Determine x-axis values
    if "gene_ratio" in df.columns:
        x_vals = pd.to_numeric(df["gene_ratio"], errors="coerce")
        x_label = "Gene Ratio"
    elif "gene_count" in df.columns:
        x_vals = df["gene_count"]
        x_label = "Gene Count"
    else:
        # No numeric measure available; use row position as placeholder
        x_vals = pd.Series(range(1, len(df) + 1), index=df.index)
        x_label = "Rank"

    # Dot size from gene_count
    if "gene_count" in df.columns:
        sizes = df["gene_count"]
        # Scale to reasonable marker sizes
        size_min, size_max = 6, 24
        if sizes.max() > sizes.min():
            scaled = (sizes - sizes.min()) / (sizes.max() - sizes.min())
            marker_sizes = size_min + scaled * (size_max - size_min)
        else:
            marker_sizes = np.full(len(sizes), (size_min + size_max) / 2)
    else:
        marker_sizes = np.full(len(df), 10)
        sizes = pd.Series(np.nan, index=df.index)

    fig = go.Figure(
        data=go.Scatter(
            x=x_vals,
            y=df["term_name"],
            mode="markers",
            marker=dict(
                size=marker_sizes,
                color=df["neg_log10_padj"],
                colorscale=SEQUENTIAL_CMAP,
                showscale=True,
                colorbar=dict(
                    title="-log10(padj)",
                    x=1.15,
                ),
            ),
            text=df["gene_count"] if "gene_count" in df.columns else None,
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{x_label}: %{{x:.3f}}<br>"
                + ("Gene count: %{text}<br>" if "gene_count" in df.columns else "")
                + "-log10(padj): %{marker.color:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="",
        legend=dict(x=1.15, y=1, xanchor="left"),
        margin=dict(l=250),
    )

    apply_plotly_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Bar plot
# ---------------------------------------------------------------------------


def create_enrichment_barplot(
    enrichment_df: pd.DataFrame,
    title: str = "",
    max_terms: int = 15,
) -> go.Figure:
    """Create a horizontal bar chart of enrichment results.

    Parameters
    ----------
    enrichment_df : pd.DataFrame
        Must contain columns ``term_name`` and ``padj``.  Optionally
        ``category`` for colour grouping.
    title : str
        Plot title.
    max_terms : int
        Maximum number of terms to display.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = _prepare_enrichment(enrichment_df, max_terms)

    fig = go.Figure()

    if "category" in df.columns:
        unique_cats = df["category"].unique()
        from itertools import cycle as _cycle
        palette = dict(zip(unique_cats, _cycle(WONG_PALETTE)))
        for cat in unique_cats:
            sub = df[df["category"] == cat]
            fig.add_trace(
                go.Bar(
                    y=sub["term_name"],
                    x=sub["neg_log10_padj"],
                    orientation="h",
                    name=str(cat),
                    marker_color=palette[cat],
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "-log10(padj): %{x:.2f}<extra></extra>"
                    ),
                )
            )
    else:
        fig.add_trace(
            go.Bar(
                y=df["term_name"],
                x=df["neg_log10_padj"],
                orientation="h",
                marker_color=WONG_PALETTE[0],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "-log10(padj): %{x:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=format_axis_label("-log10(padj)"),
        yaxis_title="",
        barmode="stack",
        legend=dict(x=1.02, y=1, xanchor="left"),
        margin=dict(l=250),
    )

    apply_plotly_theme(fig)
    return fig
