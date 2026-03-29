"""Clustered heatmap visualizations for gene expression data.

Provides ``create_clustered_heatmap`` (seaborn/matplotlib clustermap) and
``create_heatmap_plotly`` (interactive Plotly heatmap).  Both z-score
normalise rows and apply hierarchical clustering (Ward linkage on
Euclidean distances) to order genes and samples.
"""

from __future__ import annotations

import logging
from itertools import cycle

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

_logger = logging.getLogger(__name__)

from plotting.theme import (
    WONG_PALETTE,
    DIVERGING_CMAP,
    PLOTLY_DIVERGING_CMAP,
    apply_plotly_theme,
    apply_matplotlib_theme,
)


def _select_genes(
    expression_df: pd.DataFrame,
    genes: list[str] | None,
    n_top_genes: int,
) -> pd.DataFrame:
    """Select specified genes or the top N most variable genes."""
    if genes is not None:
        available = [g for g in genes if g in expression_df.index]
        return expression_df.loc[available]
    # Select top variable genes by row variance
    var = expression_df.var(axis=1)
    top = var.nlargest(min(n_top_genes, len(var))).index
    return expression_df.loc[top]


def _zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize each row (gene).

    Genes with zero variance across samples are scaled to 0 (not NaN).
    """
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    n_zero_var = int((stds == 0).sum())
    if n_zero_var > 0:
        _logger.warning(
            "%d gene(s) have zero variance across samples and will appear "
            "as zero in the z-scored heatmap.",
            n_zero_var,
        )
    stds = stds.replace(0, 1)
    return df.sub(means, axis=0).div(stds, axis=0)


# ---------------------------------------------------------------------------
# Seaborn clustermap
# ---------------------------------------------------------------------------


def create_clustered_heatmap(
    expression_df: pd.DataFrame,
    sample_groups: dict[str, str] | pd.Series | None = None,
    genes: list[str] | None = None,
    n_top_genes: int = 50,
    title: str = "",
):
    """Create a clustered heatmap using seaborn.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Genes (rows) x samples (columns) expression matrix.
    sample_groups : dict or pd.Series, optional
        Mapping of sample names to group/condition labels.
    genes : list[str], optional
        Specific genes to include.  If *None*, the top *n_top_genes* most
        variable genes are selected.
    n_top_genes : int
        Number of top variable genes when *genes* is not specified.
    title : str
        Plot title.

    Returns
    -------
    seaborn.matrix.ClusterGrid
    """
    apply_matplotlib_theme()

    sub = _select_genes(expression_df, genes, n_top_genes)
    if sub.empty:
        raise ValueError("No genes available for heatmap after filtering.")
    zscored = _zscore_rows(sub)

    # Build column color bar if groups provided
    col_colors = None
    if sample_groups is not None:
        if isinstance(sample_groups, dict):
            sample_groups = pd.Series(sample_groups)
        unique_groups = sample_groups.unique()
        palette = dict(zip(unique_groups, cycle(WONG_PALETTE)))
        col_colors = sample_groups.reindex(zscored.columns).map(palette)
        col_colors.name = "Condition"

    cg = sns.clustermap(
        zscored,
        method="ward",
        metric="euclidean",
        cmap=DIVERGING_CMAP,
        center=0,
        vmin=-3,
        vmax=3,
        col_colors=col_colors,
        figsize=(8, max(6, len(zscored) * 0.15)),
        dendrogram_ratio=(0.12, 0.12),
        cbar_pos=(1.02, 0.3, 0.02, 0.4),
        cbar_kws={"label": "Z-score"},
        xticklabels=True,
        yticklabels=True,
    )

    cg.ax_heatmap.set_xlabel("")
    cg.ax_heatmap.set_ylabel("")
    if title:
        cg.fig.suptitle(title, y=1.02, fontweight="semibold")

    return cg


# ---------------------------------------------------------------------------
# Plotly heatmap
# ---------------------------------------------------------------------------


def create_heatmap_plotly(
    expression_df: pd.DataFrame,
    sample_groups: dict[str, str] | pd.Series | None = None,
    genes: list[str] | None = None,
    n_top_genes: int = 50,
    title: str = "",
) -> go.Figure:
    """Create an interactive Plotly heatmap sorted by hierarchical cluster order.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Genes (rows) x samples (columns) expression matrix.
    sample_groups : dict or pd.Series, optional
        Mapping of sample names to group/condition labels.
    genes : list[str], optional
        Specific genes to include.
    n_top_genes : int
        Number of top variable genes when *genes* is not specified.
    title : str
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    sub = _select_genes(expression_df, genes, n_top_genes)

    if sub.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title or "Heatmap",
            annotations=[dict(
                text="No genes available for heatmap",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14, color="#7A7A7A"),
            )],
        )
        return fig

    zscored = _zscore_rows(sub)

    # Cluster rows
    if len(zscored) > 1:
        row_dist = pdist(zscored.values, metric="euclidean")
        row_link = linkage(row_dist, method="ward")
        row_order = leaves_list(row_link)
    else:
        row_order = np.arange(len(zscored))

    # Cluster columns
    if zscored.shape[1] > 1:
        col_dist = pdist(zscored.values.T, metric="euclidean")
        col_link = linkage(col_dist, method="ward")
        col_order = leaves_list(col_link)
    else:
        col_order = np.arange(zscored.shape[1])

    zscored = zscored.iloc[row_order, col_order]

    fig = go.Figure(
        data=go.Heatmap(
            z=zscored.values,
            x=zscored.columns.tolist(),
            y=zscored.index.tolist(),
            colorscale=PLOTLY_DIVERGING_CMAP,
            reversescale=True,
            zmid=0,
            zmin=-3,
            zmax=3,
            colorbar=dict(title="Z-score"),
            hovertemplate=(
                "Gene: %{y}<br>"
                "Sample: %{x}<br>"
                "Z-score: %{z:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
    )

    apply_plotly_theme(fig)
    return fig
