"""Similarity table and gene network visualizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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


# ---------------------------------------------------------------------------
# Similarity table
# ---------------------------------------------------------------------------

_SPARK_CHARS = " " + "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


def _sparkline(values: np.ndarray) -> str:
    """Return a Unicode sparkline-style mini bar string for *values*."""
    v = np.asarray(values, dtype=float)
    if np.isnan(v).all() or v.max() == v.min():
        return _SPARK_CHARS[4] * len(v)
    normed = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v))
    indices = np.clip((normed * 7).astype(int), 0, 7)
    return "".join(_SPARK_CHARS[1 + i] for i in indices)


def create_similarity_table(
    neighbors_df: pd.DataFrame,
    expression_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Format a gene-neighbours table for display.

    Parameters
    ----------
    neighbors_df : pd.DataFrame
        A neighbours table with at least columns ``gene`` and ``neighbor``
        (and optionally ``similarity``).
    expression_df : pd.DataFrame, optional
        Genes (rows) x conditions/samples (columns) expression matrix.
        When provided, a sparkline column is added showing expression across
        conditions for each neighbour gene.

    Returns
    -------
    pd.DataFrame
        A formatted DataFrame suitable for display or export.
    """
    df = neighbors_df.copy()

    # Round similarity for readability
    if "similarity" in df.columns:
        df["similarity"] = df["similarity"].round(4)

    # Add sparkline mini-bar representation
    if expression_df is not None:
        sparklines = []
        ref_col = "neighbor" if "neighbor" in df.columns else df.columns[0]
        for gene in df[ref_col]:
            if gene in expression_df.index:
                sparklines.append(_sparkline(expression_df.loc[gene].values))
            else:
                sparklines.append("")
        df["expression_profile"] = sparklines

    return df


# ---------------------------------------------------------------------------
# Gene network
# ---------------------------------------------------------------------------


def create_gene_network(
    similarity_matrix: pd.DataFrame,
    top_n: int = 200,
    gene_clusters: pd.Series | dict[str, int] | None = None,
) -> go.Figure:
    """Build a Plotly network visualization of the most connected genes.

    A force-directed-style layout is approximated by treating the similarity
    matrix as an adjacency matrix, computing a 2-D spectral embedding via the
    graph Laplacian, and drawing edges for the strongest connections.

    Parameters
    ----------
    similarity_matrix : pd.DataFrame
        Square gene-by-gene similarity matrix (symmetric, non-negative).
    top_n : int
        Number of top genes (by total connectivity / degree) to include.
    gene_clusters : pd.Series or dict, optional
        Cluster assignment for each gene.  Used to colour nodes.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    sim = similarity_matrix.copy()

    # Zero out the diagonal to ignore self-similarity
    np.fill_diagonal(sim.values, 0)

    # Select top N most connected genes
    degree = sim.sum(axis=1).sort_values(ascending=False)
    top_genes = degree.head(top_n).index.tolist()
    sim = sim.loc[top_genes, top_genes]

    n = len(top_genes)

    # -- Spectral layout (2-D) -----------------------------------------------
    adj = sim.values.copy()
    degree_vec = adj.sum(axis=1)
    degree_vec[degree_vec == 0] = 1  # avoid division by zero
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree_vec))
    laplacian = np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    # Use the 2nd and 3rd smallest eigenvectors (skip the trivial first)
    coords = eigenvectors[:, 1:3]
    x_pos = coords[:, 0]
    y_pos = coords[:, 1]

    # -- Build edges ----------------------------------------------------------
    # Keep only edges above the median similarity among the selected genes
    upper_tri = np.triu_indices(n, k=1)
    edge_weights = adj[upper_tri]
    threshold = np.percentile(edge_weights[edge_weights > 0], 50) if (edge_weights > 0).any() else 0

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for i, j in zip(*upper_tri):
        if adj[i, j] > threshold:
            edge_x.extend([x_pos[i], x_pos[j], None])
            edge_y.extend([y_pos[i], y_pos[j], None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.3, color="#CCCCCC"),
        hoverinfo="skip",
        showlegend=False,
    )

    # -- Build nodes ----------------------------------------------------------
    node_sizes = 4 + 16 * (degree.loc[top_genes].values - degree.loc[top_genes].min()) / max(
        degree.loc[top_genes].max() - degree.loc[top_genes].min(), 1
    )

    if gene_clusters is not None:
        if isinstance(gene_clusters, dict):
            gene_clusters = pd.Series(gene_clusters)
        cluster_labels = gene_clusters.reindex(top_genes).fillna(-1).astype(int)
        unique_clusters = sorted(cluster_labels.unique())
        palette = {c: WONG_PALETTE[i % len(WONG_PALETTE)] for i, c in enumerate(unique_clusters)}
        node_colors = [palette[c] for c in cluster_labels]
    else:
        node_colors = WONG_PALETTE[0]

    node_trace = go.Scatter(
        x=x_pos,
        y=y_pos,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=0.5, color="white"),
            opacity=0.85,
        ),
        text=top_genes,
        textposition="top center",
        textfont=dict(size=6),
        hovertemplate="<b>%{text}</b><br>Connectivity: %{customdata:.2f}<extra></extra>",
        customdata=degree.loc[top_genes].values,
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title="Gene Similarity Network",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False),
        hovermode="closest",
    )

    apply_plotly_theme(fig)
    return fig
