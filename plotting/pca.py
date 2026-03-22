"""PCA and UMAP scatter plot visualizations."""

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


def _confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    n_std: float = 1.96,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a 95% confidence ellipse from 2-D points.

    Parameters
    ----------
    x, y : array-like
        Coordinates.
    n_std : float
        Number of standard deviations (1.96 ≈ 95%).
    n_points : int
        Resolution of the ellipse boundary.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (ellipse_x, ellipse_y) arrays suitable for plotting.
    """
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by descending eigenvalue
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    theta = np.linspace(0, 2 * np.pi, n_points)
    half_widths = n_std * np.sqrt(np.maximum(eigenvalues, 0))

    ellipse_x = half_widths[0] * np.cos(theta)
    ellipse_y = half_widths[1] * np.sin(theta)

    # Rotate
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_x = cos_a * ellipse_x - sin_a * ellipse_y + np.mean(x)
    rot_y = sin_a * ellipse_x + cos_a * ellipse_y + np.mean(y)

    return rot_x, rot_y


# ---------------------------------------------------------------------------
# PCA scatter
# ---------------------------------------------------------------------------


def create_pca_scatter(
    pca_coords: np.ndarray | pd.DataFrame,
    variance_explained: list[float] | np.ndarray,
    sample_groups: pd.Series | dict[str, str] | None = None,
    title: str = "PCA",
) -> go.Figure:
    """Create an interactive PCA scatter plot with 95 % confidence ellipses.

    Parameters
    ----------
    pca_coords : array-like, shape (n_samples, >=2)
        PC1 and PC2 coordinates.
    variance_explained : array-like
        Proportion of variance explained for each PC (values in 0-100 or 0-1).
    sample_groups : pd.Series or dict, optional
        Group labels for each sample (index-aligned or dict keyed by sample).
    title : str
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if isinstance(pca_coords, pd.DataFrame):
        samples = pca_coords.index.tolist()
        coords = pca_coords.values
    else:
        coords = np.asarray(pca_coords)
        samples = [f"Sample_{i}" for i in range(len(coords))]

    pc1 = coords[:, 0]
    pc2 = coords[:, 1]

    # Normalise variance_explained to percentages
    ve = np.asarray(variance_explained, dtype=float)
    if ve.max() <= 1.0:
        ve = ve * 100

    fig = go.Figure()

    if sample_groups is not None:
        if isinstance(sample_groups, dict):
            sample_groups = pd.Series(sample_groups)
        groups = sample_groups.reindex(samples) if hasattr(sample_groups, "reindex") else sample_groups
        unique_groups = groups.unique()
        palette = dict(zip(unique_groups, WONG_PALETTE[: len(unique_groups)]))

        for grp in unique_groups:
            mask = (groups == grp).values if hasattr(groups, "values") else (groups == grp)
            gx, gy = pc1[mask], pc2[mask]
            fig.add_trace(
                go.Scatter(
                    x=gx,
                    y=gy,
                    mode="markers",
                    marker=dict(color=palette[grp], size=8, opacity=0.8),
                    name=str(grp),
                    text=[samples[i] for i, m in enumerate(mask) if m],
                    hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
                )
            )

            # 95% confidence ellipse (need >= 3 points)
            if len(gx) >= 3:
                ex, ey = _confidence_ellipse(gx, gy)
                fig.add_trace(
                    go.Scatter(
                        x=ex,
                        y=ey,
                        mode="lines",
                        line=dict(color=palette[grp], width=1.5, dash="dot"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
    else:
        fig.add_trace(
            go.Scatter(
                x=pc1,
                y=pc2,
                mode="markers",
                marker=dict(color=WONG_PALETTE[0], size=8, opacity=0.8),
                name="Samples",
                text=samples,
                hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=format_axis_label(f"PC1", f"{ve[0]:.1f}% variance"),
        yaxis_title=format_axis_label(f"PC2", f"{ve[1]:.1f}% variance"),
        legend=dict(x=1.02, y=1, xanchor="left"),
    )

    apply_plotly_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# UMAP scatter
# ---------------------------------------------------------------------------


def create_umap_scatter(
    umap_coords: np.ndarray | pd.DataFrame,
    sample_groups: pd.Series | dict[str, str] | None = None,
    title: str = "UMAP",
) -> go.Figure:
    """Create an interactive UMAP scatter plot.

    Parameters
    ----------
    umap_coords : array-like, shape (n_samples, >=2)
        UMAP1 and UMAP2 coordinates.
    sample_groups : pd.Series or dict, optional
        Group labels for each sample.
    title : str
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if isinstance(umap_coords, pd.DataFrame):
        samples = umap_coords.index.tolist()
        coords = umap_coords.values
    else:
        coords = np.asarray(umap_coords)
        samples = [f"Sample_{i}" for i in range(len(coords))]

    u1 = coords[:, 0]
    u2 = coords[:, 1]

    fig = go.Figure()

    if sample_groups is not None:
        if isinstance(sample_groups, dict):
            sample_groups = pd.Series(sample_groups)
        groups = sample_groups.reindex(samples) if hasattr(sample_groups, "reindex") else sample_groups
        unique_groups = groups.unique()
        palette = dict(zip(unique_groups, WONG_PALETTE[: len(unique_groups)]))

        for grp in unique_groups:
            mask = (groups == grp).values if hasattr(groups, "values") else (groups == grp)
            gx, gy = u1[mask], u2[mask]
            fig.add_trace(
                go.Scatter(
                    x=gx,
                    y=gy,
                    mode="markers",
                    marker=dict(color=palette[grp], size=8, opacity=0.8),
                    name=str(grp),
                    text=[samples[i] for i, m in enumerate(mask) if m],
                    hovertemplate="<b>%{text}</b><br>UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<extra></extra>",
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=u1,
                y=u2,
                mode="markers",
                marker=dict(color=WONG_PALETTE[0], size=8, opacity=0.8),
                name="Samples",
                text=samples,
                hovertemplate="<b>%{text}</b><br>UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="UMAP1",
        yaxis_title="UMAP2",
        legend=dict(x=1.02, y=1, xanchor="left"),
    )

    apply_plotly_theme(fig)
    return fig
