"""Overview Dashboard -- NovoView Streamlit page.

Displays high-level metrics, dimensionality-reduction scatters, sample
correlation heatmap, top-variable-gene heatmap, and per-comparison DEG
summary table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the novoview package root is importable
# ---------------------------------------------------------------------------
_NOVOVIEW_ROOT = Path(__file__).resolve().parents[2]
if str(_NOVOVIEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_NOVOVIEW_ROOT))

from pipeline.persistence import (  # noqa: E402
    load_results,
    load_expression,
    load_qc,
    load_deg,
)
from plotting.pca import create_pca_scatter, create_umap_scatter  # noqa: E402
from plotting.heatmap import create_heatmap_plotly  # noqa: E402
from app.components.filters import threshold_sliders  # noqa: E402
from app.components.download import download_figure_buttons  # noqa: E402
from app.components.shared import (  # noqa: E402
    get_data_path,
    check_data_path,
    get_sample_groups,
    fmt_count,
    table_height,
)

# ---------------------------------------------------------------------------
# Local plotting helpers
# ---------------------------------------------------------------------------


def _create_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Build a sample-sample correlation heatmap."""
    from plotting.theme import apply_plotly_theme, get_nature_colorscale

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns.tolist(),
            y=corr_df.index.tolist(),
            colorscale=get_nature_colorscale("diverging"),
            zmin=-1,
            zmax=1,
            colorbar=dict(title="r"),
            hovertemplate=(
                "Sample X: %{x}<br>"
                "Sample Y: %{y}<br>"
                "r = %{z:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Sample Correlation",
        xaxis=dict(tickangle=-45),
        height=550,
    )
    apply_plotly_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Data loading with caching
# ---------------------------------------------------------------------------

_get_data_path = get_data_path


@st.cache_data(show_spinner="Loading results...")
def _load_all_results(path: str) -> dict:
    return load_results(path)


@st.cache_data(show_spinner="Loading expression matrix...")
def _load_expression(path: str, matrix_type: str = "tpm") -> pd.DataFrame | None:
    return load_expression(path, matrix_type=matrix_type)


@st.cache_data(show_spinner="Loading QC data...")
def _load_qc(path: str) -> dict | None:
    return load_qc(path)


@st.cache_data(show_spinner="Loading DEG data...")
def _load_deg(path: str) -> dict | None:
    return load_deg(path)


_get_sample_groups = get_sample_groups


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("Overview Dashboard")
    st.caption(
        "High-level summary of your RNA-Seq experiment: sample counts, "
        "dimensionality reduction, sample correlation, and differential expression overview."
    )

    data_path = _get_data_path()
    if not check_data_path(data_path):
        return

    # Load data -- use granular loaders to avoid reading everything twice
    results = _load_all_results(data_path)
    expression_df = _load_expression(data_path, "tpm")
    qc_data = _load_qc(data_path)
    deg_all = _load_deg(data_path)
    # Note: results is used only for metadata/embeddings; expression, QC, and
    # DEG data come from the granular cached loaders above.

    metadata = results.get("metadata") or {}
    embeddings = results.get("embeddings") or {}
    samples_meta = metadata.get("samples")

    # ------------------------------------------------------------------
    # Top section: metric cards
    # ------------------------------------------------------------------
    st.header("Summary")

    n_samples = expression_df.shape[1] if expression_df is not None else 0
    n_genes = expression_df.shape[0] if expression_df is not None else 0
    n_comparisons = len(deg_all) if deg_all else 0

    # Count total DEGs across all comparisons (default thresholds)
    total_degs = 0
    if deg_all:
        for _comp, df in deg_all.items():
            if "regulation" in df.columns:
                total_degs += int((df["regulation"] != "ns").sum())
            elif "padj" in df.columns and "log2fc" in df.columns:
                total_degs += int(
                    ((df["padj"] < 0.05) & (df["log2fc"].abs() > 1.0)).sum()
                )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", fmt_count(n_samples))
    col2.metric("Total Genes", fmt_count(n_genes))
    col3.metric("Comparisons", fmt_count(n_comparisons))
    col4.metric("Total DEGs", fmt_count(total_degs))
    st.caption(
        f"Experiment summary: {n_samples} samples, {n_genes:,} measured genes, "
        f"{n_comparisons} differential expression comparisons. "
        f"Total DEGs counted at default thresholds (padj < 0.05, |log2FC| > 1)."
    )

    # ------------------------------------------------------------------
    # Middle section: PCA / UMAP + Correlation heatmap
    # ------------------------------------------------------------------
    st.divider()
    st.header("Dimensionality Reduction & Sample Correlation")
    left_col, right_col = st.columns(2)

    with left_col:
        pca_coords = embeddings.get("pca_coordinates")
        pca_variance = embeddings.get("pca_variance")
        umap_coords = embeddings.get("umap")

        available_methods: list[str] = []
        if pca_coords is not None:
            available_methods.append("PCA")
        if umap_coords is not None:
            available_methods.append("UMAP")

        if available_methods:
            method = st.selectbox(
                "Embedding method",
                options=available_methods,
                key="overview_embedding_method",
                help=(
                    "PCA shows major axes of variation (linear). "
                    "UMAP reveals local sample clusters (non-linear). "
                    "Both help visualize how samples relate to each other."
                ),
            )

            sample_names = (
                expression_df.columns.tolist() if expression_df is not None else []
            )
            sample_groups = _get_sample_groups(samples_meta, sample_names)

            if method == "PCA" and pca_coords is not None:
                # Prepare variance array
                var_array = None
                if pca_variance is not None:
                    if isinstance(pca_variance, pd.DataFrame):
                        var_array = pca_variance.values.flatten()
                    else:
                        var_array = np.asarray(pca_variance)

                fig = create_pca_scatter(
                    pca_coords,
                    variance_explained=var_array if var_array is not None else [0, 0],
                    sample_groups=sample_groups,
                    title="PCA",
                )
                st.plotly_chart(fig, use_container_width=True)
                download_figure_buttons(fig, "pca_scatter")

            elif method == "UMAP" and umap_coords is not None:
                fig = create_umap_scatter(
                    umap_coords,
                    sample_groups=sample_groups,
                    title="UMAP",
                )
                st.plotly_chart(fig, use_container_width=True)
                download_figure_buttons(fig, "umap_scatter")
        else:
            st.info("No PCA or UMAP embeddings available in the results file.")

    with right_col:
        corr_df = None
        if qc_data is not None:
            corr_df = qc_data.get("correlation")

        # Fallback: compute correlation from top 1000 variable genes (fast)
        if corr_df is None and expression_df is not None:
            _top_var = expression_df.var(axis=1).nlargest(min(1000, len(expression_df)))
            corr_df = expression_df.loc[_top_var.index].corr()

        if corr_df is not None:
            fig_corr = _create_correlation_heatmap(corr_df)
            st.plotly_chart(fig_corr, use_container_width=True)
            download_figure_buttons(fig_corr, "sample_correlation")
        else:
            st.info("No correlation data available.")

    # ------------------------------------------------------------------
    # Bottom section: top variable gene heatmap + DEG summary table
    # ------------------------------------------------------------------
    st.divider()
    st.header("Top Variable Genes & DEG Summary")
    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        st.subheader("Top 50 Most Variable Genes")
        if expression_df is not None:
            sample_names_hm = expression_df.columns.tolist()
            sample_groups_hm = _get_sample_groups(samples_meta, sample_names_hm)

            fig_hm = create_heatmap_plotly(
                expression_df,
                sample_groups=sample_groups_hm,
                genes=None,
                n_top_genes=50,
                title="Top 50 Variable Genes (z-score)",
            )
            st.plotly_chart(fig_hm, use_container_width=True)
            download_figure_buttons(fig_hm, "top50_variable_genes")
        else:
            st.info("No expression data available.")

    with bottom_right:
        st.subheader("Per-Comparison DEG Counts")
        if deg_all:
            padj_thresh, log2fc_thresh = threshold_sliders(
                key_prefix="overview_deg_",
            )

            rows = []
            for comp_name, df in sorted(deg_all.items()):
                if "padj" not in df.columns or "log2fc" not in df.columns:
                    continue
                sig_mask = (df["padj"] < padj_thresh) & (
                    df["log2fc"].abs() > log2fc_thresh
                )
                sig = df.loc[sig_mask]
                if "regulation" in sig.columns:
                    n_up = int((sig["regulation"] == "up").sum())
                    n_down = int((sig["regulation"] == "down").sum())
                else:
                    n_up = int((sig["log2fc"] > 0).sum())
                    n_down = int((sig["log2fc"] < 0).sum())
                rows.append(
                    {
                        "Comparison": comp_name,
                        "Total DEGs": len(sig),
                        "Up": n_up,
                        "Down": n_down,
                    }
                )

            if rows:
                summary_df = pd.DataFrame(rows)
                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True,
                    height=table_height(len(summary_df)),
                )
            else:
                st.info("No DEG results contain the required columns.")
        else:
            st.info("No DEG results available.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__page__" or __name__ == "__main__":
    main()
else:
    main()
