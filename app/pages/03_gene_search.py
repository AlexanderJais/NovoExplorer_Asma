"""Gene Search and Similarity Finder -- NovoView Streamlit page.

Provides gene name search with autocomplete, expression bar charts,
similar-gene discovery via cosine similarity, and basket integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the novoview package root is importable
# ---------------------------------------------------------------------------
_NOVOVIEW_ROOT = Path(__file__).resolve().parents[2]
if str(_NOVOVIEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_NOVOVIEW_ROOT))

from pipeline.persistence import (  # noqa: E402
    load_expression,
    load_similarity,
    load_deg,
    load_results,
)
from plotting.theme import (  # noqa: E402
    WONG_PALETTE,
    DIVERGING_CMAP,
    apply_plotly_theme,
)
from app.components.gene_basket import (  # noqa: E402
    init_basket,
    add_to_basket,
    get_basket,
    render_basket,
)
from app.components.download import download_csv_button  # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Gene Search", layout="wide")

# ---------------------------------------------------------------------------
# Data path helpers
# ---------------------------------------------------------------------------

DATA_PATH_KEY = "data_path"
DEFAULT_DATA_PATH = str(_NOVOVIEW_ROOT / "results" / "novoview_results.h5")


def _get_data_path() -> str:
    # Check the canonical key set by app.py first, then local fallback
    return st.session_state.get("results_path",
           st.session_state.get(DATA_PATH_KEY, DEFAULT_DATA_PATH))


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading results...")
def _load_all_results(path: str) -> dict:
    return load_results(path)


@st.cache_data(show_spinner="Loading expression matrix...")
def _load_expression(path: str, matrix_type: str = "tpm") -> pd.DataFrame | None:
    return load_expression(path, matrix_type=matrix_type)


@st.cache_data(show_spinner="Loading similarity data...")
def _load_similarity(path: str) -> dict | None:
    return load_similarity(path)


@st.cache_data(show_spinner="Loading DEG data...")
def _load_deg(path: str) -> dict | None:
    return load_deg(path)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _create_expression_bar(
    gene_name: str,
    expression_df: pd.DataFrame,
    samples_meta: pd.DataFrame | None,
) -> go.Figure | None:
    """Bar chart of a single gene's expression across samples/conditions."""
    if expression_df is None or gene_name not in expression_df.index:
        return None

    expr_values = expression_df.loc[gene_name]
    df = pd.DataFrame({
        "sample": expr_values.index,
        "expression": expr_values.values,
    })

    # Attach condition from metadata
    if samples_meta is not None and not samples_meta.empty:
        meta = samples_meta.copy()
        if "sample_id" in meta.columns:
            meta = meta.set_index("sample_id")
        for candidate in ("condition", "group", "sample_group"):
            if candidate in meta.columns:
                df["condition"] = meta[candidate].reindex(df["sample"].values).values
                break

    if "condition" not in df.columns:
        df["condition"] = "all"

    fig = px.bar(
        df,
        x="sample",
        y="expression",
        color="condition",
        color_discrete_sequence=WONG_PALETTE,
        title=f"{gene_name} Expression Across Samples",
        labels={"expression": "Expression (TPM)", "sample": "Sample"},
    )
    fig.update_layout(xaxis_tickangle=-45, bargap=0.2)
    apply_plotly_theme(fig)
    return fig


def _find_similar_genes(
    gene_name: str,
    similarity_data: dict | None,
    top_n: int = 20,
) -> pd.DataFrame | None:
    """Find the top N most similar genes using the cosine similarity matrix.

    Parameters
    ----------
    gene_name : str
        Query gene.
    similarity_data : dict or None
        Output of ``load_similarity`` containing ``cosine_matrix`` and
        optionally ``gene_clusters``.
    top_n : int
        Number of similar genes to return.

    Returns
    -------
    pd.DataFrame or None
        Columns: ``gene``, ``similarity``.  Sorted descending by similarity.
    """
    if similarity_data is None:
        return None

    cosine_matrix = similarity_data.get("cosine_matrix")
    if cosine_matrix is None:
        return None

    if gene_name not in cosine_matrix.index:
        return None

    scores = cosine_matrix.loc[gene_name].drop(gene_name, errors="ignore")
    top = scores.nlargest(top_n)

    result = pd.DataFrame({
        "gene": top.index,
        "similarity": top.values,
    })

    # Optionally attach cluster info
    gene_clusters = similarity_data.get("gene_clusters")
    if gene_clusters is not None:
        if isinstance(gene_clusters, pd.DataFrame):
            cluster_col = None
            for candidate in ("cluster", "gene_cluster", "module"):
                if candidate in gene_clusters.columns:
                    cluster_col = candidate
                    break
            if cluster_col:
                cluster_map = gene_clusters[cluster_col]
                result["cluster"] = result["gene"].map(cluster_map).values

    return result


def _get_gene_info(
    gene_name: str,
    expression_df: pd.DataFrame | None,
    similarity_data: dict | None,
    deg_all: dict | None,
) -> dict:
    """Collect summary information about a gene for the info card."""
    info: dict = {"name": gene_name}

    # Expression statistics
    if expression_df is not None and gene_name in expression_df.index:
        expr_row = expression_df.loc[gene_name]
        info["mean_expression"] = float(expr_row.mean())
        info["max_expression"] = float(expr_row.max())
        info["n_samples_detected"] = int((expr_row > 0).sum())
        info["n_samples_total"] = len(expr_row)

    # Cluster assignment
    if similarity_data is not None:
        gene_clusters = similarity_data.get("gene_clusters")
        if gene_clusters is not None and isinstance(gene_clusters, pd.DataFrame):
            for candidate in ("cluster", "gene_cluster", "module"):
                if candidate in gene_clusters.columns:
                    if gene_name in gene_clusters.index:
                        info["cluster"] = str(gene_clusters.loc[gene_name, candidate])
                    break

    # DEG summary across comparisons
    if deg_all:
        deg_hits = []
        for comp_name, df in deg_all.items():
            gene_col = "gene_name" if "gene_name" in df.columns else None
            if gene_col:
                match = df[df[gene_col] == gene_name]
            else:
                match = df.loc[[gene_name]] if gene_name in df.index else pd.DataFrame()
            if not match.empty and "padj" in match.columns:
                row = match.iloc[0]
                if row["padj"] < 0.05:
                    direction = "up" if row.get("log2fc", 0) > 0 else "down"
                    deg_hits.append(f"{comp_name} ({direction})")
        if deg_hits:
            info["significant_in"] = deg_hits

    return info


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("Gene Search & Similarity Finder")
    st.caption(
        "Search for any gene to view its expression profile, find co-expressed genes "
        "via cosine similarity, and build a gene basket for comparative analysis."
    )
    init_basket()

    data_path = _get_data_path()

    if not Path(data_path).exists():
        st.error(
            f"Results file not found: `{data_path}`.  "
            "Run the pipeline first or set the correct path in session state "
            f"(key: `{DATA_PATH_KEY}`)."
        )
        return

    # Load data
    results = _load_all_results(data_path)
    expression_df = _load_expression(data_path, "tpm")
    similarity_data = _load_similarity(data_path)
    deg_all = _load_deg(data_path)

    metadata = results.get("metadata") or {}
    samples_meta = metadata.get("samples")

    # ------------------------------------------------------------------
    # Build gene list for autocomplete
    # ------------------------------------------------------------------
    gene_names: list[str] = []
    if expression_df is not None:
        gene_names = sorted(expression_df.index.tolist())
    elif similarity_data is not None and similarity_data.get("cosine_matrix") is not None:
        gene_names = sorted(similarity_data["cosine_matrix"].index.tolist())

    if not gene_names:
        st.warning("No gene names available. Ensure expression data is loaded.")
        return

    # ------------------------------------------------------------------
    # Search box with autocomplete (selectbox)
    # ------------------------------------------------------------------
    st.subheader("Search for a Gene")

    selected_gene = st.selectbox(
        "Gene name",
        options=[""] + gene_names,
        index=0,
        key="gene_search_main",
        help="Start typing to filter the gene list",
    )

    if not selected_gene:
        st.info("Select a gene above to view its expression profile and find similar genes.")
        # Still render basket at bottom
        _render_basket_panel(expression_df, samples_meta)
        return

    # ------------------------------------------------------------------
    # Gene info card
    # ------------------------------------------------------------------
    gene_info = _get_gene_info(selected_gene, expression_df, similarity_data, deg_all)

    st.subheader(f"Gene: {selected_gene}")

    info_cols = st.columns(4)
    with info_cols[0]:
        st.metric("Gene Name", gene_info["name"])
    with info_cols[1]:
        cluster = gene_info.get("cluster", "N/A")
        st.metric("Cluster", cluster)
    with info_cols[2]:
        mean_expr = gene_info.get("mean_expression")
        st.metric("Mean Expression", f"{mean_expr:.2f}" if mean_expr is not None else "N/A")
    with info_cols[3]:
        detected = gene_info.get("n_samples_detected")
        total = gene_info.get("n_samples_total")
        if detected is not None and total is not None:
            st.metric("Detected In", f"{detected}/{total} samples")
        else:
            st.metric("Detected In", "N/A")

    # Show DEG significance info
    sig_in = gene_info.get("significant_in")
    if sig_in:
        st.caption(f"Significantly differentially expressed in: {', '.join(sig_in)}")

    st.divider()

    # ------------------------------------------------------------------
    # Two columns: expression bar chart + similar genes table
    # ------------------------------------------------------------------
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Expression Profile")
        fig_expr = _create_expression_bar(selected_gene, expression_df, samples_meta)
        if fig_expr is not None:
            st.plotly_chart(fig_expr, use_container_width=True)
        else:
            st.info(f"Gene '{selected_gene}' not found in expression matrix.")

    with right_col:
        st.subheader("Similar Genes")

        top_n = st.slider(
            "Number of similar genes",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            key="similar_genes_n",
        )

        similar_df = _find_similar_genes(selected_gene, similarity_data, top_n=top_n)

        if similar_df is not None and not similar_df.empty:
            # Display the table
            st.dataframe(
                similar_df,
                use_container_width=True,
                hide_index=True,
                height=400,
            )

            download_csv_button(
                similar_df,
                filename=f"similar_to_{selected_gene}.csv",
                label="Download Similar Genes (CSV)",
            )

            # Add-to-basket buttons
            st.caption("Add similar genes to basket")
            genes_for_basket = similar_df["gene"].tolist()
            n_cols = min(5, len(genes_for_basket))
            if n_cols > 0:
                basket_cols = st.columns(n_cols)
                for idx, gene in enumerate(genes_for_basket):
                    col = basket_cols[idx % n_cols]
                    with col:
                        if st.button(
                            f"+ {gene}",
                            key=f"gs_basket_{gene}",
                            help=f"Add {gene} to gene basket",
                        ):
                            add_to_basket(gene)
                            st.rerun()
        else:
            st.info(
                "No similarity data available. Ensure the pipeline computed "
                "cosine similarity and the gene exists in the similarity matrix."
            )

    # ------------------------------------------------------------------
    # Add searched gene to basket
    # ------------------------------------------------------------------
    st.divider()
    if st.button(
        f"Add {selected_gene} to basket",
        key=f"gs_add_main_{selected_gene}",
    ):
        add_to_basket(selected_gene)
        st.rerun()

    # ------------------------------------------------------------------
    # Gene basket panel at bottom
    # ------------------------------------------------------------------
    _render_basket_panel(expression_df, samples_meta)


def _render_basket_panel(
    expression_df: pd.DataFrame | None,
    samples_meta: pd.DataFrame | None,
) -> None:
    """Render the gene basket section at the bottom of the page."""
    # Sidebar basket widget
    render_basket()

    # Inline basket panel
    basket = get_basket()
    if not basket:
        return

    st.divider()
    st.subheader(f"Gene Basket ({len(basket)} genes)")

    # Show basket contents as chips
    st.write(", ".join(f"**{g}**" for g in basket))

    # Quick heatmap / expression overlay using basket_actions-like logic
    if expression_df is not None:
        available = [g for g in basket if g in expression_df.index]
        if available:
            with st.expander("Basket Expression Heatmap", expanded=False):
                import plotly.express as px  # noqa: F811

                subset = expression_df.loc[available]
                # Z-score
                means = subset.mean(axis=1)
                stds = subset.std(axis=1).replace(0, 1)
                z = subset.sub(means, axis=0).div(stds, axis=0)

                fig = px.imshow(
                    z,
                    labels=dict(x="Sample", y="Gene", color="Z-score"),
                    aspect="auto",
                    color_continuous_scale=DIVERGING_CMAP,
                    title="Basket Genes (z-score)",
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Basket Expression Table", expanded=False):
                display_df = expression_df.loc[available].copy()
                display_df.insert(0, "gene", display_df.index)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                download_csv_button(
                    expression_df.loc[available],
                    filename="basket_expression.csv",
                    label="Download Basket Expression (CSV)",
                )
        else:
            st.caption("None of the basket genes were found in the expression data.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__page__" or __name__ == "__main__":
    main()
else:
    main()
