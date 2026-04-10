"""Differential Expression Explorer -- NovoExplorer Streamlit page.

Interactive volcano plot, MA plot, sortable DEG table, gene search
highlighting, basket integration, and per-gene expression bar charts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the NovoExplorer package root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.persistence import (  # noqa: E402
    load_results,
    load_expression,
    load_deg,
    list_comparisons,
)
from plotting.volcano import create_volcano_plotly  # noqa: E402
from plotting.ma_plot import create_ma_plot_plotly  # noqa: E402
from app.components.filters import (  # noqa: E402
    comparison_selector,
    threshold_sliders,
    gene_search_box,
)
from app.components.download import (  # noqa: E402
    download_csv_button,
    download_figure_buttons,
)
from app.components.gene_basket import (  # noqa: E402
    init_basket,
    add_to_basket,
    get_basket,
    render_basket,
)
from app.components.shared import (  # noqa: E402
    get_data_path,
    check_data_path,
    create_expression_bar,
    fmt_pvalue,
    fmt_fc,
    table_height,
    render_empty_state,
)

# ---------------------------------------------------------------------------
# Data path helpers
# ---------------------------------------------------------------------------

_get_data_path = get_data_path


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading results...")
def _load_all_results(path: str) -> dict:
    """Load all pipeline results from the HDF5 file at *path*."""
    return load_results(path)


@st.cache_data(show_spinner="Loading expression matrix...")
def _load_expression(path: str, matrix_type: str = "tpm") -> pd.DataFrame | None:
    """Load a cached expression matrix (counts, TPM, or FPKM)."""
    return load_expression(path, matrix_type=matrix_type)


@st.cache_data(show_spinner="Loading DEG data...")
def _load_deg(path: str) -> dict | None:
    """Load cached DEG tables for all comparisons."""
    return load_deg(path)


@st.cache_data(show_spinner="Listing comparisons...")
def _list_comparisons(path: str) -> list[str]:
    """Return the list of available comparison names from the HDF5 file."""
    return list_comparisons(path)


_create_expression_bar = create_expression_bar


# ---------------------------------------------------------------------------
# Volcano with optional gene highlight
# ---------------------------------------------------------------------------


def _create_volcano_with_highlight(
    deg_df: pd.DataFrame,
    padj_threshold: float,
    log2fc_threshold: float,
    highlight_gene: str | None = None,
    title: str = "",
) -> go.Figure:
    """Volcano plot that optionally highlights a searched gene."""
    fig = create_volcano_plotly(
        deg_df,
        padj_threshold=padj_threshold,
        log2fc_threshold=log2fc_threshold,
        title=title,
        top_n_labels=10,
    )

    if highlight_gene and "gene_name" in deg_df.columns and highlight_gene in deg_df["gene_name"].values:
        row = deg_df[deg_df["gene_name"] == highlight_gene].iloc[0]
        neg_log10 = -np.log10(max(row["padj"], 1e-300))
        fig.add_trace(
            go.Scatter(
                x=[row["log2fc"]],
                y=[neg_log10],
                mode="markers+text",
                marker=dict(color="#FFD700", size=14, line=dict(color="black", width=2)),
                text=[highlight_gene],
                textposition="top center",
                textfont=dict(size=11, color="black"),
                name=f"Searched: {highlight_gene}",
                hovertemplate=(
                    f"<b>{highlight_gene}</b><br>"
                    f"log2FC: {row['log2fc']:.2f}<br>"
                    f"padj: {row['padj']:.2e}<extra></extra>"
                ),
            )
        )

    return fig


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the Differential Expression Explorer page.

    Provides an interactive volcano plot, MA plot, sortable DEG table,
    gene search highlighting, basket integration, and per-gene expression
    bar charts.  Users select a comparison and adjust thresholds via the
    sidebar.
    """
    st.title("Differential Expression Explorer")
    st.caption(
        "Visualize differential expression results with interactive volcano plots, "
        "sortable gene tables, and per-gene expression profiles. "
        "Use the sidebar to select a comparison and adjust significance thresholds."
    )
    init_basket()

    data_path = _get_data_path()
    if not check_data_path(data_path):
        return

    # Load data
    results = _load_all_results(data_path)
    expression_df = _load_expression(data_path, "tpm")
    deg_all = _load_deg(data_path)

    metadata = results.get("metadata") or {}
    samples_meta = metadata.get("samples")

    if not deg_all:
        render_empty_state("No differential expression results found", "Run the pipeline with DEG analysis to generate results.", "search")
        return

    # ------------------------------------------------------------------
    # Sidebar: comparison selector, thresholds, gene search, basket
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Filters")
        comparisons = _list_comparisons(data_path)
        if not comparisons:
            comparisons = sorted(deg_all.keys())
        selected_comparison = comparison_selector(comparisons, key="de_comparison")
        if selected_comparison is None:
            return

        padj_thresh, log2fc_thresh = threshold_sliders(key_prefix="de_")

        st.divider()

        # Collect gene names from the selected comparison
        deg_df = deg_all.get(selected_comparison)
        if deg_df is None:
            st.warning(f"Comparison '{selected_comparison}' not found in loaded data.")
            return

        gene_names = (
            deg_df["gene_name"].dropna().unique().tolist()
            if "gene_name" in deg_df.columns
            else deg_df.index.tolist()
        )
        searched_gene = gene_search_box(sorted(gene_names), key="de_gene_search")

    # Render basket in sidebar
    render_basket()

    # ------------------------------------------------------------------
    # Main area
    # ------------------------------------------------------------------

    # Ensure required columns exist
    if "padj" not in deg_df.columns or "log2fc" not in deg_df.columns:
        st.error(
            f"Comparison '{selected_comparison}' is missing required columns "
            "(`padj`, `log2fc`). Cannot render plots."
        )
        return

    # ---- Volcano plot ----
    st.subheader(f"Volcano Plot: {selected_comparison}")
    st.caption(
        f"padj threshold: {padj_thresh}, |log2FC| threshold: {log2fc_thresh}"
    )
    fig_volcano = _create_volcano_with_highlight(
        deg_df,
        padj_threshold=padj_thresh,
        log2fc_threshold=log2fc_thresh,
        highlight_gene=searched_gene,
        title=f"Volcano: {selected_comparison}",
    )
    st.plotly_chart(fig_volcano, width="stretch")
    download_figure_buttons(fig_volcano, f"volcano_{selected_comparison}")

    # ---- Searched gene spotlight ----
    if searched_gene:
        st.divider()
        st.subheader(f"Gene Spotlight: {searched_gene}")

        # Gene stats row
        if "gene_name" in deg_df.columns:
            gene_row = deg_df[deg_df["gene_name"] == searched_gene]
        else:
            gene_row = (
                deg_df.loc[[searched_gene]]
                if searched_gene in deg_df.index
                else pd.DataFrame()
            )

        if not gene_row.empty:
            row = gene_row.iloc[0]
            stat_cols = st.columns(4)
            if "log2fc" in row.index:
                stat_cols[0].metric("log2 Fold Change", fmt_fc(row["log2fc"]))
            if "padj" in row.index:
                stat_cols[1].metric("Adjusted p-value", fmt_pvalue(row["padj"]))
            if "basemean" in row.index:
                stat_cols[2].metric("Base Mean", f"{row['basemean']:.1f}")
            elif "baseMean" in row.index:
                stat_cols[2].metric("Base Mean", f"{row['baseMean']:.1f}")
            with stat_cols[3]:
                if st.button(
                    f"+ Add to basket",
                    key=f"de_add_searched_{searched_gene}",
                    help=f"Add {searched_gene} to gene basket",
                ):
                    add_to_basket(searched_gene)
                    st.rerun()
        else:
            st.info(f"**{searched_gene}** is not in the DEG results for this comparison.")

        # Expression bar chart
        fig_expr = _create_expression_bar(searched_gene, expression_df, samples_meta)
        if fig_expr is not None:
            st.plotly_chart(fig_expr, width="stretch")
        else:
            st.caption(f"Expression data not available for {searched_gene}.")

    # ---- MA plot (collapsible) ----
    with st.expander("MA Plot", expanded=False):
        # Use the imported create_ma_plot_plotly from plotting.ma_plot
        ma_df = deg_df.copy()
        if "baseMean" in ma_df.columns and "basemean" not in ma_df.columns:
            ma_df = ma_df.rename(columns={"baseMean": "basemean"})
        # Compute basemean from sample count columns when absent
        if "basemean" not in ma_df.columns and {"log2fc", "padj"}.issubset(set(ma_df.columns)):
            _meta_cols = {
                "gene_id", "gene_name", "log2fc", "pvalue", "padj", "basemean",
                "regulation", "gene_chr", "gene_start", "gene_end", "gene_strand",
                "gene_length", "gene_biotype", "gene_description", "tf_family",
            }
            count_cols = [c for c in ma_df.columns if c not in _meta_cols and pd.api.types.is_numeric_dtype(ma_df[c])]
            if count_cols:
                ma_df["basemean"] = ma_df[count_cols].mean(axis=1)
        if "basemean" in ma_df.columns:
            fig_ma = create_ma_plot_plotly(
                ma_df,
                padj_threshold=padj_thresh,
                log2fc_threshold=log2fc_thresh,
                title=f"MA Plot: {selected_comparison}",
            )
            st.plotly_chart(fig_ma, width="stretch")
            download_figure_buttons(fig_ma, f"ma_plot_{selected_comparison}")
        else:
            st.info(
                "MA plot requires a `basemean` column (mean expression). "
                "This column is not available for this comparison."
            )

    # ------------------------------------------------------------------
    # DEG table: filterable by regulation, sortable by padj
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("DEG Table")
    st.caption(
        "Genes meeting the significance thresholds, sorted by adjusted p-value. "
        "**log2fc** = log2 fold-change (positive = upregulated), "
        "**padj** = FDR-adjusted p-value, "
        "**basemean** = mean expression across all samples."
    )

    # Build display table with required columns
    display_cols = []
    for col in ("gene_name", "log2fc", "padj", "basemean", "baseMean", "regulation"):
        if col in deg_df.columns:
            display_cols.append(col)

    table_df = deg_df[display_cols].copy() if display_cols else deg_df.copy()

    # Compute regulation column BEFORE filtering so indices stay aligned
    if "regulation" not in table_df.columns and "padj" in table_df.columns and "log2fc" in table_df.columns:
        table_df["regulation"] = "ns"
        sig = table_df["padj"] < padj_thresh
        table_df.loc[sig & (table_df["log2fc"] >= log2fc_thresh), "regulation"] = "up"
        table_df.loc[sig & (table_df["log2fc"] <= -log2fc_thresh), "regulation"] = "down"

    # Filter to significant genes, then sort by padj
    if "padj" in table_df.columns and "log2fc" in table_df.columns:
        sig_mask = (table_df["padj"] < padj_thresh) & (
            table_df["log2fc"].abs() > log2fc_thresh
        )
        table_df = table_df.loc[sig_mask]

    if "padj" in table_df.columns:
        table_df = table_df.sort_values("padj", ascending=True)

    # DEG count badges
    if not table_df.empty and "regulation" in table_df.columns:
        n_up = int((table_df["regulation"] == "up").sum())
        n_down = int((table_df["regulation"] == "down").sum())
        badge_cols = st.columns(3)
        badge_cols[0].metric("Significant", len(table_df))
        badge_cols[1].metric("Up-regulated", n_up)
        badge_cols[2].metric("Down-regulated", n_down)

    # Regulation filter
    if "regulation" in table_df.columns:
        reg_options = ["all", "up", "down"]
        reg_filter = st.radio(
            "Filter by direction",
            options=reg_options,
            horizontal=True,
            key="de_reg_filter",
            help="'up' = higher expression in test vs control, 'down' = lower expression.",
        )
        if reg_filter != "all":
            table_df = table_df[table_df["regulation"] == reg_filter]

    st.caption(
        f"{len(table_df)} significant genes at padj < {padj_thresh}, "
        f"|log2FC| > {log2fc_thresh}"
    )

    if len(table_df) < 10 and len(table_df) > 0:
        st.warning(
            f"Only {len(table_df)} genes pass the current thresholds. "
            "Consider relaxing the significance criteria in the sidebar."
        )

    # Download CSV
    download_csv_button(
        table_df,
        filename=f"deg_{selected_comparison}.csv",
        label="Download DEG Table (CSV)",
    )

    # Display table with "Add to basket" buttons
    if not table_df.empty:
        st.dataframe(
            table_df,
            width="stretch",
            hide_index=True,
            height=table_height(len(table_df)),
        )

        # Add-to-basket section
        st.caption("Add genes to basket")
        gene_col = "gene_name" if "gene_name" in table_df.columns else None

        if gene_col:
            top_genes_for_basket = table_df[gene_col].head(50).tolist()
            basket_cols = st.columns(min(5, len(top_genes_for_basket) or 1))
            for idx, gene in enumerate(top_genes_for_basket):
                col = basket_cols[idx % len(basket_cols)]
                with col:
                    if st.button(
                        f"+ {gene}",
                        key=f"de_basket_{selected_comparison}_{gene}",
                        help=f"Add {gene} to gene basket",
                    ):
                        add_to_basket(gene)
                        st.rerun()
    else:
        st.info(
            f"No genes meet the current thresholds (padj < {padj_thresh}, "
            f"|log2FC| > {log2fc_thresh}). Try relaxing the thresholds in the sidebar."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

main()
