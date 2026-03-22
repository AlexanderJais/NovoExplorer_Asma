"""Gene basket using st.session_state for NovoView."""

import streamlit as st


_BASKET_KEY = "gene_basket"


def init_basket():
    """Initialize the gene basket in session state if not already present."""
    if _BASKET_KEY not in st.session_state:
        st.session_state[_BASKET_KEY] = []


def add_to_basket(gene_name):
    """Add a gene to the basket (no duplicates).

    Parameters
    ----------
    gene_name : str
        Gene name to add.
    """
    init_basket()
    if gene_name and gene_name not in st.session_state[_BASKET_KEY]:
        st.session_state[_BASKET_KEY].append(gene_name)
        st.toast(f"Added **{gene_name}** to basket")


def remove_from_basket(gene_name):
    """Remove a gene from the basket.

    Parameters
    ----------
    gene_name : str
        Gene name to remove.
    """
    init_basket()
    if gene_name in st.session_state[_BASKET_KEY]:
        st.session_state[_BASKET_KEY].remove(gene_name)


def clear_basket():
    """Clear all genes from the basket."""
    st.session_state[_BASKET_KEY] = []


def get_basket():
    """Return the current basket as a list.

    Returns
    -------
    list[str]
        List of gene names in the basket.
    """
    init_basket()
    return list(st.session_state[_BASKET_KEY])


def render_basket():
    """Display the gene basket as a sidebar panel with remove and clear buttons."""
    init_basket()
    basket_snapshot = list(st.session_state[_BASKET_KEY])

    with st.sidebar:
        st.markdown("### Gene Basket")

        if not basket_snapshot:
            st.caption("Add genes from any analysis page.")
            return

        st.caption(f"{len(basket_snapshot)} gene(s)")

        for idx, gene in enumerate(basket_snapshot):
            col_name, col_btn = st.columns([4, 1])
            with col_name:
                st.markdown(
                    f"<span style='font-size:0.85rem; font-weight:500;'>{gene}</span>",
                    unsafe_allow_html=True,
                )
            with col_btn:
                if st.button("x", key=f"basket_remove_{idx}_{gene}"):
                    remove_from_basket(gene)
                    st.rerun()

        if st.button("Clear all", key="basket_clear"):
            clear_basket()
            st.rerun()


def basket_actions(expression_df, sample_groups):
    """Render action buttons that create plots from basket genes.

    Provides buttons to generate a heatmap and an expression overlay
    plot using the genes currently in the basket.

    Parameters
    ----------
    expression_df : pandas.DataFrame
        Expression matrix with genes as rows (index) and samples as columns.
    sample_groups : dict[str, list[str]]
        Mapping of group name to list of sample/column names.

    Returns
    -------
    None
    """
    import plotly.express as px
    import pandas as pd

    init_basket()
    basket = get_basket()

    if not basket:
        st.info("Add genes to the basket to enable actions.")
        return

    # Filter expression data to basket genes that exist in the dataframe
    available_genes = [g for g in basket if g in expression_df.index]

    if not available_genes:
        st.warning("None of the basket genes were found in the expression data.")
        return

    st.subheader("Basket Actions")

    subset_df = expression_df.loc[available_genes]

    col_heatmap, col_overlay = st.columns(2)

    with col_heatmap:
        if st.button("Generate Heatmap", key="basket_heatmap"):
            fig = px.imshow(
                subset_df,
                labels=dict(x="Sample", y="Gene", color="Expression"),
                aspect="auto",
                color_continuous_scale="RdBu_r",
            )
            fig.update_layout(title="Basket Gene Heatmap")
            st.plotly_chart(fig, use_container_width=True)

    with col_overlay:
        if st.button("Expression Overlay", key="basket_overlay"):
            # Build a long-form dataframe with group annotations
            records = []
            for group_name, samples in sample_groups.items():
                cols_in_df = [s for s in samples if s in subset_df.columns]
                if cols_in_df:
                    melted = subset_df[cols_in_df].T
                    melted["Group"] = group_name
                    melted["Sample"] = melted.index
                    for gene in available_genes:
                        for _, row in melted.iterrows():
                            records.append(
                                {
                                    "Gene": gene,
                                    "Sample": row["Sample"],
                                    "Group": row["Group"],
                                    "Expression": row[gene],
                                }
                            )

            if records:
                plot_df = pd.DataFrame(records)
                fig = px.box(
                    plot_df,
                    x="Gene",
                    y="Expression",
                    color="Group",
                    title="Basket Gene Expression by Group",
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No matching samples found in expression data.")
