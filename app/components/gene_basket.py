"""Gene basket -- collect genes of interest across NovoView pages.

Stores a list of gene names in ``st.session_state`` under the key
``"gene_basket"``.  The basket persists across page navigations within
a single Streamlit session.

Public API
----------
init_basket, add_to_basket, remove_from_basket, clear_basket,
get_basket, render_basket.
"""

import html

import streamlit as st


_BASKET_KEY = "gene_basket"


def init_basket() -> None:
    """Initialise the gene basket in session state if not already present."""
    if _BASKET_KEY not in st.session_state:
        st.session_state[_BASKET_KEY] = []


def add_to_basket(gene_name: str) -> None:
    """Add a gene to the basket (no duplicates).

    Parameters
    ----------
    gene_name : str
        Gene name to add.  Ignored if empty or already present.
    """
    init_basket()
    if gene_name and gene_name not in st.session_state[_BASKET_KEY]:
        st.session_state[_BASKET_KEY].append(gene_name)
        st.toast(f"Added **{gene_name}** to basket")


def remove_from_basket(gene_name: str) -> None:
    """Remove a gene from the basket.

    Parameters
    ----------
    gene_name : str
        Gene name to remove.  No-op if not present.
    """
    init_basket()
    if gene_name in st.session_state[_BASKET_KEY]:
        st.session_state[_BASKET_KEY].remove(gene_name)


def clear_basket() -> None:
    """Remove all genes from the basket."""
    st.session_state[_BASKET_KEY] = []


def get_basket() -> list[str]:
    """Return a copy of the current basket.

    Returns
    -------
    list[str]
        Gene names currently in the basket.
    """
    init_basket()
    return list(st.session_state[_BASKET_KEY])


def basket_to_csv() -> str:
    """Return basket gene names as a CSV string."""
    basket = get_basket()
    return "\n".join(["gene_name"] + basket)


def import_to_basket(text: str) -> int:
    """Import gene names from newline/comma-separated text. Returns count added."""
    import re
    init_basket()
    genes = [g.strip() for g in re.split(r"[,\n]+", text) if g.strip()]
    added = 0
    for gene in genes:
        if gene and gene not in st.session_state[_BASKET_KEY]:
            st.session_state[_BASKET_KEY].append(gene)
            added += 1
    return added


def render_basket() -> None:
    """Render the gene basket as a sidebar panel with remove / clear buttons.

    Displays a per-gene remove button and a "Clear all" button.
    Triggers ``st.rerun()`` on mutation so the UI stays in sync.
    """
    init_basket()
    basket_snapshot = list(st.session_state[_BASKET_KEY])

    with st.sidebar:
        st.markdown("### Gene Basket")

        if not basket_snapshot:
            st.caption(
                "Collect genes of interest using the **+ Gene** buttons on any page. "
                "Your basket persists as you navigate. View basket heatmaps on the Gene Search page."
            )
            return

        st.caption(f"{len(basket_snapshot)} gene(s)")

        for idx, gene in enumerate(basket_snapshot):
            col_name, col_btn = st.columns([4, 1])
            with col_name:
                st.markdown(
                    f"<span style='font-size:0.85rem; font-weight:500;'>{html.escape(gene)}</span>",
                    unsafe_allow_html=True,
                )
            with col_btn:
                if st.button("x", key=f"basket_remove_{idx}_{gene}"):
                    remove_from_basket(gene)
                    st.rerun()

        if st.button("Clear all", key="basket_clear"):
            clear_basket()
            st.rerun()

        # Bulk actions
        with st.expander("Bulk actions", expanded=False):
            # Export
            csv_data = basket_to_csv()
            st.download_button(
                "Export basket (CSV)",
                data=csv_data.encode("utf-8"),
                file_name="gene_basket.csv",
                mime="text/csv",
                key="basket_export",
            )
            st.divider()
            # Import
            import_text = st.text_area(
                "Import genes (one per line or comma-separated)",
                key="basket_import_text",
                height=80,
                placeholder="TP53, BRCA1, MYC...",
            )
            if st.button("Import genes", key="basket_import_btn"):
                if import_text.strip():
                    n = import_to_basket(import_text)
                    st.toast(f"Added {n} gene(s) to basket")
                    st.rerun()
