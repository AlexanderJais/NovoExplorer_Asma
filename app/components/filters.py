"""Shared filter widgets for NovoView."""

import streamlit as st


def comparison_selector(comparisons_list, key="comparison"):
    """Selectbox for picking a comparison from a list.

    Parameters
    ----------
    comparisons_list : list[str]
        Available comparison names.
    key : str
        Unique Streamlit widget key.

    Returns
    -------
    str or None
        The selected comparison name, or None if the list is empty.
    """
    if not comparisons_list:
        st.warning("No comparisons available.")
        return None

    selected = st.selectbox(
        "Select comparison",
        options=comparisons_list,
        key=key,
        help="Each comparison tests differential expression between two conditions (e.g. Treatment vs Control).",
    )
    return selected


def threshold_sliders(default_padj=0.05, default_log2fc=1.0, key_prefix=""):
    """Two sliders for adjusted p-value and log2 fold-change thresholds.

    Parameters
    ----------
    default_padj : float
        Default adjusted p-value threshold.
    default_log2fc : float
        Default absolute log2 fold-change threshold.
    key_prefix : str
        Prefix for widget keys to avoid collisions when used multiple times.

    Returns
    -------
    tuple[float, float]
        (padj_threshold, log2fc_threshold)
    """
    padj = st.slider(
        "Adjusted p-value threshold",
        min_value=0.001,
        max_value=0.1,
        value=default_padj,
        step=0.001,
        format="%.3f",
        key=f"{key_prefix}padj_threshold",
        help=(
            "Genes with adjusted p-value below this threshold are considered "
            "statistically significant. Default 0.05 = 5% false discovery rate. "
            "Lower values are more stringent."
        ),
    )

    log2fc = st.slider(
        "|log2FC| threshold",
        min_value=0.0,
        max_value=3.0,
        value=default_log2fc,
        step=0.1,
        format="%.1f",
        key=f"{key_prefix}log2fc_threshold",
        help=(
            "Minimum absolute log2 fold-change for biological significance. "
            "1.0 = 2-fold change, 2.0 = 4-fold change. "
            "Set to 0 to see all statistically significant genes regardless of effect size."
        ),
    )

    return padj, log2fc


def database_selector(databases=None, key="database"):
    """Selectbox for choosing a gene set database.

    Parameters
    ----------
    databases : list[str] or None
        Available database names. Falls back to common defaults if None.
    key : str
        Unique Streamlit widget key.

    Returns
    -------
    str
        The selected database name.
    """
    if databases is None:
        databases = [
            "GO_Biological_Process",
            "GO_Molecular_Function",
            "GO_Cellular_Component",
            "KEGG",
            "Reactome",
            "MSigDB_Hallmark",
        ]

    selected = st.selectbox(
        "Gene set database",
        options=databases,
        key=key,
    )
    return selected


def gene_search_box(gene_names, key="gene_search"):
    """Text input with selectbox fallback for gene name searching.

    Provides a text input for typing a gene name. When the typed text
    matches one or more genes, a selectbox is shown with the filtered
    results so the user can pick the exact gene.

    Parameters
    ----------
    gene_names : list[str]
        All available gene names.
    key : str
        Unique Streamlit widget key.

    Returns
    -------
    str or None
        The selected gene name, or None if nothing is selected.
    """
    if not gene_names:
        st.warning("No gene names available.")
        return None

    sorted_genes = sorted(gene_names)

    query = st.text_input(
        "Search for a gene",
        key=f"{key}_text",
        placeholder="Type a gene name...",
        help="Start typing a gene symbol (e.g. TP53, BRCA1) to search. Matches partial names.",
    )

    if query:
        query_upper = query.strip().upper()
        matches = [g for g in sorted_genes if query_upper in g.upper()]

        if not matches:
            st.info("No genes match your search.")
            return None

        if len(matches) == 1:
            st.success(f"Match: **{matches[0]}**")
            return matches[0]

        selected = st.selectbox(
            f"Matching genes ({len(matches)})",
            options=matches,
            key=f"{key}_select",
        )
        return selected

    return None
