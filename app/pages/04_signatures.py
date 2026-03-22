"""Signature / Pathway Browser -- NovoView Streamlit page.

Displays enrichment dot plots, sortable enrichment tables, download
buttons, and cross-comparison signature analysis (Jaccard overlap
heatmap, core signatures, unique signatures).
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

from pipeline.persistence import load_enrichment, load_deg, load_signatures
from plotting.enrichment import create_enrichment_dotplot
from plotting.theme import apply_plotly_theme, get_nature_colorscale, WONG_PALETTE
from app.components.download import download_csv_button, download_figure_buttons
from app.components.shared import get_data_path, check_data_path, table_height

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Signatures & Pathways", layout="wide")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATABASE_LABELS = {
    "MSigDB_Hallmark_2020": "Hallmark",
    "GO_Biological_Process_2023": "GO BP",
    "GO_Biological_Process": "GO BP",
    "GO_Cellular_Component_2023": "GO CC",
    "GO_Cellular_Component": "GO CC",
    "GO_Molecular_Function_2023": "GO MF",
    "GO_Molecular_Function": "GO MF",
    "KEGG_2021_Human": "KEGG",
    "KEGG": "KEGG",
}

_DATABASE_DISPLAY_ORDER = ["Hallmark", "GO BP", "GO CC", "GO MF", "KEGG"]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

_get_data_path = get_data_path


@st.cache_data(show_spinner="Loading enrichment data...")
def _load_enrichment(path: str) -> dict | None:
    return load_enrichment(path)


@st.cache_data(show_spinner="Loading DEG data...")
def _load_deg(path: str) -> dict | None:
    return load_deg(path)


@st.cache_data(show_spinner="Loading signature data...")
def _load_signatures(path: str) -> dict | None:
    return load_signatures(path)


# ---------------------------------------------------------------------------
# Helpers -- build display-friendly enrichment table
# ---------------------------------------------------------------------------

def _normalize_enrichment_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names from various enrichment result formats to a
    standard schema: term_name, gene_count, gene_ratio, padj, genes."""
    out = pd.DataFrame()

    # Term name
    for col in ("term_name", "term", "Term", "Term_name"):
        if col in df.columns:
            out["term_name"] = df[col].values
            break
    if "term_name" not in out.columns and not df.empty:
        out["term_name"] = df.index.astype(str)

    # padj / FDR
    for col in ("padj", "fdr", "FDR", "Adjusted P-value", "FDR q-val"):
        if col in df.columns:
            out["padj"] = pd.to_numeric(df[col], errors="coerce").values
            break
    if "padj" not in out.columns:
        for col in ("pvalue", "P-value", "NOM p-val"):
            if col in df.columns:
                out["padj"] = pd.to_numeric(df[col], errors="coerce").values
                break

    # Gene count
    for col in ("gene_count", "Gene_count", "Count"):
        if col in df.columns:
            out["gene_count"] = pd.to_numeric(df[col], errors="coerce").values
            break
    if "gene_count" not in out.columns and "overlap" in df.columns:
        # overlap is often "3/50" format
        try:
            out["gene_count"] = df["overlap"].astype(str).str.split("/").str[0].astype(int).values
        except Exception:
            pass

    # Gene ratio
    for col in ("gene_ratio", "Gene_ratio", "GeneRatio"):
        if col in df.columns:
            out["gene_ratio"] = pd.to_numeric(df[col], errors="coerce").values
            break
    if "gene_ratio" not in out.columns and "overlap" in df.columns:
        try:
            parts = df["overlap"].astype(str).str.split("/")
            numerator = pd.to_numeric(parts.str[0], errors="coerce")
            denominator = pd.to_numeric(parts.str[1], errors="coerce")
            # Avoid division by zero: set ratio to NaN where denominator is 0
            ratio = numerator / denominator.where(denominator != 0, other=np.nan)
            out["gene_ratio"] = ratio.values
        except Exception:
            pass
    if "gene_ratio" not in out.columns and "gene_count" in out.columns:
        # Use gene_count directly as a proxy; avoid dividing by sum of counts
        # (which would be meaningless). Normalize to [0, 1] for display.
        max_count = out["gene_count"].max()
        if max_count > 0:
            out["gene_ratio"] = out["gene_count"] / max_count

    # Genes
    for col in ("genes", "Genes", "lead_genes", "Lead_genes"):
        if col in df.columns:
            out["genes"] = df[col].astype(str).values
            break

    # Fill missing columns with defaults
    if "gene_count" not in out.columns:
        out["gene_count"] = np.nan
    if "gene_ratio" not in out.columns:
        out["gene_ratio"] = np.nan
    if "genes" not in out.columns:
        out["genes"] = ""

    return out


def _collect_enrichment_for_comparison(
    enrichment_data: dict,
    comparison: str,
    database_filter: str | None = None,
    padj_max: float = 0.05,
) -> pd.DataFrame:
    """Gather enrichment results for a single comparison across databases,
    optionally filtering by database and padj threshold."""
    comp_data = enrichment_data.get(comparison, {})
    if not comp_data:
        return pd.DataFrame()

    frames = []
    for db_name, df in comp_data.items():
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            continue

        # If db_name maps to a known label, use it; else keep raw name
        label = _DATABASE_LABELS.get(db_name, db_name)

        if database_filter and database_filter != "All":
            if label != database_filter and db_name != database_filter:
                continue

        # Handle nested dict (gsea / ora_up / ora_down) or flat DataFrame
        if isinstance(df, dict):
            for sub_key, sub_df in df.items():
                if sub_df is None or (isinstance(sub_df, pd.DataFrame) and sub_df.empty):
                    continue
                normed = _normalize_enrichment_df(sub_df)
                normed["database"] = label
                normed["analysis"] = sub_key
                frames.append(normed)
        elif isinstance(df, pd.DataFrame):
            normed = _normalize_enrichment_df(df)
            normed["database"] = label
            normed["analysis"] = "enrichment"
            frames.append(normed)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Apply padj filter
    if "padj" in combined.columns:
        combined = combined[combined["padj"] <= padj_max].copy()

    # Sort by significance
    if "padj" in combined.columns:
        combined = combined.sort_values("padj", ascending=True).reset_index(drop=True)

    return combined


# ---------------------------------------------------------------------------
# Cross-comparison helpers
# ---------------------------------------------------------------------------

def _build_jaccard_matrix(enrichment_data: dict, padj_max: float = 0.05) -> pd.DataFrame:
    """Build a Jaccard similarity matrix of enriched term sets across comparisons."""
    comparisons = sorted(enrichment_data.keys())
    if len(comparisons) < 2:
        return pd.DataFrame()

    term_sets: dict[str, set] = {}
    for comp in comparisons:
        terms = set()
        comp_data = enrichment_data.get(comp, {})
        for db_name, df in comp_data.items():
            if isinstance(df, dict):
                for sub_df in df.values():
                    if sub_df is None or (isinstance(sub_df, pd.DataFrame) and sub_df.empty):
                        continue
                    normed = _normalize_enrichment_df(sub_df)
                    if "padj" in normed.columns and "term_name" in normed.columns:
                        sig = normed.loc[normed["padj"] <= padj_max, "term_name"]
                        terms.update(sig.dropna().unique())
            elif isinstance(df, pd.DataFrame) and not df.empty:
                normed = _normalize_enrichment_df(df)
                if "padj" in normed.columns and "term_name" in normed.columns:
                    sig = normed.loc[normed["padj"] <= padj_max, "term_name"]
                    terms.update(sig.dropna().unique())
        term_sets[comp] = terms

    n = len(comparisons)
    matrix = np.ones((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            a = term_sets[comparisons[i]]
            b = term_sets[comparisons[j]]
            union = a | b
            jaccard = len(a & b) / len(union) if union else 0.0
            matrix[i, j] = jaccard
            matrix[j, i] = jaccard

    return pd.DataFrame(matrix, index=comparisons, columns=comparisons)


def _find_core_signatures_from_enrichment(
    enrichment_data: dict,
    min_comparisons: int = 2,
    padj_max: float = 0.05,
) -> pd.DataFrame:
    """Find gene sets enriched in >= min_comparisons."""
    term_comps: dict[str, set] = {}
    for comp, comp_data in enrichment_data.items():
        for db_name, df in comp_data.items():
            dfs_to_check = []
            if isinstance(df, dict):
                dfs_to_check = [v for v in df.values() if isinstance(v, pd.DataFrame) and not v.empty]
            elif isinstance(df, pd.DataFrame) and not df.empty:
                dfs_to_check = [df]
            for sub_df in dfs_to_check:
                normed = _normalize_enrichment_df(sub_df)
                if "padj" in normed.columns and "term_name" in normed.columns:
                    sig_terms = normed.loc[normed["padj"] <= padj_max, "term_name"].dropna().unique()
                    for t in sig_terms:
                        term_comps.setdefault(t, set()).add(comp)

    rows = []
    for term, comps in term_comps.items():
        if len(comps) >= min_comparisons:
            rows.append({
                "term": term,
                "n_comparisons": len(comps),
                "comparisons": ", ".join(sorted(comps)),
            })

    if not rows:
        return pd.DataFrame(columns=["term", "n_comparisons", "comparisons"])

    return pd.DataFrame(rows).sort_values("n_comparisons", ascending=False).reset_index(drop=True)


def _find_unique_signatures_from_enrichment(
    enrichment_data: dict,
    padj_max: float = 0.05,
) -> pd.DataFrame:
    """Find gene sets enriched in exactly one comparison."""
    term_comps: dict[str, set] = {}
    for comp, comp_data in enrichment_data.items():
        for db_name, df in comp_data.items():
            dfs_to_check = []
            if isinstance(df, dict):
                dfs_to_check = [v for v in df.values() if isinstance(v, pd.DataFrame) and not v.empty]
            elif isinstance(df, pd.DataFrame) and not df.empty:
                dfs_to_check = [df]
            for sub_df in dfs_to_check:
                normed = _normalize_enrichment_df(sub_df)
                if "padj" in normed.columns and "term_name" in normed.columns:
                    sig_terms = normed.loc[normed["padj"] <= padj_max, "term_name"].dropna().unique()
                    for t in sig_terms:
                        term_comps.setdefault(t, set()).add(comp)

    rows = []
    for term, comps in term_comps.items():
        if len(comps) == 1:
            rows.append({
                "term": term,
                "comparison": sorted(comps)[0],
            })

    if not rows:
        return pd.DataFrame(columns=["term", "comparison"])

    return pd.DataFrame(rows).sort_values("comparison").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("Signatures & Pathways")
    st.caption(
        "Browse enrichment analysis results across comparisons and databases. "
        "Explore cross-comparison overlap to identify core and unique gene-set signatures."
    )

    data_path = _get_data_path()
    if not check_data_path(data_path):
        return

    enrichment_data = _load_enrichment(data_path)
    signatures_data = _load_signatures(data_path)

    if not enrichment_data:
        st.warning(
            "No enrichment results found in the results file. "
            "Run the enrichment analysis pipeline step first."
        )
        return

    comparisons = sorted(enrichment_data.keys())

    # ------------------------------------------------------------------
    # Sidebar controls
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Filters")

        comparison_options = ["All"] + comparisons
        selected_comparison = st.selectbox(
            "Comparison",
            options=comparison_options,
            key="sig_comparison",
        )

        # Detect available databases
        available_dbs = set()
        for comp_data in enrichment_data.values():
            for db_name in comp_data.keys():
                label = _DATABASE_LABELS.get(db_name, db_name)
                available_dbs.add(label)

        # Sort by preferred display order, then append any extras
        sorted_dbs = [d for d in _DATABASE_DISPLAY_ORDER if d in available_dbs]
        extras = sorted(available_dbs - set(sorted_dbs))
        sorted_dbs.extend(extras)

        database_options = ["All"] + sorted_dbs
        selected_database = st.selectbox(
            "Database",
            options=database_options,
            key="sig_database",
        )

        padj_threshold = st.slider(
            "Max adjusted p-value",
            min_value=0.001,
            max_value=0.10,
            value=0.05,
            step=0.001,
            format="%.3f",
            key="sig_padj",
            help=(
                "Only show pathways with adjusted p-value below this threshold. "
                "This filter applies to all sections on this page, including "
                "the overlap heatmap and core/unique signatures below."
            ),
        )

    # ------------------------------------------------------------------
    # Collect data for selected comparison(s)
    # ------------------------------------------------------------------
    if selected_comparison == "All":
        comps_to_show = comparisons
    else:
        comps_to_show = [selected_comparison]

    # ------------------------------------------------------------------
    # Section 1: Enrichment dot plot and table per comparison
    # ------------------------------------------------------------------
    st.header("Enrichment Results")
    st.caption(
        "Enrichment analysis identifies biological pathways and gene sets that are "
        "over-represented among your differentially expressed genes. "
        "Dot size = number of genes, color intensity = statistical significance."
    )

    for comp in comps_to_show:
        combined_df = _collect_enrichment_for_comparison(
            enrichment_data, comp,
            database_filter=selected_database if selected_database != "All" else None,
            padj_max=padj_threshold,
        )

        if combined_df.empty:
            st.info(f"No significant enrichment results for **{comp}** at padj <= {padj_threshold}.")
            continue

        with st.container(border=True):
            st.subheader(f"Comparison: {comp}")

            # --- Dot plot ---
            # Prepare data for create_enrichment_dotplot
            dotplot_df = combined_df.copy()
            if "padj" not in dotplot_df.columns:
                continue

            dotplot_df = dotplot_df.dropna(subset=["padj", "term_name"])
            if dotplot_df.empty:
                st.info("No terms to plot after filtering.")
                continue

            fig = create_enrichment_dotplot(
                dotplot_df,
                title=f"Enrichment: {comp}",
                max_terms=20,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"dotplot_{comp}")

            # Download buttons for the figure
            download_figure_buttons(fig, f"enrichment_dotplot_{comp}")

            # --- Sortable table ---
            st.markdown("#### Enrichment Table")
            display_cols = ["term_name", "gene_count", "gene_ratio", "padj", "genes"]
            table_df = combined_df[[c for c in display_cols if c in combined_df.columns]].copy()
            if "database" in combined_df.columns:
                table_df.insert(0, "database", combined_df["database"])

            # Format numeric columns
            if "padj" in table_df.columns:
                table_df["padj"] = table_df["padj"].map(
                    lambda x: f"{x:.2e}" if pd.notna(x) else ""
                )
            if "gene_ratio" in table_df.columns:
                table_df["gene_ratio"] = table_df["gene_ratio"].map(
                    lambda x: f"{x:.4f}" if pd.notna(x) else ""
                )

            st.dataframe(
                table_df,
                use_container_width=True,
                hide_index=True,
                height=table_height(len(table_df)),
            )

            # Download table as CSV
            download_csv_button(
                combined_df,
                filename=f"enrichment_{comp}.csv",
                label=f"Download enrichment table ({comp})",
            )

    # ------------------------------------------------------------------
    # Section 2: Cross-comparison analysis
    # ------------------------------------------------------------------
    if len(comparisons) < 2:
        return

    st.divider()
    st.header("Cross-Comparison Signature Analysis")

    # --- Jaccard overlap heatmap ---
    st.subheader("Signature Overlap (Jaccard Similarity)")
    st.caption(
        "How much do comparisons share the same enriched pathways? "
        "Jaccard index: 0 = completely different pathways, 1 = identical. "
        "High overlap suggests shared biological mechanisms across conditions."
    )

    # Try pre-computed overlap first, fall back to computing from enrichment
    overlap_matrix = None
    if signatures_data is not None:
        overlap_matrix = signatures_data.get("overlap_matrix")

    if overlap_matrix is None or overlap_matrix.empty:
        overlap_matrix = _build_jaccard_matrix(enrichment_data, padj_max=padj_threshold)

    if overlap_matrix is not None and not overlap_matrix.empty:
        fig_overlap = go.Figure(
            data=go.Heatmap(
                z=overlap_matrix.values,
                x=overlap_matrix.columns.tolist(),
                y=overlap_matrix.index.tolist(),
                colorscale=get_nature_colorscale("sequential"),
                zmin=0,
                zmax=1,
                colorbar=dict(title="Jaccard"),
                hovertemplate=(
                    "%{y} vs %{x}<br>"
                    "Jaccard: %{z:.3f}<extra></extra>"
                ),
            )
        )
        fig_overlap.update_layout(
            title="Signature Overlap Heatmap",
            xaxis=dict(tickangle=-45),
            height=max(400, len(overlap_matrix) * 60),
        )
        apply_plotly_theme(fig_overlap)
        st.plotly_chart(fig_overlap, use_container_width=True, key="jaccard_heatmap")
        download_figure_buttons(fig_overlap, "signature_overlap_heatmap")
    else:
        st.info("Not enough data to compute overlap heatmap.")

    # --- Core and unique signatures ---
    col_core, col_unique = st.columns(2)

    with col_core:
        st.subheader("Core Signatures")
        st.caption("Gene sets enriched across multiple comparisons.")

        # Try pre-computed core signatures, fall back to computing
        core_df = None
        if signatures_data is not None:
            core_df = signatures_data.get("core")

        if core_df is None or (isinstance(core_df, pd.DataFrame) and core_df.empty):
            min_n = st.slider(
                "Minimum comparisons",
                min_value=2,
                max_value=max(2, len(comparisons)),
                value=2,
                key="core_min_n",
            )
            core_df = _find_core_signatures_from_enrichment(
                enrichment_data,
                min_comparisons=min_n,
                padj_max=padj_threshold,
            )

        if core_df is not None and not core_df.empty:
            st.dataframe(core_df, use_container_width=True, hide_index=True)
            download_csv_button(core_df, "core_signatures.csv", "Download core signatures")
        else:
            st.info("No core signatures found at the current thresholds.")

    # --- Unique signatures ---
    with col_unique:
        st.subheader("Unique Signatures")
        st.caption("Gene sets enriched in exactly one comparison.")

        unique_df = None
        if signatures_data is not None:
            unique_df = signatures_data.get("unique")

        if unique_df is None or (isinstance(unique_df, pd.DataFrame) and unique_df.empty):
            unique_df = _find_unique_signatures_from_enrichment(
                enrichment_data, padj_max=padj_threshold,
            )

        if unique_df is not None and not unique_df.empty:
            st.dataframe(unique_df, use_container_width=True, hide_index=True)
            download_csv_button(unique_df, "unique_signatures.csv", "Download unique signatures")
        else:
            st.info("No unique signatures found at the current thresholds.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__page__" or __name__ == "__main__":
    main()
else:
    main()
