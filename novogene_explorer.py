"""Novogene RNA-Seq Explorer – Streamlit app for browsing raw Novogene deliveries.

Launch with:
    streamlit run novogene_explorer.py -- /path/to/data_folder

Or run without arguments and enter the path interactively.
"""

from __future__ import annotations

import io
import logging
import os
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure imports from the NovoExplorer package work
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# In-app log capture
# ---------------------------------------------------------------------------

class _SessionLogHandler(logging.Handler):
    """Logging handler that appends records to ``st.session_state["_log"]``."""

    _FMT = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self._FMT.format(record)
            # session_state may not be available during cache population on
            # a fresh Streamlit boot; fall back silently.
            log_list = st.session_state.setdefault("_log", [])
            log_list.append(msg)
        except Exception:
            pass


def _install_log_handler() -> None:
    """Attach the session-log handler to the ``pipeline`` logger hierarchy."""
    if "_log_handler_installed" in st.session_state:
        return
    handler = _SessionLogHandler()
    handler.setLevel(logging.DEBUG)
    # Attach to the root "pipeline" logger so every sub-module is captured
    for logger_name in ("pipeline.ingest", "pipeline.utils"):
        lg = logging.getLogger(logger_name)
        lg.addHandler(handler)
    st.session_state["_log_handler_installed"] = True


_install_log_handler()

# Ensure a log list always exists
if "_log" not in st.session_state:
    st.session_state["_log"] = []


from pipeline.ingest import (
    discover_novogene_structure,
    parse_deg_results,
    parse_enrichment_results,
    parse_sample_info,
    _is_container_dir,
    _iglob_files,
)
from pipeline.utils import (
    read_table_flexible,
    standardize_deg_columns,
    standardize_enrichment_columns,
)

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Novogene Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Color palette (colorblind-friendly, Wong et al.)
# ---------------------------------------------------------------------------
WONG = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#F0E442", "#56B4E9", "#E69F00", "#000000",
]
UP_COLOR = "#D55E00"
DOWN_COLOR = "#0072B2"
NS_COLOR = "#BBBBBB"

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Scanning folder structure…")
def load_structure(data_dir: str) -> dict:
    return discover_novogene_structure(data_dir)


@st.cache_data(show_spinner="Parsing DEG results…")
def load_deg(deg_dir: str | None) -> dict[str, pd.DataFrame]:
    return parse_deg_results(deg_dir)


@st.cache_data(show_spinner="Parsing enrichment results…")
def load_enrichment(enrichment_dir: str | None) -> dict[str, dict[str, pd.DataFrame]]:
    return parse_enrichment_results(enrichment_dir)


@st.cache_data(show_spinner="Reading diff_stat.xls…")
def load_diff_stat(deg_dir: str | None) -> pd.DataFrame | None:
    """Try to find and read diff_stat.xls from the DEG directory tree."""
    if deg_dir is None:
        return None
    deg_path = Path(deg_dir)
    # Search in deg_dir itself and one level of numbered containers
    for search_dir in [deg_path] + sorted(
        d for d in deg_path.iterdir() if d.is_dir() and _is_container_dir(d)
    ):
        candidates = _iglob_files(search_dir, ("diff_stat*",))
        if candidates:
            try:
                return read_table_flexible(candidates[0])
            except Exception:
                pass
    return None


@st.cache_data(show_spinner="Parsing sample info…")
def load_sample_info(path: str | None) -> pd.DataFrame | None:
    return parse_sample_info(path)


def _split_comparison(name: str) -> tuple[str, str]:
    """Split a comparison name into (group_a, group_b)."""
    parts = re.split(r"_vs_", name, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0], parts[1]
    parts = re.split(r"vs", name, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0], parts[1]
    return name, ""


def _all_gene_names(deg: dict[str, pd.DataFrame]) -> list[str]:
    """Collect all unique gene names across comparisons, sorted."""
    names: set[str] = set()
    for df in deg.values():
        if "gene_name" in df.columns:
            names.update(df["gene_name"].dropna().unique())
    return sorted(names)


# ---------------------------------------------------------------------------
# Sidebar: folder browser
# ---------------------------------------------------------------------------


def _list_subdirs(parent: Path) -> list[str]:
    """Return sorted subdirectory names under *parent*."""
    try:
        return sorted(
            d.name for d in parent.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
    except PermissionError:
        return []


def _looks_like_novogene(p: Path) -> bool:
    """Quick check: does this folder contain Differential/ or Enrichment/?"""
    for child in ("Differential", "differential", "Enrichment", "enrichment"):
        if (p / child).is_dir():
            return True
    return False


st.sidebar.title("Novogene Explorer")

# Determine initial path: CLI arg > session state > home directory
_cli_path = sys.argv[-1] if len(sys.argv) > 1 and Path(sys.argv[-1]).is_dir() else ""
if "data_dir" not in st.session_state:
    st.session_state["data_dir"] = _cli_path
if "browse_dir" not in st.session_state:
    st.session_state["browse_dir"] = (
        _cli_path if _cli_path else str(Path.home())
    )

# --- Folder browser ---
st.sidebar.subheader("Select data folder")


def _on_path_change():
    """Callback: user typed/pasted a new path in the text input."""
    st.session_state["browse_dir"] = st.session_state["_path_input"]


def _go_up():
    """Callback: navigate to parent directory."""
    st.session_state["browse_dir"] = str(Path(st.session_state["browse_dir"]).parent)
    st.session_state["_path_input"] = st.session_state["browse_dir"]


def _use_folder():
    """Callback: confirm current browse_dir as the data folder."""
    st.session_state["data_dir"] = st.session_state["browse_dir"]


def _on_subfolder_click(subfolder_name: str):
    """Callback: navigate into a subfolder."""
    new_path = str(Path(st.session_state["browse_dir"]) / subfolder_name)
    st.session_state["browse_dir"] = new_path
    st.session_state["_path_input"] = new_path


# Sync text input default with browse_dir
if "_path_input" not in st.session_state:
    st.session_state["_path_input"] = st.session_state["browse_dir"]

st.sidebar.text_input(
    "Folder path",
    key="_path_input",
    on_change=_on_path_change,
    help="Paste a full path and press Enter",
)

browse_path = Path(st.session_state["browse_dir"])

if browse_path.is_dir():
    # Show current path (may differ from text input during navigation)
    if str(browse_path) != st.session_state.get("_path_input", ""):
        st.sidebar.caption(f"📂 `{browse_path}`")

    # Action buttons
    col_up, col_use = st.sidebar.columns(2)
    with col_up:
        if browse_path.parent != browse_path:
            st.button("↑ Up", key="browse_up", on_click=_go_up, width="stretch")
    with col_use:
        st.button(
            "✓ Use this folder", key="browse_use", type="primary",
            on_click=_use_folder, width="stretch",
        )

    if _looks_like_novogene(browse_path):
        st.sidebar.success("Novogene data detected")

    # List subdirectories as clickable buttons
    subdirs = _list_subdirs(browse_path)
    if subdirs:
        st.sidebar.caption("Subfolders:")
        for d in subdirs:
            child = browse_path / d
            label = f"📊 {d}" if _looks_like_novogene(child) else f"📁 {d}"
            st.sidebar.button(
                label, key=f"_nav_{d}",
                on_click=_on_subfolder_click, args=(d,),
                width="stretch",
            )
else:
    if st.session_state["browse_dir"]:
        st.sidebar.warning("Path does not exist.")

st.sidebar.divider()

# Resolve selected data folder
data_dir = st.session_state.get("data_dir", "")
if not data_dir or not Path(data_dir).is_dir():
    st.title("Novogene RNA-Seq Explorer")
    st.info(
        "Navigate to your Novogene data folder using the sidebar browser, "
        "then click **Use this folder** or select a folder marked **[Novogene]**."
    )
    st.stop()

st.sidebar.success(f"**Loaded:** {Path(data_dir).name}")

# Load data
structure = load_structure(data_dir)
deg = load_deg(str(structure["deg_dir"]) if structure["deg_dir"] else None)
enrichment = load_enrichment(str(structure["enrichment_dir"]) if structure["enrichment_dir"] else None)
diff_stat = load_diff_stat(str(structure["deg_dir"]) if structure["deg_dir"] else None)
sample_info = load_sample_info(str(structure["sample_info_file"]) if structure["sample_info_file"] else None)
gene_names = _all_gene_names(deg)

# Sidebar metadata
st.sidebar.divider()
st.sidebar.metric("Comparisons", len(deg))
st.sidebar.metric("Genes tracked", f"{len(gene_names):,}")
if enrichment:
    st.sidebar.metric("Enrichment comparisons", len(enrichment))
if sample_info is not None and "group" in sample_info.columns:
    n_groups = sample_info["group"].nunique()
    st.sidebar.metric("Sample groups", n_groups)

# ---------------------------------------------------------------------------
# Sidebar: log panel
# ---------------------------------------------------------------------------

st.sidebar.divider()
with st.sidebar.expander("Log", expanded=False):
    log_lines = st.session_state.get("_log", [])
    if log_lines:
        full_log = "\n".join(log_lines)
        st.code(full_log, language="log")
        col_copy, col_clear = st.columns(2)
        with col_copy:
            st.download_button(
                "📋 Download log",
                data=full_log,
                file_name="novogene_explorer.log",
                mime="text/plain",
                key="download_log",
                width="stretch",
            )
        with col_clear:
            if st.button("Clear log", key="clear_log", width="stretch"):
                st.session_state["_log"] = []
                st.rerun()
    else:
        st.caption("No log messages yet.")


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

(tab_overview, tab_gene, tab_comparison, tab_enrichment,
 tab_ma, tab_venn, tab_ranked, tab_degsummary, tab_pathway, tab_export) = st.tabs([
    "Overview", "Gene Explorer", "Comparison Browser", "Enrichment",
    "MA Plot", "Venn / UpSet", "Ranked Genes", "DEG Summary", "Pathway Viewer", "Export",
])


# =========================================================================
# TAB 1: Overview
# =========================================================================
with tab_overview:
    st.header("Overview")

    if diff_stat is not None and not diff_stat.empty:
        st.subheader("DEG Summary (diff_stat.xls)")
        display_df = diff_stat.copy()
        # Add group columns if possible
        if "compare" in display_df.columns:
            groups = display_df["compare"].apply(lambda x: pd.Series(_split_comparison(x), index=["Group A", "Group B"]))
            display_df = pd.concat([groups, display_df], axis=1)
        st.dataframe(display_df, width="stretch", hide_index=True)

        # Bar chart of DEG counts
        if "compare" in diff_stat.columns and "up" in diff_stat.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=diff_stat["compare"], y=diff_stat["up"],
                name="Upregulated", marker_color=UP_COLOR,
            ))
            fig.add_trace(go.Bar(
                x=diff_stat["compare"], y=diff_stat.get("down", pd.Series(dtype=float)),
                name="Downregulated", marker_color=DOWN_COLOR,
            ))
            fig.update_layout(
                barmode="group", title="DEG Counts per Comparison",
                xaxis_title="Comparison", yaxis_title="Number of DEGs",
                xaxis_tickangle=-45, height=450,
            )
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("No diff_stat.xls found. Showing comparisons from folder structure.")

    # Always show the comparison list
    if deg:
        st.subheader(f"All Comparisons ({len(deg)})")
        comp_summary = []
        for comp_name, df in sorted(deg.items()):
            n_up = (df["regulation"] == "Up").sum() if "regulation" in df.columns else "–"
            n_down = (df["regulation"] == "Down").sum() if "regulation" in df.columns else "–"
            ga, gb = _split_comparison(comp_name)
            comp_summary.append({
                "Comparison": comp_name,
                "Group A": ga,
                "Group B": gb,
                "Total genes": len(df),
                "Up": n_up,
                "Down": n_down,
            })
        st.dataframe(pd.DataFrame(comp_summary), width="stretch", hide_index=True)


# =========================================================================
# TAB 2: Gene Explorer
# =========================================================================
with tab_gene:
    st.header("Gene Explorer")
    st.caption("Search for a gene and see its expression changes across all comparisons.")

    col_search, col_opts = st.columns([2, 1])
    with col_search:
        query = st.text_input(
            "Gene name", placeholder="e.g. CSF2RB, IL13RA1, STAT6…",
            key="gene_query",
        )
    with col_opts:
        padj_thresh = st.number_input("padj threshold", value=0.05, min_value=0.0, max_value=1.0, step=0.01, key="gene_padj")

    if query:
        query_upper = query.strip().upper()
        # Fuzzy match: find genes containing the query
        matches = [g for g in gene_names if query_upper in g.upper()]
        if not matches:
            st.warning(f"No genes matching **{query}** found.")
        else:
            if len(matches) > 1:
                selected_gene = st.selectbox(
                    f"Found {len(matches)} matches — select one:",
                    matches,
                    index=0 if query_upper in [m.upper() for m in matches] else 0,
                )
            else:
                selected_gene = matches[0]

            st.subheader(f"**{selected_gene}** across all comparisons")

            # Collect data across comparisons
            rows = []
            for comp_name, df in sorted(deg.items()):
                if "gene_name" not in df.columns:
                    continue
                hit = df[df["gene_name"].str.upper() == selected_gene.upper()]
                if hit.empty:
                    continue
                row = hit.iloc[0]
                ga, gb = _split_comparison(comp_name)
                rows.append({
                    "Comparison": comp_name,
                    "Group A": ga,
                    "Group B": gb,
                    "log2FC": row.get("log2fc", np.nan),
                    "padj": row.get("padj", np.nan),
                    "pvalue": row.get("pvalue", np.nan),
                    "basemean": row.get("basemean", np.nan),
                    "regulation": row.get("regulation", "–"),
                })

            if not rows:
                st.info(f"**{selected_gene}** was not found in any comparison.")
            else:
                gene_df = pd.DataFrame(rows)

                # Bar chart: log2FC across comparisons
                gene_df["significant"] = gene_df["padj"] < padj_thresh
                gene_df["color"] = gene_df.apply(
                    lambda r: UP_COLOR if r["significant"] and r["log2FC"] > 0
                    else (DOWN_COLOR if r["significant"] and r["log2FC"] < 0 else NS_COLOR),
                    axis=1,
                )
                gene_df["label"] = gene_df.apply(
                    lambda r: f"padj={r['padj']:.2e}" if pd.notna(r["padj"]) else "",
                    axis=1,
                )

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=gene_df["Comparison"],
                    y=gene_df["log2FC"],
                    marker_color=gene_df["color"],
                    text=gene_df["label"],
                    textposition="outside",
                    textfont_size=9,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "log2FC: %{y:.3f}<br>"
                        "%{text}<extra></extra>"
                    ),
                ))
                fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
                fig.update_layout(
                    title=f"{selected_gene} – log2 Fold Change",
                    xaxis_title="Comparison",
                    yaxis_title="log2 Fold Change",
                    xaxis_tickangle=-45,
                    height=max(400, 50 * len(gene_df)),
                    showlegend=False,
                )
                st.plotly_chart(fig, width="stretch")

                # Data table
                st.dataframe(
                    gene_df[["Comparison", "Group A", "Group B", "log2FC", "padj", "pvalue", "basemean", "regulation"]],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "log2FC": st.column_config.NumberColumn(format="%.3f"),
                        "padj": st.column_config.NumberColumn(format="%.2e"),
                        "pvalue": st.column_config.NumberColumn(format="%.2e"),
                        "basemean": st.column_config.NumberColumn(format="%.1f"),
                    },
                )

    # Multi-gene comparison
    st.divider()
    st.subheader("Multi-Gene Comparison")
    gene_input = st.text_area(
        "Enter multiple gene names (one per line or comma-separated)",
        placeholder="CSF2RB\nIL13RA1\nSTAT6",
        height=100,
        key="multi_gene_input",
    )

    if gene_input.strip():
        genes_requested = [g.strip() for g in re.split(r"[,\n]+", gene_input) if g.strip()]
        # Build a matrix: genes × comparisons → log2FC
        matrix_rows = []
        for gene in genes_requested:
            gene_upper = gene.upper()
            row_data: dict[str, float] = {}
            for comp_name, df in sorted(deg.items()):
                if "gene_name" not in df.columns:
                    continue
                hit = df[df["gene_name"].str.upper() == gene_upper]
                if not hit.empty:
                    row_data[comp_name] = hit.iloc[0].get("log2fc", np.nan)
            if row_data:
                row_data["gene"] = gene
                matrix_rows.append(row_data)

        if matrix_rows:
            matrix_df = pd.DataFrame(matrix_rows).set_index("gene")
            # Heatmap
            fig = px.imshow(
                matrix_df.values.astype(float),
                x=matrix_df.columns.tolist(),
                y=matrix_df.index.tolist(),
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                aspect="auto",
                labels=dict(x="Comparison", y="Gene", color="log2FC"),
                title="log2 Fold Change Heatmap",
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                height=max(350, 40 * len(matrix_df) + 200),
            )
            st.plotly_chart(fig, width="stretch")

            st.dataframe(matrix_df, width="stretch")
        else:
            st.warning("None of the entered genes were found in the data.")


# =========================================================================
# TAB 3: Comparison Browser
# =========================================================================
with tab_comparison:
    st.header("Comparison Browser")

    if not deg:
        st.warning("No DEG data loaded.")
    else:
        comp_names = sorted(deg.keys())
        selected_comp = st.selectbox("Select comparison", comp_names, key="comp_select")
        comp_df = deg[selected_comp]

        ga, gb = _split_comparison(selected_comp)
        st.caption(f"**{ga}** vs **{gb}** — {len(comp_df):,} genes")

        # Thresholds
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            padj_t = st.slider("padj threshold", 0.001, 0.1, 0.05, 0.005, key="comp_padj")
        with col_t2:
            fc_t = st.slider("|log2FC| threshold", 0.0, 5.0, 1.0, 0.25, key="comp_fc")
        with col_t3:
            top_n = st.slider("Label top N genes", 0, 30, 10, key="comp_topn")

        # Classify genes
        has_cols = {"log2fc", "padj"}.issubset(set(comp_df.columns))
        if has_cols:
            df_plot = comp_df.dropna(subset=["log2fc", "padj"]).copy()
            df_plot["neg_log10_padj"] = -np.log10(df_plot["padj"].clip(lower=1e-300))

            sig = df_plot["padj"] < padj_t
            up = sig & (df_plot["log2fc"] >= fc_t)
            down = sig & (df_plot["log2fc"] <= -fc_t)
            df_plot["category"] = "ns"
            df_plot.loc[up, "category"] = "up"
            df_plot.loc[down, "category"] = "down"

            n_up = up.sum()
            n_down = down.sum()

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Upregulated", f"{n_up:,}")
            col_m2.metric("Downregulated", f"{n_down:,}")
            col_m3.metric("Not significant", f"{(~sig).sum():,}")

            # Volcano plot
            color_map = {"up": UP_COLOR, "down": DOWN_COLOR, "ns": NS_COLOR}
            fig = go.Figure()

            for cat, color in color_map.items():
                mask = df_plot["category"] == cat
                subset = df_plot[mask]
                fig.add_trace(go.Scattergl(
                    x=subset["log2fc"],
                    y=subset["neg_log10_padj"],
                    mode="markers",
                    marker=dict(color=color, size=4, opacity=0.6),
                    name=cat.capitalize() if cat != "ns" else "NS",
                    text=subset.get("gene_name", pd.Series(dtype=str)),
                    hovertemplate="<b>%{text}</b><br>log2FC: %{x:.3f}<br>-log10(padj): %{y:.2f}<extra></extra>",
                ))

            # Label top genes
            if top_n > 0:
                sig_genes = df_plot[df_plot["category"] != "ns"].nsmallest(top_n, "padj")
                if "gene_name" in sig_genes.columns:
                    for _, row in sig_genes.iterrows():
                        fig.add_annotation(
                            x=row["log2fc"], y=row["neg_log10_padj"],
                            text=row["gene_name"], showarrow=True,
                            arrowhead=0, arrowcolor="#555", arrowwidth=1,
                            ax=0, ay=-22, font=dict(size=13),
                        )

            fig.add_vline(x=fc_t, line_dash="dash", line_color="gray", line_width=0.8)
            fig.add_vline(x=-fc_t, line_dash="dash", line_color="gray", line_width=0.8)
            fig.add_hline(y=-np.log10(padj_t), line_dash="dash", line_color="gray", line_width=0.8)
            fig.update_layout(
                title=dict(text=f"Volcano Plot — {selected_comp}", font=dict(size=18)),
                xaxis_title="log2 Fold Change",
                yaxis_title="-log10(padj)",
                height=550,
                font=dict(size=14),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=13)),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("DEG table missing required columns (log2fc, padj).")

        # Filterable table
        st.subheader("DEG Table")
        show_sig_only = st.checkbox("Show significant only", value=False, key="comp_sig_only")
        gene_filter = st.text_input("Filter by gene name", key="comp_gene_filter")

        display = comp_df.copy()
        if show_sig_only and has_cols:
            display = display[(display["padj"] < padj_t) & (display["log2fc"].abs() >= fc_t)]
        if gene_filter and "gene_name" in display.columns:
            display = display[display["gene_name"].str.contains(gene_filter, case=False, na=False)]

        st.dataframe(display, width="stretch", hide_index=True, height=500)


# =========================================================================
# TAB 4: Enrichment
# =========================================================================
with tab_enrichment:
    st.header("Enrichment Analysis")

    if not enrichment:
        st.warning("No enrichment data loaded. Check that the Enrichment/ folder exists.")
    else:
        enrich_comps = sorted(enrichment.keys())
        selected_enrich_comp = st.selectbox(
            "Select comparison", enrich_comps, key="enrich_comp_select",
        )
        comp_enrich = enrichment[selected_enrich_comp]
        available_dbs = sorted(comp_enrich.keys())

        if not available_dbs:
            st.info("No enrichment databases found for this comparison.")
        else:
            selected_db = st.radio("Database", available_dbs, horizontal=True, key="enrich_db")
            enrich_df = comp_enrich[selected_db].copy()

            # Standardize columns for display
            enrich_df = standardize_enrichment_columns(enrich_df)

            # Thresholds
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                enrich_padj = st.slider("padj threshold", 0.001, 0.5, 0.05, 0.005, key="enrich_padj")
            with col_e2:
                max_terms = st.slider("Max terms to show", 5, 50, 20, key="enrich_max_terms")

            # Filter significant
            if "padj" in enrich_df.columns:
                sig_df = enrich_df[enrich_df["padj"] < enrich_padj].copy()
            else:
                sig_df = enrich_df.copy()

            st.caption(f"{len(sig_df)} significant terms (padj < {enrich_padj})")

            if sig_df.empty:
                st.info("No significant enrichment terms at this threshold.")
            else:
                # Dot plot
                sort_col = "padj" if "padj" in sig_df.columns else "pvalue" if "pvalue" in sig_df.columns else None
                if sort_col is not None:
                    plot_df = sig_df.nsmallest(max_terms, sort_col).copy()
                else:
                    plot_df = sig_df.head(max_terms).copy()
                if "padj" in plot_df.columns:
                    plot_df["neg_log10_padj"] = -np.log10(plot_df["padj"].clip(lower=1e-300))
                elif "pvalue" in plot_df.columns:
                    plot_df["neg_log10_padj"] = -np.log10(plot_df["pvalue"].clip(lower=1e-300))
                else:
                    plot_df["neg_log10_padj"] = 1.0

                # Determine term label column
                term_col = "term_name" if "term_name" in plot_df.columns else (
                    "Description" if "Description" in plot_df.columns else plot_df.columns[0]
                )
                # Truncate long names
                plot_df["_label"] = plot_df[term_col].str[:60]

                # Gene count for dot size
                has_count = "gene_count" in plot_df.columns
                size_vals = plot_df["gene_count"] if has_count else pd.Series(8, index=plot_df.index)

                # Gene ratio for x-axis
                has_ratio = "gene_ratio" in plot_df.columns
                if has_ratio:
                    # Parse fraction strings like "5/100"
                    def _parse_ratio(val):
                        if isinstance(val, str) and "/" in val:
                            parts = val.split("/")
                            try:
                                return float(parts[0]) / float(parts[1])
                            except (ValueError, ZeroDivisionError):
                                return np.nan
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            return np.nan
                    plot_df["_ratio"] = plot_df["gene_ratio"].apply(_parse_ratio)
                    x_vals = plot_df["_ratio"]
                    x_label = "Gene Ratio"
                else:
                    x_vals = plot_df["neg_log10_padj"]
                    x_label = "-log10(padj)"

                # Sort by significance (most significant at top)
                plot_df = plot_df.sort_values("neg_log10_padj", ascending=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_vals.loc[plot_df.index],
                    y=plot_df["_label"],
                    mode="markers",
                    marker=dict(
                        size=size_vals.loc[plot_df.index].fillna(8).clip(lower=5, upper=30),
                        color=plot_df["neg_log10_padj"],
                        colorscale="Viridis",
                        colorbar=dict(title="-log10(padj)"),
                        line=dict(width=0.5, color="#333"),
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        f"{x_label}: " + "%{x:.4f}<br>"
                        "-log10(padj): %{marker.color:.2f}<extra></extra>"
                    ),
                ))
                fig.update_layout(
                    title=f"{selected_db} Enrichment — {selected_enrich_comp}",
                    xaxis_title=x_label,
                    yaxis_title="",
                    height=max(400, 25 * len(plot_df) + 150),
                    margin=dict(l=300),
                )
                st.plotly_chart(fig, width="stretch")

                # Bar plot
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=plot_df["neg_log10_padj"],
                    y=plot_df["_label"],
                    orientation="h",
                    marker_color=plot_df["neg_log10_padj"],
                    marker_colorscale="Viridis",
                    hovertemplate="<b>%{y}</b><br>-log10(padj): %{x:.2f}<extra></extra>",
                ))
                fig_bar.update_layout(
                    title=f"{selected_db} Enrichment Bar Plot",
                    xaxis_title="-log10(padj)",
                    yaxis_title="",
                    height=max(400, 25 * len(plot_df) + 150),
                    margin=dict(l=300),
                )
                st.plotly_chart(fig_bar, width="stretch")

            # Full table
            st.subheader(f"{selected_db} Enrichment Table")
            term_filter = st.text_input("Filter terms", key="enrich_term_filter")
            display_enrich = sig_df if st.checkbox("Significant only", value=True, key="enrich_sig_table") else enrich_df
            if term_filter:
                text_cols = display_enrich.select_dtypes(include="object").columns
                mask = pd.Series(False, index=display_enrich.index)
                for col in text_cols:
                    mask |= display_enrich[col].str.contains(term_filter, case=False, na=False)
                display_enrich = display_enrich[mask]
            st.dataframe(display_enrich, width="stretch", hide_index=True, height=500)

        # Cross-comparison enrichment view
        if len(enrich_comps) > 1:
            st.divider()
            st.subheader("Compare Enrichment Across Conditions")
            st.caption("Find a KEGG pathway or GO term across all comparisons.")
            term_query = st.text_input("Search term/pathway", key="enrich_cross_query")

            if term_query:
                cross_rows = []
                for comp_name, comp_data in sorted(enrichment.items()):
                    for db_name, db_df in comp_data.items():
                        edf = standardize_enrichment_columns(db_df.copy())
                        term_col = "term_name" if "term_name" in edf.columns else (
                            "Description" if "Description" in edf.columns else None
                        )
                        if term_col is None:
                            continue
                        matches = edf[edf[term_col].str.contains(term_query, case=False, na=False)]
                        for _, row in matches.iterrows():
                            cross_rows.append({
                                "Comparison": comp_name,
                                "Database": db_name,
                                "Term": row.get(term_col, ""),
                                "padj": row.get("padj", np.nan),
                                "Gene Count": row.get("gene_count", ""),
                                "Gene Ratio": row.get("gene_ratio", ""),
                            })

                if cross_rows:
                    cross_df = pd.DataFrame(cross_rows)
                    st.dataframe(cross_df, width="stretch", hide_index=True)

                    # Heatmap of padj across comparisons
                    if len(cross_df["Term"].unique()) <= 30:
                        pivot = cross_df.pivot_table(
                            index="Term", columns="Comparison",
                            values="padj", aggfunc="first",
                        )
                        fig = px.imshow(
                            -np.log10(pivot.values.astype(float).clip(min=1e-300)),
                            x=pivot.columns.tolist(),
                            y=pivot.index.tolist(),
                            color_continuous_scale="Viridis",
                            aspect="auto",
                            labels=dict(color="-log10(padj)"),
                            title=f"Enrichment Significance: '{term_query}'",
                        )
                        fig.update_layout(xaxis_tickangle=-45, height=max(350, 35 * len(pivot)))
                        st.plotly_chart(fig, width="stretch")
                else:
                    st.info(f"No enrichment terms matching **{term_query}** found.")


# =========================================================================
# TAB 5: MA Plot
# =========================================================================
with tab_ma:
    st.header("MA Plot")
    st.caption("Mean expression (baseMean) vs log2 fold change — highlights expression-dependent changes.")

    if not deg:
        st.warning("No DEG data loaded.")
    else:
        ma_comp = st.selectbox("Select comparison", sorted(deg.keys()), key="ma_comp_select")
        ma_df = deg[ma_comp]

        has_ma_cols = {"log2fc", "padj"}.issubset(set(ma_df.columns)) and "basemean" in ma_df.columns
        if not has_ma_cols:
            st.warning("DEG table missing required columns (log2fc, padj, basemean).")
        else:
            col_ma1, col_ma2 = st.columns(2)
            with col_ma1:
                ma_padj = st.slider("padj threshold", 0.001, 0.1, 0.05, 0.005, key="ma_padj")
            with col_ma2:
                ma_fc = st.slider("|log2FC| threshold", 0.0, 5.0, 1.0, 0.25, key="ma_fc")

            ma_plot = ma_df.dropna(subset=["log2fc", "padj", "basemean"]).copy()
            ma_plot["log10_basemean"] = np.log10(ma_plot["basemean"].clip(lower=1e-1))

            sig = ma_plot["padj"] < ma_padj
            up = sig & (ma_plot["log2fc"] >= ma_fc)
            down = sig & (ma_plot["log2fc"] <= -ma_fc)
            ma_plot["category"] = "ns"
            ma_plot.loc[up, "category"] = "up"
            ma_plot.loc[down, "category"] = "down"

            fig = go.Figure()
            for cat, color in {"up": UP_COLOR, "down": DOWN_COLOR, "ns": NS_COLOR}.items():
                mask = ma_plot["category"] == cat
                subset = ma_plot[mask]
                fig.add_trace(go.Scattergl(
                    x=subset["log10_basemean"],
                    y=subset["log2fc"],
                    mode="markers",
                    marker=dict(color=color, size=4, opacity=0.5),
                    name=cat.capitalize() if cat != "ns" else "NS",
                    text=subset.get("gene_name", pd.Series(dtype=str)),
                    hovertemplate="<b>%{text}</b><br>log10(baseMean): %{x:.2f}<br>log2FC: %{y:.3f}<extra></extra>",
                ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8)
            fig.add_hline(y=ma_fc, line_dash="dot", line_color="gray", line_width=0.6)
            fig.add_hline(y=-ma_fc, line_dash="dot", line_color="gray", line_width=0.6)
            fig.update_layout(
                title=dict(text=f"MA Plot — {ma_comp}", font=dict(size=18)),
                xaxis_title="log10(baseMean)",
                yaxis_title="log2 Fold Change",
                height=550,
                font=dict(size=14),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=13)),
            )
            st.plotly_chart(fig, width="stretch")

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Upregulated", f"{up.sum():,}")
            col_m2.metric("Downregulated", f"{down.sum():,}")
            col_m3.metric("Not significant", f"{(~sig).sum():,}")


# =========================================================================
# TAB 6: Venn / UpSet
# =========================================================================
with tab_venn:
    st.header("Venn / UpSet Diagram")
    st.caption("Compare significant DEG overlap across comparisons.")

    if not deg:
        st.warning("No DEG data loaded.")
    elif len(deg) < 2:
        st.info("Need at least 2 comparisons for overlap analysis.")
    else:
        venn_comps = sorted(deg.keys())
        selected_venn = st.multiselect(
            "Select comparisons (2–5 recommended)",
            venn_comps, default=venn_comps[:min(3, len(venn_comps))],
            key="venn_comps",
        )

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            venn_padj = st.slider("padj threshold", 0.001, 0.1, 0.05, 0.005, key="venn_padj")
        with col_v2:
            venn_fc = st.slider("|log2FC| threshold", 0.0, 5.0, 1.0, 0.25, key="venn_fc")

        venn_direction = st.radio(
            "Include", ["All significant", "Upregulated only", "Downregulated only"],
            horizontal=True, key="venn_dir",
        )

        if len(selected_venn) >= 2:
            # Build gene sets per comparison
            gene_sets: dict[str, set[str]] = {}
            for comp in selected_venn:
                df = deg[comp]
                if not {"log2fc", "padj", "gene_name"}.issubset(set(df.columns)):
                    continue
                sig_mask = df["padj"] < venn_padj
                if venn_direction == "All significant":
                    sig_mask &= df["log2fc"].abs() >= venn_fc
                elif venn_direction == "Upregulated only":
                    sig_mask &= df["log2fc"] >= venn_fc
                else:
                    sig_mask &= df["log2fc"] <= -venn_fc
                gene_sets[comp] = set(df.loc[sig_mask, "gene_name"].dropna())

            if not gene_sets:
                st.warning("No valid comparisons with the required columns.")
            else:
                # Compute all intersections for UpSet-style display
                set_names = list(gene_sets.keys())
                all_genes = set().union(*gene_sets.values())

                # Build membership matrix
                membership = {}
                for g in all_genes:
                    key = tuple(g in gene_sets[s] for s in set_names)
                    membership.setdefault(key, []).append(g)

                # Sort by intersection size
                intersections = sorted(membership.items(), key=lambda x: len(x[1]), reverse=True)

                # UpSet-style bar chart
                bar_labels = []
                bar_sizes = []
                bar_genes_list = []
                for key, genes in intersections:
                    label = " ∩ ".join(s for s, m in zip(set_names, key) if m)
                    bar_labels.append(label)
                    bar_sizes.append(len(genes))
                    bar_genes_list.append(genes)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=bar_labels[:20],
                    y=bar_sizes[:20],
                    marker_color=WONG[0],
                    hovertemplate="<b>%{x}</b><br>%{y} genes<extra></extra>",
                ))
                fig.update_layout(
                    title=dict(text="Intersection Sizes (UpSet-style)", font=dict(size=18)),
                    xaxis_title="", yaxis_title="Gene Count",
                    xaxis_tickangle=-45, height=500,
                    font=dict(size=14),
                )
                st.plotly_chart(fig, width="stretch")

                # Set size summary
                st.subheader("Set Sizes")
                size_df = pd.DataFrame([
                    {"Comparison": s, "Significant genes": len(g)}
                    for s, g in gene_sets.items()
                ])
                st.dataframe(size_df, width="stretch", hide_index=True)

                # Intersection detail
                st.subheader("Intersection Details")
                detail_rows = []
                for key, genes in intersections:
                    label = " ∩ ".join(s for s, m in zip(set_names, key) if m)
                    only_in = " only" if sum(key) == 1 else ""
                    for g in sorted(genes):
                        detail_rows.append({"Intersection": label + only_in, "Gene": g})
                detail_df = pd.DataFrame(detail_rows)
                st.dataframe(detail_df, width="stretch", hide_index=True, height=400)
        else:
            st.info("Select at least 2 comparisons above.")


# =========================================================================
# TAB 7: Ranked Genes
# =========================================================================
with tab_ranked:
    st.header("Ranked Gene List")
    st.caption("All genes ranked by fold change or significance, with cumulative enrichment view.")

    if not deg:
        st.warning("No DEG data loaded.")
    else:
        rank_comp = st.selectbox("Select comparison", sorted(deg.keys()), key="rank_comp_select")
        rank_df = deg[rank_comp].copy()

        if not {"log2fc", "padj", "gene_name"}.issubset(set(rank_df.columns)):
            st.warning("DEG table missing required columns.")
        else:
            rank_by = st.radio(
                "Rank by", ["log2FC (descending)", "padj (ascending)", "Absolute log2FC (descending)"],
                horizontal=True, key="rank_by",
            )
            rank_padj = st.slider("padj threshold for highlighting", 0.001, 0.1, 0.05, 0.005, key="rank_padj")

            ranked = rank_df.dropna(subset=["log2fc", "padj"]).copy()
            if rank_by == "log2FC (descending)":
                ranked = ranked.sort_values("log2fc", ascending=False)
            elif rank_by == "padj (ascending)":
                ranked = ranked.sort_values("padj", ascending=True)
            else:
                ranked = ranked.sort_values("log2fc", key=lambda x: x.abs(), ascending=False)

            ranked = ranked.reset_index(drop=True)
            ranked["rank"] = range(1, len(ranked) + 1)
            ranked["significant"] = ranked["padj"] < rank_padj

            # Waterfall-style plot: rank vs log2FC
            fig = go.Figure()
            sig_mask = ranked["significant"]
            for mask, color, name in [(sig_mask, UP_COLOR, "Significant"), (~sig_mask, NS_COLOR, "NS")]:
                subset = ranked[mask]
                fig.add_trace(go.Scattergl(
                    x=subset["rank"],
                    y=subset["log2fc"],
                    mode="markers",
                    marker=dict(color=color, size=3, opacity=0.6),
                    name=name,
                    text=subset["gene_name"],
                    hovertemplate="<b>%{text}</b><br>Rank: %{x}<br>log2FC: %{y:.3f}<extra></extra>",
                ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8)
            fig.update_layout(
                title=dict(text=f"Ranked Genes — {rank_comp}", font=dict(size=18)),
                xaxis_title="Rank",
                yaxis_title="log2 Fold Change",
                height=500,
                font=dict(size=14),
            )
            st.plotly_chart(fig, width="stretch")

            # Stats
            n_sig = sig_mask.sum()
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Total genes", f"{len(ranked):,}")
            col_r2.metric("Significant", f"{n_sig:,}")
            col_r3.metric("% Significant", f"{100 * n_sig / len(ranked):.1f}%" if len(ranked) > 0 else "0.0%")

            # Top / bottom genes tables
            col_top, col_bot = st.columns(2)
            n_show = st.slider("Show top/bottom N genes", 10, 100, 25, key="rank_n_show")
            with col_top:
                st.subheader(f"Top {n_show} upregulated")
                top_up = rank_df.dropna(subset=["log2fc", "padj"]).nlargest(n_show, "log2fc")
                display_cols = [c for c in ["gene_name", "log2fc", "padj", "basemean"] if c in top_up.columns]
                st.dataframe(top_up[display_cols], width="stretch", hide_index=True)
            with col_bot:
                st.subheader(f"Top {n_show} downregulated")
                top_down = rank_df.dropna(subset=["log2fc", "padj"]).nsmallest(n_show, "log2fc")
                st.dataframe(top_down[display_cols], width="stretch", hide_index=True)


# =========================================================================
# TAB 8: DEG Summary Table
# =========================================================================
with tab_degsummary:
    st.header("DEG Summary Table")
    st.caption("Side-by-side log2FC and padj for each gene across all comparisons.")

    if not deg:
        st.warning("No DEG data loaded.")
    else:
        summary_padj = st.slider(
            "padj threshold (highlight significant)", 0.001, 0.1, 0.05, 0.005,
            key="summary_padj",
        )
        summary_fc = st.slider(
            "|log2FC| threshold", 0.0, 5.0, 1.0, 0.25, key="summary_fc",
        )
        show_mode = st.radio(
            "Show", ["All genes", "Significant in at least 1 comparison", "Significant in all comparisons"],
            horizontal=True, key="summary_mode",
        )

        # Build wide matrix
        all_comps = sorted(deg.keys())
        fc_frames = []
        padj_frames = []
        for comp in all_comps:
            df = deg[comp]
            if "gene_name" not in df.columns:
                continue
            sub = df.drop_duplicates(subset="gene_name").set_index("gene_name")
            if "log2fc" in sub.columns:
                fc_frames.append(sub[["log2fc"]].rename(columns={"log2fc": f"{comp}|log2FC"}))
            if "padj" in sub.columns:
                padj_frames.append(sub[["padj"]].rename(columns={"padj": f"{comp}|padj"}))

        if not fc_frames:
            st.warning("No gene-level data available.")
        else:
            wide_fc = pd.concat(fc_frames, axis=1)
            wide_padj = pd.concat(padj_frames, axis=1) if padj_frames else pd.DataFrame(index=wide_fc.index)
            wide = pd.concat([wide_fc, wide_padj], axis=1)

            # Filtering
            if show_mode != "All genes":
                padj_cols = {c.rsplit("|", 1)[0]: c for c in wide.columns if c.endswith("|padj")}
                fc_cols = {c.rsplit("|", 1)[0]: c for c in wide.columns if c.endswith("|log2FC")}
                sig_per_comp = pd.DataFrame(index=wide.index)
                for comp_key in padj_cols.keys() & fc_cols.keys():
                    pc, fc = padj_cols[comp_key], fc_cols[comp_key]
                    sig_per_comp[pc] = (wide[pc] < summary_padj) & (wide[fc].abs() >= summary_fc)
                if sig_per_comp.empty:
                    wide = wide.iloc[0:0]  # no matched pairs, show nothing
                elif show_mode == "Significant in at least 1 comparison":
                    mask = sig_per_comp.any(axis=1)
                    wide = wide[mask]
                else:
                    mask = sig_per_comp.all(axis=1)
                    wide = wide[mask]

            # Reorder columns: alternate FC and padj per comparison
            ordered_cols = []
            for comp in all_comps:
                fc_col = f"{comp}|log2FC"
                padj_col = f"{comp}|padj"
                if fc_col in wide.columns:
                    ordered_cols.append(fc_col)
                if padj_col in wide.columns:
                    ordered_cols.append(padj_col)
            wide = wide[[c for c in ordered_cols if c in wide.columns]]

            st.caption(f"{len(wide):,} genes shown")

            gene_search = st.text_input("Filter by gene name", key="summary_gene_filter")
            if gene_search:
                wide = wide[wide.index.str.contains(gene_search, case=False, na=False)]

            st.dataframe(wide, width="stretch", height=600)

            # Download as Excel
            buffer = io.BytesIO()
            wide.to_excel(buffer, engine="openpyxl")
            st.download_button(
                "📥 Download as Excel",
                data=buffer.getvalue(),
                file_name="deg_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="summary_download",
            )


# =========================================================================
# TAB 9: Pathway Viewer
# =========================================================================
with tab_pathway:
    st.header("Pathway Viewer")
    st.caption("Select an enriched term and see its member genes colored by log2FC.")

    if not enrichment:
        st.warning("No enrichment data loaded.")
    elif not deg:
        st.warning("No DEG data loaded.")
    else:
        pw_comp = st.selectbox("Select comparison", sorted(enrichment.keys()), key="pw_comp")
        pw_dbs = sorted(enrichment[pw_comp].keys())

        if not pw_dbs:
            st.info("No enrichment databases for this comparison.")
        else:
            pw_db = st.radio("Database", pw_dbs, horizontal=True, key="pw_db")
            pw_df = standardize_enrichment_columns(enrichment[pw_comp][pw_db].copy())

            term_col = "term_name" if "term_name" in pw_df.columns else (
                "Description" if "Description" in pw_df.columns else None
            )
            if term_col is None:
                st.warning("No term name column found in enrichment data.")
            else:
                # Filter to significant
                pw_padj_thresh = st.slider("padj threshold", 0.001, 0.5, 0.05, 0.005, key="pw_padj")
                if "padj" in pw_df.columns:
                    pw_sig = pw_df[pw_df["padj"] < pw_padj_thresh]
                else:
                    pw_sig = pw_df

                if pw_sig.empty:
                    st.info("No significant terms at this threshold.")
                else:
                    term_options = pw_sig[term_col].dropna().unique().tolist()
                    selected_term = st.selectbox("Select pathway / GO term", term_options, key="pw_term")
                    term_matches = pw_sig[pw_sig[term_col] == selected_term]
                    if term_matches.empty:
                        st.warning("Could not find the selected term in the data.")
                        st.stop()
                    term_row = term_matches.iloc[0]

                    # Extract gene list
                    genes_col = "genes" if "genes" in pw_sig.columns else None
                    if genes_col is None:
                        st.info("No gene list column found in enrichment data for this term.")
                    else:
                        raw_genes = str(term_row[genes_col])
                        # Genes may be separated by / , ; or space
                        term_genes = [g.strip() for g in re.split(r"[/,;\s]+", raw_genes) if g.strip()]

                        if not term_genes:
                            st.info("No genes listed for this term.")
                        else:
                            st.caption(f"**{selected_term}** — {len(term_genes)} genes")
                            if "padj" in term_row.index:
                                st.caption(f"Term padj = {term_row['padj']:.2e}")

                            # Look up log2FC for these genes in the DEG data
                            comp_deg = deg.get(pw_comp, pd.DataFrame())
                            gene_rows = []
                            for g in term_genes:
                                if "gene_name" not in comp_deg.columns:
                                    break
                                hit = comp_deg[comp_deg["gene_name"].str.upper() == g.upper()]
                                if not hit.empty:
                                    r = hit.iloc[0]
                                    gene_rows.append({
                                        "Gene": g,
                                        "log2FC": r.get("log2fc", np.nan),
                                        "padj": r.get("padj", np.nan),
                                        "basemean": r.get("basemean", np.nan),
                                    })
                                else:
                                    gene_rows.append({"Gene": g, "log2FC": np.nan, "padj": np.nan, "basemean": np.nan})

                            if gene_rows:
                                pw_gene_df = pd.DataFrame(gene_rows).sort_values("log2FC", ascending=False)

                                # Bar chart colored by log2FC
                                fig = go.Figure()
                                colors = [UP_COLOR if v > 0 else DOWN_COLOR if v < 0 else NS_COLOR
                                          for v in pw_gene_df["log2FC"].fillna(0)]
                                fig.add_trace(go.Bar(
                                    x=pw_gene_df["Gene"],
                                    y=pw_gene_df["log2FC"],
                                    marker_color=colors,
                                    hovertemplate="<b>%{x}</b><br>log2FC: %{y:.3f}<extra></extra>",
                                ))
                                fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8)
                                fig.update_layout(
                                    title=dict(text=f"{selected_term}", font=dict(size=16)),
                                    xaxis_title="Gene", yaxis_title="log2 Fold Change",
                                    xaxis_tickangle=-45,
                                    height=max(400, 20 * len(pw_gene_df) + 200),
                                    font=dict(size=14),
                                )
                                st.plotly_chart(fig, width="stretch")

                                # Cross-comparison heatmap for these genes
                                if len(deg) > 1:
                                    st.subheader("Across all comparisons")
                                    matrix_rows = []
                                    for g in pw_gene_df["Gene"]:
                                        row_data: dict[str, float] = {}
                                        for cname, cdf in sorted(deg.items()):
                                            if "gene_name" not in cdf.columns:
                                                continue
                                            hit = cdf[cdf["gene_name"].str.upper() == g.upper()]
                                            if not hit.empty:
                                                row_data[cname] = hit.iloc[0].get("log2fc", np.nan)
                                        if row_data:
                                            row_data["gene"] = g
                                            matrix_rows.append(row_data)

                                    if matrix_rows:
                                        mx = pd.DataFrame(matrix_rows).set_index("gene")
                                        fig_hm = px.imshow(
                                            mx.values.astype(float),
                                            x=mx.columns.tolist(),
                                            y=mx.index.tolist(),
                                            color_continuous_scale="RdBu_r",
                                            color_continuous_midpoint=0,
                                            aspect="auto",
                                            labels=dict(x="Comparison", y="Gene", color="log2FC"),
                                            title=f"{selected_term} — log2FC across comparisons",
                                        )
                                        fig_hm.update_layout(
                                            xaxis_tickangle=-45,
                                            height=max(350, 30 * len(mx) + 200),
                                            font=dict(size=13),
                                        )
                                        st.plotly_chart(fig_hm, width="stretch")

                                # Data table
                                st.dataframe(pw_gene_df, width="stretch", hide_index=True)


# =========================================================================
# TAB 10: Export / Report
# =========================================================================
with tab_export:
    st.header("Export / Report")
    st.caption("Download filtered DEG lists, enrichment results, and data as a ZIP or Excel workbook.")

    if not deg and not enrichment:
        st.warning("No data loaded to export.")
    else:
        export_padj = st.slider("padj threshold", 0.001, 0.1, 0.05, 0.005, key="export_padj")
        export_fc = st.slider("|log2FC| threshold", 0.0, 5.0, 1.0, 0.25, key="export_fc")
        sig_only = st.checkbox("Export significant genes only", value=True, key="export_sig_only")

        st.divider()

        # --- Excel workbook ---
        st.subheader("Excel Workbook")
        st.caption("All comparisons in one Excel file, one sheet per comparison.")

        if st.button("Generate Excel workbook", key="export_excel_btn"):
            buffer = io.BytesIO()
            used_sheet_names: set[str] = set()

            def _unique_sheet_name(base: str) -> str:
                name = base[:31]
                if name not in used_sheet_names:
                    used_sheet_names.add(name)
                    return name
                for i in range(2, 100):
                    suffix = f"_{i}"
                    candidate = f"{base[:31 - len(suffix)]}{suffix}"
                    if candidate not in used_sheet_names:
                        used_sheet_names.add(candidate)
                        return candidate
                # Exhausted suffix range; add hash to guarantee uniqueness
                import hashlib
                h = hashlib.md5(base.encode()).hexdigest()[:4]
                fallback = f"{base[:26]}_{h}"
                used_sheet_names.add(fallback)
                return fallback

            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                for comp_name in sorted(deg.keys()):
                    df = deg[comp_name].copy()
                    if sig_only and {"log2fc", "padj"}.issubset(set(df.columns)):
                        df = df[(df["padj"] < export_padj) & (df["log2fc"].abs() >= export_fc)]
                    sheet_name = _unique_sheet_name(comp_name)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

                # Add enrichment sheets
                for comp_name, comp_data in sorted(enrichment.items()):
                    for db_name, db_df in comp_data.items():
                        edf = standardize_enrichment_columns(db_df.copy())
                        if sig_only and "padj" in edf.columns:
                            edf = edf[edf["padj"] < export_padj]
                        sheet_name = _unique_sheet_name(f"{comp_name[:20]}_{db_name}")
                        edf.to_excel(writer, sheet_name=sheet_name, index=False)

            st.download_button(
                "📥 Download Excel workbook",
                data=buffer.getvalue(),
                file_name="novogene_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="export_excel_download",
            )

        st.divider()

        # --- ZIP of CSVs ---
        st.subheader("ZIP Archive (CSVs)")
        st.caption("Individual CSV files per comparison and enrichment database.")

        if st.button("Generate ZIP archive", key="export_zip_btn"):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for comp_name in sorted(deg.keys()):
                    df = deg[comp_name].copy()
                    if sig_only and {"log2fc", "padj"}.issubset(set(df.columns)):
                        df = df[(df["padj"] < export_padj) & (df["log2fc"].abs() >= export_fc)]
                    csv_buf = io.StringIO()
                    df.to_csv(csv_buf, index=False)
                    zf.writestr(f"deg/{comp_name}.csv", csv_buf.getvalue())

                for comp_name, comp_data in sorted(enrichment.items()):
                    for db_name, db_df in comp_data.items():
                        edf = standardize_enrichment_columns(db_df.copy())
                        if sig_only and "padj" in edf.columns:
                            edf = edf[edf["padj"] < export_padj]
                        csv_buf = io.StringIO()
                        edf.to_csv(csv_buf, index=False)
                        zf.writestr(f"enrichment/{comp_name}_{db_name}.csv", csv_buf.getvalue())

            st.download_button(
                "📥 Download ZIP archive",
                data=buffer.getvalue(),
                file_name="novogene_results.zip",
                mime="application/zip",
                key="export_zip_download",
            )
