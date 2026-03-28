"""Novogene RNA-Seq Explorer – Streamlit app for browsing raw Novogene deliveries.

Launch with:
    streamlit run novogene_explorer.py -- /path/to/data_folder

Or run without arguments and enter the path interactively.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure imports from the novoview package work
# ---------------------------------------------------------------------------
_NOVOVIEW_ROOT = Path(__file__).resolve().parent
if str(_NOVOVIEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_NOVOVIEW_ROOT))


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


def _open_folder_dialog() -> str | None:
    """Open a native OS folder-picker dialog and return the chosen path."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()          # hide the root window
        root.wm_attributes("-topmost", 1)  # dialog on top
        folder = filedialog.askdirectory(
            title="Select your Novogene data folder",
            initialdir=st.session_state.get("browse_dir", str(Path.home())),
        )
        root.destroy()
        return folder if folder else None
    except Exception:
        return None


if st.sidebar.button("Browse...", type="primary", use_container_width=True):
    chosen_folder = _open_folder_dialog()
    if chosen_folder:
        st.session_state["browse_dir"] = chosen_folder
        if _looks_like_novogene(Path(chosen_folder)):
            st.session_state["data_dir"] = chosen_folder
        st.rerun()

# Text input fallback: paste / type a path directly
browse_dir = st.sidebar.text_input(
    "Or paste a path",
    value=st.session_state["browse_dir"],
    key="_browse_input",
    help="Type or paste the full path to your data folder, then press Enter",
)
if browse_dir != st.session_state["browse_dir"]:
    st.session_state["browse_dir"] = browse_dir

browse_path = Path(browse_dir)
if browse_path.is_dir():
    if st.sidebar.button("Use this folder", key="browse_select", use_container_width=True):
        st.session_state["data_dir"] = str(browse_path)
        st.rerun()
    if _looks_like_novogene(browse_path):
        st.sidebar.success("Novogene data detected")
else:
    if browse_dir:
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
if sample_info is not None:
    n_groups = sample_info["group"].nunique()
    st.sidebar.metric("Sample groups", n_groups)

# ---------------------------------------------------------------------------
# Sidebar: log panel
# ---------------------------------------------------------------------------

st.sidebar.divider()
with st.sidebar.expander("Log", expanded=False):
    log_lines = st.session_state.get("_log", [])
    if log_lines:
        st.code("\n".join(log_lines), language="log")
        if st.button("Clear log", key="clear_log"):
            st.session_state["_log"] = []
            st.rerun()
    else:
        st.caption("No log messages yet.")


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_overview, tab_gene, tab_comparison, tab_enrichment = st.tabs([
    "Overview", "Gene Explorer", "Comparison Browser", "Enrichment",
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
        st.dataframe(display_df, use_container_width=True, hide_index=True)

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
            st.plotly_chart(fig, use_container_width=True)
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
        st.dataframe(pd.DataFrame(comp_summary), use_container_width=True, hide_index=True)


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
                st.plotly_chart(fig, use_container_width=True)

                # Data table
                st.dataframe(
                    gene_df[["Comparison", "Group A", "Group B", "log2FC", "padj", "pvalue", "basemean", "regulation"]],
                    use_container_width=True,
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
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(matrix_df, use_container_width=True)
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
                            ax=0, ay=-18, font=dict(size=9),
                        )

            fig.add_vline(x=fc_t, line_dash="dash", line_color="gray", line_width=0.8)
            fig.add_vline(x=-fc_t, line_dash="dash", line_color="gray", line_width=0.8)
            fig.add_hline(y=-np.log10(padj_t), line_dash="dash", line_color="gray", line_width=0.8)
            fig.update_layout(
                title=f"Volcano Plot — {selected_comp}",
                xaxis_title="log2 Fold Change",
                yaxis_title="-log10(padj)",
                height=550,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
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

        st.dataframe(display, use_container_width=True, hide_index=True, height=500)


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
                plot_df = sig_df.nsmallest(max_terms, "padj" if "padj" in sig_df.columns else sig_df.columns[0]).copy()
                plot_df["neg_log10_padj"] = -np.log10(plot_df["padj"].clip(lower=1e-300))

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
                st.plotly_chart(fig, use_container_width=True)

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
                st.plotly_chart(fig_bar, use_container_width=True)

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
            st.dataframe(display_enrich, use_container_width=True, hide_index=True, height=500)

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
                    st.dataframe(cross_df, use_container_width=True, hide_index=True)

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
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No enrichment terms matching **{term_query}** found.")
