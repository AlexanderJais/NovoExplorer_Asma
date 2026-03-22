"""Multi-Condition Comparison -- NovoView Streamlit page.

Provides three analysis sections:
1. DEG Overlap -- UpSet-style plot and intersection table
2. Fold-Change Concordance -- scatter plot of log2FC between two comparisons
3. Summary Table -- gene-level table with per-comparison log2FC coloring
"""

from __future__ import annotations

import sys
from itertools import combinations
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

from pipeline.persistence import load_deg
from plotting.theme import (
    WONG_PALETTE,
    VOLCANO_COLORS,
    apply_plotly_theme,
)
from app.components.filters import threshold_sliders
from app.components.download import download_csv_button
from app.components.shared import get_data_path, check_data_path, fmt_count

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VERMILION = "#D55E00"
_BLUE = "#0072B2"
_GRAY = "#BBBBBB"
_MAX_TABLE_DISPLAY_ROWS = 200

_get_data_path = get_data_path

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading DEG data...")
def _load_deg(path: str) -> dict | None:
    return load_deg(path)


# ---------------------------------------------------------------------------
# Helpers -- extract significant gene sets
# ---------------------------------------------------------------------------


def _get_sig_genes(
    deg_all: dict[str, pd.DataFrame],
    padj_thresh: float = 0.05,
    log2fc_thresh: float = 1.0,
) -> dict[str, set[str]]:
    """Return a dict mapping comparison name -> set of significant gene names."""
    result = {}
    for comp, df in deg_all.items():
        if "padj" not in df.columns or "log2fc" not in df.columns:
            continue

        sig = df[(df["padj"] < padj_thresh) & (df["log2fc"].abs() > log2fc_thresh)]

        # Determine gene name column
        gene_col = None
        for candidate in ("gene_name", "gene_id"):
            if candidate in sig.columns:
                gene_col = candidate
                break

        if gene_col is not None:
            result[comp] = set(sig[gene_col].dropna().unique())
        else:
            # Fall back to index
            result[comp] = set(sig.index.astype(str))

    return result


def _gene_col(df: pd.DataFrame) -> str:
    """Determine the gene name column for a DEG DataFrame."""
    for candidate in ("gene_name", "gene_id"):
        if candidate in df.columns:
            return candidate
    return df.index.name or "index"


# ---------------------------------------------------------------------------
# Section 1: DEG Overlap (UpSet-style)
# ---------------------------------------------------------------------------


def _compute_upset_data(
    sig_genes: dict[str, set[str]],
) -> pd.DataFrame:
    """Compute intersection sizes for an UpSet-style visualization.

    Returns a DataFrame with columns: intersection_label, size, comparisons (frozenset).
    """
    comparisons = sorted(sig_genes.keys())
    if not comparisons:
        return pd.DataFrame()

    rows = []
    n = len(comparisons)

    # Generate all non-empty subsets
    for r in range(1, n + 1):
        for subset in combinations(comparisons, r):
            subset_set = set(subset)
            others = set(comparisons) - subset_set

            # Genes in ALL of subset but NONE of others
            intersection = set.intersection(*(sig_genes[c] for c in subset))
            for other in others:
                intersection = intersection - sig_genes[other]

            if intersection:
                label = " & ".join(subset)
                rows.append({
                    "intersection": label,
                    "size": len(intersection),
                    "n_sets": len(subset),
                    "comparisons": frozenset(subset),
                    "genes": intersection,
                })

    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)


def _create_upset_plot(upset_df: pd.DataFrame) -> go.Figure:
    """Create a horizontal bar chart approximating an UpSet plot."""
    if upset_df.empty:
        return go.Figure()

    # Show top 30 intersections
    plot_df = upset_df.head(30).copy()
    plot_df = plot_df.iloc[::-1]  # reverse for horizontal bar

    # Color by number of sets in intersection
    colors = []
    for ns in plot_df["n_sets"]:
        if ns == 1:
            colors.append(WONG_PALETTE[0])
        elif ns == 2:
            colors.append(WONG_PALETTE[1])
        elif ns == 3:
            colors.append(WONG_PALETTE[2])
        else:
            colors.append(WONG_PALETTE[min(ns - 1, len(WONG_PALETTE) - 1)])

    fig = go.Figure(
        data=go.Bar(
            y=plot_df["intersection"],
            x=plot_df["size"],
            orientation="h",
            marker_color=colors,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Genes: %{x}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="DEG Intersection Sizes",
        xaxis_title="Number of Genes",
        yaxis_title="",
        height=max(400, len(plot_df) * 28),
        margin=dict(l=250),
    )
    apply_plotly_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Section 2: Fold-Change Concordance
# ---------------------------------------------------------------------------


def _build_fc_scatter_data(
    deg_all: dict[str, pd.DataFrame],
    comp_a: str,
    comp_b: str,
    padj_thresh: float = 0.05,
    log2fc_thresh: float = 1.0,
) -> pd.DataFrame:
    """Build a DataFrame with log2FC from two comparisons for shared genes."""
    df_a = deg_all[comp_a].copy()
    df_b = deg_all[comp_b].copy()

    gene_col_a = _gene_col(df_a)
    gene_col_b = _gene_col(df_b)

    # Use gene name column as merge key
    if gene_col_a == "index":
        df_a["_gene"] = df_a.index.astype(str)
    else:
        df_a["_gene"] = df_a[gene_col_a].astype(str)

    if gene_col_b == "index":
        df_b["_gene"] = df_b.index.astype(str)
    else:
        df_b["_gene"] = df_b[gene_col_b].astype(str)

    # Select relevant columns
    cols_a = ["_gene", "log2fc"]
    cols_b = ["_gene", "log2fc"]
    if "padj" in df_a.columns:
        cols_a.append("padj")
    if "padj" in df_b.columns:
        cols_b.append("padj")

    merged = pd.merge(
        df_a[cols_a].rename(columns={"log2fc": "log2fc_a", "padj": "padj_a"}),
        df_b[cols_b].rename(columns={"log2fc": "log2fc_b", "padj": "padj_b"}),
        on="_gene",
        how="inner",
    )

    # Classify concordance
    def _classify(row):
        sig_a = row.get("padj_a", 1.0) < padj_thresh and abs(row["log2fc_a"]) > log2fc_thresh
        sig_b = row.get("padj_b", 1.0) < padj_thresh and abs(row["log2fc_b"]) > log2fc_thresh

        if sig_a and sig_b:
            if row["log2fc_a"] > 0 and row["log2fc_b"] > 0:
                return "Concordant Up"
            elif row["log2fc_a"] < 0 and row["log2fc_b"] < 0:
                return "Concordant Down"
            else:
                return "Discordant"
        return "NS"

    merged["category"] = merged.apply(_classify, axis=1)

    return merged


def _create_fc_scatter(
    scatter_df: pd.DataFrame,
    comp_a: str,
    comp_b: str,
) -> go.Figure:
    """Create a fold-change concordance scatter plot with quadrant coloring."""
    color_map = {
        "Concordant Up": _VERMILION,
        "Concordant Down": _BLUE,
        "Discordant": _GRAY,
        "NS": "#E5E5E5",
    }

    fig = go.Figure()

    # Plot each category separately for legend control
    for cat in ["NS", "Discordant", "Concordant Down", "Concordant Up"]:
        sub = scatter_df[scatter_df["category"] == cat]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["log2fc_a"],
            y=sub["log2fc_b"],
            mode="markers",
            name=cat,
            marker=dict(
                color=color_map[cat],
                size=5,
                opacity=0.6 if cat == "NS" else 0.8,
            ),
            text=sub["_gene"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"{comp_a} log2FC: %{{x:.2f}}<br>"
                f"{comp_b} log2FC: %{{y:.2f}}<extra></extra>"
            ),
        ))

    # Add reference lines
    x_range = scatter_df["log2fc_a"].abs().max() * 1.1 if not scatter_df.empty else 5
    y_range = scatter_df["log2fc_b"].abs().max() * 1.1 if not scatter_df.empty else 5
    axis_max = max(x_range, y_range, 2)

    fig.add_hline(y=0, line_dash="dash", line_color="#AAAAAA", line_width=0.8)
    fig.add_vline(x=0, line_dash="dash", line_color="#AAAAAA", line_width=0.8)

    # Compute correlations
    valid = scatter_df.dropna(subset=["log2fc_a", "log2fc_b"])
    annotations = []
    if len(valid) > 2:
        pearson_r = valid["log2fc_a"].corr(valid["log2fc_b"])
        spearman_r = valid["log2fc_a"].corr(valid["log2fc_b"], method="spearman")
        annotations.append(dict(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=f"Pearson r = {pearson_r:.3f}<br>Spearman rho = {spearman_r:.3f}",
            showarrow=False,
            font=dict(size=11, color="#2D2D2D"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#E5E5E5",
            borderwidth=1,
            borderpad=6,
            align="left",
        ))

    fig.update_layout(
        title=f"Fold-Change Concordance: {comp_a} vs {comp_b}",
        xaxis_title=f"log2FC ({comp_a})",
        yaxis_title=f"log2FC ({comp_b})",
        annotations=annotations,
        height=600,
        legend=dict(x=1.02, y=1, xanchor="left"),
    )

    apply_plotly_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Section 3: Summary Table
# ---------------------------------------------------------------------------


def _build_summary_table(
    deg_all: dict[str, pd.DataFrame],
    padj_thresh: float = 0.05,
    log2fc_thresh: float = 1.0,
) -> pd.DataFrame:
    """Build a gene-level table with log2FC per comparison."""
    all_genes = set()
    gene_data: dict[str, dict[str, dict]] = {}  # gene -> comp -> {log2fc, padj}

    for comp, df in deg_all.items():
        gc = _gene_col(df)
        if gc == "index":
            genes = df.index.astype(str)
        else:
            genes = df[gc].astype(str)

        for idx, gene in enumerate(genes):
            if pd.isna(gene) or str(gene).strip().lower() == "nan" or gene == "":
                continue
            all_genes.add(gene)
            row_data = df.iloc[idx]
            gene_data.setdefault(gene, {})[comp] = {
                "log2fc": row_data.get("log2fc", np.nan),
                "padj": row_data.get("padj", np.nan),
            }

    if not all_genes:
        return pd.DataFrame()

    comparisons = sorted(deg_all.keys())

    # Filter to genes significant in at least one comparison
    sig_genes = set()
    for gene, comp_vals in gene_data.items():
        for comp, vals in comp_vals.items():
            padj = vals.get("padj", 1.0)
            lfc = abs(vals.get("log2fc", 0.0))
            if not np.isnan(padj) and padj < padj_thresh and lfc > log2fc_thresh:
                sig_genes.add(gene)
                break

    if not sig_genes:
        return pd.DataFrame()

    # Build output rows
    rows = []
    for gene in sorted(sig_genes):
        row = {"gene": gene}
        for comp in comparisons:
            vals = gene_data.get(gene, {}).get(comp, {})
            row[f"log2fc_{comp}"] = vals.get("log2fc", np.nan)
            row[f"padj_{comp}"] = vals.get("padj", np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def _style_log2fc_cell(val, padj_val, padj_thresh, log2fc_thresh):
    """Return CSS style string for a log2FC cell."""
    if pd.isna(val) or pd.isna(padj_val):
        return "color: #BBBBBB; background-color: #F9F9F9;"
    if padj_val < padj_thresh and abs(val) > log2fc_thresh:
        if val > 0:
            return f"color: white; background-color: {_VERMILION}; font-weight: 600;"
        else:
            return f"color: white; background-color: {_BLUE}; font-weight: 600;"
    return "color: #888888; background-color: #F5F5F5;"


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("Multi-Condition Comparison")
    st.caption(
        "Compare differential expression results across multiple conditions. "
        "Identify shared and unique DEGs, assess fold-change concordance, "
        "and view a unified gene-level summary."
    )

    data_path = _get_data_path()
    if not check_data_path(data_path):
        return

    deg_all = _load_deg(data_path)

    if not deg_all or len(deg_all) < 2:
        st.warning(
            "Multi-condition comparison requires at least two comparisons. "
            f"Found: {len(deg_all) if deg_all else 0}."
        )
        return

    comparisons = sorted(deg_all.keys())

    # Sidebar thresholds
    with st.sidebar:
        st.header("Filters")
        padj_thresh, log2fc_thresh = threshold_sliders(key_prefix="multi_")

    sig_genes = _get_sig_genes(deg_all, padj_thresh, log2fc_thresh)

    # ==================================================================
    # Section 1: DEG Overlap
    # ==================================================================
    st.header("DEG Overlap")
    st.caption(
        "Intersection of significant DEGs across comparisons. "
        "Each bar represents a unique combination of comparisons and shows "
        "the number of genes significant in exactly that combination."
    )

    if sig_genes:
        upset_df = _compute_upset_data(sig_genes)

        if not upset_df.empty:
            fig_upset = _create_upset_plot(upset_df)
            st.plotly_chart(fig_upset, use_container_width=True, key="upset_plot")

            # Intersection selector
            st.markdown("#### Genes in Selected Intersection")
            intersection_options = upset_df["intersection"].tolist()
            selected_intersection = st.selectbox(
                "Select intersection",
                options=intersection_options,
                key="upset_intersection",
            )

            if selected_intersection:
                filtered = upset_df[upset_df["intersection"] == selected_intersection]
                if filtered.empty:
                    st.warning("Selected intersection no longer available.")
                    return
                selected_row = filtered.iloc[0]
                gene_list = sorted(selected_row["genes"])
                st.write(f"**{len(gene_list)} genes** in: {selected_intersection}")

                gene_df = pd.DataFrame({"gene": gene_list})
                st.dataframe(gene_df, use_container_width=True, hide_index=True, height=300)
                download_csv_button(gene_df, f"intersection_genes_{selected_intersection}.csv")
        else:
            st.info("No exclusive intersections found at the current thresholds.")
    else:
        st.info(
            f"No significant DEGs at the current thresholds (padj < {padj_thresh}, "
            f"|log2FC| > {log2fc_thresh}). Try relaxing the thresholds in the sidebar."
        )

    st.divider()

    # ==================================================================
    # Section 2: Fold-Change Concordance
    # ==================================================================
    st.header("Fold-Change Concordance")
    st.caption(
        "Compare log2 fold-changes between two comparisons. "
        "Concordant genes are colored by direction (vermilion = up, blue = down); "
        "discordant genes are gray."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        comp_a = st.selectbox(
            "Comparison A",
            options=comparisons,
            index=0,
            key="fc_comp_a",
        )
    with col_b:
        default_b = 1 if len(comparisons) > 1 else 0
        comp_b = st.selectbox(
            "Comparison B",
            options=comparisons,
            index=default_b,
            key="fc_comp_b",
        )

    if comp_a == comp_b:
        st.warning("Select two different comparisons to compare fold-changes.")
    else:
        # Check both comparisons have required columns
        if "log2fc" not in deg_all[comp_a].columns or "log2fc" not in deg_all[comp_b].columns:
            st.error("Both comparisons must have a `log2fc` column.")
        else:
            scatter_df = _build_fc_scatter_data(
                deg_all, comp_a, comp_b,
                padj_thresh=padj_thresh,
                log2fc_thresh=log2fc_thresh,
            )

            if scatter_df.empty:
                st.info("No shared genes between the two comparisons.")
            else:
                fig_scatter = _create_fc_scatter(scatter_df, comp_a, comp_b)
                st.plotly_chart(fig_scatter, use_container_width=True, key="fc_scatter")

                # Category counts
                counts = scatter_df["category"].value_counts()
                with st.container(border=True):
                    cc1, cc2, cc3, cc4 = st.columns(4)
                    cc1.metric("Concordant Up", fmt_count(counts.get("Concordant Up", 0)))
                    cc2.metric("Concordant Down", fmt_count(counts.get("Concordant Down", 0)))
                    cc3.metric("Discordant", fmt_count(counts.get("Discordant", 0)))
                    cc4.metric("Total Shared", fmt_count(len(scatter_df)))

    st.divider()

    # ==================================================================
    # Section 3: Summary Table
    # ==================================================================
    st.header("Gene-Level Summary")
    st.caption(
        f"Log2 fold-change per comparison for genes significant (padj < {padj_thresh}, "
        f"|log2FC| > {log2fc_thresh}) in at least one comparison. "
        "Cells are colored: vermilion = significantly up, "
        "blue = significantly down, gray = not significant. "
        "Adjust sidebar thresholds to change filtering."
    )

    summary_df = _build_summary_table(deg_all, padj_thresh, log2fc_thresh)

    if summary_df.empty:
        st.info("No significant DEGs at the current thresholds.")
    else:
        # Build a display-friendly version with colored HTML
        # For Streamlit dataframe, use column_config for coloring
        display_df = summary_df[["gene"]].copy()

        log2fc_cols = [c for c in summary_df.columns if c.startswith("log2fc_")]
        padj_cols = [c for c in summary_df.columns if c.startswith("padj_")]

        for lfc_col in log2fc_cols:
            comp_name = lfc_col.replace("log2fc_", "")
            display_df[comp_name] = summary_df[lfc_col].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else "---"
            )

        st.markdown(f"**{len(display_df)} genes** significant in at least one comparison.")

        # Use a styled HTML table for colored cells
        html_rows = []
        for _, row in summary_df.iterrows():
            cells = [f'<td style="font-weight:600; padding:0.5rem 0.75rem;">{row["gene"]}</td>']
            for lfc_col in log2fc_cols:
                comp_name = lfc_col.replace("log2fc_", "")
                padj_col = f"padj_{comp_name}"
                val = row[lfc_col]
                padj_val = row.get(padj_col, np.nan)
                style = _style_log2fc_cell(val, padj_val, padj_thresh, log2fc_thresh)
                if pd.notna(val):
                    # Add directional arrow for accessibility (colorblind-friendly)
                    is_sig = pd.notna(padj_val) and padj_val < padj_thresh and abs(val) > log2fc_thresh
                    if is_sig and val > 0:
                        display_val = f"&uarr; {val:.2f}"
                    elif is_sig and val < 0:
                        display_val = f"&darr; {val:.2f}"
                    else:
                        display_val = f"{val:.2f}"
                else:
                    display_val = "---"
                cells.append(
                    f'<td style="{style} text-align:center; padding:0.5rem 0.75rem; '
                    f'border-radius:4px;">{display_val}</td>'
                )
            html_rows.append("<tr>" + "".join(cells) + "</tr>")

        comp_headers = [lfc_col.replace("log2fc_", "") for lfc_col in log2fc_cols]
        header_cells = ['<th style="text-align:left; padding:0.65rem 0.75rem;">Gene</th>']
        for ch in comp_headers:
            header_cells.append(
                f'<th style="text-align:center; padding:0.65rem 0.75rem;">{ch}</th>'
            )

        # Limit display to 200 rows in HTML, offer CSV for full data
        displayed_rows = html_rows[:_MAX_TABLE_DISPLAY_ROWS]

        html_table = f"""
        <div style="max-height:500px; overflow-y:auto; border:1px solid #E5E5E5;
                    border-radius:10px; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
        <table>
        <thead>
            <tr>{"".join(header_cells)}</tr>
        </thead>
        <tbody>
            {"".join(displayed_rows)}
        </tbody>
        </table>
        </div>
        """

        if len(html_rows) > _MAX_TABLE_DISPLAY_ROWS:
            st.markdown(
                f"Showing first {_MAX_TABLE_DISPLAY_ROWS} of {len(html_rows)} genes. "
                "Download CSV for the full table."
            )

        st.markdown(html_table, unsafe_allow_html=True)

        # Download full summary
        download_csv_button(summary_df, "multi_condition_summary.csv", "Download full summary CSV")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__page__" or __name__ == "__main__":
    main()
else:
    main()
