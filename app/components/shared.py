"""Shared helper functions for NovoView Streamlit pages.

Centralises repeated patterns: data path resolution, sample group
extraction, expression bar charts, and numeric formatting.

Public API
----------
get_data_path, check_data_path, get_sample_groups,
create_expression_bar, fmt_count, fmt_pvalue, fmt_fc, table_height.
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

from plotting.theme import WONG_PALETTE, apply_plotly_theme  # noqa: E402

# ---------------------------------------------------------------------------
# Data path
# ---------------------------------------------------------------------------

_DEFAULT_DATA_PATH = str(_NOVOVIEW_ROOT / "results" / "novoview_results.h5")


def get_data_path() -> str:
    """Return the HDF5 results path stored in session state.

    Returns
    -------
    str
        Absolute path to the ``novoview_results.h5`` file.  Falls back to
        a default under ``novoview/results/`` if session state is empty.
    """
    return st.session_state.get("results_path", _DEFAULT_DATA_PATH)


def check_data_path(data_path: str) -> bool:
    """Verify that *data_path* exists; display an error if not.

    Parameters
    ----------
    data_path : str
        Path to the HDF5 results file.

    Returns
    -------
    bool
        ``True`` if the file exists; ``False`` (and ``st.error`` shown)
        otherwise.
    """
    if not Path(data_path).exists():
        st.error(
            f"Results file not found: `{data_path}`. "
            "Run the pipeline first or verify the configuration."
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Sample groups extraction
# ---------------------------------------------------------------------------


def get_sample_groups(
    samples_meta: pd.DataFrame | None,
    sample_names: list[str],
) -> pd.Series | None:
    """Extract a per-sample condition label aligned to *sample_names*.

    Looks for a column named ``condition``, ``group``, or ``sample_group``
    (in that order) in the metadata table and returns a Series indexed by
    sample name.

    Parameters
    ----------
    samples_meta : pd.DataFrame or None
        Metadata table, optionally containing a ``sample_id`` column that
        will be used as the index.
    sample_names : list[str]
        Sample names to align the output Series to.

    Returns
    -------
    pd.Series or None
        Condition labels indexed by sample name, or ``None`` if metadata
        is unavailable or no recognised condition column is found.
    """
    if samples_meta is None or samples_meta.empty:
        return None

    meta = samples_meta.copy()
    if "sample_id" in meta.columns:
        meta = meta.set_index("sample_id")

    for candidate in ("condition", "group", "sample_group"):
        if candidate in meta.columns:
            groups = meta[candidate].reindex(sample_names)
            groups.name = "condition"
            return groups

    return None


# ---------------------------------------------------------------------------
# Expression bar chart (shared between diffexp and gene search pages)
# ---------------------------------------------------------------------------


def create_expression_bar(
    gene_name: str,
    expression_df: pd.DataFrame,
    samples_meta: pd.DataFrame | None,
) -> go.Figure | None:
    """Create a bar chart of one gene's expression across samples.

    Parameters
    ----------
    gene_name : str
        Gene identifier; must be present in *expression_df*.index.
    expression_df : pd.DataFrame
        Expression matrix (genes x samples).
    samples_meta : pd.DataFrame or None
        Optional sample metadata used to colour bars by condition.

    Returns
    -------
    go.Figure or None
        Plotly bar chart, or ``None`` if *gene_name* is not found.
    """
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
        title=f"{gene_name} Expression",
        labels={"expression": "Expression (TPM)", "sample": "Sample"},
    )
    fig.update_layout(xaxis_tickangle=-45, bargap=0.2)
    apply_plotly_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Numeric formatting helpers
# ---------------------------------------------------------------------------


def fmt_count(n: int | float) -> str:
    """Format *n* as an integer with thousands separators (e.g. ``"1,234"``)."""
    return f"{int(n):,}"


def fmt_pvalue(p: float) -> str:
    """Format a p-value in scientific notation (e.g. ``"1.23e-04"``)."""
    if pd.isna(p):
        return "---"
    return f"{p:.2e}"


def fmt_fc(fc: float) -> str:
    """Format a log2 fold-change to three decimal places."""
    if pd.isna(fc):
        return "---"
    return f"{fc:.3f}"


# ---------------------------------------------------------------------------
# Dynamic table height
# ---------------------------------------------------------------------------


def table_height(n_rows: int, max_height: int = 400) -> int:
    """Calculate a pixel height for ``st.dataframe`` based on row count.

    Parameters
    ----------
    n_rows : int
        Number of rows to display.
    max_height : int, optional
        Upper bound in pixels (default 400).

    Returns
    -------
    int
        Height in pixels, capped at *max_height*.
    """
    return min(max_height, 35 * n_rows + 38)


# ---------------------------------------------------------------------------
# Empty state & stat badge
# ---------------------------------------------------------------------------


def render_empty_state(message: str, suggestion: str | None = None, icon: str = "info") -> None:
    """Render a styled empty-state placeholder."""
    _ICONS = {"info": "ℹ️", "search": "🔍", "chart": "📊", "gene": "🧬", "warning": "⚠️"}
    icon_char = _ICONS.get(icon, icon)
    suggestion_html = ""
    if suggestion:
        suggestion_html = f'<div style="font-size:0.82rem; color:#999; margin-top:0.5rem;">{suggestion}</div>'
    st.markdown(
        f"""<div style="text-align:center; padding:2.5rem 1rem; color:#888;">
            <div style="font-size:2.5rem; margin-bottom:0.5rem;">{icon_char}</div>
            <div style="font-size:0.95rem; font-weight:500; color:#666;">{message}</div>
            {suggestion_html}
        </div>""",
        unsafe_allow_html=True,
    )


def render_stat_badge(label: str, value: str | int, color: str = "#0072B2") -> None:
    """Render an inline stat badge."""
    st.markdown(
        f'<span style="display:inline-block; background:{color}15; color:{color}; '
        f'border:1px solid {color}40; border-radius:6px; padding:0.2rem 0.6rem; '
        f'font-size:0.78rem; font-weight:600; margin-right:0.4rem;">'
        f'{label}: {value}</span>',
        unsafe_allow_html=True,
    )
