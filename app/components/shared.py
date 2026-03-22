"""Shared helper functions for NovoView Streamlit pages.

Centralises repeated patterns: data path resolution, sample group
extraction, expression bar charts, and numeric formatting.
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
    """Return the HDF5 results path from session state."""
    return st.session_state.get("results_path", _DEFAULT_DATA_PATH)


def check_data_path(data_path: str) -> bool:
    """Show an error and return False if *data_path* does not exist."""
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
    """Extract a condition Series aligned to *sample_names* from metadata."""
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
    """Format an integer with thousands separator."""
    return f"{int(n):,}"


def fmt_pvalue(p: float) -> str:
    """Format a p-value in scientific notation."""
    if pd.isna(p):
        return "---"
    return f"{p:.2e}"


def fmt_fc(fc: float) -> str:
    """Format a log2 fold-change value."""
    if pd.isna(fc):
        return "---"
    return f"{fc:.3f}"


# ---------------------------------------------------------------------------
# Dynamic table height
# ---------------------------------------------------------------------------


def table_height(n_rows: int, max_height: int = 400) -> int:
    """Calculate a sensible table height based on row count."""
    return min(max_height, 35 * n_rows + 38)
