"""Nature journal-style plotting theme for matplotlib, seaborn, and plotly."""

from __future__ import annotations

import matplotlib as mpl
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

WONG_PALETTE: list[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion/orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange/amber
    "#000000",  # black
]

VOLCANO_COLORS: dict[str, str] = {
    "up": "#D55E00",
    "down": "#0072B2",
    "ns": "#BBBBBB",
}

# NOTE: "RdBu_r" works for matplotlib/seaborn but Plotly needs "RdBu" with
# reversescale=True.  Plotting code must handle the distinction.
DIVERGING_CMAP: str = "RdBu_r"
PLOTLY_DIVERGING_CMAP: str = "RdBu"
SEQUENTIAL_CMAP: str = "viridis"

# ---------------------------------------------------------------------------
# Matplotlib / seaborn
# ---------------------------------------------------------------------------


def apply_matplotlib_theme() -> None:
    """Configure matplotlib rcParams for a Nature journal style."""
    rc = {
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "sans-serif"],
        # Font sizes
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.titlesize": 9,
        "axes.titleweight": "semibold",
        # Spines – left and bottom only
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        # Ticks – outward facing, thin
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        # Grid off
        "axes.grid": False,
        # Background
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        # DPI
        "savefig.dpi": 300,
        # Axis line width
        "axes.linewidth": 0.5,
    }
    mpl.rcParams.update(rc)
    sns.set_theme(style="ticks", rc=rc)
    # sns.despine() omitted here -- it only affects the *current* axes and is
    # redundant with the axes.spines.top/right rcParams set above.  Each figure
    # should call sns.despine() after creation if needed.


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------


def get_plotly_template() -> go.layout.Template:
    """Return a Plotly template that mirrors the Nature journal style.

    Returns
    -------
    go.layout.Template
        Template with white background, thin axis lines, Wong colour
        palette, and no grid.
    """
    _charcoal = "#333333"

    template = go.layout.Template()

    template.layout = go.Layout(
        font=dict(
            family="Helvetica, Arial, sans-serif",
            size=11,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor=_charcoal,
            linewidth=1,
            ticks="outside",
            mirror=False,
        ),
        yaxis=dict(
            showgrid=False,
            showline=True,
            linecolor=_charcoal,
            linewidth=1,
            ticks="outside",
            mirror=False,
        ),
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
        ),
        colorway=WONG_PALETTE,
    )

    return template


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """Apply the Nature-style template to a Plotly figure.

    Sets the template and applies reasonable default margins.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to style.

    Returns
    -------
    plotly.graph_objects.Figure
        The same figure, modified in place and returned for chaining.
    """
    # Merge margins: only set defaults for margins the caller hasn't customized
    _defaults = {"l": 60, "r": 20, "t": 40, "b": 50}
    existing = fig.layout.margin.to_plotly_json() if fig.layout.margin else {}
    merged = {**_defaults, **{k: v for k, v in existing.items() if v is not None}}
    fig.update_layout(
        template=get_plotly_template(),
        margin=merged,
    )
    return fig


# ---------------------------------------------------------------------------
# Colour-scale helpers
# ---------------------------------------------------------------------------


def get_nature_colorscale(palette_type: str = "diverging") -> str:
    """Return a colorscale name for the given palette type.

    Parameters
    ----------
    palette_type : str
        ``"diverging"`` or ``"sequential"``.

    Returns
    -------
    str
        Colorscale name.  The ``"diverging"`` value (``"RdBu_r"``) is a
        **matplotlib / seaborn** name; for Plotly use :data:`PLOTLY_DIVERGING_CMAP`
        (``"RdBu"``) with ``reversescale=True``.  The ``"sequential"`` value
        (``"viridis"``) works in both backends.

    Raises
    ------
    ValueError
        If *palette_type* is not recognised.
    """
    if palette_type == "diverging":
        return DIVERGING_CMAP
    if palette_type == "sequential":
        return SEQUENTIAL_CMAP
    raise ValueError(
        f"Unknown palette_type {palette_type!r}; "
        "expected 'diverging' or 'sequential'."
    )


# ---------------------------------------------------------------------------
# Gene classification
# ---------------------------------------------------------------------------


def classify_genes(
    deg_df: pd.DataFrame,
    padj_threshold: float,
    log2fc_threshold: float,
) -> pd.Series:
    """Return a Series of 'up', 'down', or 'ns' for each gene."""
    sig = deg_df["padj"] < padj_threshold
    up = sig & (deg_df["log2fc"] >= log2fc_threshold)
    down = sig & (deg_df["log2fc"] <= -log2fc_threshold)
    category = pd.Series("ns", index=deg_df.index)
    category[up] = "up"
    category[down] = "down"
    return category


# ---------------------------------------------------------------------------
# Label formatting
# ---------------------------------------------------------------------------


def format_axis_label(label: str, unit: str | None = None) -> str:
    """Format an axis label, optionally appending a unit in parentheses.

    Parameters
    ----------
    label : str
        The base label text, e.g. ``"Expression"``.
    unit : str, optional
        Unit string, e.g. ``"log2 TPM"``.

    Returns
    -------
    str
        Formatted label such as ``"Expression (log2 TPM)"``.
    """
    if unit is not None:
        return f"{label} ({unit})"
    return label
