"""Nature journal-style plotting theme for matplotlib, seaborn, and plotly."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio

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

DIVERGING_CMAP: str = "RdBu_r"
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
    sns.set_style("ticks")
    sns.despine()


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------


def get_plotly_template() -> go.layout.Template:
    """Return a Plotly template that mirrors the Nature journal style."""
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
    fig.update_layout(
        template=get_plotly_template(),
        margin=dict(l=60, r=20, t=40, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Colour-scale helpers
# ---------------------------------------------------------------------------


def get_nature_colorscale(palette_type: str = "diverging") -> str:
    """Return an appropriate Plotly colorscale name.

    Parameters
    ----------
    palette_type : str
        ``"diverging"`` or ``"sequential"``.

    Returns
    -------
    str
        A Plotly-compatible colorscale name.
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
