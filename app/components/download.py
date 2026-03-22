"""Download utilities for NovoView."""

from io import BytesIO

import streamlit as st


def download_csv_button(df, filename, label="Download CSV"):
    """Render a download button for a DataFrame as CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to export.
    filename : str
        Name of the downloaded file (should end in .csv).
    label : str
        Button label text.
    """
    csv_data = df.to_csv(index=True).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_data,
        file_name=filename,
        mime="text/csv",
    )


def _is_plotly_figure(fig):
    """Check whether a figure is a Plotly figure."""
    try:
        import plotly.graph_objects as go

        return isinstance(fig, go.Figure)
    except ImportError:
        return False


def _is_matplotlib_figure(fig):
    """Check whether a figure is a Matplotlib figure."""
    try:
        import matplotlib.figure

        return isinstance(fig, matplotlib.figure.Figure)
    except ImportError:
        return False


def download_figure_buttons(fig, filename_base):
    """Render PNG and SVG download buttons for a figure.

    Handles both Plotly and Matplotlib figure objects.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure or matplotlib.figure.Figure
        The figure to export.
    filename_base : str
        Base filename without extension (e.g. "volcano_plot").
    """
    col_png, col_svg = st.columns(2)

    if _is_plotly_figure(fig):
        with col_png:
            png_bytes = fig.to_image(format="png", scale=2, width=1200, height=800)
            st.download_button(
                label="Download PNG",
                data=png_bytes,
                file_name=f"{filename_base}.png",
                mime="image/png",
            )

        with col_svg:
            svg_bytes = fig.to_image(format="svg")
            st.download_button(
                label="Download SVG",
                data=svg_bytes,
                file_name=f"{filename_base}.svg",
                mime="image/svg+xml",
            )

    elif _is_matplotlib_figure(fig):
        with col_png:
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)
            st.download_button(
                label="Download PNG",
                data=buf.getvalue(),
                file_name=f"{filename_base}.png",
                mime="image/png",
            )

        with col_svg:
            buf = BytesIO()
            fig.savefig(buf, format="svg", bbox_inches="tight")
            buf.seek(0)
            st.download_button(
                label="Download SVG",
                data=buf.getvalue(),
                file_name=f"{filename_base}.svg",
                mime="image/svg+xml",
            )

    else:
        st.error("Unsupported figure type. Provide a Plotly or Matplotlib figure.")
