#!/usr/bin/env python
"""NovoView -- Main Streamlit application entry point.

Launch with::

    streamlit run novoview/app/app.py -- --config config.yaml

The ``--config`` flag (after the Streamlit ``--`` separator) points to the
project YAML.  The app reads the config, resolves the results HDF5 path,
and stores both in ``st.session_state`` for downstream pages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so pipeline imports resolve.
# ---------------------------------------------------------------------------
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent.parent  # Xenium-Analysis/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "novoview") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "novoview"))

from pipeline.utils import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VERSION = "0.1.0"

_PAGE_DIR = _APP_DIR / "pages"

_PAGES = [
    st.Page(str(_PAGE_DIR / "01_overview.py"), title="Overview", icon="\U0001F4CA"),
    st.Page(str(_PAGE_DIR / "02_diffexp.py"), title="Differential Expression", icon="\U0001F30B"),
    st.Page(str(_PAGE_DIR / "03_gene_search.py"), title="Gene Search", icon="\U0001F9EC"),
    st.Page(str(_PAGE_DIR / "04_signatures.py"), title="Signatures & Pathways", icon="\U0001F9E9"),
    st.Page(str(_PAGE_DIR / "05_multi_condition.py"), title="Multi-Condition", icon="\U0001F504"),
]

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NovoView",
    page_icon="\U0001F52C",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inject custom CSS
# ---------------------------------------------------------------------------
_CSS_PATH = _APP_DIR / "style.css"
if _CSS_PATH.exists():
    st.markdown(
        f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Parse --config from sys.argv
# ---------------------------------------------------------------------------

def _parse_config_path() -> str:
    """Extract the ``--config`` value from sys.argv.

    Streamlit passes custom CLI args after a ``--`` separator, so they
    appear in ``sys.argv`` alongside Streamlit's own flags.
    """
    args = sys.argv[1:]  # skip script name
    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return "config.yaml"  # sensible default


# ---------------------------------------------------------------------------
# Load config & resolve results path  (cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_project_config(config_path: str) -> dict:
    """Load and cache the YAML configuration."""
    return load_config(config_path)


def _init_session_state() -> None:
    """Populate ``st.session_state`` with config, results_path, etc."""
    if "config" in st.session_state:
        return  # already initialised

    config_path = _parse_config_path()

    try:
        config = _load_project_config(config_path)
    except FileNotFoundError:
        st.error(
            f"Configuration file not found: **{config_path}**. "
            "Pass `--config path/to/config.yaml` after the Streamlit `--` separator."
        )
        st.stop()
    except Exception as exc:
        st.error(f"Failed to load configuration: **{type(exc).__name__}** -- {exc}")
        st.stop()

    # Resolve the HDF5 results file
    output_dir = config.get("output_dir", "results")
    results_file = config.get("results_file", "novoview_results.h5")
    results_path = str(Path(output_dir) / results_file)

    st.session_state["config"] = config
    st.session_state["config_path"] = config_path
    st.session_state["results_path"] = results_path


_init_session_state()

# ---------------------------------------------------------------------------
# Sidebar -- branding, navigation, about
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("# NovoView")
    st.caption("RNA-Seq Analysis Platform")

    # Project info card
    cfg = st.session_state.get("config", {})
    if cfg:
        project_name = cfg.get("project_name", cfg.get("data_dir", "Unknown"))
        organism = cfg.get("organism", "human").capitalize()
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg, #FFFFFF 0%, #F8FAFE 100%);
                        border:1px solid #DDE5EB; border-radius:10px;
                        padding:0.85rem 1rem; margin:0.5rem 0 0.75rem 0;">
                <div style="font-size:0.7rem; font-weight:700; color:#8A8A8A;
                            text-transform:uppercase; letter-spacing:0.06em;
                            margin-bottom:0.35rem;">Project</div>
                <div style="font-weight:600; font-size:0.92rem; color:#2D2D2D;
                            margin-bottom:0.25rem;">{project_name}</div>
                <div style="font-size:0.78rem; color:#7A7A7A;">
                    {organism}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Page navigation
    st.markdown("### Navigate")
    st.page_link(str(_PAGE_DIR / "01_overview.py"), label="Overview", icon="\U0001F4CA")
    st.page_link(str(_PAGE_DIR / "02_diffexp.py"), label="Differential Expression", icon="\U0001F30B")
    st.page_link(str(_PAGE_DIR / "03_gene_search.py"), label="Gene Search", icon="\U0001F9EC")
    st.page_link(str(_PAGE_DIR / "04_signatures.py"), label="Signatures & Pathways", icon="\U0001F9E9")
    st.page_link(str(_PAGE_DIR / "05_multi_condition.py"), label="Multi-Condition", icon="\U0001F504")

    st.markdown("---")

    # Version footer
    st.markdown(
        f"""
        <div style="font-size:0.72rem; color:#ACACAC; text-align:center; padding:0.25rem 0;">
            NovoView v{_VERSION}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Page routing via st.navigation
# ---------------------------------------------------------------------------

# Filter to only pages whose files exist (graceful degradation during dev)
available_pages = _PAGES

nav = st.navigation(available_pages)
nav.run()
