#!/usr/bin/env python
"""NovoView -- Main Streamlit application entry point.

Launch with::

    # Interactive mode (pick data in the browser):
    streamlit run novoview/app/app.py

    # Config mode (pre-configured project):
    streamlit run novoview/app/app.py -- --config config.yaml

When launched without ``--config``, the app shows a welcome screen where
the user can browse to a results folder or HDF5 file.  Once a valid path
is provided, the full analysis UI becomes available.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so pipeline imports resolve.
# ---------------------------------------------------------------------------
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent  # NovoExplorer/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.utils import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VERSION = "0.1.0"
_DEFAULT_RESULTS_FILENAME = "novoview_results.h5"

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
# Parse --config from sys.argv (returns None when absent)
# ---------------------------------------------------------------------------

def _parse_config_path() -> str | None:
    """Extract the ``--config`` value from sys.argv.

    Returns ``None`` when no ``--config`` flag is present, signalling
    the app should show the interactive data picker instead.
    """
    args = sys.argv[1:]  # skip script name
    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


# ---------------------------------------------------------------------------
# Load config & resolve results path  (cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_project_config(config_path: str) -> dict:
    """Load and cache the YAML configuration."""
    return load_config(config_path)


def _resolve_results_path(folder: str) -> str | None:
    """Given a folder or file path, return the HDF5 results path or None."""
    p = Path(folder).expanduser().resolve()
    if p.is_file() and p.suffix in (".h5", ".hdf5"):
        return str(p)
    if p.is_dir():
        candidate = p / _DEFAULT_RESULTS_FILENAME
        if candidate.exists():
            return str(candidate)
        # Check for any .h5 file in the directory
        h5_files = sorted(p.glob("*.h5"))
        if len(h5_files) == 1:
            return str(h5_files[0])
    return None


def _init_from_config(config_path: str) -> bool:
    """Try to initialise session state from a YAML config. Returns True on success."""
    try:
        config = _load_project_config(config_path)
    except FileNotFoundError:
        st.error(
            f"Configuration file not found: **{config_path}**. "
            "Pass `--config path/to/config.yaml` after the Streamlit `--` separator."
        )
        return False
    except Exception as exc:
        st.error(f"Failed to load configuration: **{type(exc).__name__}** -- {exc}")
        return False

    output_dir = config.get("output_dir", "results")
    results_file = config.get("results_file", _DEFAULT_RESULTS_FILENAME)
    results_path = str(Path(output_dir) / results_file)

    st.session_state["config"] = config
    st.session_state["config_path"] = config_path
    st.session_state["results_path"] = results_path
    return True


def _init_session_state() -> None:
    """Populate ``st.session_state`` -- either from CLI config or interactively."""
    if "results_path" in st.session_state:
        return  # already initialised

    config_path = _parse_config_path()

    if config_path is not None:
        if not _init_from_config(config_path):
            st.stop()
        return

    # --- Interactive mode: show the data picker ---
    _show_data_picker()
    st.stop()  # halt until user provides a valid path


# ---------------------------------------------------------------------------
# Interactive data picker (welcome / launcher screen)
# ---------------------------------------------------------------------------

def _show_data_picker() -> None:
    """Render the welcome screen with a folder/file browser."""
    st.markdown("# NovoView")
    st.caption("RNA-Seq Analysis Platform")

    st.markdown("---")

    st.markdown(
        "### Welcome! Point NovoView to your analysis results to get started."
    )
    st.markdown(
        "Enter the path to either:\n"
        f"- A **folder** containing `{_DEFAULT_RESULTS_FILENAME}`\n"
        "- An **HDF5 results file** (`.h5`) directly\n"
        "- A **config.yaml** file from a previous pipeline run"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        user_path = st.text_input(
            "Results path",
            value=st.session_state.get("_picker_path", ""),
            placeholder="/path/to/results or /path/to/novoview_results.h5",
            key="_picker_input",
        )
    with col2:
        st.markdown("<div style='height: 1.75rem'></div>", unsafe_allow_html=True)
        load_clicked = st.button("Load", type="primary", width="stretch")

    if not load_clicked or not user_path.strip():
        _show_picker_help()
        return

    user_path = user_path.strip()
    p = Path(user_path).expanduser().resolve()

    # --- Case 1: user pointed to a config.yaml ---
    if p.is_file() and p.suffix in (".yaml", ".yml"):
        if _init_from_config(str(p)):
            st.rerun()
        return

    # --- Case 2: folder or HDF5 file ---
    results_path = _resolve_results_path(user_path)
    if results_path is None:
        if not p.exists():
            st.error(f"Path does not exist: `{user_path}`")
        elif p.is_dir():
            st.error(
                f"No HDF5 results file found in `{user_path}`. "
                f"Expected `{_DEFAULT_RESULTS_FILENAME}` or a single `.h5` file."
            )
        else:
            st.error(
                f"Unrecognized file type: `{p.name}`. "
                "Please provide a `.h5` results file, a results folder, or a `config.yaml`."
            )
        return

    # Store minimal config and results path
    st.session_state["config"] = {
        "project_name": p.parent.name if p.is_file() else p.name,
        "organism": "human",
    }
    st.session_state["results_path"] = results_path
    st.session_state["_picker_path"] = user_path
    st.rerun()


def _show_picker_help() -> None:
    """Show helpful tips on the welcome screen."""
    st.markdown("---")
    with st.expander("How do I get results to load?", expanded=False):
        st.markdown(
            "1. Run the NovoView pipeline on your RNA-Seq data:\n"
            "   ```bash\n"
            "   python -m novoview.pipeline.run --config config.yaml\n"
            "   ```\n"
            f"2. This produces `{_DEFAULT_RESULTS_FILENAME}` in your output directory.\n"
            "3. Enter that directory path above and click **Load**.\n\n"
            "Alternatively, launch with a config file:\n"
            "   ```bash\n"
            "   streamlit run novoview/app/app.py -- --config config.yaml\n"
            "   ```"
        )


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
        project_name = html.escape(cfg.get("project_name", cfg.get("data_dir", "Unknown")))
        organism = html.escape(cfg.get("organism", "human").capitalize())
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

    # Change data source
    if st.button("Change data source", width="stretch"):
        for key in ("config", "config_path", "results_path", "_picker_path"):
            st.session_state.pop(key, None)
        st.rerun()

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
