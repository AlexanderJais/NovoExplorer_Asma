#!/usr/bin/env python
"""NovoExplorer -- Multi-page Streamlit app (pipeline-backed).

This is the pipeline-backed multi-page app that provides additional
analytical features (PCA/UMAP, gene similarity, GSEA, signature overlap)
on top of the main explorer (``novogene_explorer.py``).

Launch with::

    # Interactive mode (pick data folder, run pipeline in the browser):
    streamlit run app/app.py

    # Config mode (pre-configured project):
    streamlit run app/app.py -- --config config.yaml

For the main Novogene delivery explorer (no pipeline required), use::

    streamlit run novogene_explorer.py -- /path/to/novogene/results
"""

from __future__ import annotations

import html
import logging
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
_DEFAULT_RESULTS_FILENAME = "novoexplorer_results.h5"

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
    page_title="NovoExplorer",
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

def _looks_like_novogene_delivery(folder: Path) -> bool:
    """Return True if *folder* contains directories matching Novogene patterns."""
    if not folder.is_dir():
        return False
    names = {c.name.lower() for c in folder.iterdir() if c.is_dir()}
    # Also check one level down (Novogene sometimes nests under a project dir)
    for child in folder.iterdir():
        if child.is_dir():
            names |= {gc.name.lower() for gc in child.iterdir() if gc.is_dir()}
    markers = {"differential", "enrichment", "quantification"}
    # Match against known Novogene patterns
    for n in names:
        for pat in ("diff", "deg", "enrich", "quant", "readcount", "fpkm"):
            if n.startswith(pat):
                return True
    return bool(markers & names)


def _run_pipeline_in_app(config: dict) -> str | None:
    """Run the analysis pipeline inside the Streamlit app.

    Returns the path to the HDF5 results file on success, or None on
    failure.
    """
    from run_pipeline import run_pipeline

    # Suppress noisy library loggers during the run
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = config.get("output_dir", "results")
    results_path = str(Path(output_dir) / _DEFAULT_RESULTS_FILENAME)

    try:
        run_pipeline(config)
    except Exception as exc:
        st.error(f"Pipeline failed: **{type(exc).__name__}** -- {exc}")
        return None

    if Path(results_path).exists():
        return results_path
    return None


def _show_data_picker() -> None:
    """Render the welcome screen with a folder/file browser."""
    st.markdown("# NovoExplorer")
    st.caption("RNA-Seq Analysis Platform")

    st.markdown("---")

    st.markdown(
        "### Welcome! Point NovoExplorer at your data to get started."
    )
    st.markdown(
        "Enter the path to your **Novogene delivery folder**, or to "
        "previously generated results:"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        user_path = st.text_input(
            "Data path",
            value=st.session_state.get("_picker_path", ""),
            placeholder="/path/to/novogene/delivery  or  /path/to/results.h5",
            key="_picker_input",
        )
    with col2:
        st.markdown("<div style='height: 1.75rem'></div>", unsafe_allow_html=True)
        load_clicked = st.button("Load", type="primary", width="stretch")

    if not load_clicked or not user_path.strip():
        return

    user_path = user_path.strip()
    p = Path(user_path).expanduser().resolve()

    if not p.exists():
        st.error(f"Path does not exist: `{user_path}`")
        return

    # --- Case 1: user pointed to a config.yaml ---
    if p.is_file() and p.suffix in (".yaml", ".yml"):
        if _init_from_config(str(p)):
            st.rerun()
        return

    # --- Case 2: existing HDF5 results ---
    results_path = _resolve_results_path(user_path)
    if results_path is not None:
        st.session_state["config"] = {
            "project_name": p.parent.name if p.is_file() else p.name,
            "organism": "human",
        }
        st.session_state["results_path"] = results_path
        st.session_state["_picker_path"] = user_path
        st.rerun()
        return

    # --- Case 3: raw Novogene delivery folder -> offer to run pipeline ---
    if p.is_dir() and _looks_like_novogene_delivery(p):
        st.session_state["_picker_path"] = user_path
        _show_pipeline_launcher(p)
        return

    # --- Nothing recognised ---
    if p.is_dir():
        st.error(
            f"No results or Novogene data found in `{user_path}`. "
            "Make sure the folder contains Differential/, Enrichment/, or Quantification/ subdirectories."
        )
    else:
        st.error(
            f"Unrecognized file type: `{p.name}`. "
            "Please provide a Novogene delivery folder, `.h5` results file, or `config.yaml`."
        )


def _show_pipeline_launcher(data_dir: Path) -> None:
    """Show pipeline settings and a Run button for a raw Novogene folder."""
    st.success(f"Novogene delivery detected in `{data_dir}`")

    st.markdown("#### Pipeline Settings")
    st.caption("Adjust these if needed, then click **Run Pipeline**.")

    col_a, col_b = st.columns(2)
    with col_a:
        project_name = st.text_input(
            "Project name",
            value=data_dir.name,
            key="_launch_project_name",
        )
        organism = st.selectbox(
            "Organism",
            options=["human", "mouse"],
            key="_launch_organism",
        )
    with col_b:
        padj = st.number_input(
            "Adjusted p-value threshold",
            value=0.05, min_value=0.0, max_value=1.0,
            step=0.01, format="%.2f",
            key="_launch_padj",
        )
        log2fc = st.number_input(
            "log2 fold-change threshold",
            value=1.0, min_value=0.0,
            step=0.5, format="%.1f",
            key="_launch_log2fc",
        )

    output_dir = str(data_dir / "results")

    if st.button("Run Pipeline", type="primary"):
        config = {
            "project_name": project_name,
            "data_dir": str(data_dir),
            "output_dir": output_dir,
            "organism": organism,
            "padj_threshold": padj,
            "log2fc_threshold": log2fc,
        }

        with st.status("Running analysis pipeline...", expanded=True) as status:
            st.write("Ingesting Novogene data...")
            results_path = _run_pipeline_in_app(config)

            if results_path is not None:
                status.update(label="Pipeline complete!", state="complete")
                st.session_state["config"] = config
                st.session_state["results_path"] = results_path
                st.rerun()
            else:
                status.update(label="Pipeline failed", state="error")
                st.error(
                    "The pipeline did not produce a results file. "
                    "Check the terminal for detailed error logs."
                )


_init_session_state()

# ---------------------------------------------------------------------------
# Sidebar -- branding, navigation, about
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("# NovoExplorer")
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
            NovoExplorer v{_VERSION}
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
