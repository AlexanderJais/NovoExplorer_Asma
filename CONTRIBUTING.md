# Contributing to NovoExplorer

Thank you for your interest in contributing to NovoExplorer! This document provides guidelines and instructions for contributing.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/AlexanderJais/NovoExplorer_Asma.git
cd NovoExplorer_Asma

# Run the setup script
bash setup.sh

# Activate the virtual environment
source .venv/bin/activate
```

## Project Layout

NovoExplorer is organized into four main packages:

- **`pipeline/`** -- Analysis pipeline modules (ingest, normalize, QC, diffexp, similarity, signatures, persistence)
- **`plotting/`** -- Visualization builders (Plotly + Matplotlib with Nature journal styling)
- **`app/`** -- Multi-page Streamlit app (pipeline-backed, with reusable UI components)
- **`tests/`** -- Pytest test suite covering all pipeline modules

The main entry point `novogene_explorer.py` is a standalone Streamlit app that works directly with Novogene delivery folders.

## Code Style

- **Python 3.10+** features are used throughout (union types with `|`, match statements, etc.)
- **Docstrings** follow the [NumPy/SciPy style](https://numpydoc.readthedocs.io/en/latest/format.html) with `Parameters`, `Returns`, and optional `Examples` sections
- **Type hints** are expected on all public function signatures
- **Imports** use `from __future__ import annotations` for forward references

## Docstring Guidelines

All public functions and classes should have docstrings. Use the NumPy format:

```python
def my_function(param1: str, param2: int = 10) -> pd.DataFrame:
    """Short one-line summary.

    Longer description if needed, explaining behavior, edge cases,
    or important details.

    Parameters
    ----------
    param1 : str
        Description of param1.
    param2 : int, optional
        Description of param2 (default 10).

    Returns
    -------
    pd.DataFrame
        Description of return value.
    """
```

Private helper functions (prefixed with `_`) should have at least a one-line docstring.

## Running Tests

```bash
# Run the full test suite
pytest tests/

# Run a specific test module
pytest tests/test_ingest.py -v

# Run with coverage
pytest tests/ --cov=pipeline --cov=plotting --cov-report=term-missing
```

All new pipeline or plotting features should include corresponding tests.

## Making Changes

1. Create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Ensure all tests pass
4. Update documentation if your change affects the public API or user-facing behavior
5. Submit a pull request with a description of what changed and why

## Adding a New Pipeline Stage

1. Create a new module in `pipeline/` following the pattern of existing modules
2. Add a `run_<stage>()` function as the public entry point
3. Wire it into `run_pipeline.py` in the appropriate order
4. Add persistence support in `pipeline/persistence.py` if the stage produces results
5. Add tests in `tests/test_<stage>.py`

## Adding a New Visualization

1. Create or extend a module in `plotting/`
2. Use `plotting/theme.py` for consistent styling (colors, fonts, layout)
3. Support both Plotly (interactive) and Matplotlib (static) where practical
4. Return figure objects -- let the caller handle rendering

## Adding a New App Page

1. Create a new file in `app/pages/` following the `NN_name.py` naming convention
2. Register the page in `app/app.py`'s `_PAGES` list
3. Use components from `app/components/` for consistent UI (filters, downloads, gene basket)
4. Include a `main()` function as the page entry point with a module-level docstring

## Reporting Issues

When reporting bugs, please include:

- Steps to reproduce the issue
- Your Python version and platform
- The Novogene delivery folder structure (directory tree, anonymized if needed)
- Any error messages or tracebacks
