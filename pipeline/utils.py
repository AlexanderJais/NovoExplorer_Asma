"""
Utility module for the NovoView RNA-Seq analysis platform.

Provides logging setup, gene ID mapping, column standardization helpers,
flexible file reading, and configuration loading.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Package root – used to resolve resource paths
# ---------------------------------------------------------------------------
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


# ===================================================================
# 1. Logging setup
# ===================================================================

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and return a formatted logger.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__`` of the calling module).
    level : int, optional
        Logging level (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers when called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Module-level logger for internal use
_logger = setup_logger(__name__)


# ===================================================================
# 2. Gene ID mapping loader
# ===================================================================

def load_gene_id_mapping(organism: str) -> dict:
    """Load an Ensembl-to-symbol mapping from a compressed TSV.

    The file is expected at
    ``<novoview_package>/resources/gene_id_mapping_{organism}.tsv.gz``
    with two tab-separated columns: ``ensembl_id`` and ``gene_symbol``.

    Parameters
    ----------
    organism : str
        Organism key, e.g. ``"human"`` or ``"mouse"``.

    Returns
    -------
    dict
        Mapping of ensembl_id -> gene_symbol.  Returns an empty dict
        (with a logged warning) if the file is not found or unreadable.
    """
    filename = f"gene_id_mapping_{organism}.tsv.gz"
    filepath = _PACKAGE_ROOT / "resources" / filename

    try:
        df = pd.read_csv(
            filepath,
            sep="\t",
            compression="gzip",
            dtype=str,
            comment="#",
        )
        # Accept various capitalisation for the two expected columns
        col_map = {}
        for col in df.columns:
            lower = col.strip().lower()
            if lower in ("ensembl_id", "ensembl_gene_id", "gene_id", "id"):
                col_map[col] = "ensembl_id"
            elif lower in ("gene_symbol", "symbol", "gene_name", "genename"):
                col_map[col] = "gene_symbol"
        df = df.rename(columns=col_map)

        if "ensembl_id" not in df.columns or "gene_symbol" not in df.columns:
            _logger.warning(
                "Gene ID mapping file '%s' does not contain the expected "
                "columns (ensembl_id, gene_symbol). Returning empty mapping.",
                filepath,
            )
            return {}

        mapping = (
            df.dropna(subset=["ensembl_id", "gene_symbol"])
            .set_index("ensembl_id")["gene_symbol"]
            .to_dict()
        )
        _logger.info(
            "Loaded %d gene ID mappings for organism '%s'.",
            len(mapping),
            organism,
        )
        return mapping

    except FileNotFoundError:
        _logger.warning(
            "Gene ID mapping file not found: %s. Returning empty mapping.",
            filepath,
        )
        return {}
    except Exception as exc:
        _logger.warning(
            "Failed to load gene ID mapping from '%s': %s. "
            "Returning empty mapping.",
            filepath,
            exc,
        )
        return {}


# ===================================================================
# 3. Column name standardization helpers
# ===================================================================

def find_column(df: pd.DataFrame, candidates: list, required: bool = False):
    """Return the first column in *df* that matches one of *candidates*.

    Matching is case-insensitive and ignores leading/trailing whitespace in
    both the DataFrame columns and the candidate list.

    Parameters
    ----------
    df : pd.DataFrame
    candidates : list of str
    required : bool
        If ``True`` and no match is found, raise ``ValueError``.

    Returns
    -------
    str or None
        The actual column name from *df*, or ``None`` if not found and
        *required* is ``False``.
    """
    # Build a lookup: lowered-stripped actual name -> actual name
    col_lookup = {col.strip().lower(): col for col in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in col_lookup:
            return col_lookup[key]

    if required:
        raise ValueError(
            f"Required column not found. Looked for any of {candidates} "
            f"among columns {list(df.columns)}."
        )
    return None


def _rename_matched(df: pd.DataFrame, standard_name: str,
                    candidates: list) -> pd.DataFrame:
    """Rename the first matching candidate column to *standard_name*."""
    matched = find_column(df, candidates)
    if matched is not None and matched != standard_name:
        df = df.rename(columns={matched: standard_name})
    return df


def standardize_deg_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename DEG table columns to standard names.

    Standard columns (when found): ``gene_id``, ``gene_name``, ``log2fc``,
    ``pvalue``, ``padj``, ``basemean``, ``regulation``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DEG table.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns (copy).
    """
    df = df.copy()

    column_candidates = {
        "gene_id": [
            "gene_id", "Geneid", "Gene ID", "Ensembl_ID",
            "ensembl_gene_id", "Gene", "GeneID", "ID",
        ],
        "gene_name": [
            "gene_name", "Gene Name", "GeneName", "Symbol",
            "gene_symbol", "SYMBOL", "Description",
        ],
        "log2fc": [
            "log2FoldChange", "log2FC", "logFC", "log2(FC)",
            "FoldChange(log2)",
        ],
        "pvalue": ["pvalue", "PValue", "P-value", "pval"],
        "padj": [
            "padj", "FDR", "adj.P.Val", "q_value", "qvalue",
            "BH", "p_adjusted",
        ],
        "basemean": [
            "baseMean", "basemean", "AveExpr", "meanExpression",
        ],
        "regulation": [
            "regulation", "Regulation", "Direction", "regulate",
        ],
    }

    for standard_name, candidates in column_candidates.items():
        df = _rename_matched(df, standard_name, candidates)

    return df


def standardize_enrichment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename enrichment table columns to standard names.

    Standard columns (when found): ``term_id``, ``term_name``,
    ``category``, ``pvalue``, ``padj``, ``gene_count``, ``gene_ratio``,
    ``genes``.

    Parameters
    ----------
    df : pd.DataFrame
        Input enrichment table.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns (copy).
    """
    df = df.copy()

    column_candidates = {
        "term_id": [
            "Term", "GO_term", "ID", "term_id", "GO_ID",
            "KEGG_ID", "PathwayID",
        ],
        "term_name": [
            "Description", "Term", "GO_term", "Pathway",
            "pathway_name", "KEGG_pathway", "term_name",
        ],
        "pvalue": ["pvalue", "PValue", "P-value", "Pvalue"],
        "padj": [
            "padj", "FDR", "q_value", "Adjusted P-value",
            "corrected_pvalue",
        ],
        "gene_count": [
            "Count", "count", "Gene_count", "gene_count",
            "nGenes",
        ],
        "gene_ratio": [
            "GeneRatio", "gene_ratio", "Rich_Factor",
            "Rich Factor", "richFactor",
        ],
        "genes": [
            "geneID", "Genes", "gene_list", "Core_enrichment",
        ],
    }

    for standard_name, candidates in column_candidates.items():
        df = _rename_matched(df, standard_name, candidates)

    return df


# ===================================================================
# 4. File reading helper
# ===================================================================

def read_table_flexible(path) -> pd.DataFrame:
    """Read a tabular file, trying multiple strategies.

    Attempt order:
    1. Tab-separated CSV with UTF-8 encoding
    2. Tab-separated CSV with Latin-1 encoding
    3. Excel file (``pd.read_excel``)

    Comment lines starting with ``#`` are skipped for CSV reads.

    Parameters
    ----------
    path : str or Path
        Path to the file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If none of the strategies succeed.
    """
    path = Path(path)

    # Strategy 1: TSV, UTF-8
    try:
        df = pd.read_csv(path, sep="\t", encoding="utf-8", comment="#")
        if len(df.columns) > 1:
            return df
    except Exception:
        pass

    # Strategy 2: TSV, Latin-1
    try:
        df = pd.read_csv(path, sep="\t", encoding="latin-1", comment="#")
        if len(df.columns) > 1:
            return df
    except Exception:
        pass

    # Strategy 3: Excel
    try:
        df = pd.read_excel(path)
        return df
    except Exception:
        pass

    raise ValueError(
        f"Unable to read '{path}' as TSV (utf-8), TSV (latin-1), or Excel."
    )


# ===================================================================
# 5. Config loader
# ===================================================================

_DEFAULT_CONFIG = {
    "organism": "human",
    "log2fc_threshold": 1.0,
    "padj_threshold": 0.05,
    "top_n_genes": 50,
    "enrichment_databases": ["GO_BP", "GO_MF", "GO_CC", "KEGG"],
    "output_dir": "results",
    "threads": 4,
}


def load_config(path) -> dict:
    """Read a YAML configuration file and fill in defaults.

    Parameters
    ----------
    path : str or Path
        Path to a YAML file.

    Returns
    -------
    dict
        Merged configuration (file values override defaults).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        user_config = yaml.safe_load(fh)

    if user_config is None:
        user_config = {}

    config = {**_DEFAULT_CONFIG, **user_config}

    _logger.info("Loaded configuration from '%s'.", path)
    return config
