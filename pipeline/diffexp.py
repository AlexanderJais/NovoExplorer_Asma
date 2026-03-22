"""
Differential expression analysis module for the NovoView RNA-Seq platform.

Provides utilities for cleaning pre-computed Novogene DEG results, running
de novo differential expression with pyDESeq2, classifying gene regulation,
and summarising results across comparisons.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.utils import setup_logger, standardize_deg_columns

# Conditional import for pyDESeq2 -- it may not be installed in every
# environment.  Functions that require it will raise an informative error
# at call time rather than crashing at import time.
try:
    import anndata as ad
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    _HAS_PYDESEQ2 = True
except ImportError:
    _HAS_PYDESEQ2 = False

logger = setup_logger(__name__)


# ===================================================================
# 1. classify_regulation
# ===================================================================


def classify_regulation(
    log2fc: float,
    padj: float,
    log2fc_threshold: float = 1.0,
    padj_threshold: float = 0.05,
) -> str:
    """Classify a gene as up-regulated, down-regulated, or not significant.

    Parameters
    ----------
    log2fc : float
        Log2 fold-change value.
    padj : float
        Adjusted p-value.
    log2fc_threshold : float, optional
        Absolute log2FC must be at least this value (default 1.0).
    padj_threshold : float, optional
        Adjusted p-value must be at most this value (default 0.05).

    Returns
    -------
    str
        ``'up'``, ``'down'``, or ``'ns'`` (not significant).
    """
    if pd.isna(padj) or pd.isna(log2fc):
        return "ns"
    if padj <= padj_threshold and log2fc >= log2fc_threshold:
        return "up"
    if padj <= padj_threshold and log2fc <= -log2fc_threshold:
        return "down"
    return "ns"


def _vectorized_classify(
    log2fc: pd.Series,
    padj: pd.Series,
    log2fc_threshold: float = 1.0,
    padj_threshold: float = 0.05,
) -> pd.Series:
    """Vectorized version of :func:`classify_regulation` for entire columns."""
    result = pd.Series("ns", index=log2fc.index)
    sig = padj.le(padj_threshold) & padj.notna() & log2fc.notna()
    result[sig & log2fc.ge(log2fc_threshold)] = "up"
    result[sig & log2fc.le(-log2fc_threshold)] = "down"
    return result


# ===================================================================
# 2. parse_novogene_deg  (clean pre-computed DEG results)
# ===================================================================


def parse_novogene_deg(
    deg_results: Dict[str, pd.DataFrame],
    log2fc_threshold: float = 1.0,
    padj_threshold: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """Validate and clean already-parsed Novogene DEG result DataFrames.

    This function takes the dictionary produced by
    :func:`pipeline.ingest.parse_deg_results` and performs additional
    cleaning steps:

    - Ensures ``log2fc``, ``pvalue``, and ``padj`` columns are numeric.
    - Drops rows where ``padj`` is NaN.
    - Adds a ``regulation`` column (if missing) based on *log2fc_threshold*
      and *padj_threshold*.

    Parameters
    ----------
    deg_results : dict[str, DataFrame]
        Mapping of comparison name to DEG DataFrame (with standardised
        column names from ingest).
    log2fc_threshold : float, optional
        Absolute log2FC threshold for regulation classification (default 1.0).
    padj_threshold : float, optional
        Adjusted p-value threshold for regulation classification (default 0.05).

    Returns
    -------
    dict[str, DataFrame]
        Cleaned copy of the input dict.  Comparisons whose DataFrames become
        empty after cleaning are excluded.
    """
    cleaned: Dict[str, pd.DataFrame] = {}

    for comp_name, df in deg_results.items():
        logger.info("Cleaning Novogene DEG results for %s (%d rows)", comp_name, len(df))
        df = df.copy()

        # Standardise column names (defensive -- ingest should already do this)
        df = standardize_deg_columns(df)

        # Coerce numeric columns (warn if values are lost)
        for col in ("log2fc", "pvalue", "padj"):
            if col in df.columns:
                before_na = df[col].isna().sum()
                df[col] = pd.to_numeric(df[col], errors="coerce")
                coerced = df[col].isna().sum() - before_na
                if coerced > 0:
                    logger.warning(
                        "Comparison '%s': %d non-numeric values in '%s' coerced to NaN.",
                        comp_name, coerced, col,
                    )

        # Drop rows with missing padj
        if "padj" in df.columns:
            n_before = len(df)
            df = df.dropna(subset=["padj"])
            n_dropped = n_before - len(df)
            if n_dropped > 0:
                logger.info("  Dropped %d rows with NaN padj", n_dropped)
        else:
            logger.warning("  Column 'padj' not found in %s -- skipping padj filter", comp_name)

        # Add regulation column if missing (vectorized for performance)
        if "regulation" not in df.columns:
            if "log2fc" in df.columns and "padj" in df.columns:
                df["regulation"] = _vectorized_classify(
                    df["log2fc"], df["padj"],
                    log2fc_threshold=log2fc_threshold,
                    padj_threshold=padj_threshold,
                )
                logger.info("  Added regulation column")
            else:
                logger.warning(
                    "  Cannot add regulation column -- missing log2fc or padj in %s",
                    comp_name,
                )

        if df.empty:
            logger.warning("  Comparison %s is empty after cleaning -- skipping", comp_name)
            continue

        cleaned[comp_name] = df
        logger.info("  Cleaned %s: %d genes", comp_name, len(df))

    logger.info("Cleaned %d / %d Novogene DEG comparisons", len(cleaned), len(deg_results))
    return cleaned


# ===================================================================
# 3. run_pydeseq2
# ===================================================================


def run_pydeseq2(
    counts_df: pd.DataFrame,
    sample_groups: Dict[str, str] | pd.DataFrame,
    comparisons: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, pd.DataFrame]:
    """Run differential expression analysis using pyDESeq2.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Raw count matrix with genes as rows (index) and samples as columns.
        Values must be non-negative integers or integer-like floats.
    sample_groups : dict or DataFrame
        If a *dict*, maps sample name -> group label.  If a *DataFrame*,
        must contain columns ``sample_id`` and ``group``.
    comparisons : list of (str, str) tuples, optional
        Each tuple is ``(group_a, group_b)`` where the fold-change is
        computed as group_b / group_a (i.e. positive log2FC means higher in
        group_b).  If *None*, all pairwise comparisons are generated.

    Returns
    -------
    dict[str, DataFrame]
        Mapping of ``"group_a_vs_group_b"`` to a results DataFrame with
        columns: ``gene_id``, ``gene_name``, ``log2fc``, ``pvalue``,
        ``padj``, ``basemean``, ``regulation``.

    Raises
    ------
    ImportError
        If pyDESeq2 or anndata is not installed.
    ValueError
        If the input data or sample groups are inconsistent.
    """
    if not _HAS_PYDESEQ2:
        raise ImportError(
            "pyDESeq2 is required for de novo differential expression analysis. "
            "Install it with: pip install pydeseq2"
        )

    # --- normalise sample_groups to a dict ---
    if isinstance(sample_groups, pd.DataFrame):
        if "sample_id" in sample_groups.columns and "group" in sample_groups.columns:
            group_map: Dict[str, str] = dict(
                zip(sample_groups["sample_id"], sample_groups["group"])
            )
        else:
            raise ValueError(
                "sample_groups DataFrame must contain 'sample_id' and 'group' columns."
            )
    else:
        group_map = dict(sample_groups)

    # --- intersect with available samples ---
    available = set(counts_df.columns) & set(group_map.keys())
    if len(available) < 2:
        raise ValueError(
            f"Need at least 2 samples present in both the count matrix and "
            f"sample_groups.  Found {len(available)} overlapping sample(s)."
        )

    missing_from_counts = set(group_map.keys()) - set(counts_df.columns)
    if missing_from_counts:
        logger.warning(
            "%d sample(s) in sample_groups not found in count matrix: %s",
            len(missing_from_counts),
            sorted(missing_from_counts)[:5],
        )

    # Build a metadata DataFrame aligned to the count matrix columns
    meta = pd.DataFrame(
        {"group": [group_map[s] for s in counts_df.columns if s in group_map]},
        index=[s for s in counts_df.columns if s in group_map],
    )
    counts_sub = counts_df[meta.index].copy()

    groups = sorted(meta["group"].unique())
    logger.info(
        "pyDESeq2 input: %d genes x %d samples in %d groups (%s)",
        counts_sub.shape[0],
        counts_sub.shape[1],
        len(groups),
        groups,
    )

    # --- generate comparisons if not specified ---
    if comparisons is None:
        comparisons = list(combinations(groups, 2))
        logger.info("Auto-generated %d pairwise comparisons", len(comparisons))

    # --- run each comparison ---
    results: Dict[str, pd.DataFrame] = {}

    for group_a, group_b in comparisons:
        comp_name = f"{group_a}_vs_{group_b}"
        logger.info("Running pyDESeq2: %s", comp_name)

        try:
            # Subset to the two groups
            samples_ab = meta.index[meta["group"].isin({group_a, group_b})].tolist()
            if len(samples_ab) < 2:
                logger.warning(
                    "  Skipping %s -- fewer than 2 samples in the two groups",
                    comp_name,
                )
                continue

            meta_ab = meta.loc[samples_ab].copy()
            counts_ab = counts_sub[samples_ab].copy()

            # Ensure integer counts (pyDESeq2 requirement)
            counts_ab = counts_ab.round().astype(int)

            # Build AnnData (pyDESeq2 expects samples x genes)
            adata = ad.AnnData(
                X=counts_ab.T.values,
                obs=meta_ab,
                var=pd.DataFrame(index=counts_ab.index),
            )

            # Fit the DESeq2 model
            dds = DeseqDataSet(
                adata=adata,
                design_factors="group",
                refit_cooks=True,
            )
            dds.deseq2()

            # Extract statistics for this contrast
            stat = DeseqStats(
                dds,
                contrast=["group", group_b, group_a],
            )
            stat.summary()

            # Build results DataFrame
            res_df = stat.results_df.copy()
            res_df = res_df.reset_index()
            res_df.columns = [c.strip() for c in res_df.columns]

            # Map pyDESeq2 column names to our standard names
            rename_map = {
                "index": "gene_id",
                "baseMean": "basemean",
                "log2FoldChange": "log2fc",
                "pvalue": "pvalue",
                "padj": "padj",
            }
            res_df = res_df.rename(columns=rename_map)

            # Keep only the columns we need
            keep_cols = ["gene_id", "basemean", "log2fc", "pvalue", "padj"]
            keep_cols = [c for c in keep_cols if c in res_df.columns]
            res_df = res_df[keep_cols].copy()

            # Add gene_name (same as gene_id when no mapping is available)
            if "gene_name" not in res_df.columns:
                res_df["gene_name"] = res_df["gene_id"]

            # Add regulation column (vectorized for performance)
            if "log2fc" in res_df.columns and "padj" in res_df.columns:
                res_df["regulation"] = _vectorized_classify(
                    res_df["log2fc"], res_df["padj"],
                )
            else:
                res_df["regulation"] = "ns"

            results[comp_name] = res_df
            n_up = (res_df["regulation"] == "up").sum()
            n_down = (res_df["regulation"] == "down").sum()
            logger.info(
                "  %s complete: %d genes, %d up, %d down",
                comp_name,
                len(res_df),
                n_up,
                n_down,
            )

        except Exception as exc:
            logger.error(
                "  pyDESeq2 failed for %s: %s. Skipping this comparison.",
                comp_name,
                exc,
                exc_info=True,
            )

    logger.info("pyDESeq2 completed %d / %d comparisons", len(results), len(comparisons))
    return results


# ===================================================================
# 4. get_significant_genes
# ===================================================================


def get_significant_genes(
    deg_df: pd.DataFrame,
    padj_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
) -> pd.DataFrame:
    """Filter a DEG table to only significant genes.

    A gene is considered significant if ``padj < padj_threshold`` and
    ``|log2fc| > log2fc_threshold``.

    Parameters
    ----------
    deg_df : pd.DataFrame
        DEG result table with at least ``log2fc`` and ``padj`` columns.
    padj_threshold : float, optional
        Maximum adjusted p-value (default 0.05).
    log2fc_threshold : float, optional
        Minimum absolute log2 fold-change (default 1.0).

    Returns
    -------
    pd.DataFrame
        Subset of *deg_df* containing only significant genes.
    """
    if "padj" not in deg_df.columns or "log2fc" not in deg_df.columns:
        logger.warning(
            "Cannot filter significant genes -- missing 'padj' or 'log2fc' column."
        )
        return deg_df

    mask = (deg_df["padj"] <= padj_threshold) & (deg_df["log2fc"].abs() >= log2fc_threshold)
    sig = deg_df.loc[mask].copy()
    logger.info(
        "Significant genes: %d / %d (padj < %g, |log2fc| > %g)",
        len(sig),
        len(deg_df),
        padj_threshold,
        log2fc_threshold,
    )
    return sig


# ===================================================================
# 5. summarize_deg_results
# ===================================================================


def summarize_deg_results(
    deg_results: Dict[str, pd.DataFrame],
    padj_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
) -> pd.DataFrame:
    """Summarise DEG counts across all comparisons.

    Parameters
    ----------
    deg_results : dict[str, DataFrame]
        Mapping of comparison name to DEG DataFrame.
    padj_threshold : float, optional
        Adjusted p-value threshold (default 0.05).
    log2fc_threshold : float, optional
        Absolute log2FC threshold (default 1.0).

    Returns
    -------
    pd.DataFrame
        Summary table with columns: ``comparison``, ``total_deg``, ``up``,
        ``down``, ``not_significant``.
    """
    rows: List[Dict[str, Any]] = []

    for comp_name, df in sorted(deg_results.items()):
        sig = get_significant_genes(df, padj_threshold=padj_threshold, log2fc_threshold=log2fc_threshold)
        n_up = 0
        n_down = 0

        if "regulation" in sig.columns:
            n_up = (sig["regulation"] == "up").sum()
            n_down = (sig["regulation"] == "down").sum()
        elif "log2fc" in sig.columns:
            n_up = (sig["log2fc"] > 0).sum()
            n_down = (sig["log2fc"] < 0).sum()

        rows.append(
            {
                "comparison": comp_name,
                "total_deg": len(sig),
                "up": int(n_up),
                "down": int(n_down),
                "not_significant": len(df) - len(sig),
            }
        )

    summary = pd.DataFrame(rows)
    logger.info("DEG summary across %d comparisons:\n%s", len(rows), summary.to_string(index=False))
    return summary


# ===================================================================
# 6. run_diffexp  (main orchestrator)
# ===================================================================


def run_diffexp(
    counts_df: Optional[pd.DataFrame] = None,
    sample_groups: Optional[Dict[str, str] | pd.DataFrame] = None,
    novogene_deg: Optional[Dict[str, pd.DataFrame]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Main differential expression orchestrator.

    Decision logic:

    - If *novogene_deg* is provided and ``config['rerun_de']`` is ``False``
      (the default): use the cleaned Novogene results only.
    - If *novogene_deg* is provided and ``config['rerun_de']`` is ``True``:
      run pyDESeq2 and return **both** Novogene and pyDESeq2 results.
    - If *novogene_deg* is not provided: run pyDESeq2 on *counts_df*.

    Parameters
    ----------
    counts_df : DataFrame, optional
        Raw count matrix (genes x samples).  Required when pyDESeq2 will be
        run.
    sample_groups : dict or DataFrame, optional
        Sample-to-group mapping.  Required when pyDESeq2 will be run.
    novogene_deg : dict[str, DataFrame], optional
        Pre-computed DEG results from Novogene (as returned by
        :func:`pipeline.ingest.parse_deg_results`).
    config : dict, optional
        Configuration dict.  Recognised keys:

        - ``rerun_de`` (bool, default False): whether to re-run DE with
          pyDESeq2 even when Novogene results are available.
        - ``log2fc_threshold`` (float, default 1.0)
        - ``padj_threshold`` (float, default 0.05)
        - ``comparisons`` (list of [str, str] pairs, optional)

    Returns
    -------
    dict
        Keys:

        - ``'novogene'``: cleaned Novogene DEG dict (or None).
        - ``'pydeseq2'``: pyDESeq2 DEG dict (or None).
        - ``'primary'``: the dict that should be used as the primary result
          (alias pointing to one of the above).
        - ``'summary'``: summary DataFrame for the primary results.
    """
    config = config or {}
    rerun_de = config.get("rerun_de", False)
    log2fc_threshold = config.get("log2fc_threshold", 1.0)
    padj_threshold = config.get("padj_threshold", 0.05)
    comparisons = config.get("comparisons", None)

    # Convert comparisons list-of-lists to list-of-tuples if needed
    if comparisons is not None:
        comparisons = [tuple(c) for c in comparisons]

    novogene_cleaned: Optional[Dict[str, pd.DataFrame]] = None
    pydeseq2_results: Optional[Dict[str, pd.DataFrame]] = None

    # --- Clean Novogene results if available ---
    if novogene_deg:
        logger.info("Cleaning Novogene DEG results (%d comparisons)", len(novogene_deg))
        novogene_cleaned = parse_novogene_deg(
            novogene_deg,
            log2fc_threshold=log2fc_threshold,
            padj_threshold=padj_threshold,
        )

    # --- Decide whether to run pyDESeq2 ---
    should_run_pydeseq2 = False

    if novogene_cleaned is None or len(novogene_cleaned) == 0:
        # No Novogene results -- must run pyDESeq2
        should_run_pydeseq2 = True
        logger.info("No Novogene DEG results available -- will run pyDESeq2")
    elif rerun_de:
        should_run_pydeseq2 = True
        logger.info("config.rerun_de is True -- will run pyDESeq2 in addition to Novogene results")

    if should_run_pydeseq2:
        if counts_df is None or sample_groups is None:
            logger.warning(
                "Cannot run pyDESeq2: counts_df or sample_groups not provided. "
                "Falling back to Novogene results only (if available)."
            )
        else:
            logger.info("Running pyDESeq2 differential expression analysis")
            try:
                pydeseq2_results = run_pydeseq2(
                    counts_df=counts_df,
                    sample_groups=sample_groups,
                    comparisons=comparisons,
                )
            except ImportError:
                logger.error(
                    "pyDESeq2 is not installed. Install with: pip install pydeseq2"
                )
            except Exception as exc:
                logger.error("pyDESeq2 analysis failed: %s", exc, exc_info=True)

    # --- Determine primary results ---
    if pydeseq2_results and len(pydeseq2_results) > 0:
        primary = pydeseq2_results
        primary_source = "pydeseq2"
    elif novogene_cleaned and len(novogene_cleaned) > 0:
        primary = novogene_cleaned
        primary_source = "novogene"
    else:
        primary = {}
        primary_source = "none"

    # --- Build summary ---
    summary = pd.DataFrame()
    if primary:
        summary = summarize_deg_results(
            primary,
            padj_threshold=padj_threshold,
            log2fc_threshold=log2fc_threshold,
        )

    logger.info(
        "Differential expression complete. Primary source: %s, comparisons: %d",
        primary_source,
        len(primary),
    )

    return {
        "novogene": novogene_cleaned,
        "pydeseq2": pydeseq2_results,
        "primary": primary,
        "primary_source": primary_source,
        "summary": summary,
    }
