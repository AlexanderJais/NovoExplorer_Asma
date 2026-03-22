"""
GSEA, ORA, and signature overlap analysis for the NovoView RNA-Seq platform.

Provides gene-set enrichment analysis (pre-ranked GSEA), over-representation
analysis (ORA), cross-comparison signature overlap (Jaccard), and utilities
for identifying core and unique enriched gene sets.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pipeline.utils import setup_logger

# gseapy is an optional heavyweight dependency -- fail gracefully at call time
try:
    import gseapy
    _HAS_GSEAPY = True
except ImportError:
    _HAS_GSEAPY = False

logger = setup_logger(__name__)

# Default gene-set databases used when none are specified in config
_DEFAULT_GENE_SET_DBS = [
    "MSigDB_Hallmark_2020",
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
]


# ===================================================================
# 1. run_preranked_gsea
# ===================================================================

def run_preranked_gsea(
    deg_df: pd.DataFrame,
    gene_set_db: str,
    organism: str = "human",
) -> tuple:
    """Run pre-ranked GSEA using gseapy.

    Genes are ranked by ``sign(log2fc) * -log10(pvalue)``.

    Parameters
    ----------
    deg_df : pd.DataFrame
        DEG results table.  Must contain columns ``gene_name`` (or
        ``gene_id`` as fallback), ``log2fc``, and ``pvalue``.
    gene_set_db : str
        Name of the gene-set database recognised by Enrichr / gseapy
        (e.g. ``"MSigDB_Hallmark_2020"``).
    organism : str, optional
        Organism key (default ``"human"``).

    Returns
    -------
    tuple[object | None, pd.DataFrame]
        A tuple of (gseapy results object, cleaned summary DataFrame).
        The summary has columns: ``term``, ``es``, ``nes``, ``pvalue``,
        ``fdr``, ``lead_genes``.  Returns ``(None, empty DataFrame)`` on
        failure.
    """
    if not _HAS_GSEAPY:
        logger.error(
            "gseapy is required for GSEA.  Install with: pip install gseapy"
        )
        return None, pd.DataFrame()

    # --- Determine gene name column ---
    gene_col = "gene_name" if "gene_name" in deg_df.columns else "gene_id"
    if gene_col not in deg_df.columns:
        logger.error(
            "DEG DataFrame must contain 'gene_name' or 'gene_id' column."
        )
        return None, pd.DataFrame()

    # --- Build ranking ---
    df = deg_df[[gene_col, "log2fc", "pvalue"]].dropna().copy()
    if df.empty:
        logger.warning("No valid rows to rank for GSEA.")
        return None, pd.DataFrame()

    # Clamp pvalues to valid range [1e-300, 1.0]
    df["pvalue"] = df["pvalue"].clip(lower=1e-300, upper=1.0)

    df["rank_score"] = np.sign(df["log2fc"]) * -np.log10(df["pvalue"])

    # De-duplicate genes, keeping the entry with the highest absolute score
    df["abs_rank"] = df["rank_score"].abs()
    df = df.sort_values("abs_rank", ascending=False).drop_duplicates(
        subset=[gene_col], keep="first"
    )
    df = df.drop(columns=["abs_rank"])

    # gseapy.prerank expects a Series or DataFrame indexed by gene name
    ranking = df.set_index(gene_col)["rank_score"].sort_values(ascending=False)

    logger.info(
        "Running pre-ranked GSEA against '%s' with %d ranked genes.",
        gene_set_db,
        len(ranking),
    )

    try:
        res = gseapy.prerank(
            rnk=ranking,
            gene_sets=gene_set_db,
            organism=organism,
            permutation_num=1000,
            outdir=None,
            no_plot=True,
            seed=42,
            verbose=False,
        )

        # Clean results into a tidy DataFrame
        res_df = res.res2d.copy()

        summary = pd.DataFrame({
            "term": res_df["Term"],
            "es": pd.to_numeric(res_df["ES"], errors="coerce"),
            "nes": pd.to_numeric(res_df["NES"], errors="coerce"),
            "pvalue": pd.to_numeric(res_df["NOM p-val"], errors="coerce"),
            "fdr": pd.to_numeric(res_df["FDR q-val"], errors="coerce"),
            "lead_genes": res_df["Lead_genes"].astype(str),
        }).reset_index(drop=True)

        logger.info(
            "GSEA complete for '%s': %d terms evaluated, %d with FDR < 0.25.",
            gene_set_db,
            len(summary),
            (summary["fdr"] < 0.25).sum(),
        )
        return res, summary

    except Exception as exc:
        logger.error(
            "Pre-ranked GSEA failed for gene set '%s': %s",
            gene_set_db,
            exc,
            exc_info=True,
        )
        return None, pd.DataFrame()


# ===================================================================
# 2. run_ora
# ===================================================================

def run_ora(
    gene_list: List[str],
    gene_set_db: str,
    background_genes: Optional[List[str]] = None,
    organism: str = "human",
) -> pd.DataFrame:
    """Run over-representation analysis using gseapy.

    Parameters
    ----------
    gene_list : list of str
        Significant gene names to test for enrichment.
    gene_set_db : str
        Name of the gene-set database recognised by Enrichr / gseapy.
    background_genes : list of str, optional
        Background gene universe.  If *None*, the database default is used.
    organism : str, optional
        Organism key (default ``"human"``).

    Returns
    -------
    pd.DataFrame
        Cleaned summary DataFrame with columns: ``term``, ``overlap``,
        ``pvalue``, ``fdr``, ``genes``.  Returns an empty DataFrame on
        failure.
    """
    if not _HAS_GSEAPY:
        logger.error(
            "gseapy is required for ORA.  Install with: pip install gseapy"
        )
        return pd.DataFrame()

    if not gene_list:
        logger.warning("Empty gene list provided for ORA -- skipping.")
        return pd.DataFrame()

    logger.info(
        "Running ORA against '%s' with %d genes.",
        gene_set_db,
        len(gene_list),
    )

    try:
        enr = gseapy.enrich(
            gene_list=gene_list,
            gene_sets=gene_set_db,
            background=background_genes,
            organism=organism,
            outdir=None,
            no_plot=True,
            verbose=False,
        )

        res_df = enr.results.copy()

        summary = pd.DataFrame({
            "term": res_df["Term"],
            "overlap": res_df["Overlap"],
            "pvalue": pd.to_numeric(res_df["P-value"], errors="coerce"),
            "fdr": pd.to_numeric(res_df["Adjusted P-value"], errors="coerce"),
            "genes": res_df["Genes"].astype(str),
        }).reset_index(drop=True)

        logger.info(
            "ORA complete for '%s': %d terms, %d with FDR < 0.05.",
            gene_set_db,
            len(summary),
            (summary["fdr"] < 0.05).sum(),
        )
        return summary

    except Exception as exc:
        logger.error(
            "ORA failed for gene set '%s': %s",
            gene_set_db,
            exc,
            exc_info=True,
        )
        return pd.DataFrame()


# ===================================================================
# 3. run_enrichment_analysis
# ===================================================================

def run_enrichment_analysis(
    deg_results: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """Run GSEA and ORA for every comparison and gene-set database.

    For each comparison the function:

    1. Identifies significant up- and down-regulated genes.
    2. Runs pre-ranked GSEA on the full DEG table.
    3. Runs ORA separately on up-regulated and down-regulated gene lists.

    Parameters
    ----------
    deg_results : dict[str, DataFrame]
        Mapping of comparison name -> DEG DataFrame (must have standard
        columns: ``gene_name``, ``log2fc``, ``pvalue``, ``padj``).
    config : dict
        Pipeline configuration.  Recognised keys:

        * ``enrichment_databases`` – list of gene-set database names
          (default :data:`_DEFAULT_GENE_SET_DBS`).
        * ``organism`` – organism string (default ``"human"``).
        * ``padj_threshold`` – adjusted p-value cutoff (default 0.05).
        * ``log2fc_threshold`` – absolute log2FC cutoff (default 1.0).

    Returns
    -------
    dict
        Nested dict: ``comparison -> database -> {gsea, ora_up, ora_down}``.
        Each leaf value is a :class:`~pandas.DataFrame`.
    """
    databases = config.get("enrichment_databases", _DEFAULT_GENE_SET_DBS)
    organism = config.get("organism", "human")
    padj_thresh = config.get("padj_threshold", 0.05)
    log2fc_thresh = config.get("log2fc_threshold", 1.0)

    results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}

    for comp_name, deg_df in deg_results.items():
        logger.info("=== Enrichment analysis for %s ===", comp_name)
        results[comp_name] = {}

        # Determine gene name column
        gene_col = "gene_name" if "gene_name" in deg_df.columns else "gene_id"
        if gene_col not in deg_df.columns:
            logger.warning(
                "Comparison '%s' lacks gene_name/gene_id columns; skipping.",
                comp_name,
            )
            continue

        # Identify significant genes
        has_required = "padj" in deg_df.columns and "log2fc" in deg_df.columns
        sig_mask = (
            (deg_df["padj"] < padj_thresh)
            & (deg_df["log2fc"].abs() > log2fc_thresh)
        ) if has_required else pd.Series(False, index=deg_df.index)

        if has_required:
            up_genes = deg_df.loc[sig_mask & (deg_df["log2fc"] > 0), gene_col].dropna().unique().tolist()
            down_genes = deg_df.loc[sig_mask & (deg_df["log2fc"] < 0), gene_col].dropna().unique().tolist()
        else:
            up_genes = []
            down_genes = []

        # Background: all genes in this comparison
        all_genes = deg_df[gene_col].dropna().unique().tolist()

        logger.info(
            "  %s: %d up-regulated, %d down-regulated genes.",
            comp_name,
            len(up_genes),
            len(down_genes),
        )

        if not up_genes and not down_genes:
            logger.warning(
                "  No significant genes for %s -- skipping enrichment.",
                comp_name,
            )
            continue

        for db in databases:
            logger.info("  Database: %s", db)
            db_results: Dict[str, pd.DataFrame] = {}

            # Pre-ranked GSEA on full DEG table
            _, gsea_df = run_preranked_gsea(deg_df, db, organism=organism)
            db_results["gsea"] = gsea_df

            # ORA on up-regulated genes
            db_results["ora_up"] = run_ora(
                up_genes, db, background_genes=all_genes, organism=organism
            )

            # ORA on down-regulated genes
            db_results["ora_down"] = run_ora(
                down_genes, db, background_genes=all_genes, organism=organism
            )

            results[comp_name][db] = db_results

    logger.info("Enrichment analysis complete for %d comparisons.", len(results))
    return results


# ===================================================================
# 4. compute_signature_overlap
# ===================================================================

def compute_signature_overlap(
    deg_results: Dict[str, pd.DataFrame],
    padj_threshold: float = 0.05,
) -> pd.DataFrame:
    """Compute Jaccard index of significant gene sets across comparisons.

    For each pair of comparisons the Jaccard index is calculated as
    ``|A & B| / |A | B|`` where *A* and *B* are the sets of genes with
    ``padj < padj_threshold``.

    Parameters
    ----------
    deg_results : dict[str, DataFrame]
        Mapping of comparison name -> DEG DataFrame.
    padj_threshold : float, optional
        Adjusted p-value cutoff (default 0.05).

    Returns
    -------
    pd.DataFrame
        Square DataFrame (comparisons x comparisons) of Jaccard values.
        Diagonal entries are 1.0.
    """
    comp_names = sorted(deg_results.keys())
    gene_col_map: Dict[str, str] = {}
    sig_sets: Dict[str, set] = {}

    for comp in comp_names:
        df = deg_results[comp]
        gene_col = "gene_name" if "gene_name" in df.columns else "gene_id"
        gene_col_map[comp] = gene_col

        if "padj" in df.columns and gene_col in df.columns:
            sig_genes = set(
                df.loc[df["padj"] < padj_threshold, gene_col].dropna().unique()
            )
        else:
            sig_genes = set()

        sig_sets[comp] = sig_genes

    n = len(comp_names)
    matrix = np.ones((n, n), dtype=float)

    for i, j in combinations(range(n), 2):
        a = sig_sets[comp_names[i]]
        b = sig_sets[comp_names[j]]
        union = a | b
        if len(union) == 0:
            jaccard = 0.0
        else:
            jaccard = len(a & b) / len(union)
        matrix[i, j] = jaccard
        matrix[j, i] = jaccard

    overlap_df = pd.DataFrame(matrix, index=comp_names, columns=comp_names)

    logger.info(
        "Signature overlap matrix computed for %d comparisons.", n
    )
    return overlap_df


# ===================================================================
# 5. find_core_signatures
# ===================================================================

def find_core_signatures(
    enrichment_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    min_comparisons: int = 2,
    padj_threshold: float = 0.05,
) -> pd.DataFrame:
    """Find gene sets that are significantly enriched in multiple comparisons.

    Only GSEA results are inspected (``fdr`` column).

    Parameters
    ----------
    enrichment_results : dict
        Nested dict as returned by :func:`run_enrichment_analysis`.
    min_comparisons : int, optional
        Minimum number of comparisons in which a term must be significant
        (default 2).
    padj_threshold : float, optional
        FDR threshold for significance (default 0.05).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``term``, ``n_comparisons``,
        ``comparisons_list``.
    """
    # term -> list of comparisons where it is significant
    term_map: Dict[str, List[str]] = {}

    for comp_name, db_dict in enrichment_results.items():
        for db_name, analysis_dict in db_dict.items():
            gsea_df = analysis_dict.get("gsea", pd.DataFrame())
            if gsea_df.empty or "fdr" not in gsea_df.columns:
                continue

            sig_terms = gsea_df.loc[
                gsea_df["fdr"] < padj_threshold, "term"
            ].dropna().unique()

            for term in sig_terms:
                term_map.setdefault(term, []).append(comp_name)

    # Filter to terms appearing in >= min_comparisons
    rows = []
    for term, comps in term_map.items():
        # De-duplicate in case a term appears in multiple databases
        unique_comps = sorted(set(comps))
        if len(unique_comps) >= min_comparisons:
            rows.append({
                "term": term,
                "n_comparisons": len(unique_comps),
                "comparisons_list": ", ".join(unique_comps),
            })

    core_df = pd.DataFrame(rows)
    if not core_df.empty:
        core_df = core_df.sort_values("n_comparisons", ascending=False).reset_index(drop=True)

    logger.info(
        "Found %d core signatures enriched in >= %d comparisons.",
        len(core_df),
        min_comparisons,
    )
    return core_df


# ===================================================================
# 6. find_unique_signatures
# ===================================================================

def find_unique_signatures(
    enrichment_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    padj_threshold: float = 0.05,
) -> pd.DataFrame:
    """Find gene sets significant in exactly one comparison.

    Only GSEA results are inspected (``fdr`` column).

    Parameters
    ----------
    enrichment_results : dict
        Nested dict as returned by :func:`run_enrichment_analysis`.
    padj_threshold : float, optional
        FDR threshold for significance (default 0.05).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``term``, ``comparison``.
    """
    term_map: Dict[str, List[str]] = {}

    for comp_name, db_dict in enrichment_results.items():
        for db_name, analysis_dict in db_dict.items():
            gsea_df = analysis_dict.get("gsea", pd.DataFrame())
            if gsea_df.empty or "fdr" not in gsea_df.columns:
                continue

            sig_terms = gsea_df.loc[
                gsea_df["fdr"] < padj_threshold, "term"
            ].dropna().unique()

            for term in sig_terms:
                term_map.setdefault(term, []).append(comp_name)

    rows = []
    for term, comps in term_map.items():
        unique_comps = sorted(set(comps))
        if len(unique_comps) == 1:
            rows.append({
                "term": term,
                "comparison": unique_comps[0],
            })

    unique_df = pd.DataFrame(rows)
    if not unique_df.empty:
        unique_df = unique_df.sort_values("comparison").reset_index(drop=True)

    logger.info("Found %d unique signatures (significant in exactly 1 comparison).", len(unique_df))
    return unique_df


# ===================================================================
# 7. run_signatures  (main orchestrator)
# ===================================================================

def run_signatures(
    deg_results: Dict[str, pd.DataFrame],
    enrichment_results_novogene: Optional[Dict[str, Dict[str, Dict[str, pd.DataFrame]]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Main orchestrator for GSEA, ORA, and signature overlap analysis.

    Workflow:

    1. Run enrichment analysis (GSEA + ORA) if Novogene enrichment results
       are not provided.
    2. Compute pairwise signature overlap (Jaccard index).
    3. Identify core signatures (enriched in multiple comparisons).
    4. Identify unique signatures (enriched in exactly one comparison).

    Parameters
    ----------
    deg_results : dict[str, DataFrame]
        Mapping of comparison name -> DEG DataFrame (standard columns).
    enrichment_results_novogene : dict, optional
        Pre-computed enrichment results in the same nested-dict format
        returned by :func:`run_enrichment_analysis`.  When provided, the
        enrichment step is skipped and these results are used directly.
    config : dict, optional
        Pipeline configuration.  See :func:`run_enrichment_analysis` for
        recognised keys.  Additional keys:

        * ``padj_threshold`` – adjusted p-value cutoff for overlap and
          signature queries (default 0.05).

    Returns
    -------
    dict
        Dictionary with keys:

        * ``"enrichment"`` – nested enrichment results dict.
        * ``"overlap_matrix"`` – comparisons x comparisons Jaccard
          :class:`~pandas.DataFrame`.
        * ``"core_signatures"`` – :class:`~pandas.DataFrame` of terms
          enriched in multiple comparisons.
        * ``"unique_signatures"`` – :class:`~pandas.DataFrame` of terms
          enriched in exactly one comparison.
    """
    config = config or {}
    padj_threshold = config.get("padj_threshold", 0.05)

    logger.info("=== Starting signature analysis pipeline ===")

    # --- 1. Enrichment analysis ---
    if enrichment_results_novogene is not None and len(enrichment_results_novogene) > 0:
        logger.info(
            "Using provided Novogene enrichment results (%d comparisons).",
            len(enrichment_results_novogene),
        )
        enrichment = enrichment_results_novogene
    else:
        logger.info("Running enrichment analysis (GSEA + ORA).")
        enrichment = run_enrichment_analysis(deg_results, config)

    # --- 2. Signature overlap ---
    overlap_matrix = compute_signature_overlap(
        deg_results, padj_threshold=padj_threshold
    )

    # --- 3. Core signatures ---
    core_signatures = find_core_signatures(
        enrichment, min_comparisons=2, padj_threshold=padj_threshold
    )

    # --- 4. Unique signatures ---
    unique_signatures = find_unique_signatures(
        enrichment, padj_threshold=padj_threshold
    )

    logger.info("=== Signature analysis pipeline complete ===")

    return {
        "enrichment": enrichment,
        "overlap_matrix": overlap_matrix,
        "core_signatures": core_signatures,
        "unique_signatures": unique_signatures,
    }
