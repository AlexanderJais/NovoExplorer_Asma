"""Novogene file discovery and parsing module.

Walks a Novogene bulk RNA-Seq delivery directory, identifies standard
folder structures (quantification, DEG, enrichment, QC, mapping), and
parses the data files within them into pandas DataFrames.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from pipeline.utils import (
    find_column,
    read_table_flexible,
    setup_logger,
    standardize_deg_columns,
    standardize_enrichment_columns,
)

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Directory / file discovery patterns (case-insensitive)
# ---------------------------------------------------------------------------

_QUANT_PATTERNS = ("quant*", "quantif*", "readcount*", "fpkm*", "tpm*")
_DEG_PATTERNS = ("diff*", "deg*", "diffexp*")
_ENRICHMENT_PATTERNS = ("enrich*",)
_QC_PATTERNS = ("qc*", "01.qc*")
_MAPPING_PATTERNS = ("bind*", "mapping*", "02.bind*", "03.bind*", "04.bind*")
_SAMPLE_INFO_NAMES = ("sample_info.txt", "group_info.txt", "sample_group.txt")

# Expression matrix file patterns
_COUNT_PATTERNS = ("gene_count_matrix*", "readcount*")
_FPKM_PATTERNS = ("gene_fpkm_matrix*", "FPKM*")
_TPM_PATTERNS = ("gene_tpm_matrix*", "TPM*")


def _iglob_dirs(base: Path, patterns: tuple[str, ...]) -> List[Path]:
    """Return directories under *base* whose names match any pattern (case-insensitive)."""
    found: list[Path] = []
    if not base.is_dir():
        return found
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        name_lower = child.name.lower()
        for pat in patterns:
            # fnmatch-style: translate glob pattern to regex
            regex = re.compile(
                re.escape(pat).replace(r"\*", ".*").replace(r"\?", "."),
                re.IGNORECASE,
            )
            if regex.fullmatch(name_lower):
                found.append(child)
                break
    return found


def _iglob_files(base: Path, patterns: tuple[str, ...]) -> List[Path]:
    """Return files under *base* whose names match any pattern (case-insensitive).

    Searches only the immediate directory (non-recursive).
    """
    found: list[Path] = []
    if not base.is_dir():
        return found
    for child in sorted(base.iterdir()):
        if not child.is_file():
            continue
        name = child.name
        for pat in patterns:
            regex = re.compile(
                re.escape(pat).replace(r"\*", ".*").replace(r"\?", "."),
                re.IGNORECASE,
            )
            if regex.fullmatch(name):
                found.append(child)
                break
    return found


def _find_files_prefer_all(comp_dir: Path, file_patterns: tuple[str, ...]) -> List[Path]:
    """Find files in *comp_dir*, falling back to subdirectories (prefer ``all/``)."""
    files = _iglob_files(comp_dir, file_patterns)
    if files:
        return files
    reg_dirs = sorted(
        [d for d in comp_dir.iterdir() if d.is_dir()],
        key=lambda d: (0 if d.name.lower() == "all" else 1, d.name.lower()),
    )
    for reg_dir in reg_dirs:
        files = _iglob_files(reg_dir, file_patterns)
        if files:
            return files
    return []


# Regex for numbered container folders like "1.deglist", "2.cluster", "3.enrichment"
_NUMBERED_DIR_RE = re.compile(r"^\d+\.")


def _is_container_dir(subdir: Path) -> bool:
    """Check if *subdir* is an intermediate container (not an actual comparison).

    Container directories have a numbered prefix (e.g. ``1.deglist``,
    ``2.cluster``) — a convention used in raw Novogene deliveries.
    """
    return bool(_NUMBERED_DIR_RE.match(subdir.name))


def _find_sample_info_file(base: Path) -> Optional[Path]:
    """Walk *base* recursively looking for a sample/group info file."""
    for root, _dirs, files in sorted_walk(base):
        for fname in files:
            if fname.lower() in {n.lower() for n in _SAMPLE_INFO_NAMES}:
                return Path(root) / fname
    return None


def sorted_walk(base: Path):
    """os.walk replacement using pathlib, yielding (root, dirs, files) with sorted names."""
    import os

    for root, dirs, files in os.walk(base):
        dirs.sort()
        files.sort()
        yield root, dirs, files


# ---------------------------------------------------------------------------
# 1. discover_novogene_structure
# ---------------------------------------------------------------------------


def discover_novogene_structure(data_dir: str | Path) -> Dict[str, Any]:
    """Walk *data_dir* recursively and catalogue Novogene delivery folders.

    Returns a dict with keys:
        quant_dir, deg_dir, enrichment_dir, qc_dir, mapping_dir
            – first matching Path or None
        sample_info_file – Path or None
        discovered_files – flat list of every file found during the walk
    """
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        logger.error("Data directory does not exist: %s", data_dir)
        return {
            "quant_dir": None,
            "deg_dir": None,
            "enrichment_dir": None,
            "qc_dir": None,
            "mapping_dir": None,
            "sample_info_file": None,
            "discovered_files": [],
        }

    logger.info("Discovering Novogene structure under %s", data_dir)

    # Collect every file for the inventory
    discovered_files: list[Path] = []
    for root, _dirs, files in sorted_walk(data_dir):
        for f in files:
            discovered_files.append(Path(root) / f)

    # Top-level and one-level-deep search for standard folders
    search_roots = [data_dir]
    for child in sorted(data_dir.iterdir()):
        if child.is_dir():
            search_roots.append(child)

    def _first_match(patterns: tuple[str, ...]) -> Optional[Path]:
        for sr in search_roots:
            matches = _iglob_dirs(sr, patterns)
            if matches:
                return matches[0]
        return None

    result: Dict[str, Any] = {
        "quant_dir": _first_match(_QUANT_PATTERNS),
        "deg_dir": _first_match(_DEG_PATTERNS),
        "enrichment_dir": _first_match(_ENRICHMENT_PATTERNS),
        "qc_dir": _first_match(_QC_PATTERNS),
        "mapping_dir": _first_match(_MAPPING_PATTERNS),
        "sample_info_file": _find_sample_info_file(data_dir),
        "discovered_files": discovered_files,
    }

    for key in ("quant_dir", "deg_dir", "enrichment_dir", "qc_dir", "mapping_dir"):
        val = result[key]
        if val is not None:
            logger.info("  %-20s -> %s", key, val)
        else:
            logger.warning("  %-20s -> not found", key)

    if result["sample_info_file"]:
        logger.info("  sample_info_file   -> %s", result["sample_info_file"])
    else:
        logger.warning("  sample_info_file   -> not found")

    logger.info("  Total files discovered: %d", len(discovered_files))
    return result


# ---------------------------------------------------------------------------
# 2. parse_expression_matrices
# ---------------------------------------------------------------------------


def parse_expression_matrices(quant_dir: str | Path | None) -> Dict[str, Optional[pd.DataFrame]]:
    """Find and parse count, FPKM, and TPM matrices from *quant_dir*.

    Returns dict with keys ``'counts'``, ``'fpkm'``, ``'tpm'``; any may be
    ``None`` if the corresponding file was not found.
    """
    result: Dict[str, Optional[pd.DataFrame]] = {
        "counts": None,
        "fpkm": None,
        "tpm": None,
    }

    if quant_dir is None:
        logger.warning("No quantification directory provided; skipping expression matrix parsing.")
        return result

    quant_dir = Path(quant_dir).resolve()
    if not quant_dir.is_dir():
        logger.warning("Quantification directory does not exist: %s", quant_dir)
        return result

    # Search both the directory itself and one level of subdirectories
    search_dirs = [quant_dir] + [
        d for d in sorted(quant_dir.iterdir()) if d.is_dir()
    ]

    def _find_and_parse(patterns: tuple[str, ...], label: str) -> Optional[pd.DataFrame]:
        for sd in search_dirs:
            matches = _iglob_files(sd, patterns)
            if matches:
                fpath = matches[0]
                logger.info("  Parsing %s matrix: %s", label, fpath)
                try:
                    df = read_table_flexible(fpath)
                    if df is not None and not df.empty:
                        logger.info("    -> %d genes x %d samples", df.shape[0], df.shape[1])
                        return df
                    logger.warning("    -> empty or None result for %s", fpath)
                except Exception:
                    logger.warning("    -> failed to parse %s", fpath, exc_info=True)
                return None
        logger.warning("  No %s matrix found in %s", label, quant_dir)
        return None

    result["counts"] = _find_and_parse(_COUNT_PATTERNS, "count")
    result["fpkm"] = _find_and_parse(_FPKM_PATTERNS, "FPKM")
    result["tpm"] = _find_and_parse(_TPM_PATTERNS, "TPM")

    return result


# ---------------------------------------------------------------------------
# 3. parse_deg_results
# ---------------------------------------------------------------------------


def parse_deg_results(deg_dir: str | Path | None) -> Dict[str, pd.DataFrame]:
    """Parse DEG tables from each comparison subdirectory in *deg_dir*.

    Supports two layouts:

    **Flat layout** (test fixtures / cleaned deliveries)::

        deg_dir/
          GroupA_vs_GroupB/
            *.DEG.xls

    **Numbered-container layout** (raw Novogene deliveries)::

        deg_dir/
          1.deglist/
            GroupA_vs_GroupB/
              *_deg.xls
          2.cluster/
            ...

    Numbered containers (``1.xxx``, ``2.xxx``, …) are transparently
    descended into so the comparison folders within them are found.

    Returns ``{comparison_name: DataFrame}`` with standardised column names.
    """
    results: Dict[str, pd.DataFrame] = {}

    if deg_dir is None:
        logger.warning("No DEG directory provided; skipping DEG parsing.")
        return results

    deg_dir = Path(deg_dir).resolve()
    if not deg_dir.is_dir():
        logger.warning("DEG directory does not exist: %s", deg_dir)
        return results

    _DEG_FILE_PATTERNS = (
        "*.DEG.xls",
        "*.DEG_results*",
        "*DEG*.xls",
        "*deg*.xls",
        "*diffexpr*",
        "*diff_exp*",
    )

    # Collect comparison directories — unwrap numbered containers first.
    comparison_dirs: list[Path] = []
    for subdir in sorted(deg_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if _is_container_dir(subdir):
            # Descend into numbered containers like 1.deglist/
            logger.info("  Entering container directory: %s", subdir.name)
            for inner in sorted(subdir.iterdir()):
                if inner.is_dir():
                    comparison_dirs.append(inner)
        else:
            comparison_dirs.append(subdir)

    for comp_dir in comparison_dirs:
        comparison = comp_dir.name
        logger.info("  Processing DEG comparison: %s", comparison)

        # Find the DEG table in this comparison folder
        deg_files = _iglob_files(comp_dir, _DEG_FILE_PATTERNS)
        if not deg_files:
            # Also search one level deeper (some deliveries nest further)
            for nested in sorted(comp_dir.iterdir()):
                if nested.is_dir():
                    deg_files = _iglob_files(nested, _DEG_FILE_PATTERNS)
                    if deg_files:
                        break

        if not deg_files:
            logger.warning("    No DEG table found for comparison %s", comparison)
            continue

        # Prefer the full gene list (*_deg.xls) over filtered subsets
        # (*_deg_all.xls, *_deg_up.xls, *_deg_down.xls).
        preferred = [f for f in deg_files if re.fullmatch(r".*_deg\.xls", f.name, re.IGNORECASE)]
        fpath = preferred[0] if preferred else deg_files[0]
        logger.info("    Parsing: %s", fpath)
        try:
            df = read_table_flexible(fpath)
            if df is None or df.empty:
                logger.warning("    -> empty or None result for %s", fpath)
                continue
            df = standardize_deg_columns(df)
            results[comparison] = df
            logger.info("    -> %d genes", len(df))
        except Exception:
            logger.warning("    -> failed to parse %s", fpath, exc_info=True)

    logger.info("  Parsed %d DEG comparisons", len(results))

    # --- Enrich with basemean from all_compare.xls if available ---
    if deg_dir is not None:
        results = _enrich_deg_with_all_compare(Path(deg_dir).resolve(), results)

    return results


def _enrich_deg_with_all_compare(
    deg_dir: Path,
    deg_results: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Merge basemean from ``all_compare.xls`` into DEG tables when missing.

    The Novogene ``all_compare.xls`` file (found in ``1.deglist/``) contains
    per-comparison group mean columns named
    ``{comparison}_{groupA}`` and ``{comparison}_{groupB}``.  When a DEG table
    lacks a ``basemean`` column, we compute it as the mean of these two group
    means.  Count columns (``*_count``) are also used to derive an overall
    basemean when group-mean columns are not identifiable.
    """
    # Find all_compare.xls
    all_compare_path = None
    search_dirs = [deg_dir]
    for child in sorted(deg_dir.iterdir()):
        if child.is_dir() and _is_container_dir(child):
            search_dirs.append(child)
    for sdir in search_dirs:
        candidates = _iglob_files(sdir, ("all_compare*",))
        if candidates:
            all_compare_path = candidates[0]
            break
    if all_compare_path is None:
        return deg_results

    logger.info("  Found all_compare file: %s", all_compare_path)
    try:
        ac_df = read_table_flexible(all_compare_path)
    except Exception:
        logger.warning("  Failed to parse all_compare file", exc_info=True)
        return deg_results
    if ac_df is None or ac_df.empty:
        return deg_results

    ac_df = standardize_deg_columns(ac_df)

    # Identify gene key column
    gene_key = "gene_id" if "gene_id" in ac_df.columns else None
    if gene_key is None:
        return deg_results

    for comp_name, comp_df in deg_results.items():
        if "basemean" in comp_df.columns:
            continue  # already has basemean

        # Strategy 1: find the two group-mean columns for this comparison
        #   Pattern: {comp_name}_{groupA} and {comp_name}_{groupB}
        prefix = f"{comp_name}_"
        group_mean_cols = [
            c for c in ac_df.columns
            if c.startswith(prefix)
            and not c.endswith(("_log2FoldChange", "_pvalue", "_padj",
                                "_log2fc", "_log2foldchange"))
            and pd.api.types.is_numeric_dtype(ac_df[c])
        ]
        if len(group_mean_cols) >= 2:
            basemean = ac_df[[gene_key] + group_mean_cols].copy()
            basemean["basemean"] = basemean[group_mean_cols].mean(axis=1)
            basemean = basemean[[gene_key, "basemean"]]
            # Merge into comp_df
            merge_key = gene_key if gene_key in comp_df.columns else None
            if merge_key:
                merged = comp_df.merge(basemean, on=merge_key, how="left")
                deg_results[comp_name] = merged
                logger.info("    %s: added basemean from group means (%d cols)",
                            comp_name, len(group_mean_cols))
                continue

        # Strategy 2: compute from count columns
        count_cols = [c for c in ac_df.columns if c.endswith("_count")
                      and pd.api.types.is_numeric_dtype(ac_df[c])]
        if count_cols:
            basemean = ac_df[[gene_key]].copy()
            basemean["basemean"] = ac_df[count_cols].mean(axis=1)
            merge_key = gene_key if gene_key in comp_df.columns else None
            if merge_key:
                merged = comp_df.merge(basemean, on=merge_key, how="left")
                deg_results[comp_name] = merged
                logger.info("    %s: added basemean from %d count columns",
                            comp_name, len(count_cols))

    return deg_results


# ---------------------------------------------------------------------------
# 4. parse_enrichment_results
# ---------------------------------------------------------------------------


def _detect_enrichment_layout(enrichment_dir: Path) -> str:
    """Determine which enrichment folder layout is used.

    Returns ``"comparison_first"`` for the flat/cleaned layout::

        enrichment_dir/{comparison}/GO/*.xls

    Returns ``"database_first"`` for the raw Novogene layout::

        enrichment_dir/KEGG/{comparison}/{all|up|down}/*_KEGGenrich.xls
    """
    _DB_NAMES = {"go", "kegg", "disgenet", "do", "reactome", "ppi"}
    for child in enrichment_dir.iterdir():
        if child.is_dir() and child.name.lower() in _DB_NAMES:
            return "database_first"
    return "comparison_first"


def _parse_enrichment_comparison_first(
    enrichment_dir: Path,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Parse enrichment in the ``{comparison}/{database}/`` layout."""
    results: Dict[str, Dict[str, pd.DataFrame]] = {}

    _ENRICH_FILE_PATTERNS = (
        "*.xls",
        "*.xlsx",
        "*.tsv",
        "*.csv",
        "*.txt",
    )

    _DB_DIR_PATTERNS = {
        "GO": ("go*", "GO*"),
        "KEGG": ("kegg*", "KEGG*"),
        "DisGeNET": ("disgenet*", "DisGeNET*", "DISGENET*"),
        "DO": ("do", "DO"),
        "Reactome": ("reactome*", "Reactome*", "REACTOME*"),
        # PPI directories contain protein-protein interaction networks
        # (node1, node2, score), not enrichment tables — skip them.
    }

    for subdir in sorted(enrichment_dir.iterdir()):
        if not subdir.is_dir():
            continue
        comparison = subdir.name
        logger.info("  Processing enrichment comparison: %s", comparison)
        comp_results: Dict[str, pd.DataFrame] = {}

        for db_name, db_patterns in _DB_DIR_PATTERNS.items():
            db_dirs = _iglob_dirs(subdir, db_patterns)
            if not db_dirs:
                continue

            db_dir = db_dirs[0]
            enrich_files = _iglob_files(db_dir, _ENRICH_FILE_PATTERNS)
            if not enrich_files:
                logger.warning("    No enrichment files in %s", db_dir)
                continue

            fpath = enrich_files[0]
            logger.info("    Parsing %s enrichment: %s", db_name, fpath)
            try:
                df = read_table_flexible(fpath)
                if df is not None and not df.empty:
                    df = standardize_enrichment_columns(df)
                    if "category" not in df.columns:
                        df["category"] = db_name
                    comp_results[db_name] = df
                    logger.info("      -> %d terms", len(df))
                else:
                    logger.warning("      -> empty or None result for %s", fpath)
            except Exception:
                logger.warning("      -> failed to parse %s", fpath, exc_info=True)

        if comp_results:
            results[comparison] = comp_results

    return results


def _parse_enrichment_database_first(
    enrichment_dir: Path,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Parse enrichment in the raw Novogene ``{database}/{comparison}/`` layout.

    Handles the structure::

        enrichment_dir/
          KEGG/
            GroupA_vs_GroupB/
              all/  (or up/ or down/)
                *_KEGGenrich.xls
          GO/
            GroupA_vs_GroupB/
              all/
                *_GOenrich.xls

    We prefer the ``all`` (all DEGs) enrichment table for each comparison.
    Falls back to the first available regulation-direction subfolder.
    """
    results: Dict[str, Dict[str, pd.DataFrame]] = {}

    _ENRICH_FILE_PATTERNS = (
        "*enrich*.xls",
        "*enrich*.xlsx",
        "*.xls",
        "*.xlsx",
        "*.tsv",
        "*.csv",
        "*.txt",
    )

    _DB_DIR_PATTERNS = {
        "GO": ("go*", "GO*"),
        "KEGG": ("kegg*", "KEGG*"),
        "DisGeNET": ("disgenet*", "DisGeNET*", "DISGENET*"),
        "DO": ("do", "DO"),
        "Reactome": ("reactome*", "Reactome*", "REACTOME*"),
        # PPI directories contain protein-protein interaction networks
        # (node1, node2, score), not enrichment tables — skip them.
    }

    for db_name, db_patterns in _DB_DIR_PATTERNS.items():
        db_dirs = _iglob_dirs(enrichment_dir, db_patterns)
        if not db_dirs:
            continue
        db_dir = db_dirs[0]
        logger.info("  Processing database-first enrichment: %s (%s)", db_name, db_dir.name)

        for comp_dir in sorted(db_dir.iterdir()):
            if not comp_dir.is_dir():
                continue
            comparison = comp_dir.name
            logger.info("    Comparison: %s", comparison)

            enrich_files = _find_files_prefer_all(comp_dir, _ENRICH_FILE_PATTERNS)

            if not enrich_files:
                logger.warning("      No enrichment files found for %s/%s", db_name, comparison)
                continue

            fpath = enrich_files[0]
            logger.info("      Parsing: %s", fpath)
            try:
                df = read_table_flexible(fpath)
                if df is not None and not df.empty:
                    df = standardize_enrichment_columns(df)
                    if "category" not in df.columns:
                        df["category"] = db_name
                    results.setdefault(comparison, {})[db_name] = df
                    logger.info("        -> %d terms", len(df))
                else:
                    logger.warning("        -> empty or None result for %s", fpath)
            except Exception:
                logger.warning("        -> failed to parse %s", fpath, exc_info=True)

    return results


def parse_enrichment_results(
    enrichment_dir: str | Path | None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Parse enrichment results from an enrichment directory.

    Supports two layouts:

    **Comparison-first** (test fixtures / cleaned deliveries)::

        enrichment_dir/
          CompA_vs_CompB/
            GO/  *.xls
            KEGG/  *.xls

    **Database-first** (raw Novogene deliveries)::

        enrichment_dir/
          KEGG/
            CompA_vs_CompB/
              all/  *_KEGGenrich.xls
          GO/
            CompA_vs_CompB/
              all/  *_GOenrich.xls

    Returns ``{comparison: {database: DataFrame}}``.
    """
    results: Dict[str, Dict[str, pd.DataFrame]] = {}

    if enrichment_dir is None:
        logger.warning("No enrichment directory provided; skipping enrichment parsing.")
        return results

    enrichment_dir = Path(enrichment_dir).resolve()
    if not enrichment_dir.is_dir():
        logger.warning("Enrichment directory does not exist: %s", enrichment_dir)
        return results

    layout = _detect_enrichment_layout(enrichment_dir)
    logger.info("  Detected enrichment layout: %s", layout)

    if layout == "database_first":
        results = _parse_enrichment_database_first(enrichment_dir)
    else:
        results = _parse_enrichment_comparison_first(enrichment_dir)

    logger.info("  Parsed enrichment for %d comparisons", len(results))
    return results


# ---------------------------------------------------------------------------
# 5. parse_ppi_results
# ---------------------------------------------------------------------------


_PPI_DIR_PATTERNS = ("ppi*", "PPI*")

_PPI_FILE_PATTERNS = (
    "*ppi*.xls",
    "*ppi*.xlsx",
    "*ppi*.tsv",
    "*ppi*.csv",
    "*ppi*.txt",
    "*.xls",
    "*.xlsx",
)

# Expected columns (lowercase) for PPI interaction tables
_PPI_NODE1_GENE = ["node1_gene", "source_gene", "gene1", "genea"]
_PPI_NODE2_GENE = ["node2_gene", "target_gene", "gene2", "geneb"]
_PPI_NODE1_NAME = ["node1_name", "source_name", "name1", "namea"]
_PPI_NODE2_NAME = ["node2_name", "target_name", "name2", "nameb"]
_PPI_SCORE = ["score", "combined_score", "confidence", "weight"]


def parse_ppi_results(
    enrichment_dir: str | Path | None,
) -> Dict[str, pd.DataFrame]:
    """Parse PPI network tables from inside the enrichment directory.

    Novogene deliveries place PPI data alongside enrichment databases::

        Enrichment/
          PPI/
            CompA_vs_CompB/
              all/  *_ppi.xls

    Each table contains pairwise protein interactions with columns such as
    ``node1_gene``, ``node1_name``, ``node2_gene``, ``node2_name``, ``score``.

    Returns ``{comparison_name: DataFrame}`` with standardised columns:
    ``source``, ``target``, ``source_name``, ``target_name``, ``score``.
    """
    results: Dict[str, pd.DataFrame] = {}

    if enrichment_dir is None:
        return results

    enrichment_dir = Path(enrichment_dir).resolve()
    if not enrichment_dir.is_dir():
        return results

    # Locate PPI directory inside enrichment_dir
    ppi_dirs = _iglob_dirs(enrichment_dir, _PPI_DIR_PATTERNS)
    if not ppi_dirs:
        return results

    ppi_root = ppi_dirs[0]
    logger.info("  Parsing PPI networks from: %s", ppi_root)

    for comp_dir in sorted(ppi_root.iterdir()):
        if not comp_dir.is_dir():
            continue
        comparison = comp_dir.name

        ppi_files = _find_files_prefer_all(comp_dir, _PPI_FILE_PATTERNS)

        if not ppi_files:
            logger.warning("    No PPI files found for %s", comparison)
            continue

        fpath = ppi_files[0]
        logger.info("    PPI %s: %s", comparison, fpath)
        try:
            df = read_table_flexible(fpath)
        except Exception:
            logger.warning("    Failed to parse PPI file: %s", fpath, exc_info=True)
            continue

        if df is None or df.empty:
            continue

        # Standardise column names
        src_gene = find_column(df, _PPI_NODE1_GENE)
        tgt_gene = find_column(df, _PPI_NODE2_GENE)
        src_name = find_column(df, _PPI_NODE1_NAME)
        tgt_name = find_column(df, _PPI_NODE2_NAME)
        score_col = find_column(df, _PPI_SCORE)

        rename_map: dict[str, str] = {}
        if src_gene:
            rename_map[src_gene] = "source"
        if tgt_gene:
            rename_map[tgt_gene] = "target"
        if src_name:
            rename_map[src_name] = "source_name"
        if tgt_name:
            rename_map[tgt_name] = "target_name"
        if score_col:
            rename_map[score_col] = "score"

        df = df.rename(columns=rename_map)

        # Ensure at least source and target exist
        if "source" not in df.columns or "target" not in df.columns:
            # Fall back: use first two columns as source/target
            if len(df.columns) >= 2:
                df = df.rename(columns={df.columns[0]: "source", df.columns[1]: "target"})
            else:
                logger.warning("    PPI table has fewer than 2 columns; skipping %s", fpath)
                continue

        results[comparison] = df
        logger.info("      -> %d interactions", len(df))

    if results:
        logger.info("  Parsed PPI for %d comparisons", len(results))
    return results


# ---------------------------------------------------------------------------
# 6. parse_sample_info
# ---------------------------------------------------------------------------

# Recognised column names (lowercase) for the sample identifier
_SAMPLE_COL_ALIASES = {"sample_id", "sample", "sampleid", "sample_name", "samplename", "name"}
# Recognised column names (lowercase) for the group/condition
_GROUP_COL_ALIASES = {"group", "condition", "treatment", "group_id", "groupid", "class"}


def parse_sample_info(file_path: str | Path | None) -> Optional[pd.DataFrame]:
    """Parse a sample-to-group mapping file.

    The file is expected to be tab-separated with a header row.  Column names
    are matched case-insensitively against known aliases for *sample_id* and
    *group*.

    Returns a DataFrame with exactly two columns: ``sample_id``, ``group``,
    or ``None`` if parsing fails.
    """
    if file_path is None:
        return None

    file_path = Path(file_path).resolve()
    if not file_path.is_file():
        logger.warning("Sample info file does not exist: %s", file_path)
        return None

    logger.info("Parsing sample info: %s", file_path)
    try:
        df = read_table_flexible(file_path)
    except Exception:
        logger.warning("Failed to read sample info file: %s", file_path, exc_info=True)
        return None

    if df is None or df.empty:
        logger.warning("Sample info file is empty: %s", file_path)
        return None

    cols_lower = {c.lower().strip(): c for c in df.columns}

    sample_col: Optional[str] = None
    group_col: Optional[str] = None

    for alias in _SAMPLE_COL_ALIASES:
        if alias in cols_lower:
            sample_col = cols_lower[alias]
            break

    for alias in _GROUP_COL_ALIASES:
        if alias in cols_lower:
            group_col = cols_lower[alias]
            break

    # Fallback: if only two columns, assume first=sample, second=group
    if sample_col is None and group_col is None and len(df.columns) == 2:
        sample_col, group_col = df.columns[0], df.columns[1]
        logger.info("  Falling back to positional columns: sample=%s, group=%s", sample_col, group_col)
    elif sample_col is None or group_col is None:
        logger.warning(
            "Could not identify sample/group columns in %s. Columns: %s",
            file_path,
            list(df.columns),
        )
        return None

    result = df[[sample_col, group_col]].copy()
    result.columns = ["sample_id", "group"]
    result["sample_id"] = result["sample_id"].astype(str).str.strip()
    result["group"] = result["group"].astype(str).str.strip()

    logger.info("  -> %d samples in %d groups", len(result), result["group"].nunique())
    return result


# ---------------------------------------------------------------------------
# 6. infer_groups_from_comparisons
# ---------------------------------------------------------------------------


def infer_groups_from_comparisons(deg_results: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    """Extract group names from DEG comparison folder names.

    Splits each comparison name on ``'_vs_'`` (case-insensitive), falling
    back to ``'vs'`` without underscores for Novogene naming conventions
    like ``GroupAvs GroupB``.

    Returns ``{"groups": [group1, group2, ...], "comparisons": [comp1, ...]}``.
    """
    groups: set[str] = set()
    comparisons: list[str] = []

    for comp_name in sorted(deg_results.keys()):
        comparisons.append(comp_name)
        # Try _vs_ first (e.g. GroupA_vs_GroupB), then plain vs (e.g. GroupAvsGroupB)
        parts = re.split(r"_vs_", comp_name, flags=re.IGNORECASE)
        if len(parts) == 1:
            parts = re.split(r"vs", comp_name, maxsplit=1, flags=re.IGNORECASE)
        for part in parts:
            stripped = part.strip()
            if stripped:
                groups.add(stripped)

    result = {
        "groups": sorted(groups),
        "comparisons": comparisons,
    }
    logger.info("Inferred %d groups from %d comparisons: %s", len(result["groups"]), len(comparisons), result["groups"])
    return result


# ---------------------------------------------------------------------------
# 7. ingest_all  (main entry point)
# ---------------------------------------------------------------------------


def ingest_all(data_dir: str | Path, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Main ingestion entry point.

    Parameters
    ----------
    data_dir : str or Path
        Root directory of a Novogene delivery.
    config : dict, optional
        Optional configuration overrides (currently unused but reserved for
        future options such as custom glob patterns or column mappings).

    Returns
    -------
    dict
        Keys:
            structure    – output of discover_novogene_structure
            expression   – output of parse_expression_matrices
            deg          – output of parse_deg_results
            enrichment   – output of parse_enrichment_results
            sample_info  – DataFrame or None
            groups       – inferred groups dict (from sample_info or comparisons)
    """
    config = config or {}
    data_dir = Path(data_dir).resolve()
    logger.info("=" * 60)
    logger.info("Starting NovoExplorer ingestion: %s", data_dir)
    logger.info("=" * 60)

    # Step 1 – discover directory structure
    structure = discover_novogene_structure(data_dir)

    # Step 2 – parse expression matrices
    logger.info("--- Expression matrices ---")
    expression = parse_expression_matrices(structure["quant_dir"])

    # Step 3 – parse DEG results
    logger.info("--- DEG results ---")
    deg = parse_deg_results(structure["deg_dir"])

    # Step 4 – parse enrichment results
    logger.info("--- Enrichment results ---")
    enrichment = parse_enrichment_results(structure["enrichment_dir"])

    # Step 5 – parse PPI networks
    logger.info("--- PPI networks ---")
    ppi = parse_ppi_results(structure["enrichment_dir"])

    # Step 6 – parse sample info
    logger.info("--- Sample info ---")
    sample_info = parse_sample_info(structure["sample_info_file"])

    # Step 7 – infer groups
    if sample_info is not None and not sample_info.empty:
        groups: Dict[str, Any] = {
            "groups": sorted(sample_info["group"].unique().tolist()),
            "comparisons": sorted(deg.keys()),
        }
        logger.info("Groups from sample info: %s", groups["groups"])
    elif deg:
        groups = infer_groups_from_comparisons(deg)
    else:
        groups = {"groups": [], "comparisons": []}
        logger.warning("No sample info and no DEG results; cannot determine groups.")

    # Summary
    logger.info("=" * 60)
    logger.info("Ingestion summary:")
    logger.info("  Expression matrices : counts=%s, fpkm=%s, tpm=%s",
                expression["counts"] is not None,
                expression["fpkm"] is not None,
                expression["tpm"] is not None)
    logger.info("  DEG comparisons     : %d", len(deg))
    logger.info("  Enrichment results  : %d comparisons", len(enrichment))
    logger.info("  PPI networks        : %d comparisons", len(ppi))
    logger.info("  Sample info         : %s", "loaded" if sample_info is not None else "not available")
    logger.info("  Groups              : %s", groups.get("groups", []))
    logger.info("  Total files found   : %d", len(structure["discovered_files"]))
    logger.info("=" * 60)

    return {
        "structure": structure,
        "expression": expression,
        "deg": deg,
        "enrichment": enrichment,
        "ppi": ppi,
        "sample_info": sample_info,
        "groups": groups,
    }
