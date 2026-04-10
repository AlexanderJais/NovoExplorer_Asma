#!/usr/bin/env python
"""NovoExplorer analysis pipeline -- CLI entry point.

Orchestrates the full RNA-Seq analysis workflow: ingestion of Novogene
delivery data, normalisation, quality control, differential expression,
gene similarity, signature analysis, and persistence of results.

Usage
-----
    python run_pipeline.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from pipeline.utils import load_config, setup_logger

logger = setup_logger("run_pipeline")


# -------------------------------------------------------------------
# Helper: timed step runner
# -------------------------------------------------------------------

def _run_step(step_name: str, func, *args, **kwargs) -> Any:
    """Execute *func* with timing, logging, and error handling.

    Returns the function's result on success, or ``None`` on failure.
    """
    logger.info("=" * 60)
    logger.info("STEP: %s", step_name)
    logger.info("=" * 60)
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        logger.info(
            "STEP %s completed in %.1f s",
            step_name,
            elapsed,
        )
        return result
    except Exception:
        elapsed = time.time() - t0
        logger.error(
            "STEP %s FAILED after %.1f s",
            step_name,
            elapsed,
            exc_info=True,
        )
        return None


# -------------------------------------------------------------------
# Pipeline runner
# -------------------------------------------------------------------

def run_pipeline(config: Dict[str, Any]) -> None:
    """Execute every pipeline stage in order, passing outputs forward.

    Runs ingestion, normalization, QC, differential expression, similarity,
    signature analysis, and saves all results to an HDF5 file.

    Parameters
    ----------
    config : dict[str, Any]
        Pipeline configuration dict (see ``config.yaml`` for available keys).
    """

    from pipeline import ingest, normalize, qc, diffexp, similarity, signatures, persistence

    data_dir = config.get("data_dir", ".")
    output_dir = config.get("output_dir", "results")
    organism = config.get("organism", "human")
    output_file = str(Path(output_dir) / "novoexplorer_results.h5")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    pipeline_t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Ingest
    # ------------------------------------------------------------------
    ingest_data = _run_step("Ingest", ingest.ingest_all, data_dir, config)

    if ingest_data is None:
        logger.error("Ingestion failed -- cannot continue.")
        sys.exit(1)

    # Unpack commonly used artefacts
    expression = ingest_data.get("expression", {})
    counts_df = expression.get("counts") if expression else None
    deg_results = ingest_data.get("deg", {})
    sample_info = ingest_data.get("sample_info")
    groups_info = ingest_data.get("groups", {})

    # ------------------------------------------------------------------
    # 2. Normalize
    # ------------------------------------------------------------------
    norm_result: Optional[Dict[str, Any]] = None
    if counts_df is not None:
        norm_result = _run_step(
            "Normalize",
            _normalize_step,
            counts_df,
            organism,
            normalize,
        )
    else:
        logger.warning(
            "No count matrix available -- skipping Normalize step."
        )

    # Determine the expression matrix to use downstream
    norm_counts = None
    log2_expr = None
    if norm_result is not None:
        norm_counts = norm_result.get("filtered")
        log2_expr = norm_result.get("log2_expr")

    # ------------------------------------------------------------------
    # 3. QC
    # ------------------------------------------------------------------
    qc_result = None
    if norm_counts is not None:
        sample_groups = None
        if sample_info is not None and not sample_info.empty:
            sample_groups = sample_info.set_index("sample_id")["group"]
        qc_result = _run_step(
            "QC",
            qc.run_qc,
            norm_counts,
            organism=organism,
            sample_groups=sample_groups,
        )
    else:
        logger.warning(
            "No normalised count matrix available -- skipping QC step."
        )

    # ------------------------------------------------------------------
    # 4. Differential Expression
    # ------------------------------------------------------------------
    diffexp_result = _run_step(
        "Differential Expression",
        diffexp.run_diffexp,
        counts_df=norm_counts,
        sample_groups=sample_info,
        novogene_deg=deg_results if deg_results else None,
        config=config,
    )

    primary_deg = {}
    if diffexp_result is not None:
        primary_deg = diffexp_result.get("primary", {})

    # ------------------------------------------------------------------
    # 5. Similarity
    # ------------------------------------------------------------------
    similarity_result = None
    if log2_expr is not None and primary_deg:
        similarity_result = _run_step(
            "Similarity",
            similarity.run_similarity,
            expression_df=log2_expr,
            deg_results=primary_deg,
            config=config,
        )
    else:
        logger.warning(
            "Skipping Similarity step (requires log2 expression matrix "
            "and DEG results)."
        )

    # ------------------------------------------------------------------
    # 6. Signatures
    # ------------------------------------------------------------------
    signatures_result = _run_step(
        "Signatures",
        signatures.run_signatures,
        deg_results=primary_deg,
        enrichment_results_novogene=ingest_data.get("enrichment") if ingest_data else None,
        config=config,
    )

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    # Restructure pipeline outputs to match persistence.save_results() contract
    all_results = {
        # Expression matrices
        "expression": {
            "counts": expression.get("counts") if expression else None,
            "tpm": expression.get("tpm") if expression else None,
            "fpkm": expression.get("fpkm") if expression else None,
        },
        # DEG tables (dict of comparison -> DataFrame)
        "deg": primary_deg,
        # Enrichment results (dict of comparison -> {db -> DataFrame})
        "enrichment": (
            ingest_data.get("enrichment", {}) if ingest_data else {}
        ),
        # Similarity results - map run_similarity() keys to persistence keys
        "similarity": {
            "cosine_matrix": (similarity_result or {}).get("similarity_matrix"),
            "gene_clusters": (similarity_result or {}).get("cluster_labels"),
            "signature_vectors": (similarity_result or {}).get("signature_vectors"),
        },
        # QC results - map run_qc() keys to persistence keys
        "qc": {
            "library_sizes": (qc_result or {}).get("library_sizes"),
            "detection_rates": (qc_result or {}).get("detection_rates"),
            "mito_fractions": (qc_result or {}).get("mito_fractions"),
            "correlation": (qc_result or {}).get("correlation_matrix"),
        },
        # Embeddings (extracted from QC PCA/UMAP results)
        "embeddings": {
            "pca_coordinates": (qc_result.get("pca") or {}).get("coordinates") if qc_result else None,
            "pca_variance": (qc_result.get("pca") or {}).get("variance_explained") if qc_result else None,
            "umap": qc_result.get("umap") if qc_result else None,
        },
        # Signature analysis - map run_signatures() keys to persistence keys
        "signatures": {
            "overlap_matrix": (signatures_result or {}).get("overlap_matrix"),
            "core": (signatures_result or {}).get("core_signatures"),
            "unique": (signatures_result or {}).get("unique_signatures"),
        },
        # Metadata
        "metadata": {
            "samples": sample_info,
            "genes": pd.DataFrame({"gene_id": norm_counts.index.tolist()}) if norm_counts is not None else None,
            "comparisons": pd.DataFrame({"comparison": list(primary_deg.keys())}) if primary_deg else None,
            "project": {
                "project_name": config.get("project_name", ""),
                "organism": organism,
                "data_dir": data_dir,
            },
        },
    }

    _run_step(
        "Save",
        persistence.save_results,
        results=all_results,
        output_path=output_file,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed_total = time.time() - pipeline_t0

    n_samples = norm_counts.shape[1] if norm_counts is not None else 0
    n_genes = norm_counts.shape[0] if norm_counts is not None else 0
    n_comparisons = len(primary_deg)
    n_degs = 0
    if diffexp_result is not None:
        summary_df = diffexp_result.get("summary")
        if summary_df is not None and not summary_df.empty and "total_deg" in summary_df.columns:
            n_degs = int(summary_df["total_deg"].sum())

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE (%.1f s)", elapsed_total)
    logger.info("-" * 60)
    logger.info("  Samples      : %d", n_samples)
    logger.info("  Genes        : %d", n_genes)
    logger.info("  Comparisons  : %d", n_comparisons)
    logger.info("  DEGs found   : %d", n_degs)
    logger.info("  Output file  : %s", output_file)
    logger.info("=" * 60)


# -------------------------------------------------------------------
# Normalize helper (bundles several normalize calls)
# -------------------------------------------------------------------

def _normalize_step(counts_df, organism, normalize_mod) -> Dict[str, Any]:
    """Run standardization, filtering, and log2 transform."""
    standardized = normalize_mod.standardize_expression_matrix(
        counts_df, organism=organism
    )
    filtered = normalize_mod.filter_low_expression(standardized)
    log2_expr = normalize_mod.compute_log2_transform(filtered)
    return {
        "standardized": standardized,
        "filtered": filtered,
        "log2_expr": log2_expr,
    }


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------

def main() -> None:
    """CLI entry point: parse arguments, load config, and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="NovoExplorer RNA-Seq analysis pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set up root logging level
    log_level = config.get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("NovoExplorer pipeline starting with config: %s", args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
