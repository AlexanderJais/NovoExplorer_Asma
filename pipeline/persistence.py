"""
HDF5 persistence module for NovoView pipeline results.

Provides save/load functions for all pipeline outputs using pandas HDFStore
(for DataFrames) and h5py (for metadata attributes). The HDF5 layout is:

    /expression/counts, /expression/tpm, /expression/fpkm
    /metadata/samples, /metadata/genes, /metadata/comparisons
    /deg/{comparison_name}
    /enrichment/{comparison_name}/{database}
    /similarity/cosine_matrix, /similarity/gene_clusters,
        /similarity/signature_vectors
    /embeddings/pca_coordinates, /embeddings/pca_variance, /embeddings/umap
    /signatures/overlap_matrix, /signatures/core, /signatures/unique
    /qc/library_sizes, /qc/detection_rates, /qc/mito_fractions,
        /qc/correlation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd

from pipeline.utils import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_put(store: pd.HDFStore, key: str, value: pd.DataFrame) -> None:
    """Write a DataFrame to *store* at *key*, skipping None values."""
    if value is None:
        return
    if not isinstance(value, (pd.DataFrame, pd.Series)):
        logger.warning("Skipping key '%s': value is not a DataFrame/Series.", key)
        return
    store.put(key, value, format="table")


def _safe_get(store: pd.HDFStore, key: str) -> Optional[pd.DataFrame]:
    """Read a key from *store*, returning None if it does not exist."""
    try:
        if key in store:
            return store.get(key)
    except KeyError:
        pass
    return None


def _sanitize_name(name: str) -> str:
    """Sanitize a comparison/database name for use as an HDF5 group name.

    Replaces characters that are problematic in HDF5 paths with underscores.
    """
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(results: Dict[str, Any], output_path: str | Path) -> None:
    """Save all pipeline results to an HDF5 file.

    Parameters
    ----------
    results : dict
        Pipeline output dictionary with keys: ``'expression'``,
        ``'deg'``, ``'enrichment'``, ``'similarity'``, ``'qc'``,
        ``'signatures'``, ``'metadata'``.
    output_path : str or Path
        Destination ``.h5`` file path.  Parent directories are created
        automatically.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving pipeline results to '%s'.", output_path)

    with pd.HDFStore(str(output_path), mode="w") as store:
        # ---- Expression matrices ----
        expression = results.get("expression") or {}
        _safe_put(store, "/expression/counts", expression.get("counts"))
        _safe_put(store, "/expression/tpm", expression.get("tpm"))
        _safe_put(store, "/expression/fpkm", expression.get("fpkm"))

        # ---- Metadata tables ----
        metadata = results.get("metadata") or {}
        _safe_put(store, "/metadata/samples", metadata.get("samples"))
        _safe_put(store, "/metadata/genes", metadata.get("genes"))
        _safe_put(store, "/metadata/comparisons", metadata.get("comparisons"))

        # ---- DEG tables (one per comparison) ----
        deg = results.get("deg") or {}
        for comp_name, df in deg.items():
            key = f"/deg/{_sanitize_name(comp_name)}"
            _safe_put(store, key, df)

        # ---- Enrichment tables (comparison / database) ----
        # Handles two structures:
        #   1. Flat: {comparison: {database: DataFrame}}  (from ingest)
        #   2. Nested: {comparison: {database: {gsea: df, ora_up: df, ...}}} (from signatures)
        enrichment = results.get("enrichment") or {}
        for comp_name, db_dict in enrichment.items():
            if not isinstance(db_dict, dict):
                continue
            for db_name, value in db_dict.items():
                base_key = (
                    f"/enrichment/{_sanitize_name(comp_name)}"
                    f"/{_sanitize_name(db_name)}"
                )
                if isinstance(value, pd.DataFrame):
                    _safe_put(store, base_key, value)
                elif isinstance(value, dict):
                    # Nested structure: save each sub-table
                    for sub_name, sub_df in value.items():
                        if isinstance(sub_df, pd.DataFrame):
                            _safe_put(store, f"{base_key}/{_sanitize_name(sub_name)}", sub_df)
                else:
                    logger.warning("Unexpected enrichment value type for %s: %s", base_key, type(value))

        # ---- Similarity ----
        similarity = results.get("similarity") or {}
        _safe_put(store, "/similarity/cosine_matrix",
                  similarity.get("cosine_matrix"))
        _safe_put(store, "/similarity/gene_clusters",
                  similarity.get("gene_clusters"))
        _safe_put(store, "/similarity/signature_vectors",
                  similarity.get("signature_vectors"))

        # ---- Embeddings ----
        embeddings = results.get("embeddings") or {}
        _safe_put(store, "/embeddings/pca_coordinates",
                  embeddings.get("pca_coordinates"))
        _safe_put(store, "/embeddings/pca_variance",
                  embeddings.get("pca_variance"))
        _safe_put(store, "/embeddings/umap", embeddings.get("umap"))

        # ---- Signatures ----
        signatures = results.get("signatures") or {}
        _safe_put(store, "/signatures/overlap_matrix",
                  signatures.get("overlap_matrix"))
        _safe_put(store, "/signatures/core", signatures.get("core"))
        _safe_put(store, "/signatures/unique", signatures.get("unique"))

        # ---- QC ----
        qc = results.get("qc") or {}
        _safe_put(store, "/qc/library_sizes", qc.get("library_sizes"))
        _safe_put(store, "/qc/detection_rates", qc.get("detection_rates"))
        _safe_put(store, "/qc/mito_fractions", qc.get("mito_fractions"))
        _safe_put(store, "/qc/correlation", qc.get("correlation"))

    # ---- Store project-level metadata as HDF5 attributes via h5py ----
    metadata_attrs = metadata.get("project") or metadata.get("attributes") or {}
    if metadata_attrs:
        with h5py.File(str(output_path), "a") as h5:
            for attr_key, attr_val in metadata_attrs.items():
                try:
                    if isinstance(attr_val, (dict, list)):
                        h5.attrs[attr_key] = json.dumps(attr_val)
                    else:
                        h5.attrs[attr_key] = attr_val
                except TypeError:
                    h5.attrs[attr_key] = str(attr_val)

    logger.info("Results saved successfully to '%s'.", output_path)


# ---------------------------------------------------------------------------
# Full load
# ---------------------------------------------------------------------------

def load_results(output_path: str | Path) -> Dict[str, Any]:
    """Load all pipeline results from an HDF5 file.

    Parameters
    ----------
    output_path : str or Path
        Path to the ``.h5`` file produced by :func:`save_results`.

    Returns
    -------
    dict
        Dictionary with keys ``'expression'``, ``'deg'``, ``'enrichment'``,
        ``'similarity'``, ``'embeddings'``, ``'signatures'``, ``'qc'``,
        ``'metadata'``.  Missing sections are ``None``.
    """
    output_path = Path(output_path)
    logger.info("Loading pipeline results from '%s'.", output_path)

    results: Dict[str, Any] = {
        "expression": None,
        "deg": None,
        "enrichment": None,
        "similarity": None,
        "embeddings": None,
        "signatures": None,
        "qc": None,
        "metadata": None,
    }

    try:
        with pd.HDFStore(str(output_path), mode="r") as store:
            keys = store.keys()

            # ---- Expression ----
            expr: Dict[str, Any] = {}
            for mat in ("counts", "tpm", "fpkm"):
                expr[mat] = _safe_get(store, f"/expression/{mat}")
            if any(v is not None for v in expr.values()):
                results["expression"] = expr

            # ---- Metadata tables ----
            meta: Dict[str, Any] = {}
            for name in ("samples", "genes", "comparisons"):
                meta[name] = _safe_get(store, f"/metadata/{name}")
            if any(v is not None for v in meta.values()):
                results["metadata"] = meta

            # ---- DEG ----
            deg: Dict[str, pd.DataFrame] = {}
            for key in keys:
                if key.startswith("/deg/"):
                    comp_name = key.split("/deg/", 1)[1].strip("/")
                    deg[comp_name] = store.get(key)
            if deg:
                results["deg"] = deg

            # ---- Enrichment ----
            enrichment: Dict[str, Dict[str, pd.DataFrame]] = {}
            for key in keys:
                if key.startswith("/enrichment/"):
                    parts = key.split("/enrichment/", 1)[1].strip("/").split("/", 1)
                    if len(parts) == 2:
                        comp_name, db_name = parts
                        enrichment.setdefault(comp_name, {})[db_name] = store.get(key)
            if enrichment:
                results["enrichment"] = enrichment

            # ---- Similarity ----
            sim: Dict[str, Any] = {}
            for name in ("cosine_matrix", "gene_clusters", "signature_vectors"):
                sim[name] = _safe_get(store, f"/similarity/{name}")
            if any(v is not None for v in sim.values()):
                results["similarity"] = sim

            # ---- Embeddings ----
            emb: Dict[str, Any] = {}
            for name in ("pca_coordinates", "pca_variance", "umap"):
                emb[name] = _safe_get(store, f"/embeddings/{name}")
            if any(v is not None for v in emb.values()):
                results["embeddings"] = emb

            # ---- Signatures ----
            sigs: Dict[str, Any] = {}
            for name in ("overlap_matrix", "core", "unique"):
                sigs[name] = _safe_get(store, f"/signatures/{name}")
            if any(v is not None for v in sigs.values()):
                results["signatures"] = sigs

            # ---- QC ----
            qc_data: Dict[str, Any] = {}
            for name in ("library_sizes", "detection_rates",
                         "mito_fractions", "correlation"):
                qc_data[name] = _safe_get(store, f"/qc/{name}")
            if any(v is not None for v in qc_data.values()):
                results["qc"] = qc_data

    except FileNotFoundError:
        logger.error("Results file not found: '%s'.", output_path)
    except Exception as exc:
        logger.error("Failed to load results from '%s': %s", output_path, exc)

    # ---- Project-level attributes ----
    try:
        with h5py.File(str(output_path), "r") as h5:
            if h5.attrs:
                project_meta = {}
                for attr_key in h5.attrs:
                    val = h5.attrs[attr_key]
                    if isinstance(val, (bytes, np.bytes_)):
                        val = val.decode("utf-8")
                    # Try to deserialise JSON values
                    if isinstance(val, str):
                        try:
                            val = json.loads(val)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    project_meta[attr_key] = val
                if results["metadata"] is None:
                    results["metadata"] = {}
                if isinstance(results["metadata"], dict):
                    results["metadata"]["project"] = project_meta
    except (FileNotFoundError, OSError):
        pass

    logger.info("Results loaded from '%s'.", output_path)
    return results


# ---------------------------------------------------------------------------
# Targeted loaders
# ---------------------------------------------------------------------------

def load_expression(
    output_path: str | Path,
    matrix_type: str = "tpm",
) -> Optional[pd.DataFrame]:
    """Load a single expression matrix.

    Parameters
    ----------
    output_path : str or Path
        Path to the HDF5 results file.
    matrix_type : str
        One of ``'counts'``, ``'tpm'``, ``'fpkm'``.

    Returns
    -------
    pd.DataFrame or None
    """
    output_path = Path(output_path)
    key = f"/expression/{matrix_type}"
    try:
        with pd.HDFStore(str(output_path), mode="r") as store:
            return _safe_get(store, key)
    except FileNotFoundError:
        logger.error("Results file not found: '%s'.", output_path)
        return None
    except KeyError:
        logger.warning("Key '%s' not found in '%s'.", key, output_path)
        return None


def load_deg(
    output_path: str | Path,
    comparison: Optional[str] = None,
) -> Optional[Dict[str, pd.DataFrame] | pd.DataFrame]:
    """Load DEG results.

    Parameters
    ----------
    output_path : str or Path
        Path to the HDF5 results file.
    comparison : str, optional
        If provided, return only that comparison's DataFrame.
        If ``None``, return a dict of all comparisons.

    Returns
    -------
    pd.DataFrame, dict of DataFrames, or None
    """
    output_path = Path(output_path)
    try:
        with pd.HDFStore(str(output_path), mode="r") as store:
            if comparison is not None:
                key = f"/deg/{_sanitize_name(comparison)}"
                return _safe_get(store, key)

            deg: Dict[str, pd.DataFrame] = {}
            for key in store.keys():
                if key.startswith("/deg/"):
                    comp_name = key.split("/deg/", 1)[1].strip("/")
                    deg[comp_name] = store.get(key)
            return deg if deg else None

    except FileNotFoundError:
        logger.error("Results file not found: '%s'.", output_path)
        return None
    except KeyError:
        return None


def load_enrichment(
    output_path: str | Path,
    comparison: Optional[str] = None,
    database: Optional[str] = None,
) -> Optional[Dict | pd.DataFrame]:
    """Load enrichment results with optional filtering.

    Parameters
    ----------
    output_path : str or Path
        Path to the HDF5 results file.
    comparison : str, optional
        Filter to a specific comparison.
    database : str, optional
        Filter to a specific database (e.g. ``'GO_BP'``, ``'KEGG'``).

    Returns
    -------
    pd.DataFrame, nested dict, or None
        - If both *comparison* and *database* are given, a single DataFrame.
        - If only *comparison* is given, a ``{database: DataFrame}`` dict.
        - If neither is given, a ``{comparison: {database: DataFrame}}`` dict.
    """
    output_path = Path(output_path)
    try:
        with pd.HDFStore(str(output_path), mode="r") as store:
            # Specific comparison + database
            if comparison is not None and database is not None:
                key = (
                    f"/enrichment/{_sanitize_name(comparison)}"
                    f"/{_sanitize_name(database)}"
                )
                return _safe_get(store, key)

            # Collect all enrichment keys
            enrichment: Dict[str, Dict[str, pd.DataFrame]] = {}
            for key in store.keys():
                if key.startswith("/enrichment/"):
                    parts = (
                        key.split("/enrichment/", 1)[1].strip("/").split("/", 1)
                    )
                    if len(parts) == 2:
                        comp, db = parts
                        enrichment.setdefault(comp, {})[db] = store.get(key)

            if comparison is not None:
                sanitized = _sanitize_name(comparison)
                return enrichment.get(sanitized)

            return enrichment if enrichment else None

    except FileNotFoundError:
        logger.error("Results file not found: '%s'.", output_path)
        return None
    except KeyError:
        return None


def load_similarity(output_path: str | Path) -> Optional[Dict[str, Any]]:
    """Load similarity matrix and cluster results.

    Returns
    -------
    dict or None
        Keys: ``'cosine_matrix'``, ``'gene_clusters'``,
        ``'signature_vectors'``.
    """
    output_path = Path(output_path)
    try:
        with pd.HDFStore(str(output_path), mode="r") as store:
            sim: Dict[str, Any] = {}
            for name in ("cosine_matrix", "gene_clusters", "signature_vectors"):
                sim[name] = _safe_get(store, f"/similarity/{name}")
            return sim if any(v is not None for v in sim.values()) else None
    except FileNotFoundError:
        logger.error("Results file not found: '%s'.", output_path)
        return None
    except KeyError:
        return None


def load_qc(output_path: str | Path) -> Optional[Dict[str, Any]]:
    """Load QC results.

    Returns
    -------
    dict or None
        Keys: ``'library_sizes'``, ``'detection_rates'``,
        ``'mito_fractions'``, ``'correlation'``.
    """
    output_path = Path(output_path)
    try:
        with pd.HDFStore(str(output_path), mode="r") as store:
            qc: Dict[str, Any] = {}
            for name in ("library_sizes", "detection_rates",
                         "mito_fractions", "correlation"):
                qc[name] = _safe_get(store, f"/qc/{name}")
            return qc if any(v is not None for v in qc.values()) else None
    except FileNotFoundError:
        logger.error("Results file not found: '%s'.", output_path)
        return None
    except KeyError:
        return None


def load_signatures(output_path: str | Path) -> Optional[Dict[str, Any]]:
    """Load signature analysis results.

    Returns
    -------
    dict or None
        Keys: ``'overlap_matrix'``, ``'core'``, ``'unique'``.
    """
    output_path = Path(output_path)
    try:
        with pd.HDFStore(str(output_path), mode="r") as store:
            sigs: Dict[str, Any] = {}
            for name in ("overlap_matrix", "core", "unique"):
                sigs[name] = _safe_get(store, f"/signatures/{name}")
            return sigs if any(v is not None for v in sigs.values()) else None
    except FileNotFoundError:
        logger.error("Results file not found: '%s'.", output_path)
        return None
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def list_comparisons(output_path: str | Path) -> List[str]:
    """Return a list of available DEG comparison names.

    Parameters
    ----------
    output_path : str or Path
        Path to the HDF5 results file.

    Returns
    -------
    list of str
        Comparison names found under ``/deg/``.  Empty list on error.
    """
    output_path = Path(output_path)
    try:
        with pd.HDFStore(str(output_path), mode="r") as store:
            comparisons = []
            for key in store.keys():
                if key.startswith("/deg/"):
                    comp_name = key.split("/deg/", 1)[1].strip("/")
                    comparisons.append(comp_name)
            return sorted(comparisons)
    except (FileNotFoundError, KeyError, Exception) as exc:
        logger.warning("Could not list comparisons from '%s': %s", output_path, exc)
        return []


def get_project_metadata(output_path: str | Path) -> Dict[str, Any]:
    """Return project-level metadata stored as HDF5 root attributes.

    Parameters
    ----------
    output_path : str or Path
        Path to the HDF5 results file.

    Returns
    -------
    dict
        Metadata key-value pairs.  Empty dict on error.
    """
    output_path = Path(output_path)
    try:
        with h5py.File(str(output_path), "r") as h5:
            meta: Dict[str, Any] = {}
            for attr_key in h5.attrs:
                val = h5.attrs[attr_key]
                if isinstance(val, (bytes, np.bytes_)):
                    val = val.decode("utf-8")
                if isinstance(val, str):
                    try:
                        val = json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        pass
                meta[attr_key] = val
            return meta
    except FileNotFoundError:
        logger.error("Results file not found: '%s'.", output_path)
        return {}
    except Exception as exc:
        logger.warning(
            "Could not read project metadata from '%s': %s", output_path, exc
        )
        return {}
