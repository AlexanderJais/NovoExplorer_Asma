"""
Quality control module for NovoView RNA-Seq platform.

Provides per-sample QC metrics (library size, gene detection rate,
mitochondrial fraction), sample correlation, dimensionality reduction
(PCA, UMAP), outlier detection, and an orchestrating ``run_qc`` function.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import umap

from pipeline.utils import setup_logger

logger = setup_logger(__name__)


# ===================================================================
# 1. Per-sample QC metrics
# ===================================================================

def compute_library_sizes(counts_df: pd.DataFrame) -> pd.Series:
    """Return total counts per sample (column sums).

    Parameters
    ----------
    counts_df : pd.DataFrame
        Raw count matrix (genes x samples).

    Returns
    -------
    pd.Series
        Total counts for each sample, indexed by sample name.
    """
    library_sizes = counts_df.sum(axis=0)
    logger.info(
        "Computed library sizes for %d samples (median=%.0f).",
        len(library_sizes),
        library_sizes.median(),
    )
    return library_sizes


def compute_gene_detection_rate(
    counts_df: pd.DataFrame,
    threshold: float = 0,
) -> pd.Series:
    """Fraction of genes detected per sample.

    A gene is considered detected if its count is strictly greater than
    *threshold*.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Raw count matrix (genes x samples).
    threshold : float, optional
        Minimum count value (exclusive) for a gene to be considered
        detected (default 0).

    Returns
    -------
    pd.Series
        Detection rate (0--1) for each sample, indexed by sample name.
    """
    n_genes = counts_df.shape[0]
    if n_genes == 0:
        logger.warning("Empty count matrix -- returning zero detection rates.")
        return pd.Series(0.0, index=counts_df.columns)

    detected = (counts_df > threshold).sum(axis=0)
    rate = detected / n_genes
    logger.info(
        "Gene detection rates for %d samples (median=%.3f, threshold=%g).",
        len(rate),
        rate.median(),
        threshold,
    )
    return rate


def compute_mito_fraction(
    counts_df: pd.DataFrame,
    organism: str = "human",
) -> pd.Series:
    """Fraction of counts mapping to mitochondrial genes per sample.

    Mitochondrial genes are identified by index entries starting with
    ``'MT-'`` (human) or ``'mt-'`` (mouse).

    Parameters
    ----------
    counts_df : pd.DataFrame
        Raw count matrix (genes x samples).  The index should contain
        gene IDs or gene names.
    organism : str, optional
        ``'human'`` or ``'mouse'`` (default ``'human'``).

    Returns
    -------
    pd.Series
        Mitochondrial fraction (0--1) for each sample.
    """
    if organism.lower() == "human":
        prefix = "MT-"
    elif organism.lower() == "mouse":
        prefix = "mt-"
    else:
        logger.warning(
            "Unknown organism '%s' -- defaulting to human prefix 'MT-'.",
            organism,
        )
        prefix = "MT-"

    mito_mask = counts_df.index.astype(str).str.startswith(prefix)
    n_mito = mito_mask.sum()

    if n_mito == 0:
        logger.warning(
            "No mitochondrial genes found with prefix '%s'. "
            "Returning zero fractions.",
            prefix,
        )
        return pd.Series(0.0, index=counts_df.columns)

    mito_counts = counts_df.loc[mito_mask].sum(axis=0)
    total_counts = counts_df.sum(axis=0)

    # Avoid division by zero for samples with no counts
    fraction = mito_counts / total_counts.replace(0, np.nan)
    fraction = fraction.fillna(0.0)

    logger.info(
        "Mitochondrial fraction computed (%d mito genes, prefix='%s', "
        "median fraction=%.4f).",
        n_mito,
        prefix,
        fraction.median(),
    )
    return fraction


# ===================================================================
# 2. Sample correlation
# ===================================================================

def compute_sample_correlation(
    counts_df: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Pairwise sample correlation on log2-transformed counts.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Raw count matrix (genes x samples).
    method : str, optional
        Correlation method forwarded to ``DataFrame.corr`` (default
        ``'pearson'``).

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix (samples x samples).
    """
    log_counts = np.log2(counts_df + 1)
    corr = log_counts.corr(method=method)
    logger.info(
        "Computed %s sample correlation matrix (%d x %d).",
        method,
        corr.shape[0],
        corr.shape[1],
    )
    return corr


# ===================================================================
# 3. Dimensionality reduction
# ===================================================================

def compute_pca(
    counts_df: pd.DataFrame,
    n_components: int = 10,
    n_top_genes: int = 500,
) -> dict:
    """Principal component analysis on the most variable genes.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Raw count matrix (genes x samples).
    n_components : int, optional
        Number of principal components to compute (default 10).
    n_top_genes : int, optional
        Number of top variable genes to select before PCA (default 500).

    Returns
    -------
    dict
        ``'coordinates'`` : pd.DataFrame
            Samples x components matrix.
        ``'variance_explained'`` : np.ndarray
            Fraction of variance explained by each component.
        ``'loadings'`` : pd.DataFrame
            Genes x components loading matrix.
    """
    # Select top variable genes
    gene_var = counts_df.var(axis=1)
    n_select = min(n_top_genes, len(gene_var))
    top_genes = gene_var.nlargest(n_select).index
    subset = counts_df.loc[top_genes]

    # log2 transform
    log_data = np.log2(subset + 1)

    # Transpose so rows are samples, columns are genes
    X = log_data.T

    # Cap components to min(n_samples, n_genes)
    max_components = min(X.shape[0], X.shape[1])
    n_comp = min(n_components, max_components)

    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(X)

    component_labels = [f"PC{i + 1}" for i in range(n_comp)]

    coordinates = pd.DataFrame(
        coords,
        index=counts_df.columns,
        columns=component_labels,
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=top_genes[:n_select],
        columns=component_labels,
    )

    logger.info(
        "PCA complete: %d components from %d genes x %d samples "
        "(%.1f%% variance explained).",
        n_comp,
        n_select,
        X.shape[0],
        pca.explained_variance_ratio_.sum() * 100,
    )

    return {
        "coordinates": coordinates,
        "variance_explained": pca.explained_variance_ratio_,
        "loadings": loadings,
    }


def compute_umap(
    counts_df: pd.DataFrame,
    n_top_genes: int = 500,
    n_neighbors: int = 15,
    min_dist: float = 0.5,
) -> pd.DataFrame:
    """UMAP embedding of samples based on top variable genes.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Raw count matrix (genes x samples).
    n_top_genes : int, optional
        Number of top variable genes to select (default 500).
    n_neighbors : int, optional
        ``n_neighbors`` parameter for UMAP (default 15).
    min_dist : float, optional
        ``min_dist`` parameter for UMAP (default 0.5).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``UMAP1`` and ``UMAP2``, indexed by sample.
    """
    # Select top variable genes
    gene_var = counts_df.var(axis=1)
    n_select = min(n_top_genes, len(gene_var))
    top_genes = gene_var.nlargest(n_select).index
    subset = counts_df.loc[top_genes]

    # log2 transform and transpose (samples as rows)
    log_data = np.log2(subset + 1)
    X = log_data.T

    # Adjust n_neighbors if we have very few samples
    effective_neighbors = min(n_neighbors, X.shape[0] - 1)
    effective_neighbors = max(effective_neighbors, 2)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    embedding = reducer.fit_transform(X.values)

    result = pd.DataFrame(
        embedding,
        index=counts_df.columns,
        columns=["UMAP1", "UMAP2"],
    )

    logger.info(
        "UMAP embedding computed for %d samples (%d top genes, "
        "n_neighbors=%d, min_dist=%.2f).",
        X.shape[0],
        n_select,
        effective_neighbors,
        min_dist,
    )
    return result


# ===================================================================
# 4. Outlier detection
# ===================================================================

def detect_outliers(
    pca_coords: pd.DataFrame,
    sample_groups: pd.Series,
    n_sd: float = 3,
) -> list:
    """Flag samples far from their group centroid in PCA space.

    Uses the first two principal components.

    Parameters
    ----------
    pca_coords : pd.DataFrame
        PCA coordinates (samples x components) as returned by
        ``compute_pca()['coordinates']``.
    sample_groups : pd.Series
        Series mapping sample names to group labels.  Index must match
        the index of *pca_coords*.
    n_sd : float, optional
        Number of standard deviations beyond which a sample is flagged
        as an outlier (default 3).

    Returns
    -------
    list
        Sample IDs flagged as outliers.
    """
    # Use first 2 PCs
    pc_cols = pca_coords.columns[:2]
    coords_2d = pca_coords[pc_cols]

    outliers = []

    for group_label in sample_groups.unique():
        members = sample_groups[sample_groups == group_label]
        group_samples = members.index.intersection(coords_2d.index)
        if len(group_samples) < 2:
            continue

        group_coords = coords_2d.loc[group_samples].values
        centroid = group_coords.mean(axis=0, keepdims=True)

        distances = cdist(group_coords, centroid, metric="euclidean").ravel()
        mean_dist = distances.mean()
        sd_dist = distances.std(ddof=1)

        if sd_dist == 0:
            continue

        threshold = mean_dist + n_sd * sd_dist
        outlier_mask = distances > threshold
        group_outliers = group_samples[outlier_mask].tolist()
        outliers.extend(group_outliers)

    logger.info(
        "Outlier detection: %d outlier(s) found across %d groups "
        "(n_sd=%g).",
        len(outliers),
        sample_groups.nunique(),
        n_sd,
    )
    return outliers


# ===================================================================
# 5. Orchestrator
# ===================================================================

def run_qc(
    counts_df: pd.DataFrame,
    organism: str = "human",
    sample_groups: pd.Series | None = None,
) -> dict:
    """Run the full QC pipeline.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Raw count matrix (genes x samples).
    organism : str, optional
        Organism label passed to :func:`compute_mito_fraction`
        (default ``'human'``).
    sample_groups : pd.Series or None, optional
        Series mapping sample names to group labels.  Used for outlier
        detection.  If *None*, outlier detection is skipped.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'library_sizes'``: pd.Series
        - ``'detection_rates'``: pd.Series
        - ``'mito_fractions'``: pd.Series
        - ``'correlation_matrix'``: pd.DataFrame
        - ``'pca'``: dict (coordinates, variance_explained, loadings)
        - ``'umap'``: pd.DataFrame
        - ``'outliers'``: list (empty if *sample_groups* is None)
    """
    logger.info(
        "Starting QC pipeline for %d genes x %d samples.",
        counts_df.shape[0],
        counts_df.shape[1],
    )

    library_sizes = compute_library_sizes(counts_df)
    detection_rates = compute_gene_detection_rate(counts_df)
    mito_fractions = compute_mito_fraction(counts_df, organism=organism)
    correlation_matrix = compute_sample_correlation(counts_df)
    pca_result = compute_pca(counts_df)
    umap_result = compute_umap(counts_df)

    if sample_groups is not None:
        outliers = detect_outliers(pca_result["coordinates"], sample_groups)
    else:
        logger.info("No sample groups provided -- skipping outlier detection.")
        outliers = []

    logger.info("QC pipeline complete.")

    return {
        "library_sizes": library_sizes,
        "detection_rates": detection_rates,
        "mito_fractions": mito_fractions,
        "correlation_matrix": correlation_matrix,
        "pca": pca_result,
        "umap": umap_result,
        "outliers": outliers,
    }
