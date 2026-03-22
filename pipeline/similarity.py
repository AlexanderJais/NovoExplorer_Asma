"""
Gene similarity and clustering module for the NovoView RNA-Seq platform.

Computes pairwise gene similarity from expression profiles and differential
expression signatures, performs hierarchical clustering, and provides
neighbor-lookup utilities for the interactive front-end.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity

from pipeline.utils import setup_logger

logger = setup_logger(__name__)


# ===================================================================
# 1. Expression-based cosine similarity
# ===================================================================

def compute_cosine_similarity_matrix(
    expression_df: pd.DataFrame,
    top_n_genes: int = 2000,
) -> pd.DataFrame:
    """Compute pairwise cosine similarity for the most variable genes.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Log2(TPM + 1) expression matrix with shape (genes x samples).
        The index must contain gene identifiers.
    top_n_genes : int, optional
        Number of most-variable genes to retain before computing the
        similarity matrix (default 2000).

    Returns
    -------
    pd.DataFrame
        Square DataFrame (genes x genes) of cosine similarity values,
        indexed and columned by gene identifiers.
    """
    logger.info(
        "Computing cosine similarity matrix for top %d variable genes "
        "(input: %d genes x %d samples).",
        top_n_genes,
        expression_df.shape[0],
        expression_df.shape[1],
    )

    _MAX_SAFE_GENES = 10000
    if top_n_genes > _MAX_SAFE_GENES:
        logger.warning(
            "top_n_genes=%d exceeds %d; this creates a %.1f GB matrix. "
            "Capping at %d.",
            top_n_genes, _MAX_SAFE_GENES,
            (top_n_genes ** 2 * 8) / 1e9, _MAX_SAFE_GENES,
        )
        top_n_genes = _MAX_SAFE_GENES

    # Select top N most variable genes by variance across samples
    variances = expression_df.var(axis=1)
    n_select = min(top_n_genes, len(variances))
    top_genes = variances.nlargest(n_select).index
    subset = expression_df.loc[top_genes]

    logger.info("Selected %d genes by variance.", len(top_genes))

    # Compute pairwise cosine similarity (rows = genes)
    sim_values = cosine_similarity(subset.values)
    sim_df = pd.DataFrame(sim_values, index=subset.index, columns=subset.index)

    logger.info("Cosine similarity matrix shape: %s.", sim_df.shape)
    return sim_df


# ===================================================================
# 2. On-the-fly similarity for a single query gene
# ===================================================================

def compute_on_the_fly_similarity(
    query_gene: str,
    expression_df: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    """Compute cosine similarity of *query_gene* against all other genes.

    This is intended for genes that fall outside the pre-computed
    similarity matrix so that neighbours can still be retrieved at
    query time.

    Parameters
    ----------
    query_gene : str
        Gene identifier that must be present in *expression_df*.
    expression_df : pd.DataFrame
        Log2(TPM + 1) expression matrix (genes x samples).
    top_n : int, optional
        Number of most-similar genes to return (default 50).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['gene', 'similarity']`` sorted by
        descending similarity.  The query gene itself is excluded.

    Raises
    ------
    KeyError
        If *query_gene* is not found in *expression_df*.
    """
    if query_gene not in expression_df.index:
        raise KeyError(f"Query gene '{query_gene}' not found in expression matrix.")

    query_vec = expression_df.loc[[query_gene]].values  # shape (1, n_samples)
    all_vecs = expression_df.values                     # shape (n_genes, n_samples)

    similarities = cosine_similarity(query_vec, all_vecs).flatten()

    result = pd.DataFrame({
        "gene": expression_df.index,
        "similarity": similarities,
    })

    # Drop the query gene itself and sort descending
    result = (
        result[result["gene"] != query_gene]
        .sort_values("similarity", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return result


# ===================================================================
# 3. Hierarchical clustering
# ===================================================================

def cluster_genes(
    similarity_matrix: pd.DataFrame,
    min_clusters: int = 20,
    max_clusters: int = 50,
) -> tuple[pd.Series, np.ndarray]:
    """Hierarchically cluster genes from a cosine-similarity matrix.

    The similarity matrix is converted to a distance matrix
    (``1 - similarity``), condensed, and clustered with Ward's method.
    The dendrogram is cut at a height that yields a number of clusters
    between *min_clusters* and *max_clusters*.

    Parameters
    ----------
    similarity_matrix : pd.DataFrame
        Square gene-by-gene cosine similarity DataFrame.
    min_clusters : int, optional
        Minimum desired number of clusters (default 20).
    max_clusters : int, optional
        Maximum desired number of clusters (default 50).

    Returns
    -------
    cluster_labels : pd.Series
        Mapping of gene identifier -> integer cluster ID.
    linkage_matrix : np.ndarray
        The linkage matrix produced by :func:`scipy.cluster.hierarchy.linkage`.
    """
    logger.info(
        "Clustering %d genes (target %d–%d clusters).",
        len(similarity_matrix),
        min_clusters,
        max_clusters,
    )

    # Convert similarity -> distance, clip to [0, 1] for numerical safety
    distance_matrix = 1.0 - similarity_matrix.values
    np.fill_diagonal(distance_matrix, 0.0)
    distance_matrix = np.clip(distance_matrix, 0.0, 1.0)

    # Ensure perfect symmetry before condensing
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0

    condensed = squareform(distance_matrix, checks=False)

    # Average linkage (Ward requires Euclidean distances; cosine distances
    # violate its assumptions and can produce non-monotonic dendrograms)
    linkage_matrix = linkage(condensed, method="average")

    # Binary-search for a cut height that yields the desired cluster count
    heights = linkage_matrix[:, 2]
    lo, hi = float(heights.min()), float(heights.max())
    best_labels = None
    best_n = None

    for _ in range(50):
        mid = (lo + hi) / 2.0
        labels = fcluster(linkage_matrix, t=mid, criterion="distance")
        n_clusters = len(np.unique(labels))

        if best_labels is None or abs(n_clusters - (min_clusters + max_clusters) / 2) < abs(best_n - (min_clusters + max_clusters) / 2):
            best_labels = labels
            best_n = n_clusters

        if min_clusters <= n_clusters <= max_clusters:
            break

        if n_clusters > max_clusters:
            # Too many clusters – increase cut height to merge more
            lo = mid
        else:
            # Too few clusters – decrease cut height to split more
            hi = mid

    if best_labels is None:
        logger.warning("Clustering produced no valid labels; assigning all genes to cluster 0.")
        best_labels = np.zeros(len(similarity_matrix), dtype=int)
        best_n = 1

    cluster_labels = pd.Series(best_labels, index=similarity_matrix.index, name="cluster_id")
    logger.info("Produced %d clusters.", best_n)

    return cluster_labels, linkage_matrix


# ===================================================================
# 4. Expression-signature vectors from DEG results
# ===================================================================

def compute_expression_signature_vectors(
    deg_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build per-gene signature vectors from differential expression results.

    For each comparison in *deg_results*, the log2 fold-change value is
    extracted for every gene.  Genes absent from a particular comparison
    receive a value of 0.

    Parameters
    ----------
    deg_results : dict[str, pd.DataFrame]
        Mapping of comparison name -> DataFrame.  Each DataFrame must
        contain a gene identifier column (``'gene_name'`` or ``'gene_id'``)
        and a fold-change column (``'log2fc'`` or ``'log2FoldChange'``).

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (genes x comparisons) containing log2FC
        values, with 0 for missing entries.
    """
    logger.info(
        "Building expression signature vectors from %d comparisons.",
        len(deg_results),
    )

    pieces: dict[str, pd.Series] = {}
    skipped: list[tuple[str, str]] = []
    for comparison_name, df in deg_results.items():
        # Determine the gene identifier column (standardized names)
        if "gene_name" in df.columns:
            gene_col = "gene_name"
        elif "gene_id" in df.columns:
            gene_col = "gene_id"
        else:
            skipped.append((comparison_name, "missing gene_name/gene_id"))
            continue

        # Determine the log2FC column (standardized name)
        if "log2fc" in df.columns:
            fc_col = "log2fc"
        elif "log2FoldChange" in df.columns:
            fc_col = "log2FoldChange"
        else:
            skipped.append((comparison_name, "missing log2fc/log2FoldChange"))
            continue

        series = df.set_index(gene_col)[fc_col]
        # If duplicates exist, keep the first occurrence
        series = series[~series.index.duplicated(keep="first")]
        pieces[comparison_name] = series

    if skipped:
        logger.warning(
            "Skipped %d comparison(s) for signature vectors: %s",
            len(skipped),
            "; ".join(f"{c} ({r})" for c, r in skipped),
        )

    signature_df = pd.DataFrame(pieces).fillna(0.0)
    logger.info(
        "Signature vectors: %d genes x %d comparisons.",
        signature_df.shape[0],
        signature_df.shape[1],
    )
    return signature_df


# ===================================================================
# 5. Signature-based similarity search
# ===================================================================

def find_similar_by_signature(
    query_gene: str,
    signature_vectors: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    """Find the most similar genes by differential-expression signature.

    Parameters
    ----------
    query_gene : str
        Gene identifier; must be present in *signature_vectors*.
    signature_vectors : pd.DataFrame
        Genes x comparisons matrix of log2FC values (as produced by
        :func:`compute_expression_signature_vectors`).
    top_n : int, optional
        Number of most-similar genes to return (default 50).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['gene', 'similarity']``.

    Raises
    ------
    KeyError
        If *query_gene* is not found in *signature_vectors*.
    """
    if query_gene not in signature_vectors.index:
        raise KeyError(
            f"Query gene '{query_gene}' not found in signature vectors."
        )

    query_vec = signature_vectors.loc[[query_gene]].values
    all_vecs = signature_vectors.values

    similarities = cosine_similarity(query_vec, all_vecs).flatten()

    result = pd.DataFrame({
        "gene": signature_vectors.index,
        "similarity": similarities,
    })

    result = (
        result[result["gene"] != query_gene]
        .sort_values("similarity", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return result


# ===================================================================
# 6. Unified neighbor lookup
# ===================================================================

def get_gene_neighbors(
    gene: str,
    similarity_matrix: pd.DataFrame | None = None,
    expression_df: pd.DataFrame | None = None,
    top_n: int = 50,
) -> pd.DataFrame:
    """Return the nearest neighbours of *gene* by cosine similarity.

    If *gene* is present in the pre-computed *similarity_matrix*, that
    matrix is used directly.  Otherwise the similarity is computed on
    the fly from *expression_df*.

    Parameters
    ----------
    gene : str
        Gene identifier to query.
    similarity_matrix : pd.DataFrame or None, optional
        Pre-computed gene-by-gene cosine similarity matrix.
    expression_df : pd.DataFrame or None, optional
        Log2(TPM + 1) expression matrix (genes x samples), used as
        fallback when *gene* is not in the pre-computed matrix.
    top_n : int, optional
        Number of neighbours to return (default 50).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['gene', 'similarity']``.

    Raises
    ------
    ValueError
        If the gene cannot be found in either source.
    """
    # Try the pre-computed matrix first
    if similarity_matrix is not None and gene in similarity_matrix.index:
        logger.debug("Using pre-computed similarity matrix for '%s'.", gene)
        sims = similarity_matrix.loc[gene].drop(gene, errors="ignore")
        result = (
            sims.sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        result.columns = ["gene", "similarity"]
        return result

    # Fallback to on-the-fly computation
    if expression_df is not None and gene in expression_df.index:
        logger.debug("Computing similarity on the fly for '%s'.", gene)
        return compute_on_the_fly_similarity(gene, expression_df, top_n=top_n)

    raise ValueError(
        f"Gene '{gene}' not found in the similarity matrix or expression "
        "data.  Provide at least one source that contains this gene."
    )


# ===================================================================
# 7. Main orchestrator
# ===================================================================

def run_similarity(
    expression_df: pd.DataFrame,
    deg_results: dict[str, pd.DataFrame],
    config: dict,
) -> dict:
    """Run the full gene-similarity and clustering pipeline.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Log2(TPM + 1) expression matrix (genes x samples).
    deg_results : dict[str, pd.DataFrame]
        Mapping of comparison name -> DEG results DataFrame.
    config : dict
        Pipeline configuration.  Recognised keys (all optional):

        * ``top_n_genes`` – number of variable genes for the similarity
          matrix (default 2000).
        * ``min_clusters`` – minimum cluster count (default 20).
        * ``max_clusters`` – maximum cluster count (default 50).

    Returns
    -------
    dict
        Dictionary with keys:

        * ``"similarity_matrix"`` – genes x genes cosine similarity
          :class:`~pandas.DataFrame`.
        * ``"cluster_labels"`` – :class:`~pandas.Series` mapping gene
          to cluster ID.
        * ``"linkage_matrix"`` – :class:`numpy.ndarray` linkage matrix.
        * ``"signature_vectors"`` – genes x comparisons log2FC
          :class:`~pandas.DataFrame`.
    """
    logger.info("=== Starting gene similarity pipeline ===")

    top_n_genes = config.get("top_n_genes", 2000)
    min_clusters = config.get("min_clusters", 20)
    max_clusters = config.get("max_clusters", 50)

    # 1. Cosine similarity matrix for top variable genes
    similarity_matrix = compute_cosine_similarity_matrix(
        expression_df, top_n_genes=top_n_genes
    )

    # 2. Hierarchical clustering
    cluster_labels, linkage_matrix = cluster_genes(
        similarity_matrix,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
    )

    # 3. Expression signature vectors from DEG results
    signature_vectors = compute_expression_signature_vectors(deg_results)

    logger.info("=== Gene similarity pipeline complete ===")

    return {
        "similarity_matrix": similarity_matrix,
        "cluster_labels": cluster_labels,
        "linkage_matrix": linkage_matrix,
        "signature_vectors": signature_vectors,
    }
