"""
Normalization module for NovoView RNA-Seq platform.

Provides utilities for identifying expression matrix columns, standardizing
expression data, computing TPM and log2 transforms, filtering low-expression
genes, and selecting highly variable genes.
"""

import numpy as np
import pandas as pd

from pipeline.utils import find_column, load_gene_id_mapping, setup_logger

logger = setup_logger(__name__)

# Known column names for gene identifiers and gene symbols, used for
# case-insensitive matching when auto-detecting expression matrix layout.
KNOWN_GENE_ID_NAMES = [
    "gene_id",
    "Geneid",
    "Gene ID",
    "Ensembl_ID",
    "ensembl_gene_id",
    "Gene",
    "GeneID",
]

KNOWN_GENE_NAME_NAMES = [
    "gene_name",
    "Gene Name",
    "GeneName",
    "Symbol",
    "gene_symbol",
    "SYMBOL",
]


def find_expression_columns(df: pd.DataFrame) -> dict:
    """Identify gene ID, gene name, and sample data columns in an expression DataFrame.

    Uses case-insensitive matching against known gene ID and gene name column
    names.  Every remaining column whose dtype is numeric is treated as a
    sample data column.

    Parameters
    ----------
    df : pd.DataFrame
        A count or expression DataFrame whose columns may include gene
        identifiers, gene symbols, and numeric sample data.

    Returns
    -------
    dict
        A dictionary with keys:
        - ``'gene_id_col'``: str or None -- matched gene ID column name.
        - ``'gene_name_col'``: str or None -- matched gene name column name.
        - ``'sample_cols'``: list[str] -- numeric columns that are neither
          gene ID nor gene name.
    """
    columns = list(df.columns)
    columns_lower = {col: col.strip().lower() for col in columns}

    gene_id_col = None
    gene_name_col = None

    # --- gene ID column ---------------------------------------------------
    gene_id_lower = [name.lower() for name in KNOWN_GENE_ID_NAMES]
    for col, col_low in columns_lower.items():
        if col_low in gene_id_lower:
            gene_id_col = col
            logger.debug("Detected gene ID column: '%s'", gene_id_col)
            break

    # --- gene name column -------------------------------------------------
    gene_name_lower = [name.lower() for name in KNOWN_GENE_NAME_NAMES]
    for col, col_low in columns_lower.items():
        if col_low in gene_name_lower and col != gene_id_col:
            gene_name_col = col
            logger.debug("Detected gene name column: '%s'", gene_name_col)
            break

    # --- sample columns (numeric, excluding id/name columns) --------------
    non_sample = {gene_id_col, gene_name_col} - {None}
    sample_cols = [
        col
        for col in columns
        if col not in non_sample and pd.api.types.is_numeric_dtype(df[col])
    ]

    logger.info(
        "Column detection complete -- gene_id_col=%s, gene_name_col=%s, "
        "%d sample columns found.",
        gene_id_col,
        gene_name_col,
        len(sample_cols),
    )

    return {
        "gene_id_col": gene_id_col,
        "gene_name_col": gene_name_col,
        "sample_cols": sample_cols,
    }


def standardize_expression_matrix(
    df: pd.DataFrame,
    organism: str = "human",
) -> pd.DataFrame:
    """Standardize a raw expression DataFrame into a clean matrix.

    Steps performed:
    1. Identify gene ID, gene name, and sample columns via
       :func:`find_expression_columns`.
    2. Set the gene ID column as the DataFrame index.
    3. Sum duplicate gene IDs so each gene appears exactly once.
    4. Optionally map Ensembl IDs to gene symbols using the gene-ID mapping
       provided by :func:`pipeline.utils.load_gene_id_mapping`.

    Parameters
    ----------
    df : pd.DataFrame
        Raw expression / count matrix.
    organism : str, optional
        Organism label forwarded to :func:`load_gene_id_mapping` (default
        ``'human'``).

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by gene ID containing only numeric sample columns.
    """
    col_info = find_expression_columns(df)
    gene_id_col = col_info["gene_id_col"]
    sample_cols = col_info["sample_cols"]

    if gene_id_col is None:
        raise ValueError(
            "Could not identify a gene ID column. Expected one of: "
            + ", ".join(KNOWN_GENE_ID_NAMES)
        )

    if not sample_cols:
        raise ValueError("No numeric sample columns detected in the DataFrame.")

    # Keep only gene_id + sample columns
    result = df[[gene_id_col] + sample_cols].copy()
    result = result.set_index(gene_id_col)
    result.index.name = "gene_id"

    # Handle duplicate gene IDs by summing counts
    n_dupes = result.index.duplicated().sum()
    if n_dupes > 0:
        logger.warning(
            "Found %d duplicate gene IDs -- summing their counts.", n_dupes
        )
        result = result.groupby(level=0).sum()

    # Attempt Ensembl -> symbol mapping
    try:
        mapping = load_gene_id_mapping(organism=organism)
        if mapping is not None and len(mapping) > 0:
            # Build a Series for vectorized lookup instead of per-element lambda
            mapping_series = pd.Series(mapping, dtype="object")
            # Only map IDs that actually have a mapping entry
            needs_mapping = result.index.isin(mapping_series.index)
            if needs_mapping.any():
                new_index = result.index.to_series()
                new_index[needs_mapping] = new_index[needs_mapping].map(mapping_series)
                n_mapped = int(needs_mapping.sum())
                logger.info(
                    "Mapped %d / %d Ensembl IDs to gene symbols.",
                    n_mapped,
                    len(result),
                )
                result.index = pd.Index(new_index.values, name="gene_id")
                # Re-aggregate if mapping introduced duplicates
                if result.index.duplicated().any():
                    logger.warning(
                        "Symbol mapping introduced duplicates -- summing."
                    )
                    result = result.groupby(level=0).sum()
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        logger.warning(
            "Could not load gene ID mapping for organism '%s': %s",
            organism,
            exc,
        )

    logger.info(
        "Standardised expression matrix: %d genes x %d samples.",
        result.shape[0],
        result.shape[1],
    )
    return result


def compute_tpm_from_counts(
    counts_df: pd.DataFrame,
    gene_lengths: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute Transcripts Per Million (TPM) from a raw count matrix.

    TPM formula per sample:

        rate_g = counts_g / length_g
        TPM_g  = rate_g / sum(rate) * 1e6

    Parameters
    ----------
    counts_df : pd.DataFrame
        Raw count matrix (genes x samples).  Index should be gene IDs.
    gene_lengths : pd.Series or None, optional
        Gene lengths in base-pairs, indexed by gene ID.  If *None*, a uniform
        placeholder length of 1000 bp is assumed for every gene and a warning
        is logged.

    Returns
    -------
    pd.DataFrame
        TPM-normalised expression matrix with the same shape and index as
        *counts_df*.
    """
    if gene_lengths is not None:
        # Align lengths to counts index; missing genes get NaN
        lengths = gene_lengths.reindex(counts_df.index)
        missing = int(lengths.isna().sum())
        if missing > 0:
            pct = missing / len(lengths) * 100
            logger.warning(
                "%d / %d genes (%.1f%%) lack length information -- using "
                "placeholder length of 1000 bp.  TPM values for these genes "
                "will be approximate.",
                missing, len(lengths), pct,
            )
            lengths = lengths.fillna(1000)
    else:
        logger.warning(
            "No gene lengths provided -- using a uniform placeholder "
            "length of 1000 bp for all genes.  TPM values will be "
            "approximate."
        )
        lengths = pd.Series(1000, index=counts_df.index)

    # rates: counts / length (per-kilobase)
    rate = counts_df.div(lengths, axis=0)

    # Normalise each sample so that rates sum to 1e6
    rate_sum = rate.sum(axis=0)
    zero_samples = (rate_sum == 0).sum()
    if zero_samples > 0:
        logger.warning(
            "%d sample(s) have zero total counts -- TPM will be NaN for those.",
            zero_samples,
        )
    rate_sum = rate_sum.replace(0, np.nan)  # avoid division by zero
    tpm = rate.div(rate_sum, axis=1) * 1e6

    logger.info("Computed TPM for %d genes x %d samples.", *tpm.shape)
    return tpm


def compute_log2_transform(
    df: pd.DataFrame,
    pseudocount: float = 1,
) -> pd.DataFrame:
    """Apply a log2 transformation with a pseudocount.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric expression matrix (genes x samples).
    pseudocount : float, optional
        Value added before taking the log to avoid log(0) (default 1).

    Returns
    -------
    pd.DataFrame
        log2(df + pseudocount).
    """
    if pseudocount <= 0:
        raise ValueError(
            f"pseudocount must be positive, got {pseudocount}. "
            "A non-positive pseudocount produces -Inf or NaN values."
        )
    result = np.log2(df + pseudocount)
    logger.info(
        "Applied log2 transform (pseudocount=%g) to %d genes x %d samples.",
        pseudocount,
        *result.shape,
    )
    return result


def filter_low_expression(
    df: pd.DataFrame,
    min_count: float = 10,
    min_samples: int = 2,
) -> pd.DataFrame:
    """Remove genes that are not expressed above a threshold in enough samples.

    A gene is kept only if at least *min_samples* samples have a count
    >= *min_count* for that gene.

    Parameters
    ----------
    df : pd.DataFrame
        Expression / count matrix (genes x samples).
    min_count : float, optional
        Minimum count value to consider a gene "expressed" in a sample
        (default 10).
    min_samples : int, optional
        Minimum number of samples that must meet *min_count* for a gene to be
        retained (default 2).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only genes that pass the expression
        threshold.
    """
    n_before = len(df)
    expressed = (df >= min_count).sum(axis=1)
    keep = expressed >= min_samples
    filtered = df.loc[keep]
    n_removed = n_before - len(filtered)
    logger.info(
        "Filtered low-expression genes: kept %d / %d (removed %d) "
        "with min_count=%g in >= %d samples.",
        len(filtered),
        n_before,
        n_removed,
        min_count,
        min_samples,
    )
    return filtered


def get_top_variable_genes(
    df: pd.DataFrame,
    n: int = 5000,
) -> pd.DataFrame:
    """Select the top *n* most variable genes by variance across samples.

    Parameters
    ----------
    df : pd.DataFrame
        Expression matrix (genes x samples).
    n : int, optional
        Number of top variable genes to return (default 5000).  If the matrix
        has fewer genes, all genes are returned.

    Returns
    -------
    pd.DataFrame
        Subset of *df* containing only the top *n* most variable genes,
        sorted by descending variance.
    """
    if n >= len(df):
        logger.info(
            "Requested %d variable genes but only %d available -- "
            "returning all genes.",
            n,
            len(df),
        )
        return df

    gene_var = df.var(axis=1)
    # Guard against all-zero variance (constant expression across samples)
    if gene_var.max() == 0 or gene_var.dropna().empty:
        logger.warning(
            "All genes have zero variance -- returning full matrix unchanged."
        )
        return df
    top_genes = gene_var.nlargest(n).index
    result = df.loc[top_genes]
    logger.info(
        "Selected top %d variable genes (variance range: %.2f -- %.2f).",
        n,
        gene_var.loc[top_genes].min(),
        gene_var.loc[top_genes].max(),
    )
    return result
