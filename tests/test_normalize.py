"""Tests for the NovoView normalize module.

Validates expression column detection, matrix standardization, TPM
computation, log2 transformation, low-expression filtering, and
top-variable-gene selection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the novoview package root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.normalize import (
    compute_log2_transform,
    compute_tpm_from_counts,
    filter_low_expression,
    find_expression_columns,
    get_top_variable_genes,
    standardize_expression_matrix,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_expression_df(
    n_genes: int = 20,
    n_samples: int = 6,
    *,
    include_gene_id: bool = True,
    include_gene_name: bool = True,
    seed: int = 123,
) -> pd.DataFrame:
    """Build a synthetic expression DataFrame for testing."""
    rng = np.random.default_rng(seed)
    sample_names = [f"Sample{i}" for i in range(1, n_samples + 1)]
    data = rng.integers(0, 5000, size=(n_genes, n_samples))
    df = pd.DataFrame(data, columns=sample_names)

    if include_gene_id:
        df.insert(0, "gene_id", [f"ENSG{i:011d}" for i in range(1, n_genes + 1)])
    if include_gene_name:
        col_pos = 1 if include_gene_id else 0
        df.insert(col_pos, "gene_name", [f"GENE{i:03d}" for i in range(1, n_genes + 1)])

    return df


def _make_indexed_counts(n_genes: int = 20, n_samples: int = 6, seed: int = 99) -> pd.DataFrame:
    """Build a count matrix already indexed by gene_id (no annotation cols)."""
    rng = np.random.default_rng(seed)
    sample_names = [f"Sample{i}" for i in range(1, n_samples + 1)]
    data = rng.integers(0, 5000, size=(n_genes, n_samples))
    df = pd.DataFrame(data, columns=sample_names,
                      index=[f"GENE{i:03d}" for i in range(1, n_genes + 1)])
    df.index.name = "gene_id"
    return df


# -----------------------------------------------------------------------
# 1. find_expression_columns
# -----------------------------------------------------------------------


class TestFindExpressionColumns:
    """Tests for find_expression_columns."""

    def test_detects_gene_id_column(self) -> None:
        df = _make_expression_df()
        result = find_expression_columns(df)
        assert result["gene_id_col"] == "gene_id"

    def test_detects_gene_name_column(self) -> None:
        df = _make_expression_df()
        result = find_expression_columns(df)
        assert result["gene_name_col"] == "gene_name"

    def test_detects_sample_columns(self) -> None:
        df = _make_expression_df(n_samples=4)
        result = find_expression_columns(df)
        assert len(result["sample_cols"]) == 4
        for col in result["sample_cols"]:
            assert col.startswith("Sample")

    def test_no_gene_id_column(self) -> None:
        df = _make_expression_df(include_gene_id=False, include_gene_name=False)
        result = find_expression_columns(df)
        assert result["gene_id_col"] is None
        assert result["gene_name_col"] is None
        # All columns should be sample columns
        assert len(result["sample_cols"]) == 6

    def test_alternative_gene_id_name(self) -> None:
        """Columns named 'Geneid' should be detected as gene_id."""
        df = _make_expression_df(include_gene_id=False)
        df.insert(0, "Geneid", [f"ENS{i}" for i in range(len(df))])
        result = find_expression_columns(df)
        assert result["gene_id_col"] == "Geneid"

    def test_sample_cols_exclude_annotation(self) -> None:
        df = _make_expression_df()
        result = find_expression_columns(df)
        assert "gene_id" not in result["sample_cols"]
        assert "gene_name" not in result["sample_cols"]

    def test_with_fixture(self, sample_count_matrix: pd.DataFrame) -> None:
        result = find_expression_columns(sample_count_matrix)
        assert result["gene_id_col"] == "gene_id"
        assert result["gene_name_col"] == "gene_name"
        assert len(result["sample_cols"]) == 6


# -----------------------------------------------------------------------
# 2. standardize_expression_matrix
# -----------------------------------------------------------------------


class TestStandardizeExpressionMatrix:
    """Tests for standardize_expression_matrix."""

    def test_basic_standardization(self) -> None:
        df = _make_expression_df()
        result = standardize_expression_matrix(df)

        assert result.index.name == "gene_id"
        assert "gene_id" not in result.columns
        assert "gene_name" not in result.columns

    def test_shape_preserved(self) -> None:
        df = _make_expression_df(n_genes=15, n_samples=4)
        result = standardize_expression_matrix(df)

        assert result.shape[0] == 15
        assert result.shape[1] == 4

    def test_only_numeric_columns(self) -> None:
        df = _make_expression_df()
        result = standardize_expression_matrix(df)

        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col])

    def test_duplicate_gene_ids_summed(self) -> None:
        """Duplicate gene IDs should be aggregated by summing."""
        df = _make_expression_df(n_genes=10)
        # Create a duplicate gene_id
        df.loc[9, "gene_id"] = df.loc[0, "gene_id"]

        result = standardize_expression_matrix(df)
        # Should have 9 unique genes (10 - 1 duplicate)
        assert len(result) == 9

    def test_raises_without_gene_id(self) -> None:
        df = _make_expression_df(include_gene_id=False, include_gene_name=False)
        with pytest.raises(ValueError, match="gene ID column"):
            standardize_expression_matrix(df)

    def test_raises_without_numeric_columns(self) -> None:
        df = pd.DataFrame({
            "gene_id": ["A", "B", "C"],
            "description": ["desc1", "desc2", "desc3"],
        })
        with pytest.raises(ValueError, match="No numeric sample columns"):
            standardize_expression_matrix(df)

    def test_with_fixture(self, sample_count_matrix: pd.DataFrame) -> None:
        result = standardize_expression_matrix(sample_count_matrix)
        assert result.index.name == "gene_id"
        assert result.shape[0] == 20
        assert result.shape[1] == 6


# -----------------------------------------------------------------------
# 3. compute_tpm_from_counts
# -----------------------------------------------------------------------


class TestComputeTpmFromCounts:
    """Tests for compute_tpm_from_counts."""

    def test_tpm_columns_sum_to_1e6(self) -> None:
        counts = _make_indexed_counts()
        tpm = compute_tpm_from_counts(counts)

        for col in tpm.columns:
            assert abs(tpm[col].sum() - 1e6) < 1.0, (
                f"TPM sum for {col} = {tpm[col].sum()}"
            )

    def test_shape_preserved(self) -> None:
        counts = _make_indexed_counts(n_genes=30, n_samples=4)
        tpm = compute_tpm_from_counts(counts)
        assert tpm.shape == counts.shape

    def test_index_preserved(self) -> None:
        counts = _make_indexed_counts()
        tpm = compute_tpm_from_counts(counts)
        assert list(tpm.index) == list(counts.index)

    def test_non_negative_values(self) -> None:
        counts = _make_indexed_counts()
        tpm = compute_tpm_from_counts(counts)
        assert (tpm >= 0).all().all()

    def test_with_gene_lengths(self) -> None:
        counts = _make_indexed_counts(n_genes=10)
        rng = np.random.default_rng(42)
        lengths = pd.Series(
            rng.integers(500, 5000, size=10),
            index=counts.index,
        )
        tpm = compute_tpm_from_counts(counts, gene_lengths=lengths)

        for col in tpm.columns:
            assert abs(tpm[col].sum() - 1e6) < 1.0

    def test_uniform_lengths_same_as_no_lengths(self) -> None:
        """With uniform length=1000, result should match no-lengths version."""
        counts = _make_indexed_counts(n_genes=10)
        tpm_no_lengths = compute_tpm_from_counts(counts, gene_lengths=None)
        uniform_lengths = pd.Series(1000, index=counts.index)
        tpm_uniform = compute_tpm_from_counts(counts, gene_lengths=uniform_lengths)

        pd.testing.assert_frame_equal(tpm_no_lengths, tpm_uniform)

    def test_zero_count_gene_is_zero(self) -> None:
        counts = _make_indexed_counts(n_genes=5)
        counts.iloc[0, :] = 0  # Set first gene to all zeros
        tpm = compute_tpm_from_counts(counts)
        assert (tpm.iloc[0, :] == 0).all()


# -----------------------------------------------------------------------
# 4. compute_log2_transform (test_log2_transform)
# -----------------------------------------------------------------------


class TestLog2Transform:
    """Tests for compute_log2_transform."""

    def test_basic_transform(self) -> None:
        df = _make_indexed_counts(n_genes=5, n_samples=3)
        result = compute_log2_transform(df)
        expected = np.log2(df + 1)
        pd.testing.assert_frame_equal(result, expected)

    def test_custom_pseudocount(self) -> None:
        df = _make_indexed_counts(n_genes=5, n_samples=3)
        result = compute_log2_transform(df, pseudocount=0.5)
        expected = np.log2(df + 0.5)
        pd.testing.assert_frame_equal(result, expected)

    def test_shape_preserved(self) -> None:
        df = _make_indexed_counts(n_genes=20, n_samples=6)
        result = compute_log2_transform(df)
        assert result.shape == df.shape

    def test_index_preserved(self) -> None:
        df = _make_indexed_counts()
        result = compute_log2_transform(df)
        assert list(result.index) == list(df.index)
        assert list(result.columns) == list(df.columns)

    def test_zero_values_handled(self) -> None:
        """With pseudocount=1, log2(0 + 1) = 0."""
        df = pd.DataFrame(
            {"S1": [0, 0, 0], "S2": [0, 0, 0]},
            index=["A", "B", "C"],
        )
        result = compute_log2_transform(df, pseudocount=1)
        assert (result == 0).all().all()

    def test_known_values(self) -> None:
        df = pd.DataFrame(
            {"S1": [1, 3, 7, 15]},
            index=["A", "B", "C", "D"],
        )
        result = compute_log2_transform(df, pseudocount=1)
        # log2(2) = 1, log2(4) = 2, log2(8) = 3, log2(16) = 4
        expected_vals = [1.0, 2.0, 3.0, 4.0]
        np.testing.assert_allclose(result["S1"].values, expected_vals)


# -----------------------------------------------------------------------
# 5. filter_low_expression
# -----------------------------------------------------------------------


class TestFilterLowExpression:
    """Tests for filter_low_expression."""

    def test_removes_low_genes(self) -> None:
        df = pd.DataFrame(
            {
                "S1": [0, 0, 100, 200],
                "S2": [0, 0, 150, 300],
                "S3": [0, 50, 120, 250],
            },
            index=["low1", "low2", "high1", "high2"],
        )
        result = filter_low_expression(df, min_count=10, min_samples=2)
        # low1: 0 samples >= 10 -> removed
        # low2: 1 sample >= 10 -> removed
        # high1: 3 samples >= 10 -> kept
        # high2: 3 samples >= 10 -> kept
        assert list(result.index) == ["high1", "high2"]

    def test_all_pass(self) -> None:
        df = _make_indexed_counts(n_genes=10)
        # With high counts and low threshold, all should pass
        result = filter_low_expression(df, min_count=0, min_samples=1)
        assert len(result) == len(df)

    def test_none_pass(self) -> None:
        df = pd.DataFrame(
            {"S1": [0, 0], "S2": [0, 0]},
            index=["A", "B"],
        )
        result = filter_low_expression(df, min_count=10, min_samples=1)
        assert len(result) == 0

    def test_shape_columns_preserved(self) -> None:
        df = _make_indexed_counts(n_genes=20, n_samples=6)
        result = filter_low_expression(df, min_count=10, min_samples=2)
        assert result.shape[1] == df.shape[1]
        assert list(result.columns) == list(df.columns)

    def test_default_parameters(self) -> None:
        """Default min_count=10, min_samples=2."""
        df = pd.DataFrame(
            {
                "S1": [15, 5, 20],
                "S2": [12, 3, 25],
                "S3": [8, 1, 30],
            },
            index=["gene1", "gene2", "gene3"],
        )
        result = filter_low_expression(df)
        # gene1: S1=15>=10, S2=12>=10, S3=8<10 -> 2 samples pass -> kept
        # gene2: 0 pass -> removed
        # gene3: all pass -> kept
        assert "gene1" in result.index
        assert "gene2" not in result.index
        assert "gene3" in result.index

    def test_min_samples_boundary(self) -> None:
        df = pd.DataFrame(
            {"S1": [20], "S2": [5], "S3": [5]},
            index=["A"],
        )
        # Exactly 1 sample >= 10, need min_samples=1 to keep
        kept = filter_low_expression(df, min_count=10, min_samples=1)
        assert len(kept) == 1
        # Need min_samples=2 to remove
        removed = filter_low_expression(df, min_count=10, min_samples=2)
        assert len(removed) == 0


# -----------------------------------------------------------------------
# 6. get_top_variable_genes
# -----------------------------------------------------------------------


class TestGetTopVariableGenes:
    """Tests for get_top_variable_genes."""

    def test_returns_requested_number(self) -> None:
        df = _make_indexed_counts(n_genes=50, n_samples=6)
        result = get_top_variable_genes(df, n=10)
        assert len(result) == 10

    def test_returns_all_when_n_exceeds_genes(self) -> None:
        df = _make_indexed_counts(n_genes=10, n_samples=4)
        result = get_top_variable_genes(df, n=100)
        assert len(result) == 10

    def test_columns_preserved(self) -> None:
        df = _make_indexed_counts(n_genes=20, n_samples=6)
        result = get_top_variable_genes(df, n=5)
        assert list(result.columns) == list(df.columns)

    def test_highest_variance_selected(self) -> None:
        """Ensure the gene with highest variance is included."""
        df = pd.DataFrame(
            {
                "S1": [0, 10, 5],
                "S2": [1000, 10, 5],
                "S3": [0, 10, 5],
            },
            index=["high_var", "low_var1", "low_var2"],
        )
        result = get_top_variable_genes(df, n=1)
        assert "high_var" in result.index

    def test_sorted_by_descending_variance(self) -> None:
        df = _make_indexed_counts(n_genes=30, n_samples=6)
        result = get_top_variable_genes(df, n=10)
        variances = result.var(axis=1)
        # Check that variances are non-increasing
        vals = variances.values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_result_is_subset_of_input(self) -> None:
        df = _make_indexed_counts(n_genes=20, n_samples=4)
        result = get_top_variable_genes(df, n=5)
        assert set(result.index).issubset(set(df.index))

    def test_default_n_5000(self) -> None:
        """Default n=5000; with fewer genes, all should be returned."""
        df = _make_indexed_counts(n_genes=20, n_samples=4)
        result = get_top_variable_genes(df)
        assert len(result) == 20
