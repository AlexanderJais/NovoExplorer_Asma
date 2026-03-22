"""Tests for the NovoView QC module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.qc import (
    compute_gene_detection_rate,
    compute_library_sizes,
    compute_mito_fraction,
    compute_pca,
    compute_sample_correlation,
    compute_umap,
    detect_outliers,
    run_qc,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_counts(n_genes=50, n_samples=6, seed=42):
    """Build a simple count matrix (genes x samples)."""
    rng = np.random.default_rng(seed)
    samples = [f"S{i}" for i in range(1, n_samples + 1)]
    genes = [f"GENE{i:03d}" for i in range(1, n_genes + 1)]
    data = rng.integers(0, 5000, size=(n_genes, n_samples))
    return pd.DataFrame(data, index=genes, columns=samples)


def _make_counts_with_mito(n_genes=50, n_samples=6, seed=42):
    """Count matrix with some MT- prefixed genes."""
    df = _make_counts(n_genes, n_samples, seed)
    mito_names = [f"MT-GENE{i}" for i in range(1, 4)]
    rng = np.random.default_rng(seed + 1)
    mito_data = rng.integers(0, 500, size=(3, n_samples))
    mito_df = pd.DataFrame(mito_data, index=mito_names, columns=df.columns)
    return pd.concat([df, mito_df])


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestLibrarySizes:
    def test_returns_series(self):
        counts = _make_counts()
        result = compute_library_sizes(counts)
        assert isinstance(result, pd.Series)
        assert len(result) == 6

    def test_sums_are_positive(self):
        counts = _make_counts()
        result = compute_library_sizes(counts)
        assert (result > 0).all()

    def test_empty_matrix(self):
        counts = pd.DataFrame(dtype=float)
        result = compute_library_sizes(counts)
        assert len(result) == 0


class TestDetectionRate:
    def test_all_nonzero(self):
        counts = pd.DataFrame({"S1": [1, 2, 3], "S2": [4, 5, 6]}, index=["A", "B", "C"])
        rate = compute_gene_detection_rate(counts)
        assert (rate == 1.0).all()

    def test_some_zeros(self):
        counts = pd.DataFrame({"S1": [0, 0, 3], "S2": [1, 0, 0]}, index=["A", "B", "C"])
        rate = compute_gene_detection_rate(counts)
        assert rate["S1"] == pytest.approx(1 / 3)
        assert rate["S2"] == pytest.approx(1 / 3)

    def test_empty_matrix(self):
        counts = pd.DataFrame(index=[], columns=["S1"], dtype=float)
        rate = compute_gene_detection_rate(counts)
        assert rate["S1"] == 0.0


class TestMitoFraction:
    def test_with_mito_genes(self):
        counts = _make_counts_with_mito()
        frac = compute_mito_fraction(counts)
        assert isinstance(frac, pd.Series)
        assert (frac >= 0).all()
        assert (frac <= 1).all()

    def test_no_mito_genes(self):
        counts = _make_counts()
        frac = compute_mito_fraction(counts)
        assert (frac == 0.0).all()


class TestSampleCorrelation:
    def test_shape(self):
        counts = _make_counts()
        corr = compute_sample_correlation(counts)
        assert corr.shape == (6, 6)

    def test_diagonal_is_one(self):
        counts = _make_counts()
        corr = compute_sample_correlation(counts)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)


class TestPCA:
    def test_returns_dict_with_expected_keys(self):
        counts = _make_counts()
        result = compute_pca(counts, n_components=3)
        assert "coordinates" in result
        assert "variance_explained" in result
        assert "loadings" in result

    def test_coordinates_shape(self):
        counts = _make_counts()
        result = compute_pca(counts, n_components=3)
        assert result["coordinates"].shape == (6, 3)

    def test_variance_sums_le_one(self):
        counts = _make_counts()
        result = compute_pca(counts, n_components=3)
        assert result["variance_explained"].sum() <= 1.0 + 1e-10


class TestUMAP:
    def test_returns_dataframe(self):
        counts = _make_counts(n_genes=50, n_samples=6)
        result = compute_umap(counts)
        assert isinstance(result, pd.DataFrame)
        assert "UMAP1" in result.columns
        assert "UMAP2" in result.columns

    def test_fewer_than_3_samples_returns_empty(self):
        counts = _make_counts(n_genes=50, n_samples=2)
        result = compute_umap(counts)
        assert result.empty


class TestDetectOutliers:
    def test_no_outliers_in_tight_data(self):
        coords = pd.DataFrame(
            {"PC1": [1.0, 1.1, 2.0, 2.1], "PC2": [1.0, 1.1, 2.0, 2.1]},
            index=["S1", "S2", "S3", "S4"],
        )
        groups = pd.Series(["A", "A", "B", "B"], index=["S1", "S2", "S3", "S4"])
        outliers = detect_outliers(coords, groups)
        assert outliers == []

    def test_single_sample_group_skipped(self):
        coords = pd.DataFrame(
            {"PC1": [1.0, 100.0], "PC2": [1.0, 100.0]},
            index=["S1", "S2"],
        )
        groups = pd.Series(["A", "B"], index=["S1", "S2"])
        outliers = detect_outliers(coords, groups)
        assert outliers == []


class TestRunQC:
    def test_returns_all_keys(self):
        counts = _make_counts_with_mito()
        result = run_qc(counts)
        expected_keys = {
            "library_sizes", "detection_rates", "mito_fractions",
            "correlation_matrix", "pca", "umap", "outliers",
        }
        assert set(result.keys()) == expected_keys

    def test_outliers_skipped_without_groups(self):
        counts = _make_counts()
        result = run_qc(counts, sample_groups=None)
        assert result["outliers"] == []
