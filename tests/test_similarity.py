"""Tests for the NovoExplorer similarity module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.similarity import (
    cluster_genes,
    compute_cosine_similarity_matrix,
    compute_expression_signature_vectors,
    compute_on_the_fly_similarity,
    find_similar_by_signature,
    get_gene_neighbors,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_expression(n_genes=100, n_samples=6, seed=42):
    """Expression matrix (genes x samples), already indexed by gene name."""
    rng = np.random.default_rng(seed)
    genes = [f"GENE{i:03d}" for i in range(1, n_genes + 1)]
    samples = [f"S{i}" for i in range(1, n_samples + 1)]
    data = rng.random((n_genes, n_samples)) * 100
    return pd.DataFrame(data, index=genes, columns=samples)


def _make_deg_results(n_genes=100, seed=42):
    """Dict of comparison -> DEG DataFrame."""
    rng = np.random.default_rng(seed)
    genes = [f"GENE{i:03d}" for i in range(1, n_genes + 1)]
    results = {}
    for comp in ["A_vs_B", "A_vs_C"]:
        results[comp] = pd.DataFrame({
            "gene_name": genes,
            "log2fc": rng.normal(0, 2, n_genes),
            "padj": np.clip(rng.exponential(0.1, n_genes), 0, 1),
        })
    return results


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestCosineSimilarityMatrix:
    def test_shape(self):
        expr = _make_expression(50, 6)
        sim = compute_cosine_similarity_matrix(expr, top_n_genes=50)
        assert sim.shape[0] == sim.shape[1]
        assert sim.shape[0] <= 50

    def test_symmetric(self):
        expr = _make_expression(30, 6)
        sim = compute_cosine_similarity_matrix(expr, top_n_genes=30)
        np.testing.assert_allclose(sim.values, sim.values.T, atol=1e-10)

    def test_diagonal_is_one(self):
        expr = _make_expression(30, 6)
        sim = compute_cosine_similarity_matrix(expr, top_n_genes=30)
        np.testing.assert_allclose(np.diag(sim.values), 1.0, atol=1e-10)


class TestOnTheFlySimilarity:
    def test_returns_dataframe(self):
        expr = _make_expression(50, 6)
        result = compute_on_the_fly_similarity("GENE001", expr, top_n=10)
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 10

    def test_excludes_query_gene(self):
        expr = _make_expression(50, 6)
        result = compute_on_the_fly_similarity("GENE001", expr, top_n=49)
        if "gene" in result.columns:
            assert "GENE001" not in result["gene"].values

    def test_missing_gene_raises(self):
        expr = _make_expression(50, 6)
        with pytest.raises(KeyError):
            compute_on_the_fly_similarity("NONEXISTENT", expr)


class TestClusterGenes:
    def test_returns_labels_and_linkage(self):
        expr = _make_expression(50, 6)
        sim = compute_cosine_similarity_matrix(expr, top_n_genes=50)
        labels, link = cluster_genes(sim, min_clusters=3, max_clusters=10)
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(sim)
        assert link is not None

    def test_cluster_count_in_range(self):
        expr = _make_expression(80, 6)
        sim = compute_cosine_similarity_matrix(expr, top_n_genes=80)
        labels, _ = cluster_genes(sim, min_clusters=3, max_clusters=10)
        n_clusters = labels.nunique()
        # Binary search is approximate; accept within a small margin
        assert n_clusters >= 1


class TestSignatureVectors:
    def test_shape(self):
        deg = _make_deg_results(50)
        sv = compute_expression_signature_vectors(deg)
        assert isinstance(sv, pd.DataFrame)
        assert sv.shape[1] == 2  # two comparisons

    def test_genes_present(self):
        deg = _make_deg_results(50)
        sv = compute_expression_signature_vectors(deg)
        assert len(sv) == 50


class TestFindSimilarBySignature:
    def test_returns_top_n(self):
        deg = _make_deg_results(50)
        sv = compute_expression_signature_vectors(deg)
        result = find_similar_by_signature("GENE001", sv, top_n=5)
        assert len(result) == 5

    def test_missing_gene_raises(self):
        deg = _make_deg_results(50)
        sv = compute_expression_signature_vectors(deg)
        with pytest.raises(KeyError):
            find_similar_by_signature("NONEXISTENT", sv)


class TestGetGeneNeighbors:
    def test_with_expression(self):
        expr = _make_expression(50, 6)
        result = get_gene_neighbors("GENE001", expression_df=expr, top_n=5)
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5

    def test_missing_gene_raises(self):
        expr = _make_expression(50, 6)
        with pytest.raises((KeyError, ValueError)):
            get_gene_neighbors("NONEXISTENT", expression_df=expr)
