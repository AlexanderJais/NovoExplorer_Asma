"""Tests for the NovoExplorer signatures module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.signatures import (
    compute_signature_overlap,
    find_core_signatures,
    find_unique_signatures,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_deg_results(n_genes=50, seed=42):
    """Synthetic DEG dict with known significant genes."""
    rng = np.random.default_rng(seed)
    genes = [f"GENE{i:03d}" for i in range(1, n_genes + 1)]
    results = {}
    for comp in ["A_vs_B", "A_vs_C", "B_vs_C"]:
        log2fc = rng.normal(0, 2, n_genes)
        padj = np.clip(rng.exponential(0.1, n_genes), 0, 1)
        results[comp] = pd.DataFrame({
            "gene_name": genes,
            "log2fc": log2fc,
            "padj": padj,
        })
    return results


def _make_enrichment_results():
    """Synthetic enrichment results dict matching pipeline conventions."""
    def _gsea_df(terms, fdr_vals):
        return pd.DataFrame({
            "term": terms,
            "fdr": fdr_vals,
        })

    return {
        "A_vs_B": {
            "Hallmark": {
                "gsea": _gsea_df(
                    ["TERM1", "TERM2", "TERM3"],
                    [0.01, 0.04, 0.5],
                ),
            },
        },
        "A_vs_C": {
            "Hallmark": {
                "gsea": _gsea_df(
                    ["TERM1", "TERM4"],
                    [0.02, 0.03],
                ),
            },
        },
    }


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestSignatureOverlap:
    def test_shape(self):
        deg = _make_deg_results()
        overlap = compute_signature_overlap(deg)
        assert overlap.shape == (3, 3)

    def test_diagonal_is_one(self):
        deg = _make_deg_results()
        overlap = compute_signature_overlap(deg)
        np.testing.assert_allclose(np.diag(overlap.values), 1.0)

    def test_symmetric(self):
        deg = _make_deg_results()
        overlap = compute_signature_overlap(deg)
        np.testing.assert_allclose(overlap.values, overlap.values.T, atol=1e-10)

    def test_values_between_zero_and_one(self):
        deg = _make_deg_results()
        overlap = compute_signature_overlap(deg)
        assert (overlap.values >= 0).all()
        assert (overlap.values <= 1).all()

    def test_single_comparison(self):
        deg = {"A_vs_B": _make_deg_results()["A_vs_B"]}
        overlap = compute_signature_overlap(deg)
        assert overlap.shape == (1, 1)

    def test_no_significant_genes(self):
        """All padj > threshold should produce zero overlap."""
        df = pd.DataFrame({
            "gene_name": [f"G{i}" for i in range(10)],
            "log2fc": [0.1] * 10,
            "padj": [0.9] * 10,
        })
        overlap = compute_signature_overlap({"A": df, "B": df})
        # Diagonal still 1.0 (Jaccard of empty sets is defined as 1.0 on diagonal)
        # Off-diagonal should be 0
        assert overlap.loc["A", "B"] == 0.0


class TestFindCoreSignatures:
    def test_finds_shared_terms(self):
        enrich = _make_enrichment_results()
        core = find_core_signatures(enrich, min_comparisons=2)
        assert isinstance(core, pd.DataFrame)
        # TERM1 is significant (padj < 0.05) in both A_vs_B and A_vs_C
        assert len(core) > 0, "Expected at least one core signature"
        assert "TERM1" in core["term"].values

    def test_min_comparisons_filter(self):
        enrich = _make_enrichment_results()
        core = find_core_signatures(enrich, min_comparisons=3)
        # No term is in all 3 (only 2 comparisons exist)
        assert len(core) == 0


class TestFindUniqueSignatures:
    def test_finds_unique_terms(self):
        enrich = _make_enrichment_results()
        unique = find_unique_signatures(enrich)
        assert isinstance(unique, pd.DataFrame)
        # TERM4 only appears in A_vs_C (padj=0.03)
        assert len(unique) > 0, "Expected at least one unique signature"
        unique_terms = set(unique["term"].values)
        assert "TERM4" in unique_terms
