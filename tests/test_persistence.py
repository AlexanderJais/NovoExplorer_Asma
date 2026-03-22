"""Tests for the NovoView persistence module.

Requires the ``tables`` (PyTables) package for HDF5 read/write via
pandas HDFStore.  Tests are skipped automatically when the package is
not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import tables  # noqa: F401
    _HAS_PYTABLES = True
except ImportError:
    _HAS_PYTABLES = False

pytestmark = pytest.mark.skipif(
    not _HAS_PYTABLES,
    reason="pytables not installed",
)

from pipeline.persistence import (
    get_project_metadata,
    list_comparisons,
    load_results,
    save_results,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_results():
    """Build a minimal but representative results dict."""
    rng = np.random.default_rng(42)
    n_genes, n_samples = 20, 4
    genes = [f"GENE{i}" for i in range(1, n_genes + 1)]
    samples = [f"S{i}" for i in range(1, n_samples + 1)]

    counts = pd.DataFrame(
        rng.integers(0, 5000, (n_genes, n_samples)),
        index=genes, columns=samples,
    )
    tpm = counts.div(counts.sum(axis=0), axis=1) * 1e6

    deg_df = pd.DataFrame({
        "gene_name": genes[:10],
        "log2fc": rng.normal(0, 2, 10),
        "padj": np.clip(rng.exponential(0.1, 10), 0, 1),
    })

    pca_coords = pd.DataFrame(
        rng.random((n_samples, 2)),
        index=samples, columns=["PC1", "PC2"],
    )
    pca_var = np.array([0.6, 0.2])

    return {
        "expression": {"counts": counts, "tpm": tpm, "fpkm": None},
        "deg": {"A_vs_B": deg_df},
        "enrichment": {},
        "similarity": {
            "cosine_matrix": pd.DataFrame(
                np.eye(5), index=genes[:5], columns=genes[:5],
            ),
            "gene_clusters": pd.Series([0, 0, 1, 1, 2], index=genes[:5], name="cluster"),
            "signature_vectors": None,
        },
        "qc": {
            "library_sizes": pd.Series(
                counts.sum(axis=0).values, index=samples, name="library_size",
            ),
            "detection_rates": None,
            "mito_fractions": None,
            "correlation": None,
        },
        "embeddings": {
            "pca_coordinates": pca_coords,
            "pca_variance": pca_var,
            "umap": None,
        },
        "signatures": {"overlap_matrix": None, "core": None, "unique": None},
        "metadata": {
            "samples": pd.DataFrame({
                "sample_id": samples,
                "group": ["A", "A", "B", "B"],
            }),
            "genes": pd.DataFrame({"gene_id": genes}),
            "comparisons": pd.DataFrame({"comparison": ["A_vs_B"]}),
            "project": {"project_name": "Test", "organism": "human", "data_dir": "/tmp"},
        },
    }


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestSaveLoadRoundtrip:
    def test_basic_roundtrip(self, tmp_path):
        h5_path = tmp_path / "results.h5"
        results = _make_results()
        save_results(results, h5_path)
        loaded = load_results(h5_path)

        # Expression
        assert loaded["expression"]["counts"] is not None
        pd.testing.assert_frame_equal(
            loaded["expression"]["counts"],
            results["expression"]["counts"],
        )
        assert loaded["expression"]["tpm"] is not None

        # DEG
        assert "A_vs_B" in loaded["deg"]
        assert len(loaded["deg"]["A_vs_B"]) == 10
        pd.testing.assert_frame_equal(
            loaded["deg"]["A_vs_B"],
            results["deg"]["A_vs_B"],
        )

    def test_metadata_roundtrip(self, tmp_path):
        h5_path = tmp_path / "results.h5"
        results = _make_results()
        save_results(results, h5_path)
        loaded = load_results(h5_path)

        assert loaded["metadata"]["samples"] is not None
        assert len(loaded["metadata"]["samples"]) == 4

    def test_similarity_roundtrip(self, tmp_path):
        h5_path = tmp_path / "results.h5"
        results = _make_results()
        save_results(results, h5_path)
        loaded = load_results(h5_path)

        assert loaded["similarity"] is not None
        assert loaded["similarity"]["cosine_matrix"] is not None
        assert loaded["similarity"]["cosine_matrix"].shape == (5, 5)

    def test_qc_series_roundtrip(self, tmp_path):
        """Regression: Series should survive save/load via DataFrame conversion."""
        h5_path = tmp_path / "results.h5"
        results = _make_results()
        save_results(results, h5_path)
        loaded = load_results(h5_path)

        assert loaded["qc"] is not None
        lib_sizes = loaded["qc"]["library_sizes"]
        assert lib_sizes is not None
        assert len(lib_sizes) == 4

    def test_embeddings_roundtrip(self, tmp_path):
        h5_path = tmp_path / "results.h5"
        results = _make_results()
        save_results(results, h5_path)
        loaded = load_results(h5_path)

        assert loaded["embeddings"] is not None
        assert loaded["embeddings"]["pca_coordinates"] is not None
        assert loaded["embeddings"]["pca_coordinates"].shape == (4, 2)


class TestListComparisons:
    def test_lists_saved_comparisons(self, tmp_path):
        h5_path = tmp_path / "results.h5"
        save_results(_make_results(), h5_path)
        comps = list_comparisons(h5_path)
        assert "A_vs_B" in comps

    def test_missing_file_returns_empty(self, tmp_path):
        comps = list_comparisons(tmp_path / "nonexistent.h5")
        assert comps == []


class TestGetProjectMetadata:
    def test_retrieves_metadata(self, tmp_path):
        h5_path = tmp_path / "results.h5"
        save_results(_make_results(), h5_path)
        meta = get_project_metadata(h5_path)
        assert meta.get("project_name") == "Test"
        assert meta.get("organism") == "human"

    def test_missing_file_returns_empty(self, tmp_path):
        meta = get_project_metadata(tmp_path / "nonexistent.h5")
        assert meta == {}


class TestLoadMissingFile:
    def test_load_results_missing(self, tmp_path):
        loaded = load_results(tmp_path / "nonexistent.h5")
        # Should return dict with None sections, not crash
        assert isinstance(loaded, dict)
