"""Tests for the NovoView ingest module.

Validates directory discovery, expression matrix parsing, DEG parsing,
enrichment parsing, sample info parsing, group inference, and end-to-end
ingestion against synthetic Novogene-like data created by conftest fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure the novoview package root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.ingest import (
    discover_novogene_structure,
    infer_groups_from_comparisons,
    ingest_all,
    parse_deg_results,
    parse_enrichment_results,
    parse_expression_matrices,
    parse_sample_info,
)


# -----------------------------------------------------------------------
# 1. discover_novogene_structure
# -----------------------------------------------------------------------


class TestDiscoverNovogeneStructure:
    """Tests for discover_novogene_structure."""

    def test_finds_quant_dir(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        assert structure["quant_dir"] is not None
        assert structure["quant_dir"].name == "Quant"
        assert structure["quant_dir"].is_dir()

    def test_finds_differential_dir(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        assert structure["deg_dir"] is not None
        assert structure["deg_dir"].name == "Differential"
        assert structure["deg_dir"].is_dir()

    def test_finds_enrichment_dir(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        assert structure["enrichment_dir"] is not None
        assert structure["enrichment_dir"].name == "Enrichment"
        assert structure["enrichment_dir"].is_dir()

    def test_finds_sample_info_file(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        assert structure["sample_info_file"] is not None
        assert structure["sample_info_file"].name == "sample_info.txt"
        assert structure["sample_info_file"].is_file()

    def test_discovered_files_not_empty(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        assert len(structure["discovered_files"]) > 0

    def test_all_expected_keys_present(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        expected_keys = {
            "quant_dir",
            "deg_dir",
            "enrichment_dir",
            "qc_dir",
            "mapping_dir",
            "sample_info_file",
            "discovered_files",
        }
        assert set(structure.keys()) == expected_keys

    def test_missing_directory_returns_none(self, tmp_path: Path) -> None:
        """An empty directory should yield None for all standard subdirs."""
        empty = tmp_path / "empty_delivery"
        empty.mkdir()
        structure = discover_novogene_structure(empty)
        assert structure["quant_dir"] is None
        assert structure["deg_dir"] is None
        assert structure["enrichment_dir"] is None

    def test_nonexistent_path_returns_none(self, tmp_path: Path) -> None:
        structure = discover_novogene_structure(tmp_path / "does_not_exist")
        assert structure["quant_dir"] is None
        assert structure["discovered_files"] == []


# -----------------------------------------------------------------------
# 2. parse_expression_matrices
# -----------------------------------------------------------------------


class TestParseExpressionMatrices:
    """Tests for parse_expression_matrices."""

    def test_all_matrices_parsed(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        matrices = parse_expression_matrices(structure["quant_dir"])

        assert matrices["counts"] is not None
        assert matrices["fpkm"] is not None
        assert matrices["tpm"] is not None

    def test_count_matrix_shape(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        matrices = parse_expression_matrices(structure["quant_dir"])
        counts = matrices["counts"]

        # 100 genes, at least 6 sample columns (+ possible gene_id/gene_name)
        assert counts.shape[0] == 100
        assert counts.shape[1] >= 6

    def test_count_matrix_has_gene_id_column(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        matrices = parse_expression_matrices(structure["quant_dir"])
        counts = matrices["counts"]

        # The count matrix should have a gene_id column or gene_name column
        cols_lower = [c.lower() for c in counts.columns]
        assert "gene_id" in cols_lower or "gene_name" in cols_lower

    def test_fpkm_values_are_float(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        matrices = parse_expression_matrices(structure["quant_dir"])
        fpkm = matrices["fpkm"]

        # Numeric sample columns should be floats
        sample_cols = [c for c in fpkm.columns if c.startswith("Sample")]
        for col in sample_cols:
            assert pd.api.types.is_numeric_dtype(fpkm[col])

    def test_tpm_columns_sum_approximately(self, tmp_novogene_dir: Path) -> None:
        """TPM values per sample should sum to approximately 1e6."""
        structure = discover_novogene_structure(tmp_novogene_dir)
        matrices = parse_expression_matrices(structure["quant_dir"])
        tpm = matrices["tpm"]

        sample_cols = [c for c in tpm.columns if c.startswith("Sample")]
        for col in sample_cols:
            col_sum = tpm[col].sum()
            assert 9e5 < col_sum < 1.1e6, f"TPM sum for {col} = {col_sum}"

    def test_none_quant_dir_returns_all_none(self) -> None:
        matrices = parse_expression_matrices(None)
        assert matrices["counts"] is None
        assert matrices["fpkm"] is None
        assert matrices["tpm"] is None


# -----------------------------------------------------------------------
# 3. parse_deg_results
# -----------------------------------------------------------------------


class TestParseDegResults:
    """Tests for parse_deg_results."""

    def test_both_comparisons_parsed(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        deg = parse_deg_results(structure["deg_dir"])

        assert "GroupA_vs_GroupB" in deg
        assert "GroupA_vs_GroupC" in deg

    def test_deg_dataframe_shape(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        deg = parse_deg_results(structure["deg_dir"])
        df = deg["GroupA_vs_GroupB"]

        assert len(df) == 100
        assert df.shape[1] >= 5  # gene_id, gene_name, log2fc, pvalue, padj, ...

    def test_standardized_column_names(self, tmp_novogene_dir: Path) -> None:
        """DEG columns should be standardized by standardize_deg_columns."""
        structure = discover_novogene_structure(tmp_novogene_dir)
        deg = parse_deg_results(structure["deg_dir"])
        df = deg["GroupA_vs_GroupB"]

        # After standardization the columns should use lowercase names
        expected_cols = {"gene_id", "gene_name", "log2fc", "pvalue", "padj"}
        actual_cols = set(df.columns)
        assert expected_cols.issubset(actual_cols), (
            f"Missing columns: {expected_cols - actual_cols}"
        )

    def test_regulation_column_values(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        deg = parse_deg_results(structure["deg_dir"])
        df = deg["GroupA_vs_GroupB"]

        assert "regulation" in df.columns
        assert set(df["regulation"].unique()).issubset({"Up", "Down"})

    def test_none_deg_dir_returns_empty(self) -> None:
        deg = parse_deg_results(None)
        assert deg == {}

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_diff"
        empty.mkdir()
        deg = parse_deg_results(empty)
        assert deg == {}


# -----------------------------------------------------------------------
# 4. parse_enrichment_results
# -----------------------------------------------------------------------


class TestParseEnrichmentResults:
    """Tests for parse_enrichment_results."""

    def test_enrichment_comparison_found(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        enrichment = parse_enrichment_results(structure["enrichment_dir"])

        assert "GroupA_vs_GroupB" in enrichment

    def test_go_and_kegg_parsed(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        enrichment = parse_enrichment_results(structure["enrichment_dir"])
        comp = enrichment["GroupA_vs_GroupB"]

        assert "GO" in comp
        assert "KEGG" in comp

    def test_go_enrichment_shape(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        enrichment = parse_enrichment_results(structure["enrichment_dir"])
        go_df = enrichment["GroupA_vs_GroupB"]["GO"]

        assert len(go_df) == 10  # 10 GO terms in the fixture
        assert go_df.shape[1] >= 4  # ID, Description, PValue, padj, ...

    def test_kegg_enrichment_shape(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        enrichment = parse_enrichment_results(structure["enrichment_dir"])
        kegg_df = enrichment["GroupA_vs_GroupB"]["KEGG"]

        assert len(kegg_df) == 5  # 5 KEGG terms in the fixture

    def test_enrichment_has_pvalue_column(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        enrichment = parse_enrichment_results(structure["enrichment_dir"])
        go_df = enrichment["GroupA_vs_GroupB"]["GO"]

        cols_lower = [c.lower() for c in go_df.columns]
        assert "pvalue" in cols_lower or "p-value" in cols_lower

    def test_none_enrichment_dir_returns_empty(self) -> None:
        enrichment = parse_enrichment_results(None)
        assert enrichment == {}


# -----------------------------------------------------------------------
# 5. parse_sample_info
# -----------------------------------------------------------------------


class TestParseSampleInfo:
    """Tests for parse_sample_info."""

    def test_sample_info_parsed(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        sample_info = parse_sample_info(structure["sample_info_file"])

        assert sample_info is not None
        assert isinstance(sample_info, pd.DataFrame)

    def test_sample_info_columns(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        sample_info = parse_sample_info(structure["sample_info_file"])

        assert list(sample_info.columns) == ["sample_id", "group"]

    def test_sample_info_row_count(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        sample_info = parse_sample_info(structure["sample_info_file"])

        assert len(sample_info) == 6

    def test_sample_group_mapping(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        sample_info = parse_sample_info(structure["sample_info_file"])

        mapping = dict(zip(sample_info["sample_id"], sample_info["group"]))
        assert mapping["Sample1"] == "GroupA"
        assert mapping["Sample2"] == "GroupA"
        assert mapping["Sample3"] == "GroupB"
        assert mapping["Sample4"] == "GroupB"
        assert mapping["Sample5"] == "GroupC"
        assert mapping["Sample6"] == "GroupC"

    def test_three_groups(self, tmp_novogene_dir: Path) -> None:
        structure = discover_novogene_structure(tmp_novogene_dir)
        sample_info = parse_sample_info(structure["sample_info_file"])

        groups = sorted(sample_info["group"].unique())
        assert groups == ["GroupA", "GroupB", "GroupC"]

    def test_none_path_returns_none(self) -> None:
        assert parse_sample_info(None) is None

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert parse_sample_info(tmp_path / "nonexistent.txt") is None


# -----------------------------------------------------------------------
# 6. infer_groups_from_comparisons
# -----------------------------------------------------------------------


class TestInferGroupsFromComparisons:
    """Tests for infer_groups_from_comparisons."""

    def test_infer_from_two_comparisons(self) -> None:
        deg_results = {
            "GroupA_vs_GroupB": pd.DataFrame({"log2fc": [1.0]}),
            "GroupA_vs_GroupC": pd.DataFrame({"log2fc": [-0.5]}),
        }
        result = infer_groups_from_comparisons(deg_results)

        assert sorted(result["groups"]) == ["GroupA", "GroupB", "GroupC"]
        assert len(result["comparisons"]) == 2

    def test_infer_from_single_comparison(self) -> None:
        deg_results = {
            "Treatment_vs_Control": pd.DataFrame({"log2fc": [0.3]}),
        }
        result = infer_groups_from_comparisons(deg_results)

        assert sorted(result["groups"]) == ["Control", "Treatment"]
        assert result["comparisons"] == ["Treatment_vs_Control"]

    def test_empty_deg_results(self) -> None:
        result = infer_groups_from_comparisons({})

        assert result["groups"] == []
        assert result["comparisons"] == []

    def test_duplicate_groups_deduplicated(self) -> None:
        deg_results = {
            "A_vs_B": pd.DataFrame({"x": [1]}),
            "A_vs_C": pd.DataFrame({"x": [2]}),
            "B_vs_C": pd.DataFrame({"x": [3]}),
        }
        result = infer_groups_from_comparisons(deg_results)

        assert sorted(result["groups"]) == ["A", "B", "C"]

    def test_comparisons_sorted(self) -> None:
        deg_results = {
            "Z_vs_A": pd.DataFrame(),
            "A_vs_M": pd.DataFrame(),
        }
        result = infer_groups_from_comparisons(deg_results)

        assert result["comparisons"] == ["A_vs_M", "Z_vs_A"]


# -----------------------------------------------------------------------
# 7. ingest_all (end-to-end)
# -----------------------------------------------------------------------


class TestIngestAll:
    """End-to-end test for the ingest_all entry point."""

    def test_ingest_all_returns_all_keys(self, tmp_novogene_dir: Path) -> None:
        result = ingest_all(tmp_novogene_dir)

        expected_keys = {
            "structure",
            "expression",
            "deg",
            "enrichment",
            "sample_info",
            "groups",
        }
        assert set(result.keys()) == expected_keys

    def test_ingest_all_expression_matrices(self, tmp_novogene_dir: Path) -> None:
        result = ingest_all(tmp_novogene_dir)

        assert result["expression"]["counts"] is not None
        assert result["expression"]["fpkm"] is not None
        assert result["expression"]["tpm"] is not None

    def test_ingest_all_deg_comparisons(self, tmp_novogene_dir: Path) -> None:
        result = ingest_all(tmp_novogene_dir)

        assert len(result["deg"]) == 2
        assert "GroupA_vs_GroupB" in result["deg"]
        assert "GroupA_vs_GroupC" in result["deg"]

    def test_ingest_all_enrichment(self, tmp_novogene_dir: Path) -> None:
        result = ingest_all(tmp_novogene_dir)

        assert "GroupA_vs_GroupB" in result["enrichment"]
        assert "GO" in result["enrichment"]["GroupA_vs_GroupB"]
        assert "KEGG" in result["enrichment"]["GroupA_vs_GroupB"]

    def test_ingest_all_sample_info(self, tmp_novogene_dir: Path) -> None:
        result = ingest_all(tmp_novogene_dir)

        assert result["sample_info"] is not None
        assert len(result["sample_info"]) == 6

    def test_ingest_all_groups(self, tmp_novogene_dir: Path) -> None:
        result = ingest_all(tmp_novogene_dir)

        groups = sorted(result["groups"]["groups"])
        assert groups == ["GroupA", "GroupB", "GroupC"]

    def test_ingest_all_with_config(self, tmp_novogene_dir: Path, sample_config: dict) -> None:
        result = ingest_all(tmp_novogene_dir, config=sample_config)
        assert result["expression"]["counts"] is not None

    def test_ingest_all_empty_dir(self, tmp_path: Path) -> None:
        """Ingesting an empty directory should not raise."""
        empty = tmp_path / "empty"
        empty.mkdir()
        result = ingest_all(empty)

        assert result["expression"]["counts"] is None
        assert result["deg"] == {}
        assert result["enrichment"] == {}
        assert result["sample_info"] is None
        assert result["groups"]["groups"] == []
