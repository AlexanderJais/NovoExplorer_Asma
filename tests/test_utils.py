"""Tests for the NovoExplorer utils module."""

from __future__ import annotations

import logging
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.utils import (
    find_column,
    load_config,
    read_table_flexible,
    setup_logger,
    standardize_deg_columns,
    standardize_enrichment_columns,
)


# -----------------------------------------------------------------------
# setup_logger
# -----------------------------------------------------------------------

class TestSetupLogger:
    def test_returns_logger(self):
        lg = setup_logger("test_utils_logger")
        assert isinstance(lg, logging.Logger)
        assert lg.name == "test_utils_logger"

    def test_no_duplicate_handlers(self):
        lg = setup_logger("test_dup")
        n = len(lg.handlers)
        setup_logger("test_dup")
        assert len(lg.handlers) == n


# -----------------------------------------------------------------------
# find_column
# -----------------------------------------------------------------------

class TestFindColumn:
    def test_finds_exact_match(self):
        df = pd.DataFrame({"log2FC": [1], "pvalue": [0.01]})
        assert find_column(df, ["log2FC"]) == "log2FC"

    def test_case_insensitive(self):
        df = pd.DataFrame({"Log2FoldChange": [1]})
        assert find_column(df, ["log2foldchange"]) == "Log2FoldChange"

    def test_returns_none_when_missing(self):
        df = pd.DataFrame({"a": [1]})
        assert find_column(df, ["b", "c"]) is None

    def test_required_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Required column"):
            find_column(df, ["b"], required=True)

    def test_first_candidate_wins(self):
        df = pd.DataFrame({"alpha": [1], "beta": [2]})
        assert find_column(df, ["alpha", "beta"]) == "alpha"


# -----------------------------------------------------------------------
# standardize_deg_columns
# -----------------------------------------------------------------------

class TestStandardizeDegColumns:
    def test_renames_known_columns(self):
        df = pd.DataFrame({
            "log2FoldChange": [1.5],
            "PValue": [0.01],
            "FDR": [0.05],
        })
        result = standardize_deg_columns(df)
        assert "log2fc" in result.columns
        assert "pvalue" in result.columns
        assert "padj" in result.columns

    def test_preserves_unknown_columns(self):
        df = pd.DataFrame({"custom_col": [1], "pvalue": [0.01]})
        result = standardize_deg_columns(df)
        assert "custom_col" in result.columns

    def test_returns_copy(self):
        df = pd.DataFrame({"PValue": [0.01]})
        result = standardize_deg_columns(df)
        assert result is not df


# -----------------------------------------------------------------------
# standardize_enrichment_columns
# -----------------------------------------------------------------------

class TestStandardizeEnrichmentColumns:
    def test_renames_enrichment_columns(self):
        df = pd.DataFrame({
            "Description": ["term1"],
            "GeneRatio": ["5/100"],
            "Count": [5],
            "geneID": ["BRCA1/TP53"],
        })
        result = standardize_enrichment_columns(df)
        assert "term_name" in result.columns
        assert "gene_ratio" in result.columns
        assert "gene_count" in result.columns
        assert "genes" in result.columns

    def test_genes_column_not_stolen_by_gene_count(self):
        """Regression: 'Genes' should map to 'genes', not 'gene_count'."""
        df = pd.DataFrame({
            "Genes": ["BRCA1/TP53"],
            "padj": [0.01],
        })
        result = standardize_enrichment_columns(df)
        assert "genes" in result.columns
        # "Genes" should NOT have been claimed by gene_count
        assert result["genes"].iloc[0] == "BRCA1/TP53"

    def test_novogene_kegg_columns(self):
        """Novogene KEGG uses KEGGID (no underscore) and geneName."""
        df = pd.DataFrame({
            "KEGGID": ["hsa04110"],
            "Description": ["Cell cycle"],
            "GeneRatio": ["5/150"],
            "BgRatio": ["50/20000"],
            "pvalue": [0.001],
            "padj": [0.01],
            "geneID": ["BRCA1/TP53/MYC"],
            "geneName": ["BRCA1/TP53/MYC"],
            "Count": [3],
        })
        result = standardize_enrichment_columns(df)
        assert "term_id" in result.columns
        assert result["term_id"].iloc[0] == "hsa04110"
        assert "genes" in result.columns
        # geneName should be preferred over geneID
        assert result["genes"].iloc[0] == "BRCA1/TP53/MYC"

    def test_geneName_preferred_over_geneID(self):
        """When both geneID (ENSG) and geneName exist, prefer geneName."""
        df = pd.DataFrame({
            "KEGGID": ["hsa04110"],
            "Description": ["Cell cycle"],
            "pvalue": [0.001],
            "padj": [0.01],
            "geneID": ["ENSG00000012048/ENSG00000141510"],
            "geneName": ["BRCA1/TP53"],
        })
        result = standardize_enrichment_columns(df)
        assert result["genes"].iloc[0] == "BRCA1/TP53"


# -----------------------------------------------------------------------
# read_table_flexible
# -----------------------------------------------------------------------

class TestReadTableFlexible:
    def test_reads_tsv(self, tmp_path):
        p = tmp_path / "data.tsv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(p, sep="\t", index=False)
        df = read_table_flexible(p)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_raises_on_unreadable(self, tmp_path):
        p = tmp_path / "garbage.xyz"
        p.write_text("not a table at all")
        with pytest.raises(ValueError, match="Unable to read"):
            read_table_flexible(p)

    def test_skips_comment_lines(self, tmp_path):
        p = tmp_path / "commented.tsv"
        p.write_text("# header comment\na\tb\n1\t2\n")
        df = read_table_flexible(p)
        assert len(df) == 1


# -----------------------------------------------------------------------
# load_config
# -----------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("organism: mouse\nthreads: 8\n")
        cfg = load_config(p)
        assert cfg["organism"] == "mouse"
        assert cfg["threads"] == 8

    def test_defaults_applied(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("organism: mouse\n")
        cfg = load_config(p)
        assert "padj_threshold" in cfg
        assert cfg["padj_threshold"] == 0.05

    def test_empty_yaml_returns_defaults(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        cfg = load_config(p)
        assert cfg["organism"] == "mouse"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_malformed_yaml_returns_defaults(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(":\n  - :\n    invalid: [")
        cfg = load_config(p)
        assert cfg["organism"] == "mouse"
