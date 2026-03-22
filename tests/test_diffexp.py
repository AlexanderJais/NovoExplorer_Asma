"""Tests for the NovoView diffexp module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.diffexp import (
    classify_regulation,
    get_significant_genes,
    parse_novogene_deg,
    summarize_deg_results,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_deg(n=30, seed=99):
    """Create a synthetic DEG table with standard column names."""
    rng = np.random.default_rng(seed)
    log2fc = rng.normal(0, 2, size=n)
    return pd.DataFrame({
        "gene_id": [f"ENSG{i:011d}" for i in range(1, n + 1)],
        "gene_name": [f"GENE{i}" for i in range(1, n + 1)],
        "log2fc": np.round(log2fc, 4),
        "pvalue": np.clip(rng.exponential(0.05, size=n), 1e-300, 1.0),
        "padj": np.clip(rng.exponential(0.1, size=n), 0, 1.0),
        "basemean": rng.random(n) * 1000,
    })


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestClassifyRegulation:
    def test_upregulated(self):
        assert classify_regulation(2.0, 0.01) == "up"

    def test_downregulated(self):
        assert classify_regulation(-2.0, 0.01) == "down"

    def test_not_significant_padj(self):
        assert classify_regulation(2.0, 0.1) == "ns"

    def test_not_significant_log2fc(self):
        assert classify_regulation(0.5, 0.01) == "ns"

    def test_nan_padj(self):
        assert classify_regulation(2.0, float("nan")) == "ns"

    def test_nan_log2fc(self):
        assert classify_regulation(float("nan"), 0.01) == "ns"

    def test_custom_thresholds(self):
        assert classify_regulation(0.5, 0.01, log2fc_threshold=0.3) == "up"
        assert classify_regulation(-0.5, 0.01, log2fc_threshold=0.3) == "down"


class TestParseNovoGeneDeg:
    def test_returns_cleaned_dict(self):
        deg = {"CompA": _make_deg()}
        result = parse_novogene_deg(deg)
        assert "CompA" in result
        assert "padj" in result["CompA"].columns

    def test_adds_regulation_column(self):
        deg = {"CompA": _make_deg()}
        result = parse_novogene_deg(deg)
        assert "regulation" in result["CompA"].columns
        vals = set(result["CompA"]["regulation"].unique())
        assert vals.issubset({"up", "down", "ns"})

    def test_coerces_string_numeric_columns(self):
        df = _make_deg()
        df["log2fc"] = df["log2fc"].astype(str)
        result = parse_novogene_deg({"CompA": df})
        assert result["CompA"]["log2fc"].dtype == np.float64

    def test_drops_na_padj(self):
        df = _make_deg()
        df.loc[0, "padj"] = np.nan
        result = parse_novogene_deg({"CompA": df})
        assert result["CompA"]["padj"].isna().sum() == 0

    def test_empty_df_excluded(self):
        df = pd.DataFrame(columns=["gene_name", "log2fc", "padj"])
        result = parse_novogene_deg({"CompA": df})
        assert "CompA" not in result


class TestGetSignificantGenes:
    def test_filters_by_thresholds(self):
        df = _make_deg(100, seed=42)
        sig = get_significant_genes(df)
        assert len(sig) <= len(df)
        if len(sig) > 0:
            assert (sig["padj"] < 0.05).all()
            assert (sig["log2fc"].abs() > 1.0).all()

    def test_returns_all_when_columns_missing(self):
        df = pd.DataFrame({"gene_name": ["A", "B"], "value": [1, 2]})
        result = get_significant_genes(df)
        assert len(result) == 2


class TestSummarizeDegResults:
    def test_summary_shape(self):
        deg = {
            "CompA": _make_deg(50, seed=1),
            "CompB": _make_deg(50, seed=2),
        }
        summary = summarize_deg_results(deg)
        assert len(summary) == 2
        assert "comparison" in summary.columns
        assert "total_deg" in summary.columns

    def test_up_down_sum_equals_total(self):
        deg = {"CompA": _make_deg(50, seed=1)}
        summary = summarize_deg_results(deg)
        row = summary.iloc[0]
        assert row["up"] + row["down"] == row["total_deg"]
