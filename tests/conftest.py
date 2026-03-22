"""
Pytest fixtures for NovoView pipeline tests.

Creates synthetic Novogene-like data structures and DataFrames that
mirror the folder layout and file formats delivered by Novogene.
"""

import random
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Gene / sample pools
# ---------------------------------------------------------------------------
_REAL_GENE_NAMES = [
    "BRCA1", "BRCA2", "TP53", "MYC", "KRAS", "EGFR", "PTEN", "RB1",
    "APC", "VHL", "CDH1", "SMAD4", "NOTCH1", "PIK3CA", "BRAF", "NF1",
    "NF2", "WT1", "MEN1", "RET", "KIT", "PDGFRA", "FLT3", "JAK2",
    "ALK", "ROS1", "ERBB2", "FGFR1", "FGFR2", "FGFR3", "MET", "AXL",
    "IDH1", "IDH2", "DNMT3A", "TET2", "ASXL1", "EZH2", "SUZ12", "KDM5A",
    "CREBBP", "EP300", "ARID1A", "SMARCA4", "ATM", "ATR", "CHEK1",
    "CHEK2", "MDM2", "MDM4",
]

# Pad to 100 genes with synthetic IDs
_GENE_NAMES = _REAL_GENE_NAMES + [f"GENE{i:03d}" for i in range(51, 101)]

_SAMPLE_NAMES = [f"Sample{i}" for i in range(1, 7)]

_GROUP_MAP = {
    "Sample1": "GroupA",
    "Sample2": "GroupA",
    "Sample3": "GroupB",
    "Sample4": "GroupB",
    "Sample5": "GroupC",
    "Sample6": "GroupC",
}


# ---------------------------------------------------------------------------
# Helper: write a tab-separated "xls" file (Novogene style)
# ---------------------------------------------------------------------------
def _write_tsv(path: Path, df: pd.DataFrame) -> None:
    """Write a DataFrame as a tab-separated file (with any extension)."""
    df.to_csv(path, sep="\t", index=False)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def tmp_novogene_dir(tmp_path: Path) -> Path:
    """Create a realistic Novogene delivery folder tree under *tmp_path*.

    Layout::

        <tmp_path>/
        ├── Quant/
        │   ├── gene_count_matrix.txt
        │   ├── gene_fpkm_matrix.txt
        │   └── gene_tpm_matrix.txt
        ├── Differential/
        │   ├── GroupA_vs_GroupB/
        │   │   └── GroupA_vs_GroupB.DEG.xls
        │   └── GroupA_vs_GroupC/
        │       └── GroupA_vs_GroupC.DEG.xls
        ├── Enrichment/
        │   └── GroupA_vs_GroupB/
        │       ├── GO/
        │       │   └── GO_enrichment.xls
        │       └── KEGG/
        │           └── KEGG_enrichment.xls
        └── sample_info.txt

    Returns the root directory path.
    """
    root = tmp_path / "novogene_delivery"
    root.mkdir()

    # ------------------------------------------------------------------
    # 1. Quant directory
    # ------------------------------------------------------------------
    quant_dir = root / "Quant"
    quant_dir.mkdir()

    gene_ids = [f"ENSG{i:011d}" for i in range(1, 101)]

    # Raw counts (integers)
    count_data = _RNG.integers(0, 5000, size=(100, 6))
    count_df = pd.DataFrame(count_data, columns=_SAMPLE_NAMES)
    count_df.insert(0, "gene_id", gene_ids)
    count_df.insert(1, "gene_name", _GENE_NAMES)
    _write_tsv(quant_dir / "gene_count_matrix.txt", count_df)

    # FPKM (float)
    fpkm_data = _RNG.random(size=(100, 6)) * 100
    fpkm_df = pd.DataFrame(
        np.round(fpkm_data, 4), columns=_SAMPLE_NAMES,
    )
    fpkm_df.insert(0, "gene_id", gene_ids)
    fpkm_df.insert(1, "gene_name", _GENE_NAMES)
    _write_tsv(quant_dir / "gene_fpkm_matrix.txt", fpkm_df)

    # TPM (float, columns sum to ~1e6 per sample)
    tpm_raw = _RNG.random(size=(100, 6)) * 100
    tpm_norm = tpm_raw / tpm_raw.sum(axis=0) * 1e6
    tpm_df = pd.DataFrame(
        np.round(tpm_norm, 4), columns=_SAMPLE_NAMES,
    )
    tpm_df.insert(0, "gene_id", gene_ids)
    tpm_df.insert(1, "gene_name", _GENE_NAMES)
    _write_tsv(quant_dir / "gene_tpm_matrix.txt", tpm_df)

    # ------------------------------------------------------------------
    # 2. Differential directory
    # ------------------------------------------------------------------
    diff_dir = root / "Differential"
    diff_dir.mkdir()

    for comparison in ["GroupA_vs_GroupB", "GroupA_vs_GroupC"]:
        comp_dir = diff_dir / comparison
        comp_dir.mkdir()

        n_degs = 100
        log2fc = _RNG.normal(0, 2, size=n_degs)
        pvals = np.clip(_RNG.exponential(0.05, size=n_degs), 1e-300, 1.0)
        padj = np.clip(pvals * n_degs / np.arange(1, n_degs + 1), 0, 1.0)

        regulation = np.where(log2fc > 0, "Up", "Down")

        deg_df = pd.DataFrame({
            "gene_id": gene_ids,
            "gene_name": _GENE_NAMES,
            "log2FoldChange": np.round(log2fc, 4),
            "pvalue": pvals,
            "padj": padj,
            "baseMean": np.round(_RNG.random(n_degs) * 1000, 2),
            "regulation": regulation,
        })
        _write_tsv(comp_dir / f"{comparison}.DEG.xls", deg_df)

    # ------------------------------------------------------------------
    # 3. Enrichment directory
    # ------------------------------------------------------------------
    enrich_dir = root / "Enrichment" / "GroupA_vs_GroupB"
    enrich_dir.mkdir(parents=True)

    # GO enrichment
    go_dir = enrich_dir / "GO"
    go_dir.mkdir()
    go_terms = [
        ("GO:0006915", "apoptotic process"),
        ("GO:0007049", "cell cycle"),
        ("GO:0006281", "DNA repair"),
        ("GO:0006955", "immune response"),
        ("GO:0007165", "signal transduction"),
        ("GO:0008283", "cell proliferation"),
        ("GO:0006351", "transcription, DNA-templated"),
        ("GO:0006412", "translation"),
        ("GO:0006810", "transport"),
        ("GO:0007155", "cell adhesion"),
    ]
    go_df = pd.DataFrame({
        "ID": [t[0] for t in go_terms],
        "Description": [t[1] for t in go_terms],
        "GeneRatio": [f"{_RNG.integers(5, 30)}/200" for _ in go_terms],
        "PValue": np.clip(_RNG.exponential(0.02, size=len(go_terms)), 1e-20, 1.0),
        "padj": np.clip(_RNG.exponential(0.05, size=len(go_terms)), 1e-20, 1.0),
        "Count": _RNG.integers(5, 30, size=len(go_terms)),
        "geneID": [
            "/".join(_RNG.choice(_GENE_NAMES, size=5, replace=False))
            for _ in go_terms
        ],
    })
    _write_tsv(go_dir / "GO_enrichment.xls", go_df)

    # KEGG enrichment
    kegg_dir = enrich_dir / "KEGG"
    kegg_dir.mkdir()
    kegg_terms = [
        ("hsa04110", "Cell cycle"),
        ("hsa04115", "p53 signaling pathway"),
        ("hsa04151", "PI3K-Akt signaling pathway"),
        ("hsa04310", "Wnt signaling pathway"),
        ("hsa04350", "TGF-beta signaling pathway"),
    ]
    kegg_df = pd.DataFrame({
        "ID": [t[0] for t in kegg_terms],
        "Description": [t[1] for t in kegg_terms],
        "GeneRatio": [f"{_RNG.integers(3, 20)}/150" for _ in kegg_terms],
        "PValue": np.clip(
            _RNG.exponential(0.03, size=len(kegg_terms)), 1e-20, 1.0,
        ),
        "padj": np.clip(
            _RNG.exponential(0.06, size=len(kegg_terms)), 1e-20, 1.0,
        ),
        "Count": _RNG.integers(3, 20, size=len(kegg_terms)),
        "geneID": [
            "/".join(_RNG.choice(_GENE_NAMES, size=4, replace=False))
            for _ in kegg_terms
        ],
    })
    _write_tsv(kegg_dir / "KEGG_enrichment.xls", kegg_df)

    # ------------------------------------------------------------------
    # 4. sample_info.txt
    # ------------------------------------------------------------------
    info_df = pd.DataFrame({
        "sample": list(_GROUP_MAP.keys()),
        "group": list(_GROUP_MAP.values()),
    })
    _write_tsv(root / "sample_info.txt", info_df)

    return root


# ---------------------------------------------------------------------------
# Smaller, in-memory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_count_matrix() -> pd.DataFrame:
    """A small raw-count DataFrame: 20 genes x 6 samples.

    Includes ``gene_id`` and ``gene_name`` annotation columns plus six
    integer count columns (Sample1-Sample6).
    """
    genes = _GENE_NAMES[:20]
    gene_ids = [f"ENSG{i:011d}" for i in range(1, 21)]
    counts = _RNG.integers(0, 8000, size=(20, 6))

    df = pd.DataFrame(counts, columns=_SAMPLE_NAMES)
    df.insert(0, "gene_id", gene_ids)
    df.insert(1, "gene_name", genes)
    return df


@pytest.fixture()
def sample_deg_table() -> pd.DataFrame:
    """A standardised DEG DataFrame (30 genes)."""
    n = 30
    genes = _GENE_NAMES[:n]
    gene_ids = [f"ENSG{i:011d}" for i in range(1, n + 1)]
    log2fc = _RNG.normal(0, 2, size=n)

    return pd.DataFrame({
        "gene_id": gene_ids,
        "gene_name": genes,
        "log2fc": np.round(log2fc, 4),
        "pvalue": np.clip(_RNG.exponential(0.05, size=n), 1e-300, 1.0),
        "padj": np.clip(_RNG.exponential(0.1, size=n), 0, 1.0),
        "basemean": np.round(_RNG.random(n) * 1000, 2),
        "regulation": np.where(log2fc > 0, "Up", "Down"),
    })


@pytest.fixture()
def sample_config() -> dict:
    """A minimal configuration dictionary for pipeline tests."""
    return {
        "project_name": "Test RNA-Seq Experiment",
        "data_dir": "/tmp/fake_data",
        "output_dir": "/tmp/fake_results",
        "organism": "human",
        "padj_threshold": 0.05,
        "log2fc_threshold": 1.0,
        "rerun_de": False,
        "comparisons": "auto",
        "similarity_top_n": 50,
        "similarity_variable_genes": 5000,
        "signature_min_comparisons": 2,
        "gene_set_databases": [
            "MSigDB_Hallmark_2020",
            "GO_Biological_Process_2023",
            "KEGG_2021_Human",
        ],
        "app_port": 8501,
        "app_theme": "light",
        "threads": 4,
        "enrichment_databases": ["GO_BP", "GO_MF", "GO_CC", "KEGG"],
    }
