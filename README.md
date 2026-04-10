# NovoExplorer

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Interactive Streamlit application for browsing and analyzing Novogene bulk RNA-Seq deliveries.

Point it at your Novogene results folder and instantly get volcano plots, MA plots, enrichment dot plots, pathway views, gene-level exploration, and more -- no coding required.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Supported Data](#supported-data)
- [Advanced: Analysis Pipeline](#advanced-analysis-pipeline--multi-page-app)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

```bash
# 1. Run the setup script (creates venv, installs deps, downloads gene sets)
bash setup.sh

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Launch the app -- pass your Novogene delivery folder directly
streamlit run novogene_explorer.py -- /path/to/your/novogene/results
```

Or launch without arguments to pick the folder in the browser:

```bash
streamlit run novogene_explorer.py
```

No configuration files, no pipeline commands. Just point and explore.

---

## Features

NovoExplorer provides **11 interactive tabs** for exploring your RNA-Seq data:

| Tab | Description |
|-----|-------------|
| **Overview** | DEG summary statistics, bar charts of up/down-regulated gene counts per comparison |
| **Gene Explorer** | Search any gene, see its log2FC across all comparisons, multi-gene heatmap |
| **Comparison Browser** | Volcano plot with interactive gene labels, filterable DEG table |
| **Enrichment** | Dot plots, bar charts, cross-comparison enrichment search |
| **MA Plot** | baseMean vs log2FC -- reveals expression-dependent fold changes |
| **Venn / UpSet** | Overlap of significant DEGs across comparisons |
| **Ranked Genes** | Waterfall plot of all genes sorted by fold change or significance |
| **DEG Summary** | Wide table: log2FC + padj for every gene across all comparisons |
| **Pathway Viewer** | Select a GO/KEGG/Reactome term, see member genes colored by log2FC |
| **PPI Network** | Protein-protein interaction hub analysis, score filtering, gene neighborhood |
| **Export** | Download Excel workbook or ZIP of CSVs with optional significance filtering |

### Key Highlights

- **Zero configuration** -- auto-detects Novogene delivery folder structures
- **Colorblind-friendly** -- uses the Wong et al. palette throughout
- **Publication-ready** -- export any figure as PNG or SVG
- **Fast** -- Streamlit caching makes repeated exploration instant
- **Organism support** -- works with both human and mouse data

---

## Supported Data

NovoExplorer auto-detects the standard Novogene delivery folder structure:

```
your_results/
  Differential/
    1.deglist/
      GroupA_vs_GroupB/
        GroupA_vs_GroupB_deg.xls
    2.cluster/
      ...
  Enrichment/
    GO/
      GroupA_vs_GroupB/ALL/*.xls
    KEGG/
      GroupA_vs_GroupB/ALL/*.xls
    DisGeNET/
      GroupA_vs_GroupB/ALL/*.xls
    DO/
      GroupA_vs_GroupB/ALL/*.xls
    Reactome/
      GroupA_vs_GroupB/ALL/*.xls
    PPI/
      GroupA_vs_GroupB/ALL/*.xls
```

Both `database_first` (e.g. `Enrichment/GO/CompA_vs_CompB/`) and `comparison_first` (e.g. `Enrichment/CompA_vs_CompB/GO/`) layouts are detected automatically.

Numbered-prefix directories (e.g. `3.Quant/`, `4.Differential/`) are also recognized.

### Supported Enrichment Databases

| Database | Description |
|----------|-------------|
| GO | Gene Ontology (Biological Process, Molecular Function, Cellular Component) |
| KEGG | Kyoto Encyclopedia of Genes and Genomes pathways |
| DisGeNET | Disease-gene associations |
| DO | Disease Ontology |
| Reactome | Reactome pathway database |
| PPI | Protein-protein interaction networks |

---

## Advanced: Analysis Pipeline + Multi-Page App

For deeper analysis (normalization, QC, gene similarity, GSEA, signature overlap), NovoExplorer includes a computational pipeline and a multi-page app that builds on the pipeline's results.

### Running the Pipeline

```bash
# Option A: launch the multi-page app and run the pipeline from the browser
streamlit run app/app.py

# Option B: run from the command line with a config file
python run_pipeline.py --config config.yaml
streamlit run app/app.py -- --config config.yaml
```

The multi-page app (`app/app.py`) can detect a raw Novogene delivery folder, let you configure settings in the browser, and run the pipeline without touching the command line.

### Pipeline Stages

| Stage | Module | What it does |
|-------|--------|-------------|
| **1. Ingest** | `pipeline/ingest.py` | Walks the Novogene delivery folder, discovers quantification matrices, DEG tables, and enrichment results. |
| **2. Normalize** | `pipeline/normalize.py` | Filters low-count genes, computes TPM, produces a log2(TPM+1) expression matrix. |
| **3. QC** | `pipeline/qc.py` | Library sizes, detection rates, mitochondrial fractions, sample correlation, PCA, UMAP. |
| **4. Differential Expression** | `pipeline/diffexp.py` | Cleans Novogene DEG results. Optionally re-runs DE via pyDESeq2. |
| **5. Similarity** | `pipeline/similarity.py` | Gene-gene cosine similarity, hierarchical clustering, expression signature vectors. |
| **6. Signatures** | `pipeline/signatures.py` | GSEA, ORA via gseapy, Jaccard overlap, core/unique pathway signatures. |
| **7. Save** | `pipeline/persistence.py` | Persists all results to `results/novoexplorer_results.h5`. |

### Multi-Page App

The pipeline-backed app provides additional analytical features not available in the main explorer:

| Page | What it adds beyond the main explorer |
|------|---------------------------------------|
| **Overview** | PCA scatter, UMAP, sample correlation heatmap, library size / detection rate QC |
| **Differential Expression** | Per-gene expression bar charts, gene basket for collecting genes across pages |
| **Gene Search** | Similar-gene discovery via cosine similarity across expression profiles |
| **Signatures & Pathways** | GSEA dot plots, Jaccard overlap heatmap, core/unique signature identification |
| **Multi-Condition** | Fold-change concordance scatter between comparison pairs |

---

## Configuration

`config.yaml` is only needed for the pipeline (not for the main explorer). All keys are optional -- defaults are applied for anything omitted.

### Core Settings

| Key | Default | Description |
|-----|---------|-------------|
| `project_name` | `""` | Display name shown in the app sidebar. |
| `data_dir` | `"."` | Path to the Novogene delivery folder. |
| `output_dir` | `"results"` | Directory where the HDF5 results file is written. |
| `organism` | `"human"` | `"human"` or `"mouse"`. Controls gene name mappings and gene set organisms. |

### Differential Expression

| Key | Default | Description |
|-----|---------|-------------|
| `padj_threshold` | `0.05` | Adjusted p-value cutoff for significance. |
| `log2fc_threshold` | `1.0` | Absolute log2 fold-change cutoff. |
| `rerun_de` | `false` | If `true`, re-run DE from raw counts via pyDESeq2, even when Novogene DEG results exist. |
| `comparisons` | `"auto"` | Set to `"auto"` to discover comparisons from folder names, or provide an explicit list (e.g. `[["Treatment", "Control"]]`). |

### Gene Similarity

| Key | Default | Description |
|-----|---------|-------------|
| `similarity_variable_genes` | `5000` | Number of top-variable genes for the cosine similarity matrix. |

### Signature Analysis

| Key | Default | Description |
|-----|---------|-------------|
| `signature_min_comparisons` | `2` | Minimum comparisons a pathway must be enriched in to be called a "core" signature. |
| `gene_set_databases` | `["MSigDB_Hallmark_2020", ...]` | Gene-set libraries for enrichment analysis. See the [Enrichr libraries list](https://maayanlab.cloud/Enrichr/#libraries). |

### Advanced

| Key | Default | Description |
|-----|---------|-------------|
| `log_level` | `"INFO"` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

### Example `config.yaml`

```yaml
project_name: "My RNA-Seq Experiment"
data_dir: "./data"
output_dir: "./results"
organism: "mouse"

padj_threshold: 0.05
log2fc_threshold: 1.0
rerun_de: false
comparisons: "auto"

similarity_variable_genes: 5000
signature_min_comparisons: 2
gene_set_databases:
  - "MSigDB_Hallmark_2020"
  - "GO_Biological_Process_2023"
  - "KEGG_2021_Mouse"
```

---

## Project Structure

```
NovoExplorer/
  novogene_explorer.py    # Main Streamlit app (11 tabs, direct Novogene folder browsing)
  run_pipeline.py         # CLI entry point for the analysis pipeline
  config.yaml             # Pipeline configuration (not needed for main app)
  setup.sh                # One-step setup script (venv + deps + gene sets)
  requirements.txt        # Python dependencies with platform-specific pins
  app/                    # Multi-page Streamlit app (pipeline-backed, additional analyses)
    app.py                #   Entry point, data picker, in-app pipeline runner
    style.css             #   Custom theme
    pages/                #   Individual pages (overview, diffexp, gene search, ...)
    components/           #   Shared UI widgets (filters, gene basket, downloads)
  pipeline/               # Analysis pipeline modules
    ingest.py             #   Novogene folder discovery & file parsing
    normalize.py          #   Count filtering, TPM computation, log2 transform
    qc.py                 #   PCA, UMAP, correlation, outlier detection
    diffexp.py            #   DEG cleaning, pyDESeq2 re-analysis, merging
    similarity.py         #   Cosine similarity, hierarchical clustering, signatures
    signatures.py         #   GSEA, ORA, Jaccard overlap, core/unique pathways
    persistence.py        #   HDF5 save/load for all pipeline results
    utils.py              #   Column standardization, flexible file reading, config
  plotting/               # Figure builders (Plotly + Matplotlib, Nature-style theme)
    volcano.py            #   Volcano plots (matplotlib + plotly)
    ma_plot.py            #   MA plots (baseMean vs log2FC)
    heatmap.py            #   Clustered heatmaps (seaborn clustermap)
    enrichment.py         #   Enrichment dot plots and bar charts
    pca.py                #   PCA scatter plots
    similarity_viz.py     #   Similarity heatmaps and network graphs
    upset.py              #   UpSet plots for set intersections
    ppi_network.py        #   PPI network visualization
    theme.py              #   Shared color palettes and styling
  tests/                  # Pytest test suite
    conftest.py           #   Shared fixtures
    test_ingest.py        #   Ingest parsing tests
    test_normalize.py     #   Normalization and filtering tests
    test_diffexp.py       #   Differential expression tests
    test_qc.py            #   QC metric tests
    test_similarity.py    #   Similarity and clustering tests
    test_signatures.py    #   GSEA, ORA, overlap tests
    test_persistence.py   #   HDF5 roundtrip tests
    test_utils.py         #   Utility function tests
```

---

## Requirements

- **Python 3.10+** (tested on 3.10, 3.11, 3.12)
- See `requirements.txt` for the full dependency list.

Key dependencies: Streamlit, pandas, NumPy, Plotly, Matplotlib, pyDESeq2, gseapy, scikit-learn, UMAP, h5py, networkx.

### Platform Notes

`setup.sh` handles the following automatically:

- **macOS (Intel x86_64):** gseapy has no prebuilt wheel for this platform. The setup script installs a minimal Rust toolchain via [rustup](https://rustup.rs) so pip can build gseapy from source. numba/llvmlite are pinned to versions that still ship Intel Mac wheels.
- **macOS (Apple Silicon):** All dependencies install from prebuilt wheels. No extra toolchains required.
- **Linux:** All dependencies install from prebuilt wheels.

If you prefer to install manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --prefer-binary -r requirements.txt
```

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/
```

To run a specific module:

```bash
pytest tests/test_diffexp.py -v
```

To run with coverage:

```bash
pytest tests/ --cov=pipeline --cov=plotting --cov-report=term-missing
```

---

## Troubleshooting

### `setup.sh` fails with "can't find Rust compiler"

gseapy requires a Rust toolchain when building from source (no prebuilt wheel available for your platform). The setup script tries to install Rust automatically on macOS. If it fails:

```bash
# Install Rust manually
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Re-run setup
bash setup.sh
```

### `setup.sh` fails with "No such file or directory: 'cmake'"

This happens when llvmlite is built from source. The pinned versions in `requirements.txt` should provide prebuilt wheels. If you hit this anyway, install CMake:

```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt-get install cmake

# Then re-run setup
bash setup.sh
```

### Streamlit shows "No module named 'pipeline'"

Make sure you launch the app from the NovoExplorer root directory:

```bash
cd /path/to/NovoExplorer
source .venv/bin/activate
streamlit run novogene_explorer.py
```

### Plots don't export to PNG/SVG

The kaleido package is required for static image export from Plotly. It is included in `requirements.txt`, but if it's missing:

```bash
pip install kaleido
```

### The app is slow on first load

This is expected -- Streamlit caches data after the first load. Subsequent interactions with the same dataset will be significantly faster. If you're working with very large datasets, consider running the pipeline first and using the multi-page app, which loads pre-computed results from HDF5.

---

## FAQ

**Q: Do I need to run the pipeline before using the app?**
A: No. The main app (`novogene_explorer.py`) works directly with raw Novogene delivery folders. The pipeline is only needed for the advanced multi-page app features (PCA, UMAP, gene similarity, GSEA).

**Q: What organisms are supported?**
A: Human and mouse. Set the `organism` key in `config.yaml` to `"human"` or `"mouse"`.

**Q: Can I use data from a provider other than Novogene?**
A: The app expects the Novogene folder structure. If your data follows a different layout, you would need to reorganize it to match, or use the pipeline directly with a count matrix.

**Q: How do I add custom gene sets for enrichment analysis?**
A: Add the gene-set library name to the `gene_set_databases` list in `config.yaml`. Any library available in [Enrichr](https://maayanlab.cloud/Enrichr/#libraries) can be used.

**Q: Can I change the significance thresholds?**
A: Yes. In the main app, thresholds are adjustable via sidebar sliders. For the pipeline, set `padj_threshold` and `log2fc_threshold` in `config.yaml`.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines, code style, and how to submit changes.

---

## License

This project is available under the MIT License. See [LICENSE](LICENSE) for details.
