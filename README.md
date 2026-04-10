# NovoExplorer

Interactive Streamlit application for browsing and analyzing Novogene bulk RNA-Seq deliveries.

Point it at your Novogene results folder and instantly get volcano plots, MA plots, enrichment dot plots, pathway views, gene-level exploration, and more -- no coding required.

## Quick Start

```bash
# 1. Run the setup script (creates venv, installs deps, downloads gene sets)
bash setup.sh

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Launch the app
streamlit run app/app.py
```

That's it. In the browser, enter the path to your **Novogene delivery folder**, adjust settings (organism, thresholds), and click **Run Pipeline**. The full analysis runs in-app and the results load automatically.

### Advanced: CLI Pipeline + Config File

For scripted or reproducible workflows, you can run the pipeline from the
command line with a YAML configuration file:

```bash
# Edit config.yaml with your settings (data_dir, thresholds, etc.)
python run_pipeline.py --config config.yaml

# Then launch the app pointing at the config
streamlit run app/app.py -- --config config.yaml
```

### Legacy: Single-file Explorer

`novogene_explorer.py` is a standalone single-file Streamlit app that browses a
Novogene delivery folder directly without running the pipeline. It is kept for
quick point-and-click use:

```bash
streamlit run novogene_explorer.py -- /path/to/your/novogene/results
```

New features (QC dashboards, PPI network, multi-condition comparisons) live in
the multi-page app under `app/`, not in the legacy file.

## How the Pipeline Works

`run_pipeline.py --config config.yaml` executes these stages in order:

| Stage | Module | What it does |
|-------|--------|-------------|
| **1. Ingest** | `pipeline/ingest.py` | Walks the Novogene delivery folder, discovers quantification matrices, DEG tables, and enrichment results. Handles both `database_first` and `comparison_first` enrichment layouts. |
| **2. Normalize** | `pipeline/normalize.py` | Filters low-count genes, computes TPM from raw counts and gene lengths, and produces a log2(TPM+1) expression matrix for downstream analysis. |
| **3. QC** | `pipeline/qc.py` | Computes library sizes, gene detection rates, mitochondrial fractions, sample-sample correlation, PCA, and UMAP embeddings. Optionally flags outlier samples. |
| **4. Differential Expression** | `pipeline/diffexp.py` | Cleans and standardizes Novogene DEG results. If `rerun_de: true`, re-runs DE from counts using pyDESeq2 and merges with Novogene results. |
| **5. Similarity** | `pipeline/similarity.py` | Builds a gene-gene cosine similarity matrix over the top variable genes, performs hierarchical clustering, and computes per-comparison expression signature vectors. |
| **6. Signatures** | `pipeline/signatures.py` | Runs GSEA and over-representation analysis (ORA) via gseapy, computes Jaccard overlap of significant gene sets across comparisons, and identifies core (shared) and unique pathway signatures. |
| **7. Save** | `pipeline/persistence.py` | Persists all results to a single HDF5 file (`results/novoexplorer_results.h5`) for fast loading by the Streamlit app. |

## Pages (Multi-Page App)

| Page | Description |
|------|-------------|
| **Overview** | Library sizes, gene detection rates, PCA scatter, sample correlation heatmap, DEG counts per comparison. |
| **Differential Expression** | Interactive volcano and MA plots with gene highlighting, sortable DEG table, per-gene expression bar charts, gene basket. |
| **Gene Search** | Autocomplete gene search, expression profile across samples, log2FC across all comparisons, similar-gene finder (cosine similarity), gene basket management. |
| **Signatures & Pathways** | GSEA enrichment dot plots per comparison, Jaccard overlap heatmap, core signatures (enriched across multiple comparisons), unique signatures. |
| **Multi-Condition** | UpSet plot of DEG overlaps, fold-change concordance scatter between comparison pairs, cross-comparison gene summary table. |

## Tabs (Standalone Explorer)

The legacy `novogene_explorer.py` has these tabs:

| Tab | Description |
|-----|-------------|
| **Overview** | DEG summary statistics, bar charts of up/down counts per comparison |
| **Gene Explorer** | Search a gene, see its log2FC across all comparisons. Multi-gene heatmap. |
| **Comparison Browser** | Volcano plot with interactive gene labels, filterable DEG table |
| **Enrichment** | Dot plots, bar charts, cross-comparison enrichment search |
| **MA Plot** | baseMean vs log2FC -- shows expression-dependent fold changes |
| **Venn / UpSet** | Overlap of significant DEGs across comparisons |
| **Ranked Genes** | Waterfall plot of all genes sorted by FC or significance |
| **DEG Summary** | Wide table: log2FC + padj for every gene across all comparisons |
| **Pathway Viewer** | Select a GO/KEGG/Reactome term, see member genes colored by log2FC |
| **PPI Network** | Protein-protein interaction hub analysis, score filtering, gene search |
| **Export** | Download Excel workbook or ZIP of CSVs with optional significance filtering |

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

### Supported Enrichment Databases

| Database | Description |
|----------|-------------|
| GO | Gene Ontology (Biological Process, Molecular Function, Cellular Component) |
| KEGG | Kyoto Encyclopedia of Genes and Genomes pathways |
| DisGeNET | Disease-gene associations |
| DO | Disease Ontology |
| Reactome | Reactome pathway database |
| PPI | Protein-protein interaction networks |

## Configuration

Edit `config.yaml` to customize the pipeline. All keys are optional -- defaults are applied for anything omitted.

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
| `similarity_variable_genes` | `5000` | Number of top-variable genes used to build the precomputed cosine similarity matrix. Higher values capture more biology but increase memory. |

### Signature Analysis

| Key | Default | Description |
|-----|---------|-------------|
| `signature_min_comparisons` | `2` | Minimum number of comparisons a pathway must be enriched in to be called a "core" signature. |
| `gene_set_databases` | `["MSigDB_Hallmark_2020", ...]` | Gene-set libraries passed to gseapy for enrichment analysis. See the [Enrichr libraries list](https://maayanlab.cloud/Enrichr/#libraries) for available options. |

### Advanced

| Key | Default | Description |
|-----|---------|-------------|
| `log_level` | `"INFO"` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

### Example

```yaml
project_name: "My RNA-Seq Experiment"
data_dir: "./data"
output_dir: "./results"
organism: "human"
padj_threshold: 0.05
log2fc_threshold: 1.0
rerun_de: false
comparisons: "auto"

similarity_variable_genes: 5000

signature_min_comparisons: 2
gene_set_databases:
  - "MSigDB_Hallmark_2020"
  - "GO_Biological_Process_2023"
  - "KEGG_2021_Human"
```

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

## Project Structure

```
NovoExplorer/
  run_pipeline.py         # CLI entry point for the full analysis pipeline
  config.yaml             # Pipeline configuration template
  setup.sh                # One-step setup script (venv + deps + gene sets)
  requirements.txt        # Python dependencies with platform-specific pins
  novogene_explorer.py    # Legacy standalone single-file Streamlit app
  app/                    # Multi-page Streamlit app
    app.py                #   Main entry point and page routing
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

## Running Tests

```bash
source .venv/bin/activate
pytest tests/
```

To run a specific module:

```bash
pytest tests/test_diffexp.py -v
```

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

### "No differential expression results found" in the app

Make sure `data_dir` in `config.yaml` points to the root of your Novogene delivery folder (the directory that contains `Differential/`, `Enrichment/`, etc.). Then re-run the pipeline:

```bash
python run_pipeline.py --config config.yaml
```

### The app shows encoded comparison names like `Drug%20A_vs_Control`

This was fixed in commit `d2c6dbd`. Pull the latest `main` and re-run the pipeline to regenerate the HDF5 file:

```bash
git pull origin main
python run_pipeline.py --config config.yaml
```

### Streamlit shows "No module named 'pipeline'"

Make sure you launch the app from the NovoExplorer root directory:

```bash
cd /path/to/NovoExplorer
source .venv/bin/activate
streamlit run app/app.py -- --config config.yaml
```
