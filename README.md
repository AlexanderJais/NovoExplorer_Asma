# NovoExplorer

Interactive Streamlit application for browsing and analyzing Novogene bulk RNA-Seq deliveries.

Point it at your Novogene results folder and instantly get volcano plots, MA plots, enrichment dot plots, pathway views, gene-level exploration, and more -- no coding required.

## Quick Start

```bash
# 1. Run the setup script (creates venv, installs deps, downloads gene sets)
bash setup.sh

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Edit config.yaml to point at your Novogene delivery folder
# 4. Run the analysis pipeline (normalization, QC, DE, similarity, signatures)
python run_pipeline.py --config config.yaml

# 5. Launch the multi-page Streamlit app
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

## Project Structure

```
NovoExplorer/
  novogene_explorer.py    # Standalone single-file Streamlit app
  run_pipeline.py         # CLI entry point for the full analysis pipeline
  config.yaml             # Pipeline configuration
  setup.sh                # One-step setup script
  requirements.txt        # Python dependencies
  app/                    # Multi-page Streamlit app
    app.py                #   Main entry point
    style.css             #   Custom theme
    pages/                #   Individual pages
    components/           #   Shared UI components
  pipeline/               # Analysis pipeline modules
    ingest.py             #   Novogene folder discovery & parsing
    normalize.py          #   Count normalization & gene ID mapping
    qc.py                 #   Quality control (PCA, UMAP, correlation)
    diffexp.py            #   Differential expression (Novogene + pyDESeq2)
    similarity.py         #   Gene similarity & clustering
    signatures.py         #   GSEA, ORA, signature overlap
    persistence.py        #   HDF5 result storage
    utils.py              #   Column standardization, file reading, config
  plotting/               # Plotly figure builders (Nature journal styling)
  tests/                  # Pytest test suite
```

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

## Tabs (Standalone Explorer)

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

## Pages (Multi-Page App)

| Page | Description |
|------|-------------|
| **Overview** | Library sizes, detection rates, PCA, sample correlation, DEG counts |
| **Differential Expression** | Volcano/MA plots, gene spotlight, comparison selector |
| **Gene Search** | Autocomplete gene search, expression bars, similar-gene finder, basket |
| **Signatures & Pathways** | GSEA dot plots, Jaccard overlap heatmap, core/unique signatures |
| **Multi-Condition** | UpSet plot, fold-change concordance scatter, cross-comparison summary |

## Configuration

Edit `config.yaml` to customize the pipeline:

```yaml
project_name: "My RNA-Seq Experiment"
data_dir: "./data"
output_dir: "./results"
organism: "human"             # "human" or "mouse"
padj_threshold: 0.05
log2fc_threshold: 1.0
rerun_de: false               # Re-run DE with pyDESeq2 even if Novogene results exist
comparisons: "auto"           # Auto-detect from folder names, or specify explicitly
```

## Requirements

- Python 3.10+
- See `requirements.txt` for package dependencies

Key dependencies: Streamlit, pandas, Plotly, pyDESeq2, gseapy, scikit-learn, UMAP.

## Running Tests

```bash
pytest tests/
```
