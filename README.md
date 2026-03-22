# NovoView: RNA-Seq Explorer for Novogene Data

A local analysis and exploration platform for Novogene bulk RNA-Seq results. NovoView ingests Novogene deliverables, standardizes the data, computes gene-level similarity and pathway signatures, and provides an interactive Streamlit web application for exploring results.

## Features

- **Automatic Novogene ingestion**: Drop in a Novogene delivery folder and it "just works" — no manual renaming
- **Quality control**: Library sizes, gene detection rates, mitochondrial fractions, PCA/UMAP, outlier detection
- **Differential expression**: Uses Novogene results or re-runs with pyDESeq2
- **Gene similarity**: Cosine similarity across expression profiles, hierarchical clustering into modules
- **Pathway analysis**: Pre-ranked GSEA, over-representation analysis, signature overlap
- **Interactive web app**: Five-page Streamlit dashboard with Nature-quality figures
- **Publication-ready figures**: All plots follow Nature journal styling (Wong colorblind-safe palette, clean axes, proper typography)

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd novoview

# 2. Run the setup script (creates venv, installs dependencies, downloads gene sets)
bash setup.sh

# 3. Place your Novogene delivery folder in the data/ directory
cp -r /path/to/Novogene_Results ./data/

# 4. Edit config.yaml
#    - Set data_dir to point to your data
#    - Set organism to "human" or "mouse"
#    - Adjust thresholds as needed

# 5. Run the analysis pipeline
source .venv/bin/activate
python run_pipeline.py --config config.yaml

# 6. Launch the web app
streamlit run app/app.py -- --config config.yaml

# The app will open at http://localhost:8501
```

## Data Organization

Place your Novogene delivery folder(s) in the data directory:

```
data/
├── experiment_1/          # One Novogene delivery per experiment
│   ├── Quant/
│   ├── Differential/
│   ├── Enrichment/
│   └── sample_info.txt
├── experiment_2/
│   └── ...
└── shared_sample_info.txt # Optional: global metadata
```

Or simply point `data_dir` in `config.yaml` to a single Novogene delivery folder.

## Configuration

Edit `config.yaml` to customize:

- **organism**: `"human"` or `"mouse"`
- **padj_threshold**: Adjusted p-value cutoff (default: 0.05)
- **log2fc_threshold**: Log2 fold change cutoff (default: 1.0)
- **rerun_de**: Set to `true` to re-run DE with pyDESeq2 even if Novogene results exist
- **comparisons**: `"auto"` to discover from folder names, or specify explicitly
- **similarity_top_n**: Number of similar genes to return per query (default: 50)
- **gene_set_databases**: Databases for enrichment analysis

## Web App Pages

1. **Overview**: PCA/UMAP, sample correlation, top variable genes, DEG summary
2. **Differential Expression**: Volcano plots, MA plots, DEG tables per comparison
3. **Gene Search**: Find similar genes, build gene baskets, explore expression profiles
4. **Signatures**: Enrichment browser, signature overlap, core/unique signatures
5. **Multi-Condition**: UpSet plots, fold-change concordance, cross-comparison tables

## Project Structure

```
novoview/
├── pipeline/          # Analysis pipeline modules
│   ├── ingest.py      # Novogene file discovery and parsing
│   ├── normalize.py   # Expression data standardization
│   ├── qc.py          # Quality control metrics
│   ├── diffexp.py     # Differential expression analysis
│   ├── similarity.py  # Gene similarity and clustering
│   ├── signatures.py  # GSEA, ORA, signature analysis
│   ├── persistence.py # HDF5 data storage
│   └── utils.py       # Shared utilities
├── plotting/          # Nature-style plotting modules
│   ├── theme.py       # Global Nature journal theme
│   ├── volcano.py     # Volcano plots
│   ├── heatmap.py     # Clustered heatmaps
│   ├── pca.py         # PCA/UMAP scatter plots
│   ├── enrichment.py  # Enrichment dot/bar plots
│   ├── upset.py       # UpSet plots
│   ├── similarity_viz.py # Similarity networks
│   └── ma_plot.py     # MA plots
├── app/               # Streamlit web application
│   ├── app.py         # Entry point
│   ├── style.css      # Custom styling
│   ├── pages/         # App pages
│   └── components/    # Shared widgets
├── resources/         # Gene mappings and gene sets
├── tests/             # Test suite
├── config.yaml        # Configuration
├── run_pipeline.py    # CLI entry point
└── setup.sh           # Setup script
```

## Dependencies

- pandas, numpy, scipy, scikit-learn
- pyDESeq2 (differential expression)
- gseapy (enrichment analysis)
- plotly, matplotlib, seaborn (visualization)
- streamlit (web application)
- h5py (data persistence)

## License

MIT
