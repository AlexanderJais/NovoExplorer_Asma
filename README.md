# NovoView: RNA-Seq Explorer for Novogene Data

A local analysis and exploration platform for Novogene bulk RNA-Seq results. NovoView provides two tools: a **pipeline** for full analysis (normalization, DE, enrichment, persistence) and a **standalone explorer app** that reads raw Novogene delivery folders directly — no preprocessing needed.

## Novogene Explorer (Quick Start)

The fastest way to browse your Novogene results. Point it at your data folder and go.

### Requirements

```bash
pip install streamlit plotly pandas numpy
```

### Launch

```bash
cd novoview

# Option 1: pass the data folder path as an argument
streamlit run novogene_explorer.py -- /path/to/your/novogene_data

# Option 2: launch without arguments and enter the path in the sidebar
streamlit run novogene_explorer.py
```

The app opens at **http://localhost:8501**.

### Expected Folder Structure

The app auto-discovers the standard Novogene delivery layout:

```
your_data_folder/
├── Differential/
│   ├── 1.deglist/
│   │   ├── PS139_1_IL13_ALLvsCTRL_IL13_ALL/
│   │   │   ├── PS139_1_IL13_ALLvsCTRL_IL13_ALL_deg.xls      # full gene list
│   │   │   ├── PS139_1_IL13_ALLvsCTRL_IL13_ALL_deg_all.xls   # significant DEGs
│   │   │   ├── PS139_1_IL13_ALLvsCTRL_IL13_ALL_deg_up.xls    # upregulated
│   │   │   └── PS139_1_IL13_ALLvsCTRL_IL13_ALL_deg_down.xls  # downregulated
│   │   ├── PS139_10_IL13_ALLvsCTRL_IL13_ALL/
│   │   │   └── ...
│   │   └── diff_stat.xls           # summary of DEG counts per comparison
│   ├── 2.cluster/
│   └── diff_readme.txt
├── Enrichment/
│   ├── KEGG/
│   │   ├── PS139_1_IL13_ALLvsCTRL_IL13_ALL/
│   │   │   └── all/
│   │   │       └── *_KEGGenrich.xls
│   │   └── .../
│   └── GO/
│       ├── PS139_1_IL13_ALLvsCTRL_IL13_ALL/
│       │   └── all/
│       │       └── *_GOenrich.xls
│       └── .../
└── sample_info.txt                  # optional: sample-to-group mapping
```

Both naming conventions for comparisons are supported:
- `GroupA_vs_GroupB` (with underscores around `vs`)
- `GroupAvsGroupB` (Novogene default, no underscores)

Numbered container folders (`1.deglist/`, `2.cluster/`) are automatically unwrapped. The `.xls` files are expected to be tab-separated text (standard Novogene format).

### App Tabs

#### Overview

- DEG summary table from `diff_stat.xls` with up/down counts per comparison
- Bar chart of DEG counts across all comparisons
- Full comparison listing with extracted group names

#### Gene Explorer

- **Single gene search**: type a gene name (e.g., `CSF2RB`) and see its log2FC across ALL comparisons as a bar chart, colored by significance
- Detailed table with padj, pvalue, basemean, regulation per comparison
- **Multi-gene heatmap**: paste a list of genes (one per line or comma-separated) to see a log2FC heatmap across all comparisons

#### Comparison Browser

- Select a comparison from the dropdown
- **Interactive volcano plot** with adjustable padj and log2FC thresholds
- Top N significant genes are labeled directly on the plot
- Filterable DEG table with significant-only toggle and gene name search

#### Enrichment

- Select a comparison and database (KEGG or GO)
- **Dot plot**: gene ratio vs significance, dot size by gene count
- **Bar plot**: -log10(padj) for top enriched terms
- Filterable enrichment table
- **Cross-comparison search**: find a pathway or GO term across all conditions with a significance heatmap

### Sidebar

- **Data folder path**: enter or change the path to your Novogene data
- **Metrics**: number of comparisons, genes, enrichment comparisons, sample groups
- **Log panel**: collapsible expander showing all pipeline log messages (file discovery, parsing, warnings). Useful for debugging when data isn't loading as expected. Includes a "Clear log" button.

### Troubleshooting

| Problem | Solution |
|---------|----------|
| "No DEG data loaded" | Check that `Differential/` exists in the data folder and contains comparison subfolders |
| Comparisons not found | The app looks inside numbered containers like `1.deglist/`. Open the **Log** panel to see what was discovered |
| Enrichment tab empty | Check that `Enrichment/` contains `KEGG/` and/or `GO/` subfolders with comparison subdirectories |
| Gene not found | Gene names are matched case-insensitively. Check the **Overview** tab to see total gene count |
| Groups show as single name | Comparison folder must contain `vs` (e.g., `TreatvsCtrl` or `Treat_vs_Ctrl`) |

---

## Full Pipeline (Advanced)

For full analysis including normalization, QC, pyDESeq2, similarity analysis, and HDF5 persistence:

```bash
# 1. Run the setup script (creates venv, installs dependencies)
bash setup.sh

# 2. Place your Novogene delivery folder in data/
cp -r /path/to/Novogene_Results ./data/

# 3. Edit config.yaml
#    - Set data_dir to point to your data
#    - Set organism to "human" or "mouse"
#    - Adjust thresholds as needed

# 4. Run the analysis pipeline
source .venv/bin/activate
python run_pipeline.py --config config.yaml

# 5. Launch the full web app
streamlit run app/app.py -- --config config.yaml
```

### Configuration

Edit `config.yaml` to customize:

- **organism**: `"human"` or `"mouse"`
- **padj_threshold**: Adjusted p-value cutoff (default: 0.05)
- **log2fc_threshold**: Log2 fold change cutoff (default: 1.0)
- **rerun_de**: Set to `true` to re-run DE with pyDESeq2 even if Novogene results exist
- **comparisons**: `"auto"` to discover from folder names, or specify explicitly
- **similarity_top_n**: Number of similar genes to return per query (default: 50)
- **gene_set_databases**: Databases for enrichment analysis

### Full App Pages

1. **Overview**: PCA/UMAP, sample correlation, top variable genes, DEG summary
2. **Differential Expression**: Volcano plots, MA plots, DEG tables per comparison
3. **Gene Search**: Find similar genes, build gene baskets, explore expression profiles
4. **Signatures**: Enrichment browser, signature overlap, core/unique signatures
5. **Multi-Condition**: UpSet plots, fold-change concordance, cross-comparison tables

## Project Structure

```
novoview/
├── novogene_explorer.py  # Standalone Streamlit app (no preprocessing needed)
├── pipeline/             # Analysis pipeline modules
│   ├── ingest.py         # Novogene file discovery and parsing
│   ├── normalize.py      # Expression data standardization
│   ├── qc.py             # Quality control metrics
│   ├── diffexp.py        # Differential expression analysis
│   ├── similarity.py     # Gene similarity and clustering
│   ├── signatures.py     # GSEA, ORA, signature analysis
│   ├── persistence.py    # HDF5 data storage
│   └── utils.py          # Shared utilities
├── plotting/             # Nature-style plotting modules
│   ├── theme.py          # Global Nature journal theme
│   ├── volcano.py        # Volcano plots
│   ├── heatmap.py        # Clustered heatmaps
│   ├── pca.py            # PCA/UMAP scatter plots
│   ├── enrichment.py     # Enrichment dot/bar plots
│   ├── upset.py          # UpSet plots
│   ├── similarity_viz.py # Similarity networks
│   └── ma_plot.py        # MA plots
├── app/                  # Full Streamlit web application (requires pipeline run)
│   ├── app.py            # Entry point
│   ├── style.css         # Custom styling
│   ├── pages/            # App pages
│   └── components/       # Shared widgets
├── tests/                # Test suite
├── config.yaml           # Configuration
├── run_pipeline.py       # CLI entry point
├── requirements.txt      # Python dependencies
└── setup.sh              # Setup script
```

## Dependencies

For the standalone explorer (`novogene_explorer.py`):
- streamlit, plotly, pandas, numpy

For the full pipeline:
- pandas, numpy, scipy, scikit-learn
- pyDESeq2 (differential expression)
- gseapy (enrichment analysis)
- plotly, matplotlib, seaborn (visualization)
- streamlit (web application)
- h5py (data persistence)

## License

MIT
