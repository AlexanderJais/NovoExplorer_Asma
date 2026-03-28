# NovoExplorer

Interactive Streamlit app for browsing and analyzing Novogene bulk RNA-Seq deliveries.

Point it at your Novogene results folder and instantly get volcano plots, MA plots, enrichment dot plots, pathway views, gene-level exploration, and more — no coding required.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch
streamlit run novogene_explorer.py

# Or pass the data folder directly
streamlit run novogene_explorer.py -- /path/to/your/novogene/results
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
      GroupA_vs_GroupB/all/*.xls
    KEGG/
      GroupA_vs_GroupB/all/*.xls
    DisGeNET/
      GroupA_vs_GroupB/all/*.xls
    DO/
      GroupA_vs_GroupB/all/*.xls
    Reactome/
      GroupA_vs_GroupB/all/*.xls
    PPI/
      GroupA_vs_GroupB/all/*.xls
```

### Supported Enrichment Databases

| Database | Description |
|----------|-------------|
| GO | Gene Ontology (Biological Process, Molecular Function, Cellular Component) |
| KEGG | Kyoto Encyclopedia of Genes and Genomes pathways |
| DisGeNET | Disease-gene associations |
| DO | Disease Ontology |
| Reactome | Reactome pathway database |
| PPI | Protein-protein interaction networks |

## Tabs

| Tab | Description |
|-----|-------------|
| **Overview** | DEG summary statistics, bar charts of up/down counts per comparison |
| **Gene Explorer** | Search a gene, see its log2FC across all comparisons. Multi-gene heatmap. |
| **Comparison Browser** | Volcano plot with interactive gene labels, filterable DEG table |
| **Enrichment** | Dot plots, bar charts, cross-comparison enrichment search |
| **MA Plot** | baseMean vs log2FC — shows expression-dependent fold changes |
| **Venn / UpSet** | Overlap of significant DEGs across comparisons |
| **Ranked Genes** | Waterfall plot of all genes sorted by FC or significance |
| **DEG Summary** | Wide table: log2FC + padj for every gene across all comparisons |
| **Pathway Viewer** | Select a GO/KEGG/Reactome term, see member genes colored by log2FC |
| **Export** | Download Excel workbook or ZIP of CSVs with optional significance filtering |

## Requirements

- Python 3.10+
- See `requirements.txt` for package dependencies
