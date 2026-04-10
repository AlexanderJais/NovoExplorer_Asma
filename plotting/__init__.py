"""NovoExplorer plotting modules with Nature journal styling.

All figure builders return Plotly or Matplotlib figure objects.  Shared
styling (colors, fonts, layout) is defined in ``theme.py`` and applied
automatically via ``apply_plotly_theme()`` or ``apply_mpl_theme()``.

Modules
-------
volcano : Volcano plots (Matplotlib + Plotly).
ma_plot : MA plots (baseMean vs log2FC).
heatmap : Clustered heatmaps (seaborn clustermap, Plotly fallback).
enrichment : Enrichment dot plots and bar charts.
pca : PCA and UMAP scatter plots.
similarity_viz : Similarity heatmaps and network graphs.
upset : UpSet plots for set intersections.
ppi_network : Protein-protein interaction network visualization.
theme : Color palettes (Wong et al.), font settings, styling helpers.
"""
