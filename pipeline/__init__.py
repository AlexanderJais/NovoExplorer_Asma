"""NovoExplorer analysis pipeline for Novogene bulk RNA-Seq data.

Provides a seven-stage workflow for processing Novogene bulk RNA-Seq
deliveries: ingestion, normalization, quality control, differential
expression, gene similarity, signature analysis, and persistence.

Modules
-------
ingest : Novogene delivery folder discovery and file parsing.
normalize : Count filtering, TPM computation, log2 transformation.
qc : PCA, UMAP, correlation, library sizes, outlier detection.
diffexp : DEG table cleaning, optional pyDESeq2 re-analysis.
similarity : Cosine similarity, hierarchical clustering, signatures.
signatures : GSEA, ORA, Jaccard overlap, core/unique pathways.
persistence : HDF5 save/load for all pipeline outputs.
utils : Column standardization, flexible file reading, config loading.
"""
