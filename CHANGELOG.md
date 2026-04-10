# Changelog

All notable changes to NovoExplorer are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Comprehensive documentation: improved README with table of contents, badges, FAQ, and troubleshooting
- CONTRIBUTING.md with development guidelines and code style reference
- This CHANGELOG.md
- Module-level and function-level docstrings across all Python packages

## [0.1.0] - 2025

### Added
- **Diagnostics tab** with directory tree, discovery summary, and downloadable log
- **In-app pipeline runner** so users never touch YAML or CLI
- Comprehensive README and config reference documentation

### Fixed
- Repeated expression parsing and noisy DEG warnings from non-DEG containers
- `diff_stat.xls` not found: skip PDF/PNG siblings with same name prefix
- Expression file patterns and diff_stat search for new delivery format
- Config aliases, URL-decode on load

### Changed
- Adapted NovoExplorer for mouse (Mus musculus) and numbered directory layout
- `novogene_explorer.py` is the primary app (not legacy)

### Initial Features
- **Main explorer** (`novogene_explorer.py`) with 11 interactive tabs:
  Overview, Gene Explorer, Comparison Browser, Enrichment, MA Plot,
  Venn/UpSet, Ranked Genes, DEG Summary, Pathway Viewer, PPI Network, Export
- **Analysis pipeline** with 7 stages: ingest, normalize, QC, differential
  expression, similarity, signatures, persistence
- **Multi-page Streamlit app** (`app/app.py`) with pipeline-backed pages:
  Overview, Differential Expression, Gene Search, Signatures & Pathways,
  Multi-Condition
- **Plotting library** with Nature journal-style themes, supporting both Plotly
  (interactive) and Matplotlib (static) outputs
- Auto-detection of Novogene delivery folder structures (database-first and
  comparison-first layouts, numbered-prefix directories)
- Human and mouse organism support
- Colorblind-friendly Wong et al. palette
- Publication-ready PNG/SVG figure export
- HDF5 persistence for pipeline results
- Pytest test suite covering all pipeline modules
