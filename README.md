# Electride ML Predictor

This repository develops a staged workflow for identifying and characterizing interstitial electron localization in crystalline materials.

## Research pipeline

```text
Crystal structures and DFT/ELF data
        ↓
Candidate-void detection
        ↓
Atom/void graph-dataset construction
        ↓
Heterogeneous GNN and baseline models
        ↓
Architecture and feature studies
```

## Main areas

- [`void_detection/`](void_detection/): exploratory periodic candidate-void generation and synthetic validation.
- [`elf_void_analysis/`](elf_void_analysis/): POSCAR/ELFCAR indexing, ELF labeling, graph construction, HDF5 shards, and family-safe splits.
- [`elf_void_gnn/`](elf_void_gnn/): graph loading, metrics, model implementations, training, cluster jobs, and research experiments.
- [`dft/`](dft/): PBE and CP2K electronic-structure workflows and ELF post-processing.
- [`data/`](data/) and [`A2BC2/`](A2BC2/): structure collections and processed source data.
- [`elfnet_data_analysis/`](elfnet_data_analysis/): exploratory notebooks and visual inspection exports.

## Current GNN research

- [`elf_void_gnn/README.md`](elf_void_gnn/README.md): modeling entry point.
- [`elf_void_gnn/DATASET_PROVENANCE.md`](elf_void_gnn/DATASET_PROVENANCE.md): source, processed dataset, split, and label provenance.
- [`elf_void_gnn/REPRODUCIBILITY.md`](elf_void_gnn/REPRODUCIBILITY.md): extraction, training, architecture-sweep, and reporting commands.
- [`elf_void_gnn/architecture_test/research.md`](elf_void_gnn/architecture_test/research.md): architecture-study design and literature rationale.
- [`elf_void_gnn/architecture_test/RESULTS.md`](elf_void_gnn/architecture_test/RESULTS.md): concise architecture results.
- [`elf_void_gnn/architecture_test/DETAILED_RESULTS.md`](elf_void_gnn/architecture_test/DETAILED_RESULTS.md): full architecture report.
- [`elf_void_gnn/feature_study/PLAN.md`](elf_void_gnn/feature_study/PLAN.md): fixed-architecture feature-ablation plan.
- [`elf_void_gnn/feature_study/RESULTS.md`](elf_void_gnn/feature_study/RESULTS.md): completed feature-study summary.

## Data policy

Raw ELFNet files, generated graph shards, model checkpoints, Slurm logs, and per-void prediction files are intentionally kept outside Git. Their canonical locations and fingerprints are documented in `elf_void_gnn/DATASET_PROVENANCE.md`.
