# Electride ML Predictor

Finding **electrides**: crystals where some electrons sit in interstitial voids instead of on atoms. The project treats each *candidate void* separately and asks two questions:

1. **Geometry: is there room?** How is the open space around a void connected (0D pocket, 1D channel, 2D layer, 3D network)? Answered from structure alone by the **void connectivity identifier**.
2. **Electronic structure: is an electron there?** How strongly is density localized in that void? Answered by a **heterogeneous atom/void GNN** that predicts the DFT electron localization function (ELF) at each void.

```text
Crystal structure (POSCAR/CIF)
   │
   ▼
Candidate-void detection ──► Void connectivity (0D/1D/2D/3D)   elf_void_analysis/
   │
   ▼
Atom/void graph (HDF5 shards, family-safe splits)              elf_void_analysis/
   │
   ▼
Heterogeneous GNN → ELF at each void                           elf_void_gnn/
```

## Headline results

Family-held-out test set: 5,000 structures, 9 phase families, ELF RMSE at void centers (lower is better).

| Model | Test RMSE |
| --- | ---: |
| **Atom–void GNN, atomic features only (width 64)** | **0.059 ± 0.003** |
| Atom–void GNN, all void descriptors (width 128) | 0.070 ± 0.001 |
| Gated MPNN | 0.070 ± 0.003 |
| Gradient-boosted tree | 0.077 |
| Fixed-context MLP | 0.085 ± 0.002 |
| Void-only MLP | 0.140 ± 0.000 |

- Learned atom/void message passing clearly beats void-only and fixed-context baselines. Width 128 vs 64 gives a small gain; gated recurrence gives none.
- After fixing the void-descriptor bug (dataset `voidv2`), the all-descriptor models no longer fail on individual families and win on validation, but the **atom-only model still wins on test** on every seed. Hand-built void descriptors show no consistent gain.
- Caveat: one family split, three seeds; seed spread is optimization variability, not a confidence interval.

Details: [architecture test](elf_void_gnn/architecture_test/RESULTS.md) · [feature study](elf_void_gnn/feature_study/RESULTS.md) · [confirmation study (voidv2)](elf_void_gnn/confirmation_study/results_voidv2/README.md) · [open reruns](docs/TODO.md)

![Model comparison](poster/figures/gnn_model_comparison.png)
![Predicted vs DFT ELF](poster/figures/gnn_parity_atom_only.png)

## Repository layout

| Path | Contents |
| --- | --- |
| [`elf_void_analysis/`](elf_void_analysis/) | Void labeling from ELF, exact power-diagram connectivity (`power_void_connectivity.py`), adaptive connectivity, graph/dataset build, HDF5 shards, tests, case studies |
| [`elf_void_gnn/`](elf_void_gnn/) | Data loader, models, trainer, metrics, and the architecture / feature / confirmation studies |
| [`void_detection/`](void_detection/) | Periodic candidate-void finder used by the graph builder |
| [`data/`](data/) | Case-study CIFs and the electride registry |
| [`dft/`](dft/) | Quantum ESPRESSO PBE and CP2K convergence workflows for six reference materials, ELF post-processing, source CIFs in `structures/` |
| [`poster/`](poster/) | Research poster (SVG/PDF), figures, and the scripts that build them |
| [`docs/`](docs/) | [Project scope](docs/PROJECT_SCOPE.md), [write-up](docs/project_writeup.md), [timeline](docs/project_timeline.md), [TODO](docs/TODO.md) |

## Quickstart

```bash
pip install -r elf_void_analysis/requirements.txt          # analysis / connectivity
pip install -r elf_void_gnn/requirements-rocm63.txt        # GNN (install torch first; see file header)

# Unit tests
PYTHONPATH=. pytest elf_void_gnn/tests

# Build a graph dataset from ELFNet POSCAR/ELFCAR pairs, then train
python elf_void_analysis/build_dataset.py --help
python elf_void_gnn/train.py --help
```

Full commands for extraction, training, architecture sweeps, and reporting: [`elf_void_gnn/REPRODUCIBILITY.md`](elf_void_gnn/REPRODUCIBILITY.md). Data sources and fingerprints: [`elf_void_gnn/DATASET_PROVENANCE.md`](elf_void_gnn/DATASET_PROVENANCE.md).

## Data policy

Raw ELFNet files, generated graph shards, model checkpoints, Slurm logs and per-void prediction files are kept **outside Git** (on the Negishi cluster); locations and hashes are recorded in the provenance document. The earlier exploratory material (Voronoi void finder, A2BC2 set, CIF collections, ELFNet notebook) was removed from the tree and remains in git history.
