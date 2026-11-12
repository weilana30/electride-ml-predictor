# Electride ML Predictor: Project Scope

Status as of 2026-11-03. For the original plan and staged roadmap, see
[`project_writeup.md`](project_writeup.md) and
[`project_timeline.md`](project_timeline.md).

## Goal

Identify and characterize **electrides**: crystals in which some electrons
occupy interstitial voids instead of atom-centered states. The project works
at the level of individual **candidate voids** and separates two questions:

1. **Geometry: is there room?** Does a void exist, and how is the open space
   around it connected (isolated 0D pocket, 1D channel, 2D layer, or 3D
   network)?
2. **Electronic structure: is an electron there?** Given the surrounding atoms,
   how strongly localized is the electron density in that void?

Two components answer these questions. The **void connectivity identifier**
answers the first from structure alone. The **heterogeneous GNN** answers the
second by predicting a DFT-derived ELF value at each void.

```text
Crystal structure (POSCAR/CIF)
        │
        ▼
Candidate-void detection  ── void_detection/, elf_void_analysis/
        │
        ├──► Void connectivity identification (geometry: 0D/1D/2D/3D)
        │        adaptive_void_connectivity.py, power_void_connectivity.py
        │
        └──► Atom/void heterogeneous graph ──► GNN: ELF at each void
                 elf_void_analysis/            elf_void_gnn/
        │
        ▼
Combined void-level picture → crystal-level electride assessment (future)
```

## Data

| Dataset | Contents | Role |
| --- | --- | --- |
| ELFNet corpus | 50,000 paired VASP POSCAR/ELFCAR structures | GNN training and evaluation |
| Processed graph dataset | `/home/weilana/elfnet-datasets/elfnet-gnn-2026-10-23/`, HDF5 shards | Atom/void graphs and labels |
| DFT reference set (`dft/`) | PBE calculations for Ba2As, Ba3CrN3, Ca2N (electrides) and MgS, BaSn3, Cu2SiSe3 (controls) | Physical reference cases |
| Case-study CIFs (`data/cifs/`) | K4Al3(SiO4)3 (0D), Ba3CrN3 (1D), Ca2N (2D) | Connectivity validation against literature |

The graph dataset uses a **family-safe split** that keeps named parent-phase
families in one split:

| Split | Samples | Families |
| --- | ---: | ---: |
| Train | 40,031 | 37 |
| Validation | 4,969 | 9 |
| Test | 5,000 | 9 |

Split SHA256: `cdb3fc87b80cb05ce1cfdd8b94267696cfb0bb9e810004512144b0c2f2aac7f4`.
Related pressure or space-group variants can still cross splits, so results
are held-out *named phase* results, not unseen-composition results. Raw data,
graph shards, checkpoints, logs and predictions stay outside Git; see
[`elf_void_gnn/DATASET_PROVENANCE.md`](../elf_void_gnn/DATASET_PROVENANCE.md).

## Component 1: Candidate voids

Code: [`void_detection/find_voronoi_voids.py`](../void_detection/find_voronoi_voids.py)
and [`elf_void_analysis/`](../elf_void_analysis/).

- Candidates come from pymatgen Voronoi interstitials plus local maxima of
  the distance-to-nearest-atom field on a 0.5 Å grid.
- Each candidate is moved to its nearby clearance maximum, merged with
  symmetry-equivalent candidates, then expanded back to full symmetry orbits.
- Each void gets an ID, coordinates, clearance and structural descriptors.
  Both the GNN and the connectivity tools use these same void sites.

## Component 2: Heterogeneous GNN (electronic question)

Code: [`elf_void_gnn/`](../elf_void_gnn/).

**Task.** Per-void regression of `elf_at_center`, the ELF value interpolated
at each candidate void center. Model inputs are structure-derived only. ELF
quantities are labels, never inputs. The dataset also stores local ELF maxima
within 0.5, 1.0 and 1.5 Å, their offsets, and a geometric dimensionality
label, all of which are available for future targets.

**Graph.** Atom nodes and void nodes, with periodic atom–atom, void–atom,
atom–void and void–void edges. Edge distances are encoded with eight radial
basis channels. Models register through `forward(batch) -> one ELF value per
void`, using a shared loader and trainer. Normalization is fit on training
rows only, checkpoints are selected by validation RMSE, and test is evaluated
once after selection.

### Architecture study (complete)

25 predeclared runs: 8 neural configurations × 3 seeds, plus a gradient-
boosted tree. Details are in
[`architecture_test/RESULTS.md`](../elf_void_gnn/architecture_test/RESULTS.md).

| Model | Validation RMSE | Test RMSE |
| --- | ---: | ---: |
| **Hetero MPNN, width 128, depth 2 (selected)** | **0.0606 ± 0.0010** | **0.0652 ± 0.0007** |
| Hetero MPNN, width 64, depth 2 | 0.0619 ± 0.0038 | 0.0668 ± 0.0013 |
| Gated MPNN | 0.0625 ± 0.0013 | 0.0698 ± 0.0032 |
| Gradient-boosted tree | 0.0725 | 0.0766 |
| Fixed-context MLP | 0.0823 ± 0.0034 | 0.0848 ± 0.0020 |
| Void-only MLP | 0.1341 ± 0.0004 | 0.1400 ± 0.0004 |

Main finding: message passing between atoms and voids matters. Void-only
models are roughly twice as wrong.

### Feature study (complete)

Same width-64 Hetero MPNN with different inputs. Details are in
[`feature_study/RESULTS.md`](../elf_void_gnn/feature_study/RESULTS.md).

| Variant | Test RMSE | Test R² |
| --- | ---: | ---: |
| Baseline (atom + void + topology + edges) | 0.0668 ± 0.0013 | 0.774 |
| Minimal (Z, void distances, edges) | 0.0817 ± 0.0024 | 0.662 |
| **Atom-only (void descriptors zeroed)** | **0.0576 ± 0.0021** | **0.832** |

Main finding: the explicit void descriptors did not help. The model learns
void environments better from atom descriptors and graph geometry.

### Direct confirmation study (complete)

A repeat on the same split with three new seeds and three predeclared
configurations: width-128 baseline, width-128 atom-only and width-64
atom-only. The plan is in
[`confirmation_study/PLAN.md`](../elf_void_gnn/confirmation_study/PLAN.md).
Results are in
[`confirmation_study/results/README.md`](../elf_void_gnn/confirmation_study/results/README.md).

| Configuration | Validation RMSE | Test RMSE | Test R² |
| --- | ---: | ---: | ---: |
| Width 128, baseline | 0.0629 ± 0.0022 | 0.0701 ± 0.0011 | 0.751 |
| Width 128, atom-only | 0.0636 ± 0.0013 | 0.0603 ± 0.0017 | 0.816 |
| **Width 64, atom-only (selected)** | **0.0600 ± 0.0008** | **0.0593 ± 0.0028** | **0.822** |

Main finding: validation selected width-64 atom-only, the same winner as
the feature study. Its test RMSE of 0.0593 is close to that study's 0.0576.
It beats the width-128 baseline on all three test seeds; the
family-bootstrap interval excludes zero on validation and test. Width 128
gives no gain over width 64. The caveat is that this study used the same
corpus, the same family split, and a test partition that earlier studies
had already used.

## Component 3: Void connectivity identification (geometric question)

Code: [`elf_void_analysis/`](../elf_void_analysis/). Case studies are in
[`elf_void_analysis/case_studies/`](../elf_void_analysis/case_studies/).

**Question.** Is the open space a probe can reach around each candidate void
an isolated pocket (0D), a channel (1D), a layer (2D) or a 3D network? The
dimension is the number of independent periodic directions the open region
wraps around (its *winding rank*). This is atom-geometry dimensionality,
which is related to, but different from, the electronic dimensionality
reported in the electride literature.

### Methods

| Tool | Method | Output |
| --- | --- | --- |
| `label_void_dimensionality.py` | Uniform voxel grid flood fill | Training-dataset geometric label; depends on grid resolution |
| `adaptive_void_connectivity.py` | Adaptive octree with certified open/blocked blocks; VESTA `.xsf` export | Certified rank; often a `lower_bound` because boundary blocks stay unresolved |
| **`power_void_connectivity.py`** | **Exact power-diagram (radical Voronoi) network** | **Exact 0D–3D label for every region and void; no unresolved regions** |

**Exact method.** Each atom excludes a ball of radius *element radius +
probe*. Inside each atom's power cell, the union of all balls is exactly that
atom's ball, so the open space can be shrunk onto the open edges and vertices
of the power diagram without changing its connectivity. Connectivity and
winding rank of that finite network are exact. Runs take under 0.1 s per case
study.

**Validation.**

- 7 unit tests: analytic cubic and CsCl-type thresholds, degenerate
  atom-wall lattices, and reports.
- A cross-check against brute-force grids on 40 random structures with mixed
  radii. The two disagreements were both grid errors: a narrow neck the grid
  missed, and a pocket about 0.2 Å across.

**Element-dependent radii.** `--radius-scheme none | ionic | atomic | custom`.

- `ionic` uses Shannon crystal radii. Oxidation states are assigned per
  element because electrides are not charge balanced.
- `none` reproduces the old uniform atom-center cutoff.
- When an element has no Shannon radius, the atomic radius is used and the
  report says so.

### Case-study findings

The probe ranges below are where each structure shows its literature
dimension as its highest open dimension:

| Material (literature) | Uniform cutoff | Ionic radius + probe |
| --- | --- | --- |
| K4Al3(SiO4)3 (0D) | 2.00–3.00 Å | 0.75–1.45 Å |
| Ba3CrN3 (1D) | 2.00–2.70 Å | 0.65–1.20 Å |
| Ca2N (2D) | 1.65–2.40 Å | 0.50–1.25 Å |
| **Shared window** | **2.00–2.40 Å** | **0.75–1.20 Å** |

- The earlier uniform 2.0 Å choice sits right on the edge of two windows: at
  1.95 Å, both K4Al3(SiO4)3 and Ba3CrN3 become 3D.
- Ionic radii + a 1.0 Å probe is at least 0.2 Å inside every window. At that
  setting the controls MgS and Cu2SiSe3 have no open space and BaSn3 has only
  0D pockets.
- Three electrides and three controls are too few to fix a cutoff. This is a
  starting point for calibration.

### Current role

Connectivity labels are **report-only**. They group voids into regions
(`region_uid`, `connectivity_label`, `label_status`) but do not change the
GNN dataset or predictions.

## How the components combine

The intended end state follows the two-question design:

1. **Geometry filter.** The connectivity identifier finds voids that are open
   at a calibrated probe and labels their region dimension.
2. **Electronic score.** The GNN predicts how strongly electrons localize in
   each void.
3. **Joint picture.** High predicted ELF in voids of one connected region
   suggests an electride of that region's dimension. High ELF with no open
   region, or open space with low ELF, are informative mismatches and
   candidates for DFT follow-up.

## Open work

| Area | Next step |
| --- | --- |
| GNN confirmation | Freeze validation selection, run the single final test, write up results |
| Connectivity calibration | Choose the radius scheme and probe on many ELF-labeled structures, not 3 case studies |
| Geometry vs. electrons | Compare geometric region labels with ELF-basin connectivity and literature electronic dimensionality |
| Radii | Better anion radii (As3-, Sn4-), coordination-aware Shannon radii, a policy for intermetallics |
| Visualization | VESTA export for the exact method (currently octree only, uniform radii only) |
| Dataset integration | Regenerate dataset dimensionality labels with the exact method and decide whether to use them as targets or features |
| Generalization | Composition- or prototype-held-out splits; current results are named-phase held-out only |
| Crystal level | Pool void predictions and region labels into a crystal-level electride score |
| Ba2As label | Settle 0D vs 2D with a phase-matched electronic comparison |

## Key documents

- [`README.md`](../README.md): repository map and data policy
- [`elf_void_gnn/README.md`](../elf_void_gnn/README.md): GNN entry point
- [`elf_void_gnn/REPRODUCIBILITY.md`](../elf_void_gnn/REPRODUCIBILITY.md): commands for every study
- [`elf_void_analysis/README.md`](../elf_void_analysis/README.md): void, label and connectivity tools
- [`elf_void_analysis/case_studies/README.md`](../elf_void_analysis/case_studies/README.md): literature case studies
- [`dft/README.md`](../dft/README.md): PBE/CP2K workflows
