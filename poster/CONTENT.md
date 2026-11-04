# Poster content

## Story in one paragraph

Electrides are crystals whose extra electrons sit in the empty space between
atoms. Finding them, and telling whether those electrons are trapped (0D) or
connected (1D/2D), normally needs a DFT calculation per crystal. We built two
structure-only tools: a graph neural network that predicts the electron
localization function (ELF) at each void, and an exact geometric method that
labels how the empty space connects. Each works on its own; combining them is
the next step toward a fast electride screen.

## Framing

- **End goal:** a fast, structure-only tool that says whether a crystal is likely
  an electride, where its extra electrons sit, and whether they are trapped (0D)
  or connected (1D/2D/3D), without DFT on every candidate.
- **This poster:** the two parts, built and tested separately.
- **Do not claim:** that the GNN finds electrides. It was trained and tested on
  ELFNet (metal frameworks from compressed hydrides), which contains no known
  electrides.

## Title

**Where Do the Extra Electrons Go?**
Predicting interstitial electron localization and void dimensionality for
electride screening

## Research questions

| | Question | Status |
| --- | --- | --- |
| RQ1 | Can a GNN predict ELF at candidate voids from the crystal structure alone? | Answered |
| RQ2 | Can geometry alone recover an electride's dimensionality (0D/1D/2D)? | Answered |
| RQ3 | How do the two combine into an electride screen? | Proposed (schematic only) |

## Key numbers and where they come from

### RQ1: GNN (source: `elf_void_gnn/poster_figures/summary.json`, `architecture_test/RESULTS.md`, `feature_study/RESULTS.md`)

All are test-set numbers: 5,000 structures, 88,680 voids, 9 families never
seen in training. Mean ± SD over 3 seeds.

| Model | Test RMSE (ELF) |
| --- | ---: |
| Hetero GNN, atom-only (best) | 0.058 ± 0.002 |
| Hetero GNN, all descriptors (selected in architecture study) | 0.065 ± 0.001 |
| Gated MPNN | 0.070 ± 0.003 |
| Gradient-boosted tree (1 run) | 0.077 |
| Fixed-context MLP | 0.085 ± 0.002 |
| Void-only MLP | 0.140 |
| Predict the average ELF | 0.141 |

- Atom-only GNN: R² 0.83 ± 0.01, MAE 0.038.
- Of voids with true ELF ≥ 0.5, it predicts ≥ 0.5 for 63% (precision 0.83).
  Of the rare voids with ELF ≥ 0.7 (0.7% of voids), it catches only 6%.
- The two GNN rows come from different studies (width 64 vs 128) on the same
  split. The confirmation study comparing them directly is in progress.

### RQ2: dimensionality (source: `elf_void_analysis/case_studies/registry_test/summary.json`)

- 260 known electrides scored (167 0D, 62 1D, 29 2D, 2 3D); 28 with
  conflicting labels and the 3 used to pick the settings were left out.
- Ionic radii + 1.0 Å probe: **57% balanced accuracy** (chance 33%), 62% exact
  (always guessing 0D gives 64% exact).
- Recall: 0D 65%, 1D 65%, 2D 41%.
- Same size for every atom (2.0 Å): 45% balanced, 38% exact.
- The literature dimension appears at some probe size for 99% of structures.
- Main error: 50 of 167 0D electrides come out 2D/3D.
- Tuning the probe on half and testing on the other half: 68% ± 3% exact.
- Only 9 of the 260 are experimentally confirmed.

### Data

- ELFNet: 50,000 DFT structures with ELF; family split 40,031 / 4,969 / 5,000.
- Electride registry: `data/electrides/final_dataset/inorganic_electride_registry_matched.csv`,
  mostly from JACS 2023 (magnetic electrides), Matter 2019, Chem. Mater. 2018.

## Text drafts (as on the poster)

**Background**
- Electrides are crystals whose extra electrons sit in the empty space between
  atoms and act as anions.
- How those electrons connect sets their properties: isolated 0D cages, 1D
  channels, or 2D layers.
- ELF (electron localization function, 0–1): a high value in a void marks a
  trapped electron, the electride fingerprint.
- Finding them needs a DFT calculation per crystal: too slow to screen large
  databases.

**Approach**
- Voids: Voronoi interstitials plus points farthest from atoms, expanded by symmetry.
- GNN: atoms and voids are nodes joined by distance edges; message passing
  predicts ELF at each void.
- Void network: each atom blocks a sphere (ionic radius + 1.0 Å probe); an exact
  power-diagram network counts how many directions the open space repeats
  through the crystal.

**RQ1 take-home:** Learning from the surrounding atoms predicts void ELF: test
RMSE 0.058, R² 0.83.

**RQ2 take-home:** On 260 known electrides: 57% balanced accuracy (chance 33%).

**RQ3:** Same open space can hold connected electrons (delocalized) or separate
pockets (localized 0D); ELF at the narrow necks between voids tells them apart.

**Conclusions**
- Void ELF can be predicted from structure alone (R² 0.83).
- Geometry finds electride dimensionality well above chance; its errors show
  where ELF is needed.

**Next steps**
- Run DFT on known electrides to test the GNN on real ones.
- Predict ELF at void necks → electronic dimensionality.
- Crystal-level electride score; test on unseen compositions.

**Caveat line (RQ2):** Literature labels come from computational screens (JACS
2023, Matter 2019, Chem. Mater. 2018), compiled with AI-assisted search; not
fully hand-checked.

## Talking points for questions

- Why balanced accuracy? Most electrides are 0D, so always guessing 0D scores
  64% exact but only 33% balanced.
- Why does geometry miss 0D electrides? The empty space connects, but the
  electrons stay in pockets; geometry cannot see electrons. That is why RQ3
  adds ELF at the necks.
- Why no known electrides in the GNN data? ELFNet is metal frameworks from
  compressed hydrides; testing on real electrides is the first next step.
- How exact is the void network? It uses the power (radical Voronoi) diagram,
  so the connectivity is exact for the chosen radii, with no grid resolution
  limit. Related tools: Zeo++, CAVD, ToposPro.
