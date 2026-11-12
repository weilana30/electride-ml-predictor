# TODO

Open items from the November 2026 code review of void detection, the GNN and
its studies, and the connectivity workflow. Already fixed (commit 96889b7):
cell-dependent void features, leaky nearest-distance ablations, the hydride
radius, and the slow periodic edge search. Of the reruns those fixes need,
only the poster's baseline bars were redone (see "Reruns still owed").

## Reruns still owed after the void-feature fix

The fixed dataset is on Negishi at
`/home/weilana/elfnet-datasets/elfnet-gnn-2026-11-10-voidv2` (same graphs,
targets and split; only `voids/features` changed). Run trees and job scripts
for every rerun are ready under `/home/weilana/elfnet-runs/voidv2-2026-11-10/`
(`packed_commands.txt` lists all 56 trainings; `jobs.txt` logs submissions).

- [ ] **Feature study** (Stages A–C, final test, correlations): every result
  that uses void descriptors is from the broken v1 features, including
  "atom-only beats baseline" and the derived-feature variants. The
  nearest-distance ablations also need rerunning with the corrected masks.
  Re-freeze the shortlist from new validation results before any test run.
- [ ] **Architecture test**: rerun so far: the fixed-context MLP, void-only
  MLP and tree (poster bars), plus the all-feature hetero GNN at widths 64
  and 128. The w32, d1, d3 and gated-MPNN configurations still use v1
  features, so the architecture choice is untested.
- [x] **Confirmation study**: all-descriptor GNNs rerun on v2 at widths 64
  and 128 with the original seeds (`confirmation_study/results_voidv2/`).
  The per-family failures are gone; all-descriptor wins on validation but
  atom-only still wins on test. Atom-only runs needed no rerun.
- [ ] Update `RESULTS.md` / `DETAILED_RESULTS.md` / `confirmation_report.md`
  after the reruns, and note in each that earlier numbers used v1 features.

- [ ] `elf_void_gnn/poster_figures/` still holds the v1 comparison figure
  and numbers from the first poster; the current figure is
  `poster/figures/gnn_model_comparison` (from `poster/build/make_gnn_figures.py`).
  Regenerate or remove the old copies.

## Evaluation protocol

- [ ] **Test set was reused across studies.** The architecture test scored
  test in all runs, the feature study reused those runs, and the
  confirmation study evaluated the same 9 test families. Make
  `validation-only` the default in `train.py` and hold out new families (or
  use nested family CV) for any final generalization claim.
- [ ] **Confirmation report bug:** `comparison_readout` in
  `confirmation_study/report.py` reads only validation comparisons, so the
  "Test interpretation" paragraph repeats validation numbers (it prints
  +0.000684; the test delta is −0.0098). Pass the split in and regenerate.
- [ ] **Coarse split:** only 9 validation and 9 test families, and related
  composition/pressure variants cross splits. Report per-family spread and
  consider grouping by prototype.
- [ ] **Pooled metrics over-weight symmetry copies and near-duplicates.**
  Add per-orbit or per-structure weighted metrics.

## GNN training and model

- [ ] **Batches come from one family.** Shards follow the path-sorted
  manifest (140 of 196 shards hold one family) and `ShardShuffleSampler`
  draws each batch from one shard. Shuffle globally or interleave shards.
- [ ] **Mean aggregation with a hard 6 Å cutoff** dilutes the nearest atoms
  (~37 neighbours per void) and makes outputs jump at the cutoff. Try sum or
  attention aggregation with a smooth cosine envelope.
- [ ] **Distance-only messages** (no angles or directions). Consider an
  angular or equivariant model.
- [ ] **Shared trainer hard-codes `hetero_mpnn` and 80,001 parameters**
  (`train.py`), contrary to the README's generic registry; move those checks
  into the feature-study entry point.
- [ ] Output is unbounded although ELF is in [0, 1] (optional sigmoid).
- [ ] No DataLoader worker seeding; `best_model.pt` is never written if every
  validation RMSE is NaN.
- [ ] Derived-feature variants normalize only the first 5 atom channels;
  `_edge_context` degree features change with the supercell.

## Void detection and targets

- [ ] **Near-duplicate voids:** 8 of 40 sampled structures have void pairs
  closer than 0.5 Å (0.2 Å merge tolerance; Nelder–Mead stops on flat
  ridges). Merge at ~0.5 Å or by refined clearance.
- [ ] **Fixed absolute finder thresholds** (grid peaks ≥ 1.5 Å, Voronoi
  min_dist 0.9 Å) on compressed cells; make them relative or keep all
  candidates with clearance as a feature.
- [ ] When candidates merge, the first candidate's position is kept but the
  maximum clearance is stored (≤ 0.0055 Å mismatch). Keep the coordinates
  of the highest-clearance candidate.
- [ ] **`elf_at_center` is sensitive to void position**; finder refinement
  error becomes label noise. Consider the local maximum within 0.5 Å or a
  small-sphere average as the target.
- [ ] Local ELF maxima are not masked against atomic basins (documented).
- [ ] Guard against spin-polarized ELFCARs (pymatgen's "total" is spin-up
  only); ELFNet is unaffected (0 of 400 sampled files).
- [ ] Unreachable symmetry fallback in `_expand_candidate_orbits`.
- [ ] `validate_dataset.py` checks structure only, not values (ELF range,
  feature consistency).

## Dataset: `void_dimension` label (not used by the GNN or the poster)

`build_dataset.py` stores a per-void `void_dimension` from
`label_void_dimensionality.py`; nothing reads it.

- With the bare 1.0 Å cutoff from atom centers
  (`void_dimension_probe_radius_angstrom`), every sampled ELFNet void is 3D.
- The grid tool reads the grid point at the lower corner of the void's cell
  (`np.floor`, `label_void_dimensionality.py:161`), up to ~0.9 Å away; it
  disagrees with the exact power-diagram method on 12–22% of test points.

Choose one:

- [ ] **Remove:** stop computing `void_dimension` / `void_dimension_valid`
  in `build_dataset.py`, and update `schema.json`, `validate_dataset.py`
  and the README.
- [ ] **Fix:** relabel with `power_void_connectivity.py` at settings that
  matched the literature (2.0 Å uniform, or ionic radius + 1.0 Å), store
  the probe settings, and consider several probes.

## Connectivity (Tool 2)

- [ ] **Decide the hydrogen radius.** The fix uses 1.40 Å (ionic H⁻), but the
  registry hydrides are mostly metallic (CrH, VH, TiH, YH2, ...). At 1.40 Å
  + 1.0 Å probe, 12 of 16 scored hydrides have no open space, and the
  default balanced accuracy drops from 57% to 55%. The committed
  `case_studies/registry_test/` results and the poster Sankey still use
  0.25 Å, while the code now defaults to 1.40 Å; rerun or revert once decided.
- [ ] Noble gases crash `element_radii` (`float(None)` atomic radius).
- [ ] **Registry scoring** uses the highest-dimension open region, not the
  region holding the electrons (not fixable without ELF; that is RQ3).
- [ ] **Poster Sankey rule is lenient:** a structure counts as "Channels" if
  1D appears at any probe in the sweep. State it, or show a fixed probe.

## Scope and housekeeping

- [ ] **Domain gap:** ELFNet has no known electrides (metal frameworks from
  compressed hydrides). Run DFT on known electrides to test the GNN.
- [ ] `pytest elf_void_analysis` fails to collect: `registry_dimensionality_test.py`
  matches `*_test.py` and the folder has no `__init__.py`. Rename the file
  and add a package init.
- [ ] `architecture_test/train_architectures.py` default `--config` path
  does not exist.
- [ ] `dft/cp2k/find_voids.py` (not used by the GNN) uses only ±1 images for
  shortest distances, which is not exact for skewed cells.
- [ ] Checkout sync: the `Documents/ChatGPT/...` checkout is behind `main`
  with uncommitted poster edits, and the `Andrews_Lab_Research` main
  checkout has uncommitted figure-script, `CONTENT.md` and registry-test
  changes.
