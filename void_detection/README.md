# Void detection experiments (exploratory)

## Scope

Single-structure prototype using the extracted ELFNet DFT-reference example `BaH6_Imm2_200/Li_Ir` (2 atoms and a 40 x 24 x 24 ELF grid), plus analytical periodic simple-cubic, BCC, FCC, and skewed triclinic cells. The small set makes this a smoke test, not a general performance estimate. ELF maxima are not equivalent to excess electron density; without charge density, labels only describe ELF features.

## Methods and variations tested

- Periodic nearest-atom clearance sampled on a grid, then 6- or 26-neighbor maxima detection; absolute clearance cutoffs 0.0/0.5/0.8/1.0 A; duplicate-merging radii 0.3/0.55/0.8 A; grid resolutions 12^3 to 48^3; and continuous pattern-search refinement from grid seeds.
- Periodic empty-sphere centers from atom-image quadruplets (a brute-force Delaunay/Voronoi-family baseline) on small cells.
- ELF-grid local maxima after nuclear exclusion, used as a reference/upper-bound method rather than a structure-only predictor.
- Analytical SC/BCC/FCC targets; a skewed-cell stress test; and a BCC isotropic cell-scaling sweep to test whether absolute clearance cutoffs survive compression.

## Targets and metrics

For the ELFNet example, default reference targets are local ELF-grid maxima >=0.35, merged within 0.35 A, and farther than 0.8 A from nuclei. This low exploratory threshold captures its weak interstitial peaks; target count is swept over ELF thresholds 0.30–0.50 and nucleus exclusions 0.8/1.0/1.2 A. Synthetic targets are known high-symmetry void sites. Recall, one-to-one precision, candidate budgets/counts, periodic Cartesian match tolerances, and resolutions are saved in `results.json`.

## Main observations

- On the ideal SC and BCC cells, 26-neighbor clearance maxima recovered all specified sites with 1 and 12 candidates, respectively. FCC recall was complete but precision was 0.75 because 12 candidates covered 9 specified sites. At these high-symmetry settings, 6-neighbor detection produced many extra maxima (e.g. 212 vs 12 on BCC).
- On the skewed triclinic cell, grid candidate recall varied non-monotonically with resolution and missed some reference vertices. Pattern-search refinement helped at one resolution but did not behave monotonically, so the grid method still needs better skew-cell testing and robust geometric refinement.
- For the ELFNet sample, the single largest ELF peak is only 0.046 A from a nucleus. With the 0.35 threshold and 0.8 A nuclear exclusion there are five weak interstitial peaks (ELF 0.367–0.458). The 24-candidate clearance grid matched 4/5 within 1.0 A but none within 0.5 A; its top five matched 3/5 within 1.0 A. Wider merge tolerance reduced count to 6 but also reduced 1-A recall to 3/5.
- Raising ELF threshold to 0.40 leaves two targets, both matched within 1 A; at 0.50 there are no interstitial targets. ELF threshold is therefore a label choice, not an innocuous setting.
- In the BCC scaling sweep, a fixed 0.8 A cutoff removed all targets at a 1.0 A cell edge; a 1.0 A cutoff removed them at a 1.5 A edge. Preserve continuous clearance as a feature and avoid unvalidated absolute filters in compressed structures.
- Empty-sphere enumeration found candidates on these tiny cells, but degenerate point sets produce duplicate/unstable sites. The project already has `dft/cp2k/find_voids.py`; it was not executed here because this runtime lacks its SciPy/spglib dependencies and the sample is POSCAR rather than CIF. Use that established implementation after setting up its environment for broader tests.

## Limitations and next step

This only tests one ELFNet structure and four idealized geometries. A matching ELF maximum is not proof of electride character. Expand to a diverse extracted subset, include low-symmetry and pressure-related distortions, and group evaluation by parent structure to avoid leakage. Use actual charge-density data if the target is integrated excess basin charge; ELF grids alone cannot provide it.

For the atom-plus-void GNN, the early candidate-set choice should favor recall over aggressive pruning: take the union of periodic Voronoi/Delaunay sites and clearance-grid maxima, merge conservatively, and keep clearance as a continuous void-node feature. Then use spglib to collapse equivalent sites when helpful while retaining orbit multiplicity. A cache keyed only by space group is too coarse; cache only exact prototype/internal-coordinate patterns with compatible normalized cell metrics. Validate the union against ELF peaks before using its sites to define ELF basin targets.

## Reproduce

Run `python benchmark.py --sample /path/to/ELFNet/sample` with NumPy installed. The default points to `/private/tmp/elfnet_sample/...`; the `--sample` option lets you select a stable extracted path. The benchmark writes `results.json` alongside this file.
