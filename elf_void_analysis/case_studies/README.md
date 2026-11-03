# Electride geometry case studies

These are exploratory outputs of `adaptive_void_connectivity.py` for three
structures already present in `data/cifs/`. The papers classify the **electronic**
electride states; this tool measures **atom-center free-space connectivity**.
They are different quantities. Each structure was analyzed with the same
2.0 Å minimum distance from an atom center. This cutoff was selected after a
short sweep, so agreement with literature here is illustrative, not validation
of a universal cutoff.

| Material | Literature | Geometric result at 2.0 Å | Files |
| --- | --- | --- | --- |
| K4Al3(SiO4)3 | [Predicted 0D electride](https://doi.org/10.1016/j.matt.2019.06.017) | Two separate certified components, each with winding rank 0 | [`K4Al3SiO4_0D/`](K4Al3SiO4_0D/) |
| Ba3CrN3 | [1D electron channel along the hexagonal axis](https://doi.org/10.1103/PhysRevLett.123.206402) | Region 2 has certified winding rank 1 | [`Ba3CrN3_1D/`](Ba3CrN3_1D/) |
| Ca2N | [Experimentally established 2D electride](https://doi.org/10.1038/nature11812) | Region 0 has certified winding rank 2 | [`Ca2N_2D/`](Ca2N_2D/) |

All three reports have `dimension: null`: uncertain boundary blocks could hide
additional connections, so the certified winding ranks are lower bounds. The
reports now include `region_uid`, `connectivity_label`, `label_status`, and
`void_ids` for grouping candidate voids by certified region. These identifiers
and labels were added to the saved reports without changing the computed
regions or XSF grids. The
`region-*.xsf` files show certified open space; `unresolved.xsf` shows those
boundary blocks. In VESTA, display the `accessible_space` isosurface at 0.5.
For the intended examples, open `region-000.xsf` or `region-001.xsf` for
K4Al3(SiO4)3, `region-002.xsf` for Ba3CrN3, and `region-000.xsf` for Ca2N.

The minimum block side-vector target was 0.5 Å for K4Al3(SiO4)3 and Ba3CrN3.
Ca2N needed 0.125 Å blocks to certify two winding directions in its thin
interlayer region. All XSF grids used 0.25 Å approximate export spacing except
Ca2N, which used 0.125 Å. Full per-void assignments, resource counts, and
settings are in each `report.json`. The source structure path in each copied
report was made repository relative; the numerical results and XSF grids were
not changed.

At a 1.0 Å atom-center cutoff, all three structures had certified rank 3.
At 2.0 Å, a separate uniform-grid analysis at 0.25 Å gave ranks 0, 1, and 2
for the literature-relevant voids. The 2D Ca2N component persisted when the
uniform grid spacing was reduced to 0.125 Å. These checks show that the cutoff
matters strongly and that a single structure can contain voids with different
geometric labels.

`dft/selected_structures.csv` labels Ba2As as 0D. The linked registry record cites an
[I4/mmm Ba2As screening result](https://doi.org/10.1021/jacs.3c00284), and
the local `mp-2768.cif` resolves to I4/mmm. A separate study discusses
[layered Ba2As among 2D electride candidates](https://doi.org/10.1103/PhysRevB.111.045120).
Composition alone therefore cannot settle that label; the existing label was
left unchanged pending a phase-matched electronic comparison.

The three CIFs are local reference structures. No exact Ca2N, Ba3CrN3, or
K4Al3(SiO4)3 phase name was found in the separate 50,000-structure GNN
manifest. These case studies do not test GNN predictions.

Reproduce one run from the repository root with, for example:

```bash
python -m elf_void_analysis.adaptive_void_connectivity data/cifs/mp-12905.cif \
  --output /path/outside/repository/Ba3CrN3_probe2p0 \
  --probe-radius 2.0 --min-block-width 0.5 --export-spacing 0.25 \
  --max-blocks 50000
```
