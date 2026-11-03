# Power-diagram VESTA case studies

Exact void connectivity from `power_void_connectivity.py` for five
experimentally confirmed electrides in
`data/electrides/final_dataset/inorganic_electride_registry_matched.csv`.
Each structure was run with two settings:

- `uniform_2p0/`: `--radius-scheme none --probe-radius 2.0` (every atom blocks 2.0 Å)
- `ionic_1p0/`: `--radius-scheme ionic --probe-radius 1.0` (Shannon crystal radius + 1.0 Å)

Both used `--export-spacing 0.15`. The literature dimension is electronic;
these labels are geometric. The probe settings were chosen on three other
case studies (K4Al3(SiO4)3, Ba3CrN3, Ca2N), so Ca2N is not independent.

| Material | Literature | Uniform 2.0 Å | Ionic + 1.0 Å | Open this file |
| --- | --- | --- | --- | --- |
| Ca2N (mp-2686) | 2D | one 2D region | one 2D region | `region-000-2D.xsf` |
| Y2C (mp-1334) | 2D | one 2D region | one 2D region | `region-000-2D.xsf` |
| Sr3CrN3 (mp-12906) | 1D | 1D channel + two 0D pockets | 1D channel + four tiny 0D pockets | `region-*-1D.xsf` |
| Y5Si3 (mp-2538) | 1D | 1D channel + 48 0D pockets | 1D channel + 12 0D pockets | `region-*-1D.xsf` |
| Ca6Al7O16 (mp-721592) | 0D | six separate 0D cages | six separate 0D cages | any `region-*-0D.xsf` |

All ten runs contain a region with the literature dimension, and none has a
higher-dimensional open region. Every grid component mapped to a single
exact region, and no open grid point was left unplaced.

## Viewing in VESTA

Open an `.xsf` file, then set the `accessible_space` isosurface level to 0.5.
`open_space.xsf` shows all open space together; each `region-NNN-<dim>.xsf`
shows one exact region. Region labels come from the exact calculation; the
0.15 Å grid only draws them. Regions smaller than the grid have
`xsf_file: null` in `report.json` (four Sr3CrN3 pockets in `ionic_1p0`, twelve
Y5Si3 pockets in `uniform_2p0`).

## Caveats

- Y2C and Y5Si3 have no Shannon radius for C4- or Si4-, so `ionic_1p0` used
  empirical atomic radii (C 0.70 Å, Si 1.10 Å); see `element_radius_sources`.
- Y uses its Y3+ radius (1.04 Å), although these compounds have excess
  electrons.
- Small 0D pockets next to a 1D channel are separate geometric regions, not
  claims about where electrons sit.

Reproduce one run from the repository root:

```bash
python -m elf_void_analysis.power_void_connectivity data/cifs/mp-2538.cif \
  --output /path/outside/repository/Y5Si3_ionic \
  --radius-scheme ionic --probe-radius 1.0 --export-spacing 0.15
```
