# Poster: electride void ELF and dimensionality

Research poster, 36 × 48 in portrait, built in Inkscape in Purdue colors
(black, Aged gold #8E6F3E for Tool 1, Rush gold #DAAA00 for Tool 2).

| File | What it is |
| --- | --- |
| [`poster_v3.svg`](poster_v3.svg) | **The current poster.** Open and edit it in Inkscape. Each section is its own layer. |
| [`poster_v3.pdf`](poster_v3.pdf) | Print export of `poster_v3.svg` |
| [`poster.svg`](poster.svg), [`poster.pdf`](poster.pdf) | Earlier version, kept for reference |
| [`CONTENT.md`](CONTENT.md) | Story, every number on the poster with its source, and text drafts |
| [`TODO.md`](TODO.md) | What still needs doing before printing |
| [`figures/`](figures/) | Plots (PNG 300 dpi + PDF), linked into the poster |
| [`diagrams/`](diagrams/) | Editable Inkscape schematics, and the Blender crystal renders in `diagrams/blender/crystals/` |
| [`build/`](build/) | Scripts that produced the poster and every figure |

## Editing

Edit `poster_v3.svg` directly in Inkscape. The three schematics (basic electride,
ELF prediction, void connectivity) are pasted in as native, editable objects.
The plots and crystal renders are linked images, so keep the `figures/` and
`diagrams/` folders next to `poster.svg`.

Export the PDF after editing:

```bash
/Applications/Inkscape.app/Contents/MacOS/inkscape poster/poster_v3.svg --export-type=pdf --export-filename=poster/poster_v3.pdf
```

The schematics in `diagrams/*.svg` are copies. Editing one does not change the
poster unless you rebuild it (below) or paste the change in by hand.

## Rebuilding

`build/build_poster_v3.py` produced the first draft of `poster_v3.svg`, which has
since been edited by hand. **Do not rerun it over `poster_v3.svg`; the hand edits
would be lost.** Pass another file name to build a copy instead:

```bash
python poster/build/build_poster_v3.py /tmp/poster_rebuild.svg
```

## Where the figures come from

| File | Script | Input |
| --- | --- | --- |
| `figures/gnn_model_comparison`, `figures/gnn_parity_atom_only` | `build/make_gnn_figures.py --feature-predictions test_predictions.parquet` | GNN run results in `elf_void_gnn/` (the predictions parquet is on Negishi) |
| `figures/gnn_model_comparison` (current poster version) | `build/make_comparison_from_summary.py elf_void_gnn/architecture_test/results_voidv2/poster_summary.json poster/figures/gnn_model_comparison 4.4 6.8` | the committed summary written by `make_gnn_figures.py`, redrawn taller |
| `figures/gnn_parity_confirmation` | `build/make_parity_confirmation.py test_predictions.parquet` | Confirmation-study predictions (w64 atom-only, seed 20261028; on Negishi) |
| `figures/connectivity_flow_sweep` | `build/connectivity_flow_figure.py elf_void_analysis/case_studies/registry_test` | Output of `python -m elf_void_analysis.registry_dimensionality_test` |
| `figures/pipes_vs_electrons` | `build/make_pipes_figure.py` | none (schematic) |
| `diagrams/blender/crystals/*.png` | `build/build_structures.py GEOM_DIR`, then `Blender -b diagrams/blender/crystals/electride_crystals_v2.blend --python build/render_crystals.py -- GEOM_DIR diagrams/blender/crystals SAMPLES save 0D,1D,2D` | CIFs in `data/cifs/` (mp-721592, mp-2538, mp-2686) |
| `diagrams/*.svg` | edited by hand in Inkscape | none |

Each script writes into `figures/` (or the folder given) by default.

### Crystal renders

- **0D:** mayenite, Ca6Al7O16 (mp-721592). Cages are drawn as equal spheres, and O atoms are hidden.
- **1D:** Y5Si3 (mp-2538).
- **2D:** Ca2N (mp-2686).

The gold surfaces show geometric void space (Shannon ionic radii plus a 1.0 Å
probe), not ELF or charge density. Atom sizes are chosen for readability.
