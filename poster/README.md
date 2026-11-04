# Poster: electride void ELF and dimensionality

Working folder for the research poster (48 × 36 in, landscape, one slide).

| File | What it is |
| --- | --- |
| [`poster.pptx`](poster.pptx) | The poster itself: one 48 × 36 in slide with all sections, figures, and dashed **TO DO** boxes for missing pieces |
| [`CONTENT.md`](CONTENT.md) | Story, research questions, every number on the poster with its source, and the text drafts |
| [`LAYOUT.md`](LAYOUT.md) | Formatting plan: grid, section positions and sizes, fonts, colors, figure slots |
| [`TODO.md`](TODO.md) | What still needs doing before printing |
| [`figures/`](figures/) | Result figures copied from the analysis folders (PNG 300 dpi + PDF) |
| [`diagrams/`](diagrams/) | Schematic diagrams and the script that draws them |
| [`build_poster.js`](build_poster.js) | Script that generated the first draft (do not rerun; see below) |

## Editing

**`poster.pptx` is the master copy and is edited by hand in PowerPoint.**
`build_poster.js` only produced the first draft. Running it again overwrites
`poster.pptx` and discards every hand edit, so don't, unless you want to start
over from the original draft.

To update a figure, regenerate it (table below), copy it into `figures/` or
`diagrams/`, then in PowerPoint right-click the picture → Change Picture → From a
File, so it keeps its size and position.

```bash
python poster/diagrams/make_diagrams.py        # redraw the diagrams
```

## Where the figures come from

| Poster figure | Source | Regenerate with |
| --- | --- | --- |
| `figures/accuracy_vs_probe`, `figures/confusion_ionic_1p0` | `elf_void_analysis/case_studies/registry_test/` | `python -m elf_void_analysis.registry_dimensionality_test` |
| `figures/gnn_model_comparison`, `figures/gnn_parity_atom_only` | `elf_void_gnn/poster_figures/` | `python elf_void_gnn/poster_figures/make_figures.py` |
| `diagrams/*` | this folder | `python poster/diagrams/make_diagrams.py` |

If the source figures change, copy them into `figures/` again.
