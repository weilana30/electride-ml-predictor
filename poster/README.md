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
| [`build_poster.js`](build_poster.js) | Rebuilds `poster.pptx` from scratch |

## Rebuilding

Edits made by hand in PowerPoint are lost if the poster is rebuilt, so pick one
way of working: either edit `poster.pptx` directly from now on, or change
`build_poster.js` and rebuild.

```bash
python poster/diagrams/make_diagrams.py        # diagrams (matplotlib)
npm install pptxgenjs@3.12.0                   # once
node poster/build_poster.js                    # poster.pptx
```

## Where the figures come from

| Poster figure | Source | Regenerate with |
| --- | --- | --- |
| `figures/accuracy_vs_probe`, `figures/confusion_ionic_1p0` | `elf_void_analysis/case_studies/registry_test/` | `python -m elf_void_analysis.registry_dimensionality_test` |
| `figures/gnn_model_comparison`, `figures/gnn_parity_atom_only` | `elf_void_gnn/poster_figures/` | `python elf_void_gnn/poster_figures/make_figures.py` |
| `diagrams/*` | this folder | `python poster/diagrams/make_diagrams.py` |

If the source figures change, copy them into `figures/` again.
