# Poster layout plan

## Format

- **Size:** 48 in wide × 36 in tall (4 ft × 3 ft, the maximum allowed), landscape, one slide.
- **Margins:** 0.75 in on all sides; 0.6 in between cards.
- **Reading order:** title band, then three columns read top to bottom, left to right.

## Grid

```text
┌──────────────────────────── Title band (0–4.9 in) ─────────────────────┬──────┬──────┐
│ Title · subtitle · authors                                             │ logo │  QR  │
├──────────────────────┬────────────────────────────┬─────────────────────┴──────┴──────┤
│ Column 1 (13.5 in)   │ Column 2 (16.2 in)         │ Column 3 (15.6 in)                │
│                      │                            │                                   │
│ Background   13.9 in │ Approach           10.7 in │ RQ2 · Dimensionality      15.4 in │
│  bullets             │  pipeline diagram          │  take-home line                   │
│  concept sketch      │  3 method bullets          │  accuracy vs probe                │
│  3 VESTA renders     │                            │  confusion table + bullets        │
│                      │ RQ1 · GNN results  18.4 in │  caveat line                      │
│ Research Qs   6.4 in │  take-home line            │                                   │
│                      │  model comparison chart    │ RQ3 · Putting together     7.4 in │
│ Data          4.4 in │  parity plot + bullets     │  pipes-vs-electrons diagram       │
│                      │                            │                                   │
│ References    rest   │                            │ Conclusions + next steps    rest  │
└──────────────────────┴────────────────────────────┴───────────────────────────────────┘
```

Column x positions: 0.75, 14.85, 31.65 in. Cards start at y = 5.6 in and end
at y = 35.25 in.

## Style

| Element | Setting |
| --- | --- |
| Font | Arial throughout (safe in PowerPoint and when printing) |
| Title | 80 pt bold white on navy band |
| Subtitle | 38 pt, ice blue `CADCFC` |
| Authors | 28 pt white |
| Card headers | 40 pt bold navy `13294B` |
| Take-home lines | 28 pt bold blue `2A78D6` |
| Body | 24 pt (22 pt next to figures) |
| Captions and caveats | 18 pt italic gray `52514E` |
| Card background | light blue-gray `F1F5FA`, rounded corners, no border |
| Badges | RQ1/RQ2 blue `2A78D6` circles, RQ3 orange `EB6834` (future work) |
| TO DO boxes | white with dashed gray border, italic label |

Colors match the figures: blue = main result/series, orange = electrons and
future work, gray = baselines. Same 0D/1D/2D wording everywhere.

## Figure slots

| Slot | File | Size on poster (in) |
| --- | --- | --- |
| Background concept | `diagrams/electride_concept.png` | 4.4 × 3.5 |
| Background VESTA × 3 | **TO DO** (mayenite 0D, Y5Si3 1D, Ca2N 2D) | 4.0 × 3.65 each |
| Approach pipeline | `diagrams/pipeline.png` | 15.4 × 5.6 |
| RQ1 model comparison | `figures/gnn_model_comparison.png` | 14.0 × 7.65 |
| RQ1 parity | `figures/gnn_parity_atom_only.png` | 7.3 × 6.8 |
| RQ2 accuracy vs probe | `figures/accuracy_vs_probe.png` | 14.6 × 5.6 |
| RQ2 confusion | `figures/confusion_ionic_1p0.png` | 7.2 × 5.2 |
| RQ3 pipes vs electrons | `diagrams/pipes_vs_electrons.png` | 13.2 × 5.4 |
| Logo, QR code | **TO DO** | 3.9 × 3.5, 4.25 × 3.5 |

Diagrams are drawn at their printed size, so their font sizes are the printed
sizes. The result figures were drawn smaller and are scaled up about 2×, so
their text prints at roughly 20–30 pt.

## Rules to keep

- One take-home sentence at the top of each results card.
- Lead RQ2 with balanced accuracy, not exact accuracy.
- Keep the caveat lines (ELFNet has no known electrides; literature labels are
  computational and not hand-checked).
- No text below 18 pt.
