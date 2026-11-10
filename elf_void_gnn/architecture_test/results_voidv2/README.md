# Architecture-test reruns on v2 void features

Reruns of the architecture-test baselines on the dataset with corrected void
descriptors (`elfnet-gnn-2026-11-10-voidv2`, written by
`elf_void_analysis/recompute_void_features.py`; same graphs, targets and
split as `elfnet-gnn-2026-10-23`). Same configs, seeds (20261023–25) and
trainer as `../results/`.

| Model | Test RMSE, v2 | Test RMSE, v1 |
| --- | ---: | ---: |
| Boosted decision trees (1 run) | 0.0763 | 0.077 |
| Fixed-context MLP | 0.0827 ± 0.0020 | 0.085 ± 0.002 |
| Void-only MLP | 0.1393 ± 0.0006 | 0.140 |

`run_metrics.csv` is the `report.py` output for these runs;
`poster_summary.json` is what `poster/build/make_gnn_figures.py` wrote for
the poster chart. Runs and checkpoints are on Negishi under
`/home/weilana/elfnet-runs/voidv2-2026-11-10/architecture-test/`.

Fixing the void descriptors barely changed these baselines: void geometry
alone still predicts almost nothing (void-only R² ≈ 0.02).
