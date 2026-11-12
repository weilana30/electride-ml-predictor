# Direct confirmation study results

Nine runs: three predeclared configurations × seeds 20261028, 20261029 and
20261030, on the existing family split (`splits.parquet`, SHA256
`cdb3fc87…aac7f4`; dataset fingerprint `3928d5a9…14e428`, both checked
before the report). Plan: [`../PLAN.md`](../PLAN.md). Full report, logs,
checkpoints and predictions are in
`/home/USER/elfnet-runs/elf_void_gnn/confirmation-study-2026-10-28/`.

- Validation-only training: Slurm array `44756430`.
- Validation report run and selection frozen 2026-11-03
  (`validation_stage.json`, plus `validation_selection_frozen.json` in each
  run directory).
- Final test, run once per checkpoint: Slurm array `45021833` (9/9
  COMPLETED, exit 0, no errors in logs).

## Selected configuration

**`w64_d2_atom_only`** (width 64, depth 2, explicit void descriptors zeroed,
80,001 parameters). It has the lowest mean pooled validation RMSE across the
three seeds. The rule was fixed before any test evaluation in this study.

## Pooled and macro-family metrics (mean ± seed SD, n = 3)

Validation (4,969 samples, 80,750 voids, 9 families):

| Configuration | Params | RMSE | MAE | R² | Macro-family RMSE | Macro-family MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `w128_d2_baseline` | 307,457 | 0.0629 ± 0.0022 | 0.0436 ± 0.0009 | 0.817 ± 0.013 | 0.0947 ± 0.0068 | 0.0758 ± 0.0062 |
| `w128_d2_atom_only` | 307,457 | 0.0636 ± 0.0013 | 0.0441 ± 0.0018 | 0.813 ± 0.007 | 0.0636 ± 0.0034 | 0.0445 ± 0.0034 |
| **`w64_d2_atom_only`** | 80,001 | **0.0600 ± 0.0008** | **0.0412 ± 0.0012** | **0.834 ± 0.005** | **0.0615 ± 0.0046** | **0.0435 ± 0.0045** |

Test, descriptive only (5,000 samples, 88,680 voids, 9 families):

| Configuration | RMSE | MAE | R² | Macro-family RMSE | Macro-family MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `w128_d2_baseline` | 0.0701 ± 0.0011 | 0.0469 ± 0.0005 | 0.751 ± 0.008 | 0.1004 ± 0.0040 | 0.0757 ± 0.0047 |
| `w128_d2_atom_only` | 0.0603 ± 0.0017 | 0.0398 ± 0.0012 | 0.816 ± 0.011 | 0.0651 ± 0.0023 | 0.0448 ± 0.0018 |
| **`w64_d2_atom_only`** | **0.0593 ± 0.0028** | **0.0392 ± 0.0020** | **0.822 ± 0.017** | **0.0638 ± 0.0021** | **0.0440 ± 0.0026** |

Mean training time per run: 2,253 s (w128 baseline), 2,790 s (w128
atom-only), 2,831 s (w64 atom-only).

## Paired comparisons

Each delta is model minus comparator, so a negative value favors the model.
The family-RMSE delta averages per-family differences across seeds, and the
95% interval comes from a bootstrap over the nine held-out families.

| Comparison | Split | Seeds favoring model | Pooled RMSE Δ | Family RMSE Δ [95% CI] |
| --- | --- | ---: | ---: | --- |
| Atom-only vs baseline (w128) | validation | 2/3 | +0.0007 | −0.031 [−0.075, +0.001] |
| | test | 3/3 | −0.0098 | −0.035 [−0.066, −0.007] |
| w128 vs w64 (atom-only) | validation | 0/3 | +0.0036 | +0.0021 [+0.0005, +0.0037] |
| | test | 0/3 | +0.0010 | +0.0013 [+0.00004, +0.0029] |
| w64 atom-only vs w128 baseline | validation | 2/3 | −0.0029 | −0.033 [−0.076, −0.001] |
| | test | 3/3 | −0.0108 | −0.037 [−0.068, −0.011] |

- **Feature effect.** Zeroing the explicit void descriptors makes little
  difference to pooled validation RMSE. It sharply lowers per-family error:
  the baseline's macro-family RMSE is about 0.095–0.100, against about 0.064
  for atom-only. On test, atom-only is better on every seed, and the family
  interval excludes zero.
- **Width effect.** Under atom-only features, width 64 matches or beats
  width 128 on every seed, at a quarter of the parameters. The gap is small
  (≈0.001–0.004 RMSE), but the family interval is above zero on both splits.
- **Combined.** The selected w64 atom-only model beats the w128 baseline on
  both splits. Its family interval is below zero on both.

## Comparison with the feature study

In the feature study, atom-only at width 64 reached a test RMSE of
0.0576 ± 0.0021 (R² 0.832); this is the poster's current number. Here the
same configuration, with new seeds, gives 0.0593 ± 0.0028 (R² 0.822). That
is about 0.0017 higher, within roughly one seed SD. Both studies rank the
same configuration first.

## Caveats

- This is a repeat on the same corpus and the **same family split** as the
  architecture and feature studies. It is not an independent or
  composition-held-out confirmation. Related compositions and pressure or
  space-group variants can cross splits.
- The architecture and feature studies **already used this test partition**,
  so these test numbers are descriptive, not an unbiased estimate.
- Only named-phase families are held out. Nine validation and nine test
  families make the family-bootstrap intervals wide and coarse.
- `confirmation_report.md` has a known defect. Its test-section
  "Interpretation" paragraphs repeat the *validation* readouts, because
  `report.py` passes validation comparisons to `comparison_readout`. The
  test tables in that file and in this README are correct. The bootstrap is
  also unseeded, so interval endpoints shift slightly (≈1e-4) between
  report runs.

## Files

Copied from the external `report/` directory:
`confirmation_report.md`, `selection_summary.csv`, `run_metrics.csv`,
`model_comparison.csv`, `paired_comparison_summary.csv`,
`validation_group_metrics.csv`, `test_group_metrics.csv`,
`split_summary.csv`, `validation_stage.json`, `validation_rmse.png`,
`test_rmse.png` and `paired_group_deltas.png`. The test prediction
scatter (`prediction_diagnostics.png`) and the per-family delta table
(`paired_group_deltas.csv`) remain in the external report directory only.
