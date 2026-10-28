# Feature-study results

The completed fixed-architecture feature study is archived on Negishi under:

```text
/home/weilana/elfnet-runs/feature-study/analysis/final-report/
```

The final report is generated from the frozen final-test artifact and does not re-evaluate test structures. The report and final prediction artifact are external because they include generated tables, figures, and predictions.

## Protocol

The study keeps the same HeteroMPNN topology, feature widths, and 80,001 trainable parameters across feature variants. Validation is used for feature selection; the frozen shortlist is evaluated on test once.

- `baseline`: original atom, void, topology, and edge-distance inputs.
- `minimal`: atomic number, nearest-atom void distances, and edge distances.
- `atom-only`: full atom descriptors and topology/edge-distance inputs while zeroing void descriptors.

All reported neural values are means and sample standard deviations across three seeds.

## Pooled results

| Split | Variant | RMSE | MAE | R² |
| --- | --- | ---: | ---: | ---: |
| Validation | Baseline | 0.06192 ± 0.0038 | 0.04249 ± 0.0047 | 0.8223 ± 0.022 |
| Validation | Minimal | 0.08082 ± 0.0010 | 0.05896 ± 0.0016 | 0.6980 ± 0.007 |
| Validation | Atom-only | **0.05859 ± 0.0006** | **0.03954 ± 0.0012** | **0.8413 ± 0.003** |
| Test | Baseline | 0.06682 ± 0.0013 | 0.04482 ± 0.0027 | 0.7740 ± 0.009 |
| Test | Minimal | 0.08172 ± 0.0024 | 0.05804 ± 0.0023 | 0.6619 ± 0.020 |
| Test | Atom-only | **0.05759 ± 0.0021** | **0.03771 ± 0.0014** | **0.8320 ± 0.012** |

## Interpretation

The atom-only variant was the strongest frozen feature variant under this study's validation criterion and remained strongest in the final descriptive test evaluation. The minimal anchor was substantially worse, showing that the retained atom descriptors carry important signal. These are feature-association results on the current family split, not proof that individual correlated descriptors are causally necessary.

## Caveats and external artifacts

Seed spread is optimization variability on one split, not a confidence interval. The split holds out named phase families, but related composition or pressure variants may cross splits. The external final-report directory contains `pooled_metrics.csv`, `test_family_metrics.csv`, ablation tables, correlation tables, figures, and the frozen test prediction artifact.
