# Confirmation study report

## Validation-based selection

Configurations are ranked by mean pooled validation RMSE across the three predeclared seeds. The selected configuration is **w64_d2_atom_only**. No test metric enters this rule.

### Validation performance across seeds

| configuration | pooled_rmse_mean | pooled_rmse_seed_sd | pooled_mae_mean | pooled_r2_mean | macro_family_rmse_mean | macro_family_mae_mean | parameters | training_seconds_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w128_d2_atom_only | 0.0636 | 0.0013 | 0.0441 | 0.813 | 0.0636 | 0.0445 | 307457 | 2790.2182 |
| w128_d2_baseline | 0.0629 | 0.0022 | 0.0436 | 0.8169 | 0.0947 | 0.0758 | 307457 | 2252.6474 |
| w64_d2_atom_only | 0.06 | 0.0008 | 0.0412 | 0.8335 | 0.0615 | 0.0435 | 80001 | 2830.5868 |

## Study specification

- Dataset path: /home/USER/elfnet-datasets/elfnet-gnn-2026-10-23
- Dataset build fingerprint: 3928d5a96a2edf614c7b0f5fe93b70f11da6e7f8c312c208e1af6820e814e428
- Existing family split path and SHA256: /home/USER/elfnet-datasets/elfnet-gnn-2026-10-23/splits.parquet; cdb3fc87b80cb05ce1cfdd8b94267696cfb0bb9e810004512144b0c2f2aac7f4
- Grouping definition: existing family_id split; no family overlaps splits
- Seeds: 20261028, 20261029, 20261030; configurations: w128_d2_baseline, w128_d2_atom_only, w64_d2_atom_only
- Selection rule: minimum mean pooled validation RMSE across three seeds.
- Parameter and cost figures are per-run trainable parameter count and recorded training seconds.

### Paired validation comparisons

RMSE deltas are model minus comparator; negative values favor the first model. Intervals resample families.

| model | comparator | mean_pooled_rmse_delta | mean_macro_family_rmse_delta | mean_pooled_mae_delta | family_rmse_delta | ci95_low | ci95_high | seed_sd_pooled_rmse_delta | parameter_count_model | parameter_count_comparator | mean_training_seconds_model | mean_training_seconds_comparator | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w128_d2_atom_only | w128_d2_baseline | 0.0007 | -0.0311 | 0.0005 | -0.0311 | -0.0746 | 0.0012 | 0.0032 | 307457 | 307457 | 2790.2182 | 2252.6474 | validation comparison |
| w128_d2_atom_only | w64_d2_atom_only | 0.0036 | 0.0021 | 0.0028 | 0.0021 | 0.0005 | 0.0037 | 0.0006 | 307457 | 80001 | 2790.2182 | 2830.5868 | validation comparison |
| w64_d2_atom_only | w128_d2_baseline | -0.0029 | -0.0332 | -0.0024 | -0.0332 | -0.076 | -0.0009 | 0.0026 | 80001 | 307457 | 2830.5868 | 2252.6474 | validation comparison |

### Validation interpretation

**Atom-only versus baseline:** 2/3 seeds have lower pooled RMSE for w128_d2_atom_only; mean pooled RMSE delta is 0.000684, and the paired family-RMSE delta is -0.03114 (95% CI -0.07456 to 0.001182; the family-bootstrap interval includes zero).

**Width 128 versus width 64 with atom-only features:** 0/3 seeds have lower pooled RMSE for w128_d2_atom_only; mean pooled RMSE delta is 0.003581, and the paired family-RMSE delta is 0.002081 (95% CI 0.0004676 to 0.003669; the family-bootstrap interval is above zero).

**Combined width-64 atom-only model versus baseline:** 2/3 seeds have lower pooled RMSE for w64_d2_atom_only; mean pooled RMSE delta is -0.002897, and the paired family-RMSE delta is -0.03322 (95% CI -0.07597 to -0.0009398; the family-bootstrap interval is below zero). Assess the combined configuration against its validation pooled and macro-family errors, parameter count, and training cost before carrying it forward.

## Test-set descriptive confirmation

Test results are descriptive only and do not alter the selected configuration, features, hyperparameters, or split. All comparisons below use paired group differences; the 95% percentile intervals resample held-out family IDs, not individual void rows, after averaging per-family differences across seeds. Negative deltas favor the named model.

### Test model comparisons (descriptive only)

| model | comparator | mean_pooled_rmse_delta | mean_macro_family_rmse_delta | mean_pooled_mae_delta | family_rmse_delta | ci95_low | ci95_high | seed_sd_pooled_rmse_delta | parameter_count_model | parameter_count_comparator | mean_training_seconds_model | mean_training_seconds_comparator | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w128_d2_atom_only | w128_d2_baseline | -0.0098 | -0.0353 | -0.0071 | -0.0353 | -0.0665 | -0.0075 | 0.0022 | 307457 | 307457 | 2790.2182 | 2252.6474 | descriptive only |
| w128_d2_atom_only | w64_d2_atom_only | 0.001 | 0.0013 | 0.0006 | 0.0013 | 0.0 | 0.0029 | 0.0011 | 307457 | 80001 | 2790.2182 | 2830.5868 | descriptive only |
| w64_d2_atom_only | w128_d2_baseline | -0.0108 | -0.0366 | -0.0077 | -0.0366 | -0.0683 | -0.0112 | 0.0032 | 80001 | 307457 | 2830.5868 | 2252.6474 | descriptive only |

### Interpretation

**Atom-only effect:** 2/3 seeds have lower pooled RMSE for w128_d2_atom_only; mean pooled RMSE delta is 0.000684, and the paired family-RMSE delta is -0.03114 (95% CI -0.07456 to 0.001182; the family-bootstrap interval includes zero). A negative delta indicates an improvement.

**Width effect:** 0/3 seeds have lower pooled RMSE for w128_d2_atom_only; mean pooled RMSE delta is 0.003581, and the paired family-RMSE delta is 0.002081 (95% CI 0.0004676 to 0.003669; the family-bootstrap interval is above zero). This compares width 128 with width 64 at atom-only features.

**Combined model:** 2/3 seeds have lower pooled RMSE for w64_d2_atom_only; mean pooled RMSE delta is -0.002897, and the paired family-RMSE delta is -0.03322 (95% CI -0.07597 to -0.0009398; the family-bootstrap interval is below zero). This compares the width-64 atom-only model with baseline. Carry it forward only if validation pooled and macro-family errors support the gain at its recorded parameter and training cost; the test rows provide descriptive confirmation only.

### Limitations

This confirmation reuses the same underlying corpus and existing family-based split, so it is not an independent external dataset or a new split. Any prior exposure of the test split during earlier model development limits the strength of the test confirmation.

## Results files

See `run_metrics.csv`, `validation_group_metrics.csv`, `test_group_metrics.csv` (after final evaluation), `paired_group_deltas.csv`, `seed_summary.csv`, `model_comparison.csv`, and `split_summary.csv`.
