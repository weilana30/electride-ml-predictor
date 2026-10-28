# ELF void architecture benchmark results

## Outcome

All 25 predeclared runs completed successfully: eight neural configurations with three seeds each, plus one histogram-gradient-boosted tree baseline. Every run has finite metrics and a nonempty held-out prediction file.

The strongest configuration by the prespecified selection metric, mean pooled validation RMSE, was the two-layer heterogeneous MPNN with width 128:

- validation pooled RMSE: **0.06056 ± 0.00103**
- test pooled RMSE: **0.06524 ± 0.00070**
- test pooled MAE: **0.04307 ± 0.00049**
- test macro-family RMSE: **0.09080 ± 0.00295**

The three-layer width-64 MPNN had the lowest descriptive mean test pooled RMSE (0.06430) and test macro-family RMSE (0.08478), but it was not the validation winner and its family-level seed variability was much larger. It should not replace the validation-selected width-128 model based on this test comparison.

## Architecture comparison

Values are mean ± sample standard deviation over three seeds, except the tree baseline, which has one run.

| Configuration | Validation pooled RMSE | Test pooled RMSE | Test macro-family RMSE |
| --- | ---: | ---: | ---: |
| Hetero MPNN, width 32, depth 2 | 0.06486 ± 0.00114 | 0.07164 ± 0.00149 | 0.09978 ± 0.00278 |
| Hetero MPNN, width 64, depth 2 | 0.06192 ± 0.00384 | 0.06682 ± 0.00128 | 0.09494 ± 0.00491 |
| **Hetero MPNN, width 128, depth 2** | **0.06056 ± 0.00103** | **0.06524 ± 0.00070** | **0.09080 ± 0.00295** |
| Hetero MPNN, width 64, depth 1 | 0.06110 ± 0.00102 | 0.06615 ± 0.00325 | 0.09166 ± 0.00422 |
| Hetero MPNN, width 64, depth 3 | 0.06152 ± 0.00250 | 0.06430 ± 0.00321 | 0.08478 ± 0.01143 |
| Void-only MLP | 0.13405 ± 0.00043 | 0.13997 ± 0.00044 | 0.16671 ± 0.00112 |
| Fixed-context MLP | 0.08228 ± 0.00339 | 0.08478 ± 0.00200 | 0.11889 ± 0.00083 |
| Gated MPNN | 0.06250 ± 0.00130 | 0.06979 ± 0.00316 | 0.09466 ± 0.00258 |
| Histogram gradient-boosted tree | 0.07250 | 0.07662 | 0.09182 |

The results support three practical conclusions:

1. Learned atom/void message passing materially improves over void-only and fixed-context baselines.
2. Increasing width from 64 to 128 gives a small, repeatable improvement in the pooled metric.
3. Added gated recurrence did not improve on the standard heterogeneous MPNN under the matched training protocol.

The tree achieved the best single-run validation macro-family RMSE (0.07991), but its pooled validation RMSE was worse than every heterogeneous GNN and it has no seed replication.

## Evaluation scope

The fixed split contains 40,031 training samples from 37 families, 4,969 validation samples from 9 families, and 5,000 test samples from 9 families. The three neural seeds measure initialization and training-order variability on this one split; they are not independent test splits.

The split holds out exact named phase folders. Related pressure or space-group variants can occur across splits, so these results do not establish generalization to unseen compositions or pressures. Test results are descriptive; architecture selection used validation pooled RMSE.

## Reproducibility

- benchmark commit: `ad3a175599a380c170c4932a92440943823339db`
- split SHA256: `cdb3fc87b80cb05ce1cfdd8b94267696cfb0bb9e810004512144b0c2f2aac7f4`
- Slurm GPU array: `44681694` (24/24 completed, exit 0)
- Slurm tree baseline: `44681695` (completed, exit 0)

The Negishi checkout advanced to `eb7dc34` at 20:43 and `eebabc4` at 20:50 while the array was running. The benchmark trainer, model registry, variant implementations, manifest, and configurations did not change. The only model-file change added an optional `edge_features` branch to `hetero_mpnn`; this dataset does not provide that field, so execution follows the original radial-distance path. Early tasks loaded `ad3a175`; later tasks used the behavior-equivalent fallback from `eebabc4`.

## Contents

- `results/manifest.csv`: sanitized frozen run manifest
- `results/run_metrics.csv`: one row per run with aggregate metrics and runtime information
- `results/test_family_metrics_and_paired_deltas.csv`: family-level test metrics and paired differences
- `results/mean_paired_family_deltas.csv`: mean family-level differences by configuration
- `results/*.png`: selected comparison and family-difference figures
- `runs/*/metrics.json`: complete metrics for every run
- `runs/*/test_predictions.parquet`: per-void held-out predictions for every run
- `results/manifest.csv`: sanitized frozen run matrix; the complete resolved YAML archive remains in the external result package
