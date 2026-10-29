# Direct confirmation study: frozen plan

This study reuses the 50,000 processed structures and the **existing family-safe `splits.parquet`** from the architecture and feature studies. Do not generate or substitute a new split. The persistent dataset is `/home/weilana/elfnet-datasets/elfnet-gnn-2026-10-23/`, with build fingerprint `3928d5a96a2edf614c7b0f5fe93b70f11da6e7f8c312c208e1af6820e814e428`. The original split SHA256 is `cdb3fc87b80cb05ce1cfdd8b94267696cfb0bb9e810004512144b0c2f2aac7f4` and must be checked at execution.

The existing partition contains 40,031 train samples in 37 families, 4,969 validation samples in nine families, and 5,000 test samples in nine families. It separates `family_id` across splits, but related compositions and pressure or space-group variants may cross splits. This is a direct repeat on the same corpus and partition with new seeds, not an independent or composition-held-out confirmation. Prior architecture and feature studies have already used this test partition; report that history when interpreting the descriptive test results.

## Predeclared matrix

| Configuration | Width | Depth | Feature representation | Expected parameters |
| --- | ---: | ---: | --- | ---: |
| `w128_d2_baseline` | 128 | 2 | Original baseline | 307,457 |
| `w128_d2_atom_only` | 128 | 2 | Full 14-channel atom input; nine explicit void descriptors zeroed; topology and eight radial edge channels retained | 307,457 |
| `w64_d2_atom_only` | 64 | 2 | Same atom-only representation | 80,001 |

Run all nine configuration/seed combinations, using seeds `20261028`, `20261029`, and `20261030`. Keep the optimizer, learning rate, weight decay, batch size, maximum epochs, patience, target scaling, dataset, and split identical across runs. Fit normalization only on existing train samples; select checkpoints using pooled validation RMSE. The frozen settings are in `sweep_spec.yaml`.

## Evaluation order

1. Verify the persistent dataset fingerprint, existing split hash, sample coverage, and family isolation. Freeze the three configurations and three seeds before training.
2. Train all nine runs in validation-only mode. Do not construct the test dataset or write test metrics at this stage.
3. Aggregate validation results. The predeclared selection rule is lowest **mean pooled validation RMSE across the three new seeds**. Record the selected configuration and freeze the rule before this study's test evaluations.
4. Load each validation-selected checkpoint and evaluate the existing test partition once per run. Test results are descriptive; they cannot revise selection, features, architecture, seeds, hyperparameters, or the split.
5. Report pooled MAE/RMSE/R², macro-family metrics, per-family errors, seed variation, paired family deltas, 95% bootstrap intervals resampling families, parameter counts, and training cost. Describe the three comparisons: atom-only versus baseline at width 128; width 128 versus 64 under atom-only features; and the combined width/feature result against both comparators.

The external run root is `/home/weilana/elfnet-runs/elf_void_gnn/confirmation-study-2026-10-28/`. Keep configs, logs, predictions, checkpoints, and the full report there. Commit only lightweight code, specifications, hashes, aggregate tables, plots, and a short result summary.
