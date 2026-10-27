# Per-void ELF architecture benchmark

## Question and rationale

This experiment compares a compact set of model classes for predicting the scalar ELF value at each detected void center. Each graph contains atom and void nodes with physical descriptors and distance-bearing atom/void relations. All learned models must use the existing graph batch, target, saved split, training statistics, optimizer, and trainer contract. The benchmark includes (1) width/depth controls for the existing heterogeneous MPNN, (2) local-feature and fixed-context MLP baselines, and (3) a distance-gated recurrent MPNN. This isolates whether atom/void message passing, recurrent gated updates, or simple void/context descriptors explain predictive performance.

The standard crystal GNN benchmark found CGCNN, MPNN, SchNet, and MEGNet broadly competitive on its regression tasks, and noted hyperparameter optimization can matter as much as the graph operator. That supports this focused, matched-budget comparison rather than an unbounded architecture search ([benchmark paper](https://www.nature.com/articles/s41524-021-00554-0)). SchNet motivates continuous distance-conditioned filters for atomistic interactions ([NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html)); ALIGNN shows explicit angle/line-graph message passing can encode geometrical information, but requires triplets not present in the current fixed batch ([npj paper](https://www.nature.com/articles/s41524-021-00650-1)). PNA motivates multiple neighborhood aggregators for continuous node features ([NeurIPS paper](https://papers.neurips.cc/paper_files/paper/2020/hash/99cad265a1768cc2dd013f0e740300ae-Abstract.html)); it is a possible later follow-up, outside this predeclared matrix.

Descriptor-based models are meaningful controls because materials models using physically meaningful features and feature selection can be competitive, especially when labeled data are limited ([MODNet paper](https://www.nature.com/articles/s41524-021-00552-2)). In this benchmark `void_mlp` and `fixed_context_mlp` use the same saved graph inputs, with the former using only void-node features and the latter adding fixed, distance-weighted neighboring atom summaries. They do not receive labels or target-derived features in their inputs.

The matrix also contains one deterministic CPU `HistGradientBoostingRegressor` comparator over exactly the same 26 input fields as `fixed_context_mlp`: nine void features, the distance-weighted mean of 14 neighboring-atom features, weighted distance mean and standard deviation, and the log sum of distance weights. This gives a direct tree-versus-neural comparison without adding graph-level descriptors. It uses the same saved split and a predeclared 200-iteration budget; the neural-network Adam optimizer/epoch budget does not apply to this tree estimator. Its single run is a reference baseline, not a replicated seed comparison. HistGradientBoosting consumes raw values while the MLP standardizes fields; this is a numeric preprocessing difference, not additional input information.

`train_architectures.py` is a standalone copy of the repository trainer at the benchmark's creation point. The array runner uses it directly so this benchmark keeps checkpoint-by-validation-RMSE and final test prediction behavior even if the shared trainer later changes for another study.

## Fixed protocol

- The split assignment in the existing dataset is immutable. Do not resample, rebalance, or regenerate it per model. Every model and seed gets identical train/validation/test material assignments.
- Use three predeclared seeds per configuration, the same batch size 8, Adam learning rate `1e-3`, weight decay `1e-5`, 30-epoch maximum, patience 5, and checkpoint selection by validation pooled RMSE. No model-specific tuning is permitted in this matrix.
- Scaling is fit on training rows only by the shared trainer. The target remains excluded from model inputs and input statistics.
- Record pooled void-level MAE, RMSE, R² and macro-family MAE, RMSE, R². Also report per-family metrics and the shared training-mean baseline. Pooled metrics weight crystals in proportion to their number of void targets; macro-family metrics give each family equal weight. Scikit-learn defines MAE/RMSE/R² and notes R² may be negative, so do not use R² alone ([metrics guide](https://scikit-learn.org/stable/modules/model_evaluation.html)).
- The saved split holds out exact named phase folders, but related pressure or space-group variants can occur on both sides. This estimates generalization to held-out named phases under the current split, not strict unseen-composition or unseen-pressure generalization. The unit of splitting should remain the crystal/family because void rows from one graph are dependent; grouped cross-validation is the standard way to keep dependent groups together ([scikit-learn CV guide](https://scikit-learn.org/stable/modules/cross_validation.html)).
- Keep test outputs for the fixed matrix as a final descriptive comparison only. Architecture choices, hyperparameter decisions, and any follow-up shortlist must be based on validation results or a new independently designed evaluation. Do not sort, rank, select, or declare a test-set winner in the report. Matbench uses nested CV to separate model selection from performance estimation ([npj paper](https://www.nature.com/articles/s41524-020-00406-3)).

## Predeclared matrix

| Model family | Settings | Seeds |
| --- | --- | --- |
| `hetero_mpnn` | width 32/64/128 at depth 2; width 64 at depth 1/3 | 20261023, 20261024, 20261025 |
| `void_mlp` | width 64, depth 2 | same |
| `fixed_context_mlp` | width 64, depth 2 | same |
| `gated_mpnn` | width 64, depth 2 | same |
| `tree_baseline` | `HistGradientBoostingRegressor`, 200 boosting iterations | 20261023 only |

This is 8 neural configurations × 3 seeds plus one tree baseline = 25 runs. Run directories should be outside the repository (default remote location: `/home/weilana/elfnet-runs/elf_void_gnn/architecture-test`). The generated manifest freezes all run identifiers, model settings, seeds, and paths before submission.

The held-out set contains nine families. Plots show individual predeclared runs; the three-neural-seed spread is initialization variability on this one fixed split, not variability across independent test splits. The report prominently shows macro-family metrics and paired per-family test-error differences against the same-seed width-64/depth-2 hetero MPNN. These are descriptive results for the fixed split and do not establish generalization to new compositions.

## Use

Install `architecture_test/requirements.txt` into `/home/weilana/elfnet-runs/architecture_test/python-deps` on the compute host; it contains scikit-learn for the tree model and pandas/matplotlib for reporting. Then submit the predeclared jobs with `bash architecture_test/submit_sweep.sh`. It submits neural runs 0–23 as a GPU array with a maximum concurrency of three and the tree baseline as a separate CPU job. GNN jobs have a two-hour time limit; the tree baseline has a two-hour CPU limit. Configs, manifest, run artifacts, and logs are written outside the repository under `EXPERIMENT_ROOT` (which defaults to `/home/weilana/elfnet-runs/elf_void_gnn/architecture-test`). After runs finish, call `/home/weilana/.conda/envs/elfnet-train/bin/python architecture_test/report.py --manifest <experiment-root>/configs/manifest.csv --out-dir <experiment-root>/report`. The report includes validation and held-out test results for all completed runs, with the test split clearly labeled and never used to rank or select a model.

## Compute expectations

The MLP baselines avoid graph message passing and should be cheaper than the heterogeneous models. Width/depth growth increases MPNN parameter count and compute; the depth-3 model also propagates atom states for more layers. Gated recurrent updates add gate/GRU computation, so expect higher per-step cost than the standard relation MPNN. Measure wall time and peak GPU memory from the trainer output rather than assuming equal cost. No model is run by this scaffolding.
