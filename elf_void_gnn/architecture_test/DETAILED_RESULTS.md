# ELF Void Architecture Benchmark: Deep Technical Report

## Executive summary

This study tested whether the existing heterogeneous graph neural network (GNN) was the right architectural family for predicting the electron localization function (ELF) at void centers, and whether its width, depth, or update mechanism should change. It also tested whether graph message passing was necessary at all by comparing the GNN against local multilayer perceptrons (MLPs) and a non-neural gradient-boosted tree.

The experiment completed all 25 predeclared runs: eight neural configurations with three seeds each, plus one CPU tree baseline. All runs completed successfully, all aggregate metrics were finite, and every run produced a nonempty held-out prediction file.

The principal result is that learned atom/void message passing is essential for this task. A void-only MLP had essentially no held-out explanatory power (test pooled R2 = 0.008), while the original heterogeneous GNN architecture retrained under the benchmark protocol reached 0.774. A fixed-context MLP and a boosted tree recovered some of the signal, but neither matched the strongest GNNs on the prespecified pooled validation criterion.

The recommended primary model is the two-layer heterogeneous MPNN with hidden width 128. It won the prespecified selection metric, mean pooled validation RMSE, and was stable across seeds. The width-64, depth-3 model is a promising secondary hypothesis because it had the strongest descriptive test and family-level means, but it was not the validation winner and showed substantially greater seed variability. It should be evaluated on a new grouped split rather than promoted using the already-inspected test set.

## 1. Scientific question

The target is a scalar ELF value at each detected void center. Each material is represented as a heterogeneous graph containing atom nodes, void nodes, physical node descriptors, and distance-bearing relations between node types.

The study addressed four questions:

1. Is learned graph message passing necessary, or can local void descriptors predict ELF adequately?
2. If atom context matters, is a fixed distance-weighted summary sufficient, or must atom/void interactions be learned?
3. Does the original heterogeneous MPNN benefit from greater width or greater message-passing depth?
4. Does gated recurrent updating improve on the original relation-specific MPNN?

The matrix was intentionally focused rather than an unrestricted architecture search. Published materials benchmarks show that common crystal GNNs can be broadly competitive and that hyperparameter choices can matter as much as the nominal graph operator. Continuous distance-conditioned filters motivated keeping physical distances in the graph model, while descriptor-based materials models motivated including non-graph controls. Relevant background is documented in the benchmark's [research design](research.md), including the materials GNN benchmark, SchNet, ALIGNN, PNA, MODNet, and Matbench.

## 2. Code and artifact scope

The implementation was committed as `ad3a175599a380c170c4932a92440943823339db` (`Add ELF void architecture benchmark`). It added 1,182 lines across 12 tracked files while keeping large run artifacts outside Git.

### 2.1 Benchmark code

- `research.md`: scientific rationale, fixed protocol, evaluation rules, and literature references.
- `sweep_spec.yaml`: predeclared architecture matrix and fixed seeds.
- `generate_sweep.py`: expands the specification into a frozen 25-row manifest and resolved YAML configurations.
- `train_architectures.py`: standalone benchmark trainer copied from the project trainer at benchmark creation time. This prevents unrelated later trainer changes from silently changing the experiment.
- `tree_baseline.py`: deterministic histogram-gradient-boosting comparator.
- `run_array.sbatch`: one neural run per Slurm GPU-array task.
- `tree_cpu.sbatch`: separate CPU execution for the tree model.
- `submit_sweep.sh`: validates the manifest and submits both Slurm jobs.
- `report.py`: aggregates run metrics, family metrics, paired differences, prediction diagnostics, and figures.
- `requirements.txt`: isolated report and tree dependencies.
- `elf_void_gnn/models/variants.py`: void-only MLP, fixed-context MLP, and gated MPNN implementations.
- `elf_void_gnn/models/__init__.py`: model registry entries.

### 2.2 Result artifacts

The complete result package is archived outside the repository in `architecture_test_results_2026-10-27/`; selected lightweight summaries and figures are versioned here. The external package contains:

- frozen manifest and resolved configuration files;
- one `metrics.json` per run;
- one `test_predictions.parquet` per run;
- aggregate run metrics;
- family-level test metrics and paired differences;
- mean paired family deltas;
- seed-level metric figures;
- family-difference heatmaps;
- prediction parity and residual figures;
- this report and the shorter results README.

## 3. Dataset and split

All configurations used the same saved feature batch and immutable split:

| Split | Samples | Families |
| --- | ---: | ---: |
| Training | 40,031 | 37 |
| Validation | 4,969 | 9 |
| Test | 5,000 | 9 |

The split fingerprint is:

`cdb3fc87b80cb05ce1cfdd8b94267696cfb0bb9e810004512144b0c2f2aac7f4`

The split holds out exact named phase folders. However, related pressure or space-group variants may occur in different splits. Therefore, the benchmark estimates generalization to held-out named phases under this particular split; it does not establish strict generalization to unseen compositions, pressure regimes, or structural families.

Void targets belonging to the same graph are dependent observations. The training loss averages over voids within each graph so that graphs, rather than crystals with more void targets, control the training weight. Pooled reporting metrics weight all held-out void targets equally, while macro-family metrics give each family equal weight.

## 4. Architectures tested

### 4.1 Standard heterogeneous MPNN

The standard model encodes atom and void features, expands relation distances with radial basis functions, applies relation-specific message functions, updates atom and void states, and predicts one value per void. Five size/depth configurations were tested:

- width 32, depth 2;
- width 64, depth 2 (the original architecture, retrained under this benchmark);
- width 128, depth 2;
- width 64, depth 1;
- width 64, depth 3.

### 4.2 Void-only MLP

The void-only MLP uses the nine void-node descriptors and no atom information or message passing. It tests whether the target can be predicted from the void representation alone.

### 4.3 Fixed-context MLP

The fixed-context MLP uses 26 features: nine void features, distance-weighted means of 14 neighboring-atom features, the weighted distance mean and standard deviation, and the log sum of distance weights. It tests whether a fixed local summary can replace learned graph interaction.

### 4.4 Gated MPNN

The gated model uses relation-specific distance gates and GRU-style recurrent updates. It tests whether a more elaborate recurrent update mechanism improves the standard MPNN.

### 4.5 Histogram-gradient-boosted tree

The tree uses the same 26 fields as the fixed-context MLP, but consumes raw values and fits 200 boosting iterations. It is a fast non-neural comparator rather than a replicated neural architecture. It was run once.

## 5. Predeclared experiment matrix

The seeds were `20261023`, `20261024`, and `20261025`.

| Configuration | Neural seeds | Parameter count |
| --- | ---: | ---: |
| Hetero MPNN, width 32, depth 2 | 3 | 21,569 |
| Hetero MPNN, width 64, depth 2 | 3 | 80,001 |
| Hetero MPNN, width 128, depth 2 | 3 | 307,457 |
| Hetero MPNN, width 64, depth 1 | 3 | 36,097 |
| Hetero MPNN, width 64, depth 3 | 3 | 123,905 |
| Void-only MLP, width 64, depth 2 | 3 | 4,865 |
| Fixed-context MLP, width 64, depth 2 | 3 | 5,953 |
| Gated MPNN, width 64, depth 2 | 3 | 151,873 |
| Histogram-gradient-boosted tree | 1 | not applicable |

This produced 24 GPU neural runs and one CPU tree run.

## 6. Matched training protocol

Every neural configuration used:

- the identical saved train/validation/test assignment;
- batch size 8;
- Adam optimizer;
- learning rate `1e-3`;
- weight decay `1e-5`;
- a maximum of 30 epochs;
- early stopping after five validation epochs without improvement;
- checkpoint selection by pooled validation RMSE;
- feature and target scaling fitted only on training rows;
- no target-derived input features;
- no model-specific hyperparameter tuning.

The test split was evaluated after selecting each run's best validation checkpoint. The architecture-selection rule was fixed in advance: compare mean pooled validation RMSE across the three seeds. Test metrics are descriptive and were not the declared selection criterion.

## 7. Compute execution

The neural experiments ran as Slurm GPU array job `44681694`, with 24 tasks and a maximum concurrency of three. The tree ran as CPU job `44681695`. All 25 tasks completed with exit code zero.

Average neural training times varied because early stopping ended runs at different epochs:

| Configuration | Mean training time | Approximate peak GPU memory |
| --- | ---: | ---: |
| Hetero width 32, depth 2 | 1,376 s | recorded per run |
| Hetero width 64, depth 2 | 2,162 s | recorded per run |
| Hetero width 128, depth 2 | 2,031 s | about 548 MB |
| Hetero width 64, depth 1 | 2,242 s | recorded per run |
| Hetero width 64, depth 3 | 3,178 s | about 436 MB |
| Void-only MLP | 958 s | recorded per run |
| Fixed-context MLP | 1,884 s | recorded per run |
| Gated MPNN | 2,124 s | recorded per run |

The tree's fitted estimator took approximately 2.6 seconds; feature extraction plus fitting and reporting took approximately 494 seconds.

## 8. Tests and verification performed

This work primarily consisted of a controlled computational experiment and integration validation, not a conventional unit-test campaign. The following checks were performed or enforced by the benchmark code.

### 8.1 Pre-submission checks

- The sweep definition was expanded into a frozen manifest.
- Submission required exactly 25 manifest rows.
- Every row had a resolved configuration, seed, run identifier, and output path.
- Neural and CPU rows were routed to the appropriate execution script.

### 8.2 Trainer safeguards

- Neural runs required GPU availability.
- Empty or missing training batches caused failure rather than silent completion.
- Prediction counts, metadata counts, and void identifiers were checked for alignment.
- Feature and target statistics were derived only from training data.
- The best validation checkpoint was saved atomically.
- Metrics were written atomically to avoid partially valid JSON files.
- Run outputs were isolated by configuration and seed.

### 8.3 Scheduler and completion checks

- All 24 GPU-array tasks completed successfully.
- The CPU tree job completed successfully.
- Slurm exit codes were zero.
- The expected count of `metrics.json` files was 25/25.
- The expected count of test prediction files was 25/25.
- No run was listed as incomplete.

### 8.4 Result integrity checks

- All aggregate metrics were finite.
- All test prediction Parquet files were nonempty.
- Per-family results were generated.
- Same-seed paired family differences were computed against the original width-64/depth-2 architecture.
- Prediction parity and residual diagnostics were generated.
- Parameter count, training time, and peak memory were recorded for neural runs.

No separate benchmark-specific unit-test suite was recorded as having been executed. The strongest verification is therefore successful end-to-end execution of every predeclared configuration plus artifact and metric integrity checks.

## 9. Metrics

Three regression metrics were reported:

- MAE: mean absolute prediction error; lower is better.
- RMSE: square-error-sensitive prediction error; lower is better.
- R2: fraction of target variance explained relative to a constant baseline; higher is better and values can be negative.

Each was computed in two aggregation modes:

- pooled: all void targets pooled together;
- macro-family: compute within each held-out family and average families equally.

Macro-family metrics are important because pooled metrics can be dominated by families containing more samples or void targets.

## 10. Complete aggregate results

Values are mean plus or minus sample standard deviation over three seeds. The tree is one run and therefore has no seed standard deviation.

### 10.1 Pooled metrics

| Configuration | Validation MAE | Validation RMSE | Validation R2 | Test MAE | Test RMSE | Test R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hetero w32 d2 | 0.04553 +/- 0.00201 | 0.06486 +/- 0.00114 | 0.8055 +/- 0.0068 | 0.05067 +/- 0.00183 | 0.07164 +/- 0.00149 | 0.7402 +/- 0.0108 |
| Hetero w64 d2 (original architecture, retrained) | 0.04249 +/- 0.00466 | 0.06192 +/- 0.00384 | 0.8223 +/- 0.0219 | 0.04482 +/- 0.00271 | 0.06682 +/- 0.00128 | 0.7740 +/- 0.0087 |
| **Hetero w128 d2** | **0.04095 +/- 0.00148** | **0.06056 +/- 0.00103** | **0.8304 +/- 0.0058** | 0.04307 +/- 0.00049 | 0.06524 +/- 0.00070 | 0.7846 +/- 0.0046 |
| Hetero w64 d1 | 0.04137 +/- 0.00057 | 0.06110 +/- 0.00102 | 0.8273 +/- 0.0058 | 0.04502 +/- 0.00282 | 0.06615 +/- 0.00325 | 0.7782 +/- 0.0215 |
| Hetero w64 d3 | 0.04118 +/- 0.00299 | 0.06152 +/- 0.00250 | 0.8248 +/- 0.0142 | **0.04297 +/- 0.00235** | **0.06430 +/- 0.00321** | **0.7904 +/- 0.0209** |
| Gated MPNN | 0.04287 +/- 0.00176 | 0.06250 +/- 0.00130 | 0.8194 +/- 0.0075 | 0.04795 +/- 0.00223 | 0.06979 +/- 0.00316 | 0.7532 +/- 0.0226 |
| Fixed-context MLP | 0.06143 +/- 0.00265 | 0.08228 +/- 0.00339 | 0.6866 +/- 0.0255 | 0.06149 +/- 0.00214 | 0.08478 +/- 0.00200 | 0.6361 +/- 0.0171 |
| Void-only MLP | 0.10754 +/- 0.00076 | 0.13405 +/- 0.00043 | 0.1692 +/- 0.0054 | 0.11160 +/- 0.00045 | 0.13997 +/- 0.00044 | 0.0085 +/- 0.0063 |
| Tree HGB context, n=1 | 0.05235 | 0.07250 | 0.7570 | 0.05518 | 0.07662 | 0.7029 |

### 10.2 Macro-family metrics

| Configuration | Validation MAE | Validation RMSE | Validation R2 | Test MAE | Test RMSE | Test R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hetero w32 d2 | 0.07661 +/- 0.00328 | 0.09581 +/- 0.00310 | 0.6034 +/- 0.0208 | 0.07725 +/- 0.00275 | 0.09978 +/- 0.00278 | 0.6162 +/- 0.0221 |
| Hetero w64 d2 (original architecture, retrained) | 0.07181 +/- 0.00380 | 0.09138 +/- 0.00310 | 0.6357 +/- 0.0236 | 0.07201 +/- 0.00376 | 0.09494 +/- 0.00491 | 0.6520 +/- 0.0459 |
| **Hetero w128 d2** | 0.06850 +/- 0.00292 | 0.08859 +/- 0.00340 | 0.6496 +/- 0.0254 | 0.06617 +/- 0.00204 | 0.09080 +/- 0.00295 | 0.6852 +/- 0.0211 |
| Hetero w64 d1 | 0.06596 +/- 0.00231 | 0.08551 +/- 0.00103 | **0.6825 +/- 0.0070** | 0.06907 +/- 0.00532 | 0.09166 +/- 0.00422 | 0.6777 +/- 0.0316 |
| Hetero w64 d3 | **0.06478 +/- 0.01255** | 0.08480 +/- 0.01199 | 0.6605 +/- 0.0851 | **0.06244 +/- 0.01018** | **0.08478 +/- 0.01143** | **0.7194 +/- 0.0681** |
| Gated MPNN | 0.06715 +/- 0.00077 | 0.08730 +/- 0.00112 | 0.6625 +/- 0.0194 | 0.07054 +/- 0.00179 | 0.09466 +/- 0.00258 | 0.6561 +/- 0.0162 |
| Fixed-context MLP | 0.09263 +/- 0.00257 | 0.11490 +/- 0.00500 | 0.4510 +/- 0.0452 | 0.09334 +/- 0.00103 | 0.11889 +/- 0.00083 | 0.4723 +/- 0.0056 |
| Void-only MLP | 0.13481 +/- 0.00040 | 0.15789 +/- 0.00085 | -0.0664 +/- 0.0079 | 0.13813 +/- 0.00030 | 0.16671 +/- 0.00112 | -0.0653 +/- 0.0114 |
| Tree HGB context, n=1 | **0.05756** | **0.07991** | **0.7130** | 0.06723 | 0.09182 | 0.6640 |

The tree's strong validation macro-family result is noteworthy, but it is one unreplicated run and its pooled validation RMSE is worse than all heterogeneous GNNs.

## 11. Paired family analysis

For each neural seed, family-level test errors were paired against the original width-64/depth-2 GNN with the same seed. Negative deltas favor the alternative model.

| Configuration | Mean paired family MAE delta | Mean paired family RMSE delta |
| --- | ---: | ---: |
| Hetero w128 d2 | -0.00584 | -0.00414 |
| Hetero w64 d1 | -0.00294 | -0.00327 |
| Hetero w64 d3 | -0.00957 | -0.01016 |
| Gated MPNN | -0.00147 | -0.00028 |
| Hetero w32 d2 | +0.00524 | +0.00485 |
| Fixed-context MLP | +0.02133 | +0.02396 |
| Void-only MLP | +0.06612 | +0.07177 |
| Tree HGB context, unreplicated | -0.00515 | -0.00193 |

The paired analysis reinforces that width 128 and depth 3 generally improve family error relative to the original architecture. It does not remove the need for independent evaluation because these are the same nine inspected test families.

## 12. Main findings

### 12.1 Message passing is necessary

The void-only MLP failed to explain held-out target variance: test pooled R2 was 0.008 and macro-family R2 was negative. Adding fixed neighboring-atom summaries raised test pooled R2 to 0.636, showing that atom context matters. Learned graph message passing raised performance further to approximately 0.74-0.79 pooled R2 across standard GNN configurations.

The gap between the fixed-context MLP and GNN demonstrates that the useful signal is not merely the presence of average neighboring-atom descriptors. The learned relation-aware interaction is contributing materially.

### 12.2 Width 128 gives a small, repeatable improvement

Relative to the retrained original width-64/depth-2 architecture, width 128 improved:

- mean validation pooled RMSE from 0.06192 to 0.06056;
- mean test pooled RMSE from 0.06682 to 0.06524;
- mean test macro-family RMSE from 0.09494 to 0.09080;
- mean test pooled R2 from 0.7740 to 0.7846.

The gain is modest, but its test pooled RMSE standard deviation was only 0.00070 and its test macro-family RMSE standard deviation was 0.00295. This is the most stable high-performing configuration.

### 12.3 Depth 3 is promising but unstable

The width-64/depth-3 model had the best descriptive mean test values: pooled RMSE 0.06430, pooled R2 0.7904, macro-family RMSE 0.08478, and macro-family R2 0.7194. However:

- it did not win the prespecified mean pooled validation RMSE criterion;
- its test pooled RMSE standard deviation was 0.00321, about 4.6 times the width-128 value;
- its test macro-family RMSE standard deviation was 0.01143, about 3.9 times the width-128 value;
- it trained longer on average.

This suggests a real research hypothesis--deeper propagation may help certain families--but not yet a sufficiently stable replacement.

### 12.4 One layer is a strong efficiency control

The width-64/depth-1 model used only 36,097 parameters and achieved validation pooled RMSE 0.06110 and test pooled RMSE 0.06615. It was close to the larger models and had the best stable validation macro-family behavior. It is a useful low-cost control and may be attractive when throughput or parameter efficiency matters.

### 12.5 Gated recurrence did not help

The gated MPNN used 151,873 parameters but produced test pooled RMSE 0.06979 and R2 0.7532, worse than the standard width-64/depth-2 model. Under this matched budget, added recurrent gating increased complexity without increasing predictive accuracy.

### 12.6 Classical baselines remain useful controls

The tree substantially outperformed the MLP with the same summarized inputs, demonstrating that model class matters even without message passing. It remained worse than the GNNs on pooled validation RMSE, but its competitive macro-family values justify retaining it as a sanity check in future studies. Its single-run status prevents a fair stability comparison.

## 13. Clarification about the "original" model

Two distinct results have been discussed:

1. The historical checkpoint that existed before the architecture sweep. Its test RMSE was approximately 0.06826. On the same target distribution, this corresponds to a pooled R2 of approximately 0.764.
2. The original architecture retrained under the new matched three-seed benchmark: heterogeneous MPNN, width 64, depth 2. Its mean test RMSE was 0.06682 and mean pooled R2 was 0.7740.

Therefore, labels in benchmark figures should be read as "original architecture, newly retrained," not as the historical checkpoint itself. The extremely poor R2 value of 0.008 belongs to the void-only MLP, not the original GNN.

## 14. Model recommendation

### Primary research model

Use the heterogeneous MPNN with hidden width 128 and two message-passing layers.

This recommendation follows the protocol rather than the inspected test ranking:

- it had the lowest mean pooled validation RMSE;
- it had the highest mean validation pooled R2;
- it improved pooled and macro-family test metrics over the original architecture;
- it showed low seed variability;
- it retains the standard, interpretable relation-specific message-passing structure;
- the added capacity produced a repeatable gain, unlike the gated update.

### Secondary hypotheses

Retain the following models for the next study:

- width 64, depth 3: tests whether deeper propagation improves family-level generalization;
- width 64, depth 1: tests how much performance can be retained with a smaller, more efficient network;
- width 64, depth 2: preserves the original architecture as the experimental control.

The depth-3 model should not be selected as primary based on this test set. Its attractiveness became clearer after inspecting test results, so confirmation now requires a new untouched evaluation.

## 15. Recommended next experiment

A research-grade follow-up should compare the width-128/depth-2 primary model, width-64/depth-3 hypothesis, width-64/depth-1 efficiency control, and original width-64/depth-2 control using grouped cross-validation or a new strictly held-out split.

The grouping should be designed around the scientific generalization claim of interest, for example:

- unseen chemical composition;
- unseen pressure regime;
- unseen structural family;
- combinations of composition and pressure.

Model selection should remain nested inside validation folds, with an untouched outer evaluation. This is especially important because the current test set has already been inspected multiple times and contains only nine families.

For the depth-3 hypothesis, record family-wise effects and seed stability rather than relying only on pooled averages. If the depth-3 advantage repeats on new groups without the current variability, it would become a credible replacement. If not, width 128/depth 2 remains the better research default.

## 16. Limitations

1. Only one fixed data split was used. Three seeds quantify optimization variability, not dataset-split uncertainty.
2. The test set contains only nine families.
3. Related pressure or space-group variants may cross split boundaries.
4. The experiment was a matched-budget comparison, not exhaustive hyperparameter optimization for every model family.
5. The tree was run once and cannot be compared fairly on seed stability.
6. No angle or triplet features were available in the fixed batch, so angle-aware architectures such as ALIGNN were not tested.
7. PNA-style multi-aggregator models and modern equivariant atomistic networks were outside the predeclared matrix.
8. The test set is now descriptive only; subsequent model choices should be validated independently.
9. No separate benchmark-specific unit-test suite was recorded, although all end-to-end runs and output-integrity checks passed.

## 17. Reproducibility and provenance

- Benchmark commit: `ad3a175599a380c170c4932a92440943823339db`
- Split SHA256: `cdb3fc87b80cb05ce1cfdd8b94267696cfb0bb9e810004512144b0c2f2aac7f4`
- GPU Slurm array: `44681694`, 24/24 complete, exit 0
- CPU tree job: `44681695`, complete, exit 0
- Remote run root: `/home/weilana/elfnet-runs/elf_void_gnn/architecture-test`
- Local result package: `architecture_test_results_2026-10-27`

The remote checkout advanced during execution because of concurrent feature-study work. Early tasks used `ad3a175`; later tasks saw a subsequent change that added optional `edge_features` support to `hetero_mpnn.py`. The benchmark dataset did not supply that optional field, so the same original radial-distance path executed. The benchmark trainer, model variants, registry, manifest, and configurations did not change during the sweep.

## 18. Key files

- [Concise results summary](RESULTS.md)
- [All per-run metrics](results/run_metrics.csv)
- [Family metrics and paired deltas](results/test_family_metrics_and_paired_deltas.csv)
- [Mean paired family deltas](results/mean_paired_family_deltas.csv)
- [Complete architecture comparison figure](results/architecture_headline_means.png)
- [Individual seed comparison](results/split_metric_seed_points.png)
- [Family delta heatmap](results/paired_family_mae_delta.png)
- [Full external result archive](../../../architecture_test_results_2026-10-27/)
- [Frozen manifest](results/manifest.csv)

## Final conclusion

The architecture study supports continuing with a heterogeneous atom/void message-passing model. Graph interaction is not a marginal improvement; it is the dominant architectural requirement demonstrated by the controls. The width-128, depth-2 MPNN is the most defensible primary model because it won the predeclared validation criterion and combined strong accuracy with low seed variability. The depth-3 model is the most interesting next hypothesis, particularly for family-balanced performance, but it requires confirmation on a new grouped evaluation before it should replace the validation-selected model.
