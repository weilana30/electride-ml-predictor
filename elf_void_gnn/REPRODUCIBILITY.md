# GNN reproducibility guide

The repository contains the source code and experiment definitions. Large source data, graph shards, checkpoints, logs, and prediction files remain in external storage documented in [`DATASET_PROVENANCE.md`](DATASET_PROVENANCE.md).

## 1. Build and validate the graph dataset

The extraction code is in `elf_void_analysis/`. On Negishi, configure the environment and source/output locations using `elf_void_analysis/cluster/negishi/negishi.env`, then submit the manifest, extraction, and merge/validation stages:

```bash
bash elf_void_analysis/cluster/negishi/setup_environment.sh
bash elf_void_analysis/cluster/negishi/submit_prepare.sh
bash elf_void_analysis/cluster/negishi/submit_node.sh
```

The full source tree and generated dataset must be placed outside the repository. Use a new run ID for a clean rebuild.

## 2. Run the standard GNN

From the repository root on Negishi:

```bash
sbatch elf_void_gnn/cluster/smoke.sbatch
sbatch elf_void_gnn/cluster/train.sbatch
```

The standard run selects its checkpoint by validation RMSE and writes a model checkpoint, metrics, predictions, and training log under the configured external run root.

## 3. Run the architecture benchmark

The architecture benchmark is under `elf_void_gnn/architecture_test/`:

```bash
bash elf_void_gnn/architecture_test/submit_sweep.sh
```

This generates a frozen 25-row manifest, submits 24 neural GPU tasks and one CPU tree task, and writes runs outside Git. To regenerate the report after completion:

```bash
python elf_void_gnn/architecture_test/report.py \
  --manifest /home/USER/elfnet-runs/elf_void_gnn/architecture-test/configs/manifest.csv \
  --out-dir /home/USER/elfnet-runs/elf_void_gnn/architecture-test/report
```

The benchmark commit and split hash are recorded in `architecture_test/DETAILED_RESULTS.md`.

## 4. Reproduce the feature study

The fixed-architecture feature study is under `elf_void_gnn/feature_study/`. Its staged screening, correlation, shortlist, and final-test commands are documented in [`PLAN.md`](feature_study/PLAN.md) and [`REPORTING.md`](feature_study/REPORTING.md). The completed external report is summarized in [`feature_study/RESULTS.md`](feature_study/RESULTS.md).

## 5. Reproducibility rules

- Keep the saved split immutable for a given comparison.
- Select architectures and features using validation data or a new independent evaluation, not an already-inspected test set.
- Preserve the exact seed, configuration, dataset version, and split hash for every run.
- Keep raw data, checkpoints, predictions, logs, and environments outside Git; commit lightweight manifests, aggregate results, and scientific reports.
