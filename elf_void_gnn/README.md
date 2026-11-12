# ELF-center GNN

`hetero_mpnn` is the first registered per-void regressor. The pipeline consumes the saved train/validation/test assignment, derives normalization from training rows only, selects the checkpoint by validation RMSE, and evaluates the test rows once after selection. It reports pooled and per-family MAE, RMSE, and R² against the training-mean baseline. The split holds out exact named phase folders; related composition or pressure variants can appear on both sides, so this is preliminary held-out named-phase evaluation.

The [direct confirmation study](confirmation_study/README.md) repeats three predeclared models over three new seeds on the existing family split used by the architecture and feature studies. It reports the partition's generalization limits and uses validation-only selection before descriptive test evaluation.

Inputs are atom structural features plus validity flags and void structural features. Edge distances are encoded with eight radial basis channels. Target values are only used as labels. The registry interface is `forward(batch) -> one raw-scale ELF prediction per void`; add a model in `models/`, register its name in `models/__init__.py`, and keep the shared loader/trainer unchanged.

## Research records

- [`DATASET_PROVENANCE.md`](DATASET_PROVENANCE.md) records the raw ELFNet source, processed graph dataset, split fingerprint, graph contract, and label separation.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) records the extraction, training, architecture-sweep, and feature-study workflows.
- [`architecture_test/research.md`](architecture_test/research.md) documents the architecture benchmark design and literature rationale.
- [`architecture_test/RESULTS.md`](architecture_test/RESULTS.md) and [`architecture_test/DETAILED_RESULTS.md`](architecture_test/DETAILED_RESULTS.md) contain the benchmark results and interpretation.
- [`feature_study/RESULTS.md`](feature_study/RESULTS.md) summarizes the completed fixed-architecture feature study.

## Negishi

Create the separate environment using PyTorch 2.7.1 ROCm 6.3 and the pins in `requirements-rocm63.txt`; leave the extraction environment alone. From the repository root, create `/home/USER/elfnet-runs/elf_void_gnn/logs`, then submit the GPU smoke script followed by the training script only after smoke succeeds:

```sh
sbatch elf_void_gnn/cluster/smoke.sbatch
sbatch elf_void_gnn/cluster/train.sbatch
```

The training run is written to the configured run directory. It contains `config.yaml`, `best_model.pt`, `metrics.json`, `test_predictions.parquet`, and `train.log`. Slurm stdout/stderr stay in the shared `logs/` directory.

## Local checks

```sh
python -m unittest discover -s elf_void_gnn/tests -v
python -m elf_void_gnn.train --help
```
