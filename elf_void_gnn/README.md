# ELF-center GNN

`hetero_mpnn` is the first registered per-void regressor. The pipeline consumes the saved train/validation/test assignment, derives normalization from training rows only, selects the checkpoint by validation RMSE, and evaluates the test rows once after selection. It reports pooled and per-family MAE, RMSE, and R² against the training-mean baseline. The split holds out exact named phase folders; related composition or pressure variants can appear on both sides, so this is preliminary held-out named-phase evaluation.

Inputs are atom structural features plus validity flags and void structural features. Edge distances are encoded with eight radial basis channels. Target values are only used as labels. The registry interface is `forward(batch) -> one raw-scale ELF prediction per void`; add a model in `models/`, register its name in `models/__init__.py`, and keep the shared loader/trainer unchanged.

## Negishi

Create the separate environment using PyTorch 2.7.1 ROCm 6.3 and the pins in `requirements-rocm63.txt`; leave the extraction environment alone. From the repository root, create `/home/weilana/elfnet-runs/elf_void_gnn/logs`, then submit the GPU smoke script followed by the training script only after smoke succeeds:

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
