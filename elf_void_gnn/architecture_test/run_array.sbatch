#!/bin/bash
#SBATCH --account=andre176
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=elf-arch
#SBATCH --output=/home/weilana/elfnet-runs/elf_void_gnn/architecture-test/logs/arch-%A_%a.out
#SBATCH --error=/home/weilana/elfnet-runs/elf_void_gnn/architecture-test/logs/arch-%A_%a.err
set -euo pipefail
module load rocm/6.3.4
cd /home/weilana/electride-ml-predictor
export PYTHONPATH="/home/weilana/elfnet-runs/architecture_test/python-deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
MANIFEST="${MANIFEST:?Set MANIFEST to the generated architecture-test manifest.csv}"
INDEX="${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"
eval "$(/home/weilana/.conda/envs/elfnet-train/bin/python - "$MANIFEST" "$INDEX" <<'PY'
import csv, shlex, sys
with open(sys.argv[1], newline='') as stream:
    rows = list(csv.DictReader(stream))
row = rows[int(sys.argv[2])]
print('CONFIG=' + shlex.quote(row['config_path']))
print('RUN_DIR=' + shlex.quote(row['run_dir']))
print('DATASET_ROOT=' + shlex.quote(row['dataset_root']))
print('MODEL_TYPE=' + shlex.quote(row['model_type']))
print('SEED=' + shlex.quote(str(row['seed'])))
PY
)"
if [ "$MODEL_TYPE" = "tree_baseline" ]; then
  /home/weilana/.conda/envs/elfnet-train/bin/python elf_void_gnn/architecture_test/tree_baseline.py \
    --dataset-root "$DATASET_ROOT" --run-dir "$RUN_DIR" --seed "$SEED"
else
  /home/weilana/.conda/envs/elfnet-train/bin/python elf_void_gnn/architecture_test/train_architectures.py \
    --require-gpu --config "$CONFIG" --dataset-root "$DATASET_ROOT" --run-dir "$RUN_DIR"
fi
