#!/bin/bash
#SBATCH --account=andre176
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=elf-confirm
#SBATCH --output=/home/weilana/elfnet-runs/elf_void_gnn/confirmation-study-2026-10-28/logs/confirm-%A_%a.out
#SBATCH --error=/home/weilana/elfnet-runs/elf_void_gnn/confirmation-study-2026-10-28/logs/confirm-%A_%a.err
set -euo pipefail
module load rocm/6.3.4
cd /home/weilana/electride-ml-predictor
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
MANIFEST="${MANIFEST:?Set MANIFEST to the generated confirmation manifest}"
INDEX="${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"
eval "$(/home/weilana/.conda/envs/elfnet-train/bin/python - "$MANIFEST" "$INDEX" <<'PY'
import csv, shlex, sys
with open(sys.argv[1], newline="") as stream:
    rows = list(csv.DictReader(stream))
row = rows[int(sys.argv[2])]
print("CONFIGURATION_ID=" + shlex.quote(row["configuration_id"]))
print("RUN_DIR=" + shlex.quote(row["run_dir"]))
print("SEED=" + shlex.quote(row["seed"]))
PY
)"
/home/weilana/.conda/envs/elfnet-train/bin/python elf_void_gnn/confirmation_study/train_confirmation.py \
  --require-gpu --evaluation-mode validation-only \
  --configuration-id "$CONFIGURATION_ID" --seed "$SEED" --run-dir "$RUN_DIR"
