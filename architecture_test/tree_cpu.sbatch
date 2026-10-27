#!/bin/bash
#SBATCH --account=andre176
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=elf-tree
#SBATCH --output=/home/weilana/elfnet-runs/elf_void_gnn/architecture-test/logs/tree-%j.out
#SBATCH --error=/home/weilana/elfnet-runs/elf_void_gnn/architecture-test/logs/tree-%j.err
set -euo pipefail
cd /home/weilana/electride-ml-predictor
PYTHON=/home/weilana/.conda/envs/elfnet-train/bin/python
export PYTHONPATH="/home/weilana/elfnet-runs/architecture_test/python-deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
MANIFEST="${MANIFEST:?Set MANIFEST to the generated architecture-test manifest.csv}"
eval "$("$PYTHON" - "$MANIFEST" <<'PY'
import csv, shlex, sys
with open(sys.argv[1], newline='') as stream:
    rows = list(csv.DictReader(stream))
row = rows[24]
if row['model_type'] != 'tree_baseline' or row['configuration'] != 'tree_hgb_context':
    raise SystemExit('Manifest row 24 is not the predeclared tree baseline')
print('RUN_DIR=' + shlex.quote(row['run_dir']))
print('DATASET_ROOT=' + shlex.quote(row['dataset_root']))
print('SEED=' + shlex.quote(str(row['seed'])))
PY
)"
"$PYTHON" architecture_test/tree_baseline.py \
  --dataset-root "$DATASET_ROOT" --run-dir "$RUN_DIR" --seed "$SEED"
