#!/bin/bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/home/USER/elfnet-runs/elf_void_gnn/confirmation-study-2026-10-28}"
mkdir -p "$EXPERIMENT_ROOT/configs" "$EXPERIMENT_ROOT/logs" "$EXPERIMENT_ROOT/runs"
PYTHON="/home/USER/.conda/envs/elfnet-train/bin/python"
"$PYTHON" "$REPO_ROOT/elf_void_gnn/confirmation_study/generate_manifest.py" \
  --manifest "$EXPERIMENT_ROOT/configs/manifest.csv"
MANIFEST="$EXPERIMENT_ROOT/configs/manifest.csv"
RUN_COUNT="$("$PYTHON" -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))' "$MANIFEST")"
if [ "$RUN_COUNT" -ne 9 ]; then
  echo "Expected exactly 9 validation-only jobs; got $RUN_COUNT" >&2
  exit 1
fi
sbatch --account="${SLURM_ACCOUNT:?set SLURM_ACCOUNT to your Slurm allocation}" --array="0-8%3" --export="ALL,MANIFEST=$MANIFEST" \
  --output="$EXPERIMENT_ROOT/logs/confirm-%A_%a.out" \
  --error="$EXPERIMENT_ROOT/logs/confirm-%A_%a.err" \
  "$REPO_ROOT/elf_void_gnn/confirmation_study/run_array.sbatch"
