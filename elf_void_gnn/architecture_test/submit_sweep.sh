#!/bin/bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/home/USER/elfnet-runs/elf_void_gnn/architecture-test}"
CONFIG_ROOT="$EXPERIMENT_ROOT/configs"
RUNS_ROOT="$EXPERIMENT_ROOT/runs"
mkdir -p "$EXPERIMENT_ROOT/logs"
/home/USER/.conda/envs/elfnet-train/bin/python "$REPO_ROOT/elf_void_gnn/architecture_test/generate_sweep.py" \
  --config-root "$CONFIG_ROOT" --runs-root "$RUNS_ROOT"
MANIFEST="$CONFIG_ROOT/manifest.csv"
RUN_COUNT="$(/home/USER/.conda/envs/elfnet-train/bin/python -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))) )' "$MANIFEST")"
if [ "$RUN_COUNT" -lt 1 ]; then
  echo "Manifest contains no jobs" >&2
  exit 1
fi
if [ "$RUN_COUNT" -ne 25 ]; then
  echo "Expected the predeclared 25 jobs; got $RUN_COUNT" >&2
  exit 1
fi
sbatch --array="0-23%3" --export="ALL,MANIFEST=$MANIFEST" \
  --output="$EXPERIMENT_ROOT/logs/arch-%A_%a.out" \
  --error="$EXPERIMENT_ROOT/logs/arch-%A_%a.err" \
  "$REPO_ROOT/elf_void_gnn/architecture_test/run_array.sbatch"
sbatch --export="ALL,MANIFEST=$MANIFEST" \
  --output="$EXPERIMENT_ROOT/logs/tree-%j.out" \
  --error="$EXPERIMENT_ROOT/logs/tree-%j.err" \
  "$REPO_ROOT/elf_void_gnn/architecture_test/tree_cpu.sbatch"
