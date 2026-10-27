#!/usr/bin/env bash
# Submit manually only after Stage A passes its reproduction and integrity gates.
# The array entries are read, in order, from stage_b_variants.txt.
# Before submission, create /home/weilana/elfnet-runs/feature-study for Slurm logs.
#SBATCH --job-name=elf-feat-stageb
#SBATCH --account=andre176
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-13%3
#SBATCH --output=/home/weilana/elfnet-runs/feature-study/stage-b-slurm-%A_%a.out

set -euo pipefail
module load rocm/6.3.4
cd /home/weilana/electride-ml-predictor

variant_file="elf_void_gnn/feature_study/stage_b_variants.txt"
mapfile -t variants < <(awk 'NF && $1 !~ /^#/ {print $1}' "$variant_file")
if [[ "${#variants[@]}" -ne 14 ]]; then
  echo "Expected 14 Stage B variants in ${variant_file}; found ${#variants[@]}" >&2
  exit 2
fi
if [[ ! "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] || (( SLURM_ARRAY_TASK_ID >= ${#variants[@]} )); then
  echo "Invalid Stage B array task index: ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

variant="${variants[$SLURM_ARRAY_TASK_ID]}"
if [[ ! "$variant" =~ ^[a-z0-9_]+$ ]]; then
  echo "Unsafe Stage B variant name: ${variant}" >&2
  exit 2
fi

seed=20261023
echo "Stage B variant ${SLURM_ARRAY_TASK_ID}: ${variant} (seed ${seed})"
/home/weilana/.conda/envs/elfnet-train/bin/python -m elf_void_gnn.train \
  --config elf_void_gnn/feature_study/screening.yaml \
  --feature-variant "$variant" --evaluation-mode validation-only --require-gpu \
  --run-dir "/home/weilana/elfnet-runs/feature-study/stage-b-job-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}/${variant}/seed-${seed}"
