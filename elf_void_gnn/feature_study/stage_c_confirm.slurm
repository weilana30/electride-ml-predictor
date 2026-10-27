#!/usr/bin/env bash
# Run only after Stage A/B results have passed integrity checks and the
# three-variant validation shortlist has been frozen.
# The versioned shortlist was frozen from validation results before any test
# evaluation. This array adds two seeds for each shortlisted variant.
#SBATCH --job-name=elf-feat-confirm
#SBATCH --account=andre176
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-5%3
#SBATCH --output=/home/weilana/elfnet-runs/feature-study/stage-c-%A_%a.out

set -euo pipefail
module load rocm/6.3.4
cd /home/weilana/electride-ml-predictor

shortlist="elf_void_gnn/feature_study/frozen_shortlist.txt"
if [[ ! -f "$shortlist" ]]; then
  echo "Missing frozen shortlist file: $shortlist" >&2
  exit 2
fi
mapfile -t variants < <(awk 'NF && $1 !~ /^#/ {print $1}' "$shortlist")
if [[ "${#variants[@]}" -ne 3 ]]; then
  echo "Expected exactly 3 frozen feature variants; found ${#variants[@]}" >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${variants[@]}" | sort -u | wc -l | tr -d ' ')" -ne 3 ]]; then
  echo "The frozen shortlist must contain three unique variants" >&2
  exit 2
fi
if ! printf '%s\n' "${variants[@]}" | grep -Fxq baseline; then
  echo "The frozen shortlist must include baseline" >&2
  exit 2
fi
if ! printf '%s\n' "${variants[@]}" | grep -Fxq atomic_number_nearest_void_distances_edge_distances; then
  echo "The frozen shortlist must include the required minimal variant" >&2
  exit 2
fi

task_index=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
variant_index=$((task_index / 2))
seed_index=$((task_index % 2))
if (( variant_index >= ${#variants[@]} )); then
  echo "Invalid array index: ${task_index}" >&2
  exit 2
fi
variant="${variants[$variant_index]}"
if [[ ! "$variant" =~ ^[a-z0-9_]+$ ]]; then
  echo "Invalid variant name: ${variant}" >&2
  exit 2
fi
seeds=(20261024 20261025)
seed="${seeds[$seed_index]}"

/home/weilana/.conda/envs/elfnet-train/bin/python -m elf_void_gnn.train \
  --config elf_void_gnn/feature_study/screening.yaml \
  --feature-variant "$variant" --seed "$seed" --evaluation-mode validation-only --require-gpu \
  --run-dir "/home/weilana/elfnet-runs/feature-study/stage-c/job-${SLURM_ARRAY_JOB_ID}-${task_index}/${variant}/seed-${seed}"
