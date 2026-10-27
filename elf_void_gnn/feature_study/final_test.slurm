#!/usr/bin/env bash
# Submit manually only after freezing the shortlisted variant and seed.
# Before submission, create /home/weilana/elfnet-runs/feature-study for Slurm logs.
#SBATCH --job-name=elf-feat-final
#SBATCH --account=andre176
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/weilana/elfnet-runs/feature-study/final-%j.out

set -euo pipefail
module load rocm/6.3.4
cd /home/weilana/electride-ml-predictor
: "${FEATURE_VARIANT:?Set FEATURE_VARIANT to a frozen shortlisted variant}"
: "${SEED:=20261023}"
/home/weilana/.conda/envs/elfnet-train/bin/python -m elf_void_gnn.train \
  --config elf_void_gnn/feature_study/screening.yaml \
  --feature-variant "$FEATURE_VARIANT" --seed "$SEED" --evaluation-mode final-test --require-gpu \
  --run-dir "/home/weilana/elfnet-runs/feature-study/final/job-${SLURM_JOB_ID}/${FEATURE_VARIANT}/seed-${SEED}"
