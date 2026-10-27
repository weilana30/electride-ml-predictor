#!/usr/bin/env bash
# Submit manually only after freezing the shortlist and the checkpoint manifest.
# This job evaluates the listed best_model.pt checkpoints; it does not retrain.
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
manifest="${FROZEN_MANIFEST:-/home/weilana/elfnet-runs/feature-study/final-test-manifest.csv}"
/home/weilana/.conda/envs/elfnet-train/bin/python -m elf_void_gnn.feature_study.evaluate_frozen \
  --manifest "$manifest" \
  --dataset-root /home/weilana/elfnet-datasets/elfnet-gnn-2026-10-23 \
  --out-dir "/home/weilana/elfnet-runs/feature-study/final-test/job-${SLURM_JOB_ID}" \
  --require-gpu
