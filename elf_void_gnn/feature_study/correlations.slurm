#!/usr/bin/env bash
# Train-only descriptor correlation and missingness audit.
# Submit manually with sbatch after creating the external log/output directory.
# The default analysis uses a deterministic hash-selected sample, capped at
# 100 structures per training family, to bound per-structure rank calculations.
# Set MAX_PER_FAMILY=0 for a full split audit.
#SBATCH --job-name=elf-feat-corr
#SBATCH --account=andre176
#SBATCH --qos=normal
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/weilana/elfnet-runs/feature-study/correlations-%j.out

set -euo pipefail
cd /home/weilana/electride-ml-predictor
mkdir -p /home/weilana/elfnet-runs/feature-study/analysis
max_per_family="${MAX_PER_FAMILY:-100}"
/home/weilana/.conda/envs/elfnet-train/bin/python -u -m elf_void_gnn.feature_study.correlation \
  --dataset-root /home/weilana/elfnet-datasets/elfnet-gnn-2026-10-23 \
  --out-dir /home/weilana/elfnet-runs/feature-study/analysis \
  --max-per-family "$max_per_family"
