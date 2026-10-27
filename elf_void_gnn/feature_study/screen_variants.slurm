#!/usr/bin/env bash
# Submit manually with sbatch; array screens validation only and never opens test shards.
# Before submission, create /home/weilana/elfnet-runs/feature-study for Slurm logs.
#SBATCH --job-name=elf-feat-screen
#SBATCH --account=andre176
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-8%3
#SBATCH --output=/home/weilana/elfnet-runs/feature-study/slurm-%A_%a.out

set -euo pipefail
module load rocm/6.3.4
cd /home/weilana/electride-ml-predictor
variants=(
  baseline
  atomic_number_nearest_void_distances_edge_distances
  atomic_number_void_geometry
  void_full_geometry
  atom_full_chemistry_and_periodicity
  no_edge_distance
  no_atom_atom_distance
  no_void_atom_distance
  no_void_void_distance
)
variant="${variants[$SLURM_ARRAY_TASK_ID]}"
seed=20261023
/home/weilana/.conda/envs/elfnet-train/bin/python -m elf_void_gnn.train \
  --config elf_void_gnn/feature_study/screening.yaml \
  --feature-variant "$variant" --evaluation-mode validation-only --require-gpu \
  --run-dir "/home/weilana/elfnet-runs/feature-study/job-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}/${variant}/seed-${seed}"
