#!/bin/bash
#SBATCH --job-name=MgS_pbe_relax_submit
#SBATCH --account=andre176
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1G
#SBATCH --partition=cpu

set -euo pipefail
parent_dir=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}
sbatch --chdir="$parent_dir/relax" \
  --output="$parent_dir/relax/slurm-%x-%j.out" \
  --error="$parent_dir/relax/slurm-%x-%j.err" \
  "$parent_dir/relax/submit.slurm"
