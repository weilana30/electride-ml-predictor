#!/bin/bash
#SBATCH --job-name=Ba2As_pbe_ecutwfc_submit
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

echo "Submitting Ba2As PBE ecutwfc cases"
sbatch --chdir="$parent_dir/low" --output="$parent_dir/low/slurm-%x-%j.out" --error="$parent_dir/low/slurm-%x-%j.err" "$parent_dir/low/submit.slurm"
sbatch --chdir="$parent_dir/base" --output="$parent_dir/base/slurm-%x-%j.out" --error="$parent_dir/base/slurm-%x-%j.err" "$parent_dir/base/submit.slurm"
sbatch --chdir="$parent_dir/high" --output="$parent_dir/high/slurm-%x-%j.out" --error="$parent_dir/high/slurm-%x-%j.err" "$parent_dir/high/submit.slurm"
