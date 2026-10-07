#!/bin/bash
#SBATCH --job-name=Ba2As_pbe_smearing_or_conv_thr_submit
#SBATCH --account=andre176
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1G
#SBATCH --partition=cpu

set -euo pipefail
parent_dir=$(pwd -P)

echo "Submitting Ba2As PBE smearing_or_conv_thr cases"
sbatch --chdir="$parent_dir/degauss_0p01" --output="$parent_dir/degauss_0p01/slurm-%x-%j.out" --error="$parent_dir/degauss_0p01/slurm-%x-%j.err" "$parent_dir/degauss_0p01/submit.slurm"
sbatch --chdir="$parent_dir/degauss_0p02" --output="$parent_dir/degauss_0p02/slurm-%x-%j.out" --error="$parent_dir/degauss_0p02/slurm-%x-%j.err" "$parent_dir/degauss_0p02/submit.slurm"
sbatch --chdir="$parent_dir/degauss_0p04" --output="$parent_dir/degauss_0p04/slurm-%x-%j.out" --error="$parent_dir/degauss_0p04/slurm-%x-%j.err" "$parent_dir/degauss_0p04/submit.slurm"
