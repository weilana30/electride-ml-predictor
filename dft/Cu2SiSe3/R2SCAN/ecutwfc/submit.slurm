#!/bin/bash
#SBATCH --job-name=Cu2SiSe3_r2scan_ecutwfc_submit
#SBATCH --account=andre176
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --partition=cpu

set -euo pipefail
script_dir="$PWD"

echo "Submitting Cu2SiSe3 R2SCAN ecutwfc SCF cases"
for job in submit_scf_*.slurm; do
  sbatch --chdir="$script_dir" "$script_dir/$job"
done
