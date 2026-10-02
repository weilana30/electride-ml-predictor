#!/bin/bash
#SBATCH --job-name=Cu2SiSe3_pbesol_kpoints_submit
#SBATCH --account=andre176
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --partition=cpu

set -euo pipefail
script_dir="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$script_dir"

echo "Submitting Cu2SiSe3 PBEsol kpoints SCF cases"
for job in submit_scf_*.slurm; do
  sbatch --chdir="$script_dir" "$script_dir/$job"
done
