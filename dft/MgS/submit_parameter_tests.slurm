#!/bin/bash
#SBATCH --job-name=MgS_parameter_tests
#SBATCH --account=andre176
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1G
#SBATCH --partition=cpu

set -euo pipefail
script_dir="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$script_dir"

echo "Submitting SCF parameter sweeps for MgS"
for functional in PBE PBEsol R2SCAN; do
  for parameter in ecutwfc kpoints smearing_or_conv_thr; do
    sweep_dir="${script_dir}/${functional}/${parameter}"
    sbatch --chdir="$sweep_dir" "$sweep_dir/submit.slurm"
  done
done
