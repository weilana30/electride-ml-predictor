#!/bin/bash
#SBATCH --job-name=Ba2As_parameter_tests
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

echo "Submitting SCF parameter cases for Ba2As"
for functional in PBE R2SCAN; do
  for parameter in ecutwfc kpoints smearing_or_conv_thr; do
    sweep_dir="${script_dir}/${functional}/${parameter}"
    for job in "$sweep_dir"/submit_scf_*.slurm; do
      sbatch --chdir="$sweep_dir" "$job"
    done
  done
done
