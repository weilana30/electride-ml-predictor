#!/bin/bash
#SBATCH --job-name=Ba2As_parameter_tests
#SBATCH --account=andre176
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1G
#SBATCH --partition=cpu

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

echo "Submitting SCF parameter sweeps for Ba2As"
for functional in PBE PBEsol R2SCAN; do
  for parameter in ecutwfc kpoints smearing_or_conv_thr; do
    sbatch "${functional}/${parameter}/submit.slurm"
  done
done
