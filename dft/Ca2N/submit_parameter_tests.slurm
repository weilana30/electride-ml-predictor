#!/bin/bash
#SBATCH --job-name=Ca2N_pbe_parameter_tests
#SBATCH --account=andre176
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1G
#SBATCH --partition=cpu

set -euo pipefail
material_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for parameter in ecutwfc kpoints smearing_or_conv_thr; do
  bash "$material_dir/PBE/$parameter/submit.slurm"
done
