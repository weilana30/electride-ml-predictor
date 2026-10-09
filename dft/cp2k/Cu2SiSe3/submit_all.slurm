#!/bin/bash
#SBATCH --job-name=Cu2SiSe3_cp2k_submit
#SBATCH --account=andre176
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1G
#SBATCH --partition=cpu
set -euo pipefail
for case in cutoff/* kpoints/* basis/*; do (cd "$case" && sbatch submit.slurm); done
