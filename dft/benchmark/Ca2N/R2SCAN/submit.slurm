#!/bin/bash
#SBATCH --job-name=Ca2N_r2scan_relax
#SBATCH --account=andre176
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=20G
#SBATCH --partition=cpu

set -euo pipefail

module load quantum-espresso
mkdir -p out
srun pw.x -in r2scan_vc-relax.in > r2scan_vc-relax.out
