#!/bin/bash
#SBATCH --job-name=Ba3CrN3_pbe_relax
#SBATCH --account=andre176
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=40G
#SBATCH --partition=cpu

set -euo pipefail

module load quantum-espresso
mkdir -p out
srun pw.x -in pbe_vc-relax.in > pbe_vc-relax.out
