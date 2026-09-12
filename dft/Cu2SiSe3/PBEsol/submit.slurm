#!/bin/bash
#SBATCH --job-name=Cu2SiSe3_pbesol_relax
#SBATCH --account=andre176
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=32G
#SBATCH --partition=cpu

set -euo pipefail

module load quantum-espresso
mkdir -p out
srun pw.x -in pbesol_vc-relax.in > pbesol_vc-relax.out
