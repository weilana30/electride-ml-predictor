#!/bin/bash
#SBATCH --job-name=Cu2SiSe3_pbesol_scf_ecutwfc_high
#SBATCH --account=andre176
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH --mem=16G
#SBATCH --partition=cpu

set -euo pipefail

module load quantum-espresso
script_dir="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$script_dir"

format_elapsed() {
  local total_seconds="$1"
  printf '%02d:%02d:%02d' \
    "$((total_seconds / 3600))" \
    "$(((total_seconds % 3600) / 60))" \
    "$((total_seconds % 60))"
}
job_start=$(date +%s)
echo "Starting Cu2SiSe3 PBEsol scf_ecutwfc_high: $(date)"
echo "Running scf_ecutwfc_high.in"
srun pw.x -in "scf_ecutwfc_high.in" > "scf_ecutwfc_high.out"
job_elapsed=$(( $(date +%s) - job_start ))
echo "Completed calculation at $(date)"
echo "Total job elapsed time: $(format_elapsed "$job_elapsed")"
