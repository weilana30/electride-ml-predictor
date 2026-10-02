#!/bin/bash
#SBATCH --job-name=BaSn3_pbesol_scf_kpoints_low
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
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,tcp
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
echo "Starting BaSn3 PBEsol scf_kpoints_low: $(date)"
echo "Running scf_kpoints_low.in"
srun pw.x -in "scf_kpoints_low.in" > "scf_kpoints_low.out"
job_elapsed=$(( $(date +%s) - job_start ))
echo "Completed calculation at $(date)"
echo "Total job elapsed time: $(format_elapsed "$job_elapsed")"
