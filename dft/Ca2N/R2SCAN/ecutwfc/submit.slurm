#!/bin/bash
#SBATCH --job-name=Ca2N_r2scan_ecutwfc_scf
#SBATCH --account=andre176
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH --mem=4G
#SBATCH --partition=cpu

set -euo pipefail

module load quantum-espresso
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

format_elapsed() {
  local total_seconds="$1"
  printf '%02d:%02d:%02d' \
    "$((total_seconds / 3600))" \
    "$(((total_seconds % 3600) / 60))" \
    "$((total_seconds % 60))"
}
job_start=$(date +%s)
echo "Starting Ca2N R2SCAN ecutwfc SCF sweep: $(date)"

for input in scf_*.in; do
  stem="${input%.in}"
  output="${stem}.out"
  case_start=$(date +%s)
  echo "[$(date)] Running ${input}"
  srun pw.x -in "$input" > "$output"
  case_elapsed=$(( $(date +%s) - case_start ))
  echo "[$(date)] ${input} elapsed time: $(format_elapsed "$case_elapsed")"
done

job_elapsed=$(( $(date +%s) - job_start ))
echo "Completed sweep at $(date)"
echo "Total job elapsed time: $(format_elapsed "$job_elapsed")"
