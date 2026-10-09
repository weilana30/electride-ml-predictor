#!/bin/bash
#SBATCH --job-name=MgS_cutoff_600
#SBATCH --account=andre176
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH --mem=24G
#SBATCH --partition=cpu

set -euo pipefail
module load cp2k/2022.1
exe=${CP2K_EXECUTABLE:-cp2k.psmp}
command -v "$exe" >/dev/null || { echo "CP2K executable not found: $exe" >&2; exit 127; }
case_dir=$(pwd -P)
scratch_base="${RCAC_SCRATCH:-/scratch/negishi/$USER}/electride-ml-predictor/cp2k"
scratch_dir="$scratch_base/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$scratch_dir"
cp cp2k.inp "$scratch_dir/"
finish() {
  status=$?
  if [[ -f "$scratch_dir/cp2k.out" ]]; then cp "$scratch_dir/cp2k.out" "$case_dir/"; fi
  for output in "$scratch_dir"/*-ELF*.cube; do
    if [[ -f "$output" ]]; then cp "$output" "$case_dir/"; fi
  done
  if [[ "$status" -eq 0 ]]; then
    rm -rf "$scratch_dir"
  else
    echo "CP2K scratch retained at $scratch_dir" >&2
  fi
  exit "$status"
}
trap finish EXIT
cd "$scratch_dir"
srun "$exe" -i cp2k.inp -o cp2k.out
