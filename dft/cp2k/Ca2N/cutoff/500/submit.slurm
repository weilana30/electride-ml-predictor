#!/bin/bash
#SBATCH --job-name=Ca2N_cutoff_500
#SBATCH --account=andre176
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH --mem=20G
#SBATCH --partition=cpu

set -euo pipefail
module load cp2k/2022.1
exe=${CP2K_EXECUTABLE:-cp2k.psmp}
command -v "$exe" >/dev/null || { echo "CP2K executable not found: $exe" >&2; exit 127; }
case_dir=$(pwd -P)
scratch_base="${RCAC_SCRATCH:-/scratch/negishi/$USER}/electride-ml-predictor/cp2k"
scratch_dir="$scratch_base/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
workflow_dir=$(cd "$case_dir/../../.." && pwd -P)
candidate_file="$case_dir/../../structure/void_candidates.csv"
mkdir -p "$scratch_dir"
cp cp2k.inp "$scratch_dir/"
finish() {
  status=$?
  if [[ -f "$scratch_dir/cp2k.out" ]]; then cp "$scratch_dir/cp2k.out" "$case_dir/"; fi
  for report in "$scratch_dir"/*.void_elf.csv; do
    if [[ -f "$report" ]]; then cp "$report" "$case_dir/"; fi
  done
  echo "CP2K scratch retained at $scratch_dir" >&2
  exit "$status"
}
trap finish EXIT
cd "$scratch_dir"
srun "$exe" -i cp2k.inp -o cp2k.out
found_cube=0
for cube in "$scratch_dir"/*-ELF*.cube; do
  [[ -f "$cube" ]] || continue
  found_cube=1
  report="$scratch_dir/$(basename "$cube").void_elf.csv"
  python3 "$workflow_dir/analyze_void_elf.py" "$cube" "$candidate_file" --output "$report"
done
if [[ "$found_cube" -eq 0 ]]; then
  echo "CP2K completed without producing an ELF cube in $scratch_dir" >&2
  exit 1
fi
