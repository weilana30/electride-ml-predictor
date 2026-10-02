#!/bin/bash
set -euo pipefail
script_dir="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$script_dir"

echo "Submitting Ca2N R2SCAN smearing_or_conv_thr SCF cases"
for job in submit_scf_*.slurm; do
  sbatch --chdir="$script_dir" "$script_dir/$job"
done
