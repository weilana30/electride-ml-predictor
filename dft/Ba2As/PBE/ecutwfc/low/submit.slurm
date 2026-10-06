#!/bin/bash
#SBATCH --job-name=Ba2As_pbe_ecutwfc_low
#SBATCH --account=andre176
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH --mem=16G
#SBATCH --partition=cpu

set -euo pipefail
case_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "$case_dir/../../../../.." && pwd)
exec bash "$project_root/dft/run_scf_case.sh" "$project_root" "$case_dir" "scf.in" "Ba2As" "ecutwfc" "low"
