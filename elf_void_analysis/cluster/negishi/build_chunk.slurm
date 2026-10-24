#!/usr/bin/env bash
#SBATCH --job-name=elfnet-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$script_dir/negishi.env"
cd "$ELFNET_REPO"
module load conda
conda activate "$ELFNET_CONDA_PREFIX"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

task_index="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task}"
task_name="task-$(printf '%05d' "$task_index")"
task_output="$ELFNET_RUN_ROOT/chunks/$task_name"
if [[ -f "$task_output/build_report.json" ]]; then
  echo "Task already completed: $task_output"
  exit 0
fi
if [[ -e "$task_output" ]]; then
  echo "Partial task output already exists at $task_output; use a fresh run ID or remove that task directory after inspection." >&2
  exit 2
fi

python -m elf_void_analysis.build_dataset \
  --source-root "$ELFNET_SOURCE_ROOT" \
  --manifest "$ELFNET_RUN_ROOT/manifest.parquet" \
  --chunk-index "$task_index" \
  --samples-per-task "$ELFNET_SAMPLES_PER_TASK" \
  --output "$task_output" \
  --progress-every 25
