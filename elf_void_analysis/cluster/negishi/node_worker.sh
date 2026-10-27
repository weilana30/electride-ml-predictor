#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$script_dir/negishi.env"
cd "$ELFNET_REPO"

worker_id="${SLURM_PROCID:?Must be launched by srun}"
worker_count="${SLURM_NTASKS:?Missing Slurm task count}"
chunk_count="${ELFNET_CHUNK_COUNT:?Missing chunk count}"
worker_status=0

for ((chunk_index = worker_id; chunk_index < chunk_count; chunk_index += worker_count)); do
  task_name="task-$(printf '%05d' "$chunk_index")"
  task_output="$ELFNET_RUN_ROOT/chunks/$task_name"
  if [[ -f "$task_output/build_report.json" ]]; then
    echo "Worker $worker_id: already complete: $task_name"
    continue
  fi
  if [[ -e "$task_output" ]]; then
    echo "Worker $worker_id: partial output exists: $task_output" >&2
    worker_status=1
    continue
  fi

  echo "Worker $worker_id: processing $task_name"
  if python -m elf_void_analysis.build_dataset \
    --source-root "$ELFNET_SOURCE_ROOT" \
    --manifest "$ELFNET_RUN_ROOT/manifest.parquet" \
    --chunk-index "$chunk_index" \
    --samples-per-task "$ELFNET_SAMPLES_PER_TASK" \
    --output "$task_output" \
    --progress-every 25; then
    :
  elif [[ -f "$task_output/build_report.json" ]]; then
    echo "Worker $worker_id: $task_name completed with sample failures; see build_report.json" >&2
  else
    echo "Worker $worker_id: $task_name failed before writing a build report" >&2
    worker_status=1
  fi
done

exit "$worker_status"
