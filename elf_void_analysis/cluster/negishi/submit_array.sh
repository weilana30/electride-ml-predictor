#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$script_dir/negishi.env"
export ELFNET_CLUSTER_DIR="$script_dir"
manifest="$ELFNET_RUN_ROOT/manifest.parquet"
if [[ ! -s "$manifest" ]]; then
  echo "Manifest is missing. Wait for prepare_manifest.slurm to finish successfully." >&2
  exit 2
fi
mkdir -p "$ELFNET_RUN_ROOT/logs" "$ELFNET_RUN_ROOT/chunks"
module load conda
conda activate "$ELFNET_CONDA_PREFIX"
row_count="$(python - "$manifest" <<'PY'
import sys
import pyarrow.parquet as pq
print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)
PY
)"
task_count=$(((row_count + ELFNET_SAMPLES_PER_TASK - 1) / ELFNET_SAMPLES_PER_TASK))
if (( task_count < 1 )); then
  echo "Manifest has no sample rows" >&2
  exit 2
fi
if [[ -e "$ELFNET_RUN_ROOT/array_job_id.txt" ]]; then
  echo "Array job already submitted for this run. Use a fresh run ID to resubmit from scratch." >&2
  exit 2
fi

array_job_id="$(sbatch --parsable \
  --export=ALL \
  --account="$ELFNET_SLURM_ACCOUNT" \
  --qos="$ELFNET_SLURM_QOS" \
  --partition="$ELFNET_SLURM_PARTITION" \
  --array="0-$((task_count - 1))%$ELFNET_ARRAY_CONCURRENCY" \
  --output="$ELFNET_RUN_ROOT/logs/build-%A_%a.out" \
  --error="$ELFNET_RUN_ROOT/logs/build-%A_%a.err" \
  "$script_dir/build_chunk.slurm")"
array_job_id="${array_job_id%%;*}"
printf '%s\n' "$array_job_id" > "$ELFNET_RUN_ROOT/array_job_id.txt"
merge_job_id="$(sbatch --parsable \
  --export=ALL \
  --account="$ELFNET_SLURM_ACCOUNT" \
  --qos="$ELFNET_SLURM_QOS" \
  --partition="$ELFNET_SLURM_PARTITION" \
  --dependency="afterany:$array_job_id" \
  --output="$ELFNET_RUN_ROOT/logs/merge-%j.out" \
  --error="$ELFNET_RUN_ROOT/logs/merge-%j.err" \
  "$script_dir/merge.slurm")"
printf '%s\n' "$merge_job_id" > "$ELFNET_RUN_ROOT/merge_job_id.txt"
echo "Manifest rows: $row_count"
echo "Chunk size: $ELFNET_SAMPLES_PER_TASK"
echo "Array tasks: $task_count, max concurrent: $ELFNET_ARRAY_CONCURRENCY"
echo "Submitted array $array_job_id and dependent merge/validation job $merge_job_id"
