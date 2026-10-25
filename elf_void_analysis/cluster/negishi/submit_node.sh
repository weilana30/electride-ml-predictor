#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$script_dir/negishi.env"
export ELFNET_CLUSTER_DIR="$script_dir"
manifest="$ELFNET_RUN_ROOT/manifest.parquet"
if [[ ! -s "$manifest" ]]; then
  echo "Manifest is missing. Submit and finish submit_prepare.sh first." >&2
  exit 2
fi
if [[ -e "$ELFNET_RUN_ROOT/node_job_id.txt" || -e "$ELFNET_RUN_ROOT/merge_job_id.txt" ]]; then
  echo "Build or merge already submitted for this run. Use a new ELFNET_RUN_ID for a clean run." >&2
  exit 2
fi
mkdir -p "$ELFNET_RUN_ROOT/logs"

node_job_id="$(sbatch --parsable \
  --export=ALL \
  --account="$ELFNET_SLURM_ACCOUNT" \
  --qos="$ELFNET_SLURM_QOS" \
  --partition="$ELFNET_SLURM_PARTITION" \
  --output="$ELFNET_RUN_ROOT/logs/build-node-%j.out" \
  --error="$ELFNET_RUN_ROOT/logs/build-node-%j.err" \
  "$script_dir/build_node.slurm")"
node_job_id="${node_job_id%%;*}"
printf '%s\n' "$node_job_id" > "$ELFNET_RUN_ROOT/node_job_id.txt"

merge_job_id="$(sbatch --parsable \
  --export=ALL \
  --account="$ELFNET_SLURM_ACCOUNT" \
  --qos="$ELFNET_SLURM_QOS" \
  --partition="$ELFNET_SLURM_PARTITION" \
  --dependency="afterany:$node_job_id" \
  --output="$ELFNET_RUN_ROOT/logs/merge-%j.out" \
  --error="$ELFNET_RUN_ROOT/logs/merge-%j.err" \
  "$script_dir/merge.slurm")"
printf '%s\n' "$merge_job_id" > "$ELFNET_RUN_ROOT/merge_job_id.txt"

echo "Submitted single-node build job $node_job_id (128 workers)"
echo "Submitted dependent merge/validation job $merge_job_id"
