#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$script_dir/negishi.env"
mkdir -p "$ELFNET_RUN_ROOT/logs"
if [[ -e "$ELFNET_RUN_ROOT/manifest.parquet" ]]; then
  echo "Run manifest already exists. Set a new ELFNET_RUN_ID or inspect the existing run." >&2
  exit 2
fi
job_id="$(sbatch --parsable \
  --account="$ELFNET_SLURM_ACCOUNT" \
  --qos="$ELFNET_SLURM_QOS" \
  --partition="$ELFNET_SLURM_PARTITION" \
  --output="$ELFNET_RUN_ROOT/logs/manifest-%j.out" \
  --error="$ELFNET_RUN_ROOT/logs/manifest-%j.err" \
  "$script_dir/prepare_manifest.slurm")"
printf '%s\n' "$job_id" > "$ELFNET_RUN_ROOT/manifest_job_id.txt"
echo "Submitted manifest job $job_id"
