#!/usr/bin/env bash
#SBATCH --job-name=elfnet-manifest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00

set -euo pipefail
: "${ELFNET_CLUSTER_DIR:?Submit through submit_prepare.sh so cluster paths are exported}"
cd "$ELFNET_REPO"
module load conda
conda activate "$ELFNET_CONDA_PREFIX"

if [[ -e "$ELFNET_RUN_ROOT/manifest.parquet" ]]; then
  echo "Prepared manifest already exists: $ELFNET_RUN_ROOT/manifest.parquet" >&2
  exit 2
fi
mkdir -p "$ELFNET_RUN_ROOT/chunks" "$ELFNET_RUN_ROOT/logs"
hash_args=()
if [[ "$ELFNET_NO_HASHES" == "1" ]]; then
  hash_args+=(--no-hashes)
fi
python -m elf_void_analysis.build_manifest \
  --source-root "$ELFNET_SOURCE_ROOT" \
  --output "$ELFNET_RUN_ROOT/manifest.parquet" \
  "${hash_args[@]}" \
  --json-report "$ELFNET_RUN_ROOT/manifest_report.json"
