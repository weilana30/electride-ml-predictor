#!/usr/bin/env bash
#SBATCH --job-name=elfnet-merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00

set -euo pipefail
: "${ELFNET_CLUSTER_DIR:?Submit through submit_node.sh so cluster paths are exported}"
cd "$ELFNET_REPO"
module load conda
conda activate "$ELFNET_CONDA_PREFIX"

python -m elf_void_analysis.merge_dataset_chunks \
  --manifest "$ELFNET_RUN_ROOT/manifest.parquet" \
  --chunks-root "$ELFNET_RUN_ROOT/chunks" \
  --output "$ELFNET_FINAL_OUTPUT" \
  --source-root "$ELFNET_SOURCE_ROOT" \
  --config "$ELFNET_REPO/elf_void_analysis/dataset_config.json" \
  --samples-per-task "$ELFNET_SAMPLES_PER_TASK"

python -m elf_void_analysis.validate_dataset \
  --shards "$ELFNET_FINAL_OUTPUT/shards/*.h5" \
  --manifest "$ELFNET_FINAL_OUTPUT/manifest.parquet" \
  --splits "$ELFNET_FINAL_OUTPUT/splits.parquet" \
  --schema "$ELFNET_FINAL_OUTPUT/schema.json" \
  --source-root "$ELFNET_SOURCE_ROOT" \
  --require-full-coverage \
  --report-json "$ELFNET_FINAL_OUTPUT/validation.json"
