#!/usr/bin/env bash
#SBATCH --job-name=elfnet-build
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$script_dir/negishi.env"
cd "$ELFNET_REPO"
module load conda
conda activate "$ELFNET_CONDA_PREFIX"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

manifest="$ELFNET_RUN_ROOT/manifest.parquet"
if [[ ! -s "$manifest" ]]; then
  echo "Manifest is missing: $manifest" >&2
  exit 2
fi

chunk_count="$(python - "$manifest" "$ELFNET_SAMPLES_PER_TASK" <<'PY'
import math
import sys
import pyarrow.parquet as pq

row_count = pq.ParquetFile(sys.argv[1]).metadata.num_rows
chunk_size = int(sys.argv[2])
print(math.ceil(row_count / chunk_size))
PY
)"
if (( chunk_count < 1 )); then
  echo "Manifest contains no structures" >&2
  exit 2
fi
export ELFNET_CHUNK_COUNT="$chunk_count"

mkdir -p "$ELFNET_RUN_ROOT/chunks"
echo "Running $chunk_count chunks across $SLURM_NTASKS workers on one Negishi node"
srun --ntasks="$SLURM_NTASKS" --cpus-per-task=1 --kill-on-bad-exit=0 \
  bash "$script_dir/node_worker.sh"
