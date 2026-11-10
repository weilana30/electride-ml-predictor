#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$script_dir/negishi.env"

module load conda
if [[ ! -d "$ELFNET_CONDA_PREFIX/conda-meta" ]]; then
  conda create --prefix "$ELFNET_CONDA_PREFIX" python=3.11 pip -y
fi
conda activate "$ELFNET_CONDA_PREFIX"
python -m pip install -r "$ELFNET_REPO/elf_void_analysis/requirements.txt"
python - <<'PY'
from importlib.metadata import version
import h5py, numpy, pyarrow, scipy
print("ELFNet extraction environment is ready")
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("pymatgen", version("pymatgen"))
print("h5py", h5py.__version__)
print("pyarrow", pyarrow.__version__)
PY
