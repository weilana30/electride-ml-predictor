"""Parity plot in the original poster style, drawn from the confirmation-study predictions.

Uses the atom-only width-64 GNN, seed 20261028 (the median-RMSE seed of three). The per-void
predictions are kept outside Git, on Negishi:
  /home/weilana/elfnet-runs/elf_void_gnn/confirmation-study-2026-10-28/runs/w64_d2_atom_only-seed20261028/test_predictions.parquet
Copy that file locally, then:
    python poster/build/make_parity_confirmation.py PATH/TO/test_predictions.parquet [OUT_STEM]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_gnn_figures import parity

src = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "figures" / "gnn_parity_confirmation"
frame = pd.read_parquet(src, columns=["y_true", "y_pred"])
assert len(frame) == 88680
y, p = frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()
print(f"R2 {1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2):.3f}  RMSE {np.sqrt(np.mean((y - p) ** 2)):.4f}")
parity(frame, out)
print("wrote", out)
