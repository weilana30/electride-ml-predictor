"""Redraw the poster model-comparison bar chart from a committed summary JSON.

The full pipeline (make_gnn_figures.py) needs prediction files kept on Negishi.
Its summary.json already holds every number the bar chart shows, so the chart
can be redrawn locally, for example at a different aspect ratio:
    python poster/build/make_comparison_from_summary.py [SUMMARY_JSON] [OUT_STEM] [WIDTH HEIGHT]
SUMMARY_JSON defaults to elf_void_gnn/architecture_test/results_voidv2/poster_summary.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_gnn_figures import comparison

ROOT = Path(__file__).resolve().parents[2]
SHOWN = {"Boosted decision trees", "Averaged-atom neural network", "Void-only neural network"}

summary_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "elf_void_gnn/architecture_test/results_voidv2/poster_summary.json"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "poster/figures/gnn_model_comparison"
figsize = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (7.0, 6.2)
summary = json.loads(summary_path.read_text())
# Same selection as make_gnn_figures.main: the selected graph model and the non-graph baselines.
models = ([m for m in summary["confirmation_comparison"] if m["label"] == "Atom–void graph network"]
          + [m for m in summary["models"] if m["label"] in SHOWN])
assert len(models) == 4, [m["label"] for m in models]
comparison(models, summary["predict_average_rmse"], out, figsize)
print("wrote", out, [(m["label"], round(m["test_rmse"][0], 3)) for m in models])
