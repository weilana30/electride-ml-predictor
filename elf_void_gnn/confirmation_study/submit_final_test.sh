#!/bin/bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/home/weilana/elfnet-runs/elf_void_gnn/confirmation-study-2026-10-28}"
MANIFEST="$EXPERIMENT_ROOT/configs/manifest.csv"
REPORT_DIR="$EXPERIMENT_ROOT/report"
PYTHON="/home/weilana/.conda/envs/elfnet-train/bin/python"

# Refuse the launch unless the report and each predeclared validation run are complete.
"$PYTHON" - "$MANIFEST" "$REPORT_DIR/validation_stage.json" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

manifest_path, stage_path = map(Path, sys.argv[1:])
if not manifest_path.is_file():
    raise SystemExit(f"Missing manifest: {manifest_path}")
if not stage_path.is_file():
    raise SystemExit(f"Missing completed validation report marker: {stage_path}")
rows = list(csv.DictReader(manifest_path.open(newline="")))
if len(rows) != 9 or len({row["run_id"] for row in rows}) != 9:
    raise SystemExit(f"Expected exactly nine unique manifest runs; found {len(rows)}")
stage = json.loads(stage_path.read_text())
expected_rule = "mean pooled validation RMSE across three predeclared seeds"
if stage.get("manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
    raise SystemExit("Validation report marker does not match the current manifest")
if stage.get("selection_rule") != expected_rule or stage.get("selected_configuration") not in {
    "w128_d2_baseline", "w128_d2_atom_only", "w64_d2_atom_only"
}:
    raise SystemExit("Validation report marker lacks the frozen predeclared selection")
for row in rows:
    run = Path(row["run_dir"])
    metrics_path = run / "metrics.json"
    checkpoint = run / "best_model.pt"
    prediction = run / "validation_predictions.parquet"
    frozen_path = run / "validation_selection_frozen.json"
    for required in (metrics_path, checkpoint, prediction, frozen_path):
        if not required.is_file() or required.stat().st_size == 0:
            raise SystemExit(f"Incomplete run {row['run_id']}: missing/empty {required}")
    metrics = json.loads(metrics_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    if (metrics.get("status") != "completed"
            or metrics.get("evaluation_mode") != "validation-only"
            or metrics.get("test") is not None
            or metrics.get("test_evaluation") is not None
            or metrics.get("seed") != int(row["seed"])
            or metrics.get("configuration_id") != row["configuration_id"]
            or not metrics.get("validation")
            or metrics.get("best_epoch") is None
            or metrics.get("best_validation_rmse") is None):
        raise SystemExit(f"Validation metrics are incomplete or inconsistent for {row['run_id']}")
    if (frozen.get("status") != "frozen"
            or frozen.get("selection_rule") != expected_rule
            or frozen.get("selected_configuration") != stage["selected_configuration"]
            or frozen.get("configuration_id") != row["configuration_id"]
            or frozen.get("seed") != int(row["seed"])
            or frozen.get("split_sha256") != row["split_sha256"]):
        raise SystemExit(f"Frozen selection marker is incomplete or inconsistent for {row['run_id']}")
    if (run / "test_predictions.parquet").exists():
        raise SystemExit(f"Test evaluation already exists for {row['run_id']}; refusing to repeat it")
print("Preflight passed: validation stage and all nine frozen validation runs are complete.")
PY

mkdir -p "$EXPERIMENT_ROOT/logs"
sbatch --account="${SLURM_ACCOUNT:?set SLURM_ACCOUNT to your Slurm allocation}" --array="0-8%2" --export="ALL,MANIFEST=$MANIFEST" \
  --output="$EXPERIMENT_ROOT/logs/final-test-%A_%a.out" \
  --error="$EXPERIMENT_ROOT/logs/final-test-%A_%a.err" \
  "$REPO_ROOT/elf_void_gnn/confirmation_study/final_test_array.sbatch"
