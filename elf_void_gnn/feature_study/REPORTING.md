# Screening run reports

After Stage A or Stage B finishes, create the validation report from the Negishi login node (or another host with the run directory and the same dataset split file):

```bash
cd /home/weilana/electride-ml-predictor
/home/weilana/.conda/envs/elfnet-train/bin/python \
  -m elf_void_gnn.feature_study.report_validation \
  --runs-root /home/weilana/elfnet-runs/feature-study \
  --splits /home/weilana/elfnet-datasets/elfnet-gnn-2026-10-23/splits.parquet \
  --out-dir /home/weilana/elfnet-runs/feature-study/analysis/validation
```

The script discovers Stage A/B run directories recursively, reads split counts and validation family IDs from the supplied `splits.parquet`, and includes only completed runs with `evaluation_mode: validation-only`, no test metrics, matching saved `config.yaml` dataset root, matching split counts and validation family IDs, and the complete fixed HeteroMPNN configuration/signature (including its 80,001 parameters). Incomplete or ineligible runs are listed in `excluded_runs.csv`. Each variant is paired to the baseline with the same seed when present. The `split_id` in the generated manifest is a fingerprint of the caller-supplied split file, not an in-run fingerprint. Existing metrics did not record a split-file hash, so matching the saved dataset path, split counts, and validation family IDs checks consistency but cannot prove that the file contents were unchanged during the historical run. The report will fail loudly if a run has duplicate variant/seed results or if metrics are inconsistent with the aggregate schema.

Outputs are deliberately compact: `run_manifest.csv`, `excluded_runs.csv`, `run_metrics.csv`, `paired_effects.csv`, `interactions.csv`, `performance.png`, and `feature_effects.png`. The tables report pooled and macro-family metrics plus paired per-family bootstrap intervals. Screening always uses validation. `aggregate_runs.py --split test` remains a separate manual operation for a frozen final-test manifest after the shortlist is locked; do not use `report_validation.py` to inspect test results.

Install the plotting/data dependencies into the training environment once, if needed, with:

```bash
/home/weilana/.conda/envs/elfnet-train/bin/python -m pip install \
  -r elf_void_gnn/feature_study/analysis-requirements.txt
```

After the frozen test evaluation and compact summary are complete, generate the
ranked redundancy and predicted-versus-actual diagnostics with:

```bash
python -m elf_void_gnn.feature_study.plot_final_diagnostics \
  --report-dir /home/weilana/elfnet-runs/feature-study/analysis/final-report \
  --test-predictions /home/weilana/elfnet-runs/feature-study/final-test/job-44702592/test_predictions.parquet
```

This reads the existing report tables and frozen prediction artifact without
training or evaluating a model. It writes `top_feature_redundancies.png` and
`test_predicted_vs_actual.png` into the report directory.
