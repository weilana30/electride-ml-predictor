# Direct confirmation study

This study repeats three predeclared models over three new seeds on the [existing family split](PLAN.md) used by the architecture and feature studies. It does not create a new split. Training uses the existing 40,031/4,969/5,000 train/validation/test assignment and reports its known limits.

The frozen configurations and protocol are in [sweep_spec.yaml](sweep_spec.yaml). `train_confirmation.py` handles validation-only fitting and checkpoint-only final evaluation; `report.py` aggregates the nine runs and family-level paired comparisons. External artifacts are under `/home/USER/elfnet-runs/elf_void_gnn/confirmation-study-2026-10-28/`.
