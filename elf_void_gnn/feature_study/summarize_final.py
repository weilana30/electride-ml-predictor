#!/usr/bin/env python3
"""Summarize frozen feature-study validation/test metrics without model evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VARIANTS = {
    "baseline": "baseline",
    "atomic_number_nearest_void_distances_edge_distances": "minimal",
    "atom_full_chemistry_and_periodicity": "atom-only",
}
ORDER = ("baseline", "minimal", "atom-only")
METRICS = ("rmse", "mae", "r2")
EXPECTED_SEEDS = {"20261023", "20261024", "20261025"}


def read_validation(roots: list[Path]) -> dict[tuple[str, str], dict]:
    results = {}
    for root in roots:
        paths = [root] if root.is_file() else sorted(root.rglob("metrics.json"))
        for path in paths:
            data = json.loads(path.read_text())
            variant = data.get("feature_variant")
            label = VARIANTS.get(variant)
            val = data.get("validation")
            if label and isinstance(val, dict) and isinstance(val.get("pooled"), dict):
                key = (label, str(data.get("seed")))
                if key in results:
                    raise ValueError(f"duplicate validation run: {key}")
                results[key] = val
    return results


def read_test(path: Path) -> dict[tuple[str, str], dict]:
    data = json.loads(path.read_text())
    out = {}
    for checkpoint in data.get("checkpoints", []):
        label = VARIANTS.get(checkpoint.get("variant"))
        if label:
            key = (label, str(checkpoint.get("seed")))
            if key in out:
                raise ValueError(f"duplicate final-test result: {key}")
            out[key] = checkpoint["test"]
    return out


def mean_sd(vals):
    vals = np.asarray(vals, dtype=float)
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def summarize(results, split):
    rows = []
    for variant in ORDER:
        runs = [m for (v, _), m in results.items() if v == variant]
        if not runs:
            continue
        row = {"split": split, "variant": variant, "n_seeds": len(runs)}
        for metric in METRICS:
            mean, sd = mean_sd([r["pooled"][metric] for r in runs])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
        rows.append(row)
    return rows


def write_csv(path, rows):
    cols = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--final-test-metrics", type=Path, required=True)
    p.add_argument("--validation-runs-root", type=Path, action="append", required=True,
                   help="Can be repeated; recursively reads metrics.json files")
    p.add_argument("--ablation-csv", type=Path, required=True)
    p.add_argument("--correlation-csv", type=Path, required=True)
    p.add_argument("--correlation-heatmap", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    validation, test = read_validation(a.validation_runs_root), read_test(a.final_test_metrics)
    val_rows, test_rows = summarize(validation, "validation"), summarize(test, "test")
    required = set(ORDER)
    for split, rows in (("validation", val_rows), ("test", test_rows)):
        if {r["variant"] for r in rows} != required:
            raise ValueError(f"{split}: expected exactly {sorted(required)}; found {[r['variant'] for r in rows]}")
        if {r["n_seeds"] for r in rows} != {3}:
            raise ValueError(f"{split}: expected three seeds per variant")
    for split, results in (("validation", validation), ("test", test)):
        for variant in ORDER:
            seeds = {seed for (label, seed) in results if label == variant}
            if seeds != EXPECTED_SEEDS:
                raise ValueError(f"{split}: incorrect seeds for {variant}: {sorted(seeds)}")
    write_csv(a.out_dir / "pooled_metrics.csv", val_rows + test_rows)

    family_rows = []
    family_sets = []
    for variant in ORDER:
        per_seed = [m["per_family"] for (v, _), m in test.items() if v == variant]
        seed_family_sets = [set(f) for f in per_seed]
        if len({tuple(sorted(ids)) for ids in seed_family_sets}) != 1:
            raise ValueError(f"test family IDs differ between seeds for {variant}")
        ids = seed_family_sets[0]
        if len(ids) != 9:
            raise ValueError(f"expected nine test families for {variant}; found {len(ids)}")
        family_sets.append(ids)
        for family in sorted(ids):
            row = {"variant": variant, "family_id": family}
            for metric in METRICS:
                mean, sd = mean_sd([f[family][metric] for f in per_seed])
                row[f"{metric}_mean"] = mean
                row[f"{metric}_sd"] = sd
            family_rows.append(row)
    write_csv(a.out_dir / "test_family_metrics.csv", family_rows)
    if len({tuple(sorted(s)) for s in family_sets}) != 1:
        raise ValueError("test family IDs differ between variants")

    # Preserve source evidence beside the derived summary.
    shutil.copy2(a.ablation_csv, a.out_dir / "feature_ablation.csv")
    shutil.copy2(a.correlation_csv, a.out_dir / "feature_correlations.csv")
    shutil.copy2(a.correlation_heatmap, a.out_dir / "train_feature_correlation_heatmap.png")
    correlation_rows = list(csv.DictReader(a.correlation_csv.open(newline="")))
    structure_correlations = []
    for row in correlation_rows:
        if row.get("statistic") != "spearman" or row.get("estimator") != "pooled_structure_means":
            continue
        try:
            value = float(row["correlation"])
        except (KeyError, ValueError):
            continue
        structure_correlations.append((abs(value), value, row))
    strongest_correlations = [
        dict(row, absolute_correlation=absolute, correlation=value)
        for absolute, value, row in sorted(
            structure_correlations, key=lambda item: item[0], reverse=True
        )[:20]
    ]
    if strongest_correlations:
        write_csv(a.out_dir / "top_correlations.csv", strongest_correlations)
    try:
        ab = list(csv.DictReader(a.ablation_csv.open(newline="")))
        candidates = [(r, float(r["pooled_rmse_delta"])) for r in ab if r.get("pooled_rmse_delta") not in (None, "")]
    except (KeyError, ValueError):
        candidates = []
    if candidates:
        def variant_label(row):
            if row.get("variant"):
                return row["variant"]
            parts = Path(row.get("run_id", "")).parts
            return parts[-2] if len(parts) >= 2 else row.get("run_id", "")

        write_csv(
            a.out_dir / "ablation_rmse_deltas.csv",
            [dict(variant=variant_label(row), pooled_rmse_delta=delta)
             for row, delta in candidates],
        )
        fig, ax = plt.subplots(figsize=(7.2, max(3, len(candidates)*.24)))
        ordered = sorted(candidates, key=lambda item: item[1])
        labels = [variant_label(row) for row, _ in ordered]
        vals = [d for _, d in ordered]
        ax.barh(np.arange(len(vals)), vals, color="#4477AA")
        ax.set_yticks(np.arange(len(vals)), labels)
        ax.axvline(0, color="black", lw=.8)
        ax.set_xlabel("Validation pooled RMSE delta (variant − baseline)")
        ax.grid(axis="x", alpha=.22)
        fig.tight_layout()
        fig.savefig(a.out_dir / "validation_ablation_deltas.png", dpi=300)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7), sharey=True)
    for ax, rows, split in ((axes[0], val_rows, "Validation"), (axes[1], test_rows, "Test")):
        means = [r["rmse_mean"] for r in rows]
        sds = [r["rmse_sd"] for r in rows]
        ax.errorbar(ORDER, means, yerr=sds, marker="o", capsize=3, color="#4477AA")
        ax.set_title(split)
        ax.grid(axis="y", alpha=.25)
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Pooled RMSE (mean ± sample SD across 3 seeds)")
    fig.tight_layout()
    fig.savefig(a.out_dir / "pooled_rmse.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, max(3, len(family_rows)*.24)))
    for i, variant in enumerate(ORDER):
        rows = [r for r in family_rows if r["variant"] == variant]
        y = np.arange(len(rows)) + (i-1)*.22
        ax.errorbar([r["rmse_mean"] for r in rows], y, xerr=[r["rmse_sd"] for r in rows], fmt="o", capsize=2, label=variant)
    ids = sorted(family_sets[0])
    ax.set_yticks(np.arange(len(ids)), ids)
    ax.set_xlabel("Test family RMSE (mean ± sample SD across seeds)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(a.out_dir / "test_family_rmse.png", dpi=300)
    plt.close(fig)

    lines = ["# Final feature study summary", "", "Pooled metrics are mean ± sample SD across three seeds. Test values are read from the supplied frozen `final_test_metrics.json`; this report does not evaluate models or read test structures.", "", "## Pooled metrics", "", "| Split | Variant | RMSE | MAE | R² |", "|---|---|---:|---:|---:|"]
    for r in val_rows + test_rows:
        lines.append(f"| {r['split']} | {r['variant']} | {r['rmse_mean']:.4g} ± {r['rmse_sd']:.2g} | {r['mae_mean']:.4g} ± {r['mae_sd']:.2g} | {r['r2_mean']:.4g} ± {r['r2_sd']:.2g} |")
    lines += [
        "",
        "The fixed-architecture study keeps the same graph topology, feature tensor widths, and 80,001-parameter HeteroMPNN across variants. The minimal anchor retains atomic number, nearest-atom void distances, and edge distances. Atom-only zeroes the void descriptors while retaining the full atom descriptors and edge-distance/topology inputs. Validation is used for feature selection; test metrics are shown once from the frozen evaluation artifact.",
    ]
    if strongest_correlations:
        lines += [
            "",
            "## Strongest training-only correlations",
            "",
            "| Feature A | Feature B | Spearman ρ |",
            "|---|---|---:|",
        ]
        for row in strongest_correlations[:5]:
            lines.append(
                f"| {row['feature_a']} | {row['feature_b']} | "
                f"{float(row['correlation']):.3f} |"
            )
    lines += [
        "",
        "## Limitations",
        "",
        "Seed-to-seed SD describes variability across these three fixed runs; it is not a confidence interval. Per-family test summaries require the same nine family IDs in every seed. Ablation differences are validation evidence and remain associational: correlated inputs, finite data, and retraining can affect the result. The existing split holds out named phase families, but related composition or pressure variants may cross splits.",
        "",
        "Inputs copied into this report include feature ablation and training-only correlation tables. Figures: `pooled_rmse.png`, `test_family_rmse.png`, `validation_ablation_deltas.png`, and `train_feature_correlation_heatmap.png`.",
    ]
    (a.out_dir / "report.md").write_text("\n".join(lines)+"\n")
    print(f"Wrote final feature-study report to {a.out_dir}")


if __name__ == "__main__":
    main()
