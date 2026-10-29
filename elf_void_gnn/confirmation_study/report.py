"""Aggregate the predeclared nine-run confirmation study.

Validation summaries implement the frozen selection rule. Test summaries are
descriptive only and are never used to choose a configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONFIGURATIONS = ("w128_d2_baseline", "w128_d2_atom_only", "w64_d2_atom_only")
SEEDS = (20261028, 20261029, 20261030)
COMPARISONS = (("w128_d2_atom_only", "w128_d2_baseline"),
               ("w128_d2_atom_only", "w64_d2_atom_only"))
METRICS = ("mae", "rmse", "r2")


def _metric_frame(pred: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """One row per independent group; each group contributes equally to macro metrics."""
    required = {group_col, "y_true", "y_pred"}
    if not required.issubset(pred):
        raise ValueError(f"predictions require columns {sorted(required)}")
    if pred.empty or pred[list(required)].isna().any().any():
        raise ValueError("predictions must be non-empty and complete")
    if not np.isfinite(pred[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        raise ValueError("predictions contain non-finite values")
    rows = []
    for group, frame in pred.groupby(group_col, sort=True):
        error = frame.y_pred.to_numpy(float) - frame.y_true.to_numpy(float)
        rows.append({"group_id": group, "count": len(frame), "mae": float(np.abs(error).mean()),
                     "rmse": float(np.sqrt(np.mean(error ** 2)))})
    return pd.DataFrame(rows)


def grouped_metrics(pred: pd.DataFrame, group_col: str) -> dict:
    """Compute pooled row metrics and macro group metrics from prediction rows."""
    groups = _metric_frame(pred, group_col)
    error = pred.y_pred.to_numpy(float) - pred.y_true.to_numpy(float)
    truth = pred.y_true.to_numpy(float)
    ss_tot = float(np.square(truth - truth.mean()).sum())
    return {
        "pooled_mae": float(np.abs(error).mean()),
        "pooled_rmse": float(np.sqrt(np.mean(error ** 2))),
        "pooled_r2": float(1 - np.square(error).sum() / ss_tot) if ss_tot else float("nan"),
        "macro_mae": float(groups.mae.mean()), "macro_rmse": float(groups.rmse.mean()),
        "group_count": len(groups), "groups": groups,
    }


def grouped_bootstrap(values: np.ndarray, draws: int = 10000, seed: int = 20261028) -> tuple[float, float]:
    """Percentile interval by resampling group-level values, never individual rows."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap requires finite one-dimensional group values")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(draws, len(values)))
    lo, hi = np.quantile(values[samples].mean(axis=1), [0.025, 0.975])
    return float(lo), float(hi)


def validate_prediction_identity(frame: pd.DataFrame, run_id: str, split: str) -> None:
    """Validate row identity; a sample may legitimately contain many voids."""
    identity = ["sample_id", "void_id"]
    if not set(identity).issubset(frame) or frame[identity].isna().any().any() or frame.duplicated(identity).any():
        raise ValueError(f"{run_id}: duplicate or missing (sample_id, void_id) in {split} predictions")
    if frame.empty:
        raise ValueError(f"{run_id}: empty {split} predictions")


def validate_manifest(manifest: pd.DataFrame) -> None:
    needed = {"run_id", "configuration", "seed", "run_dir"}
    if not needed.issubset(manifest.columns):
        raise ValueError(f"manifest requires columns {sorted(needed)}")
    if manifest.duplicated(["configuration", "seed"]).any():
        raise ValueError("duplicate model/seed combination")
    if manifest.run_id.isna().any() or manifest.run_id.duplicated().any():
        raise ValueError("run_id values must be present and unique")
    expected = {(c, s) for c in CONFIGURATIONS for s in SEEDS}
    found = {(str(r.configuration), int(r.seed)) for r in manifest.itertuples()}
    if found != expected:
        raise ValueError(f"expected exactly nine declared runs; missing={sorted(expected-found)}, extra={sorted(found-expected)}")


def _load_predictions(manifest: pd.DataFrame, split: str, *, required: bool) -> dict[tuple[str, int], pd.DataFrame]:
    loaded = {}
    for r in manifest.itertuples():
        path = Path(r.run_dir) / f"{split}_predictions.parquet"
        if not path.is_file():
            if not required:
                continue
            raise ValueError(f"missing {split} predictions for {r.run_id}: {path}")
        frame = pd.read_parquet(path)
        validate_prediction_identity(frame, str(r.run_id), split)
        loaded[(str(r.configuration), int(r.seed))] = frame
    return loaded


def _run_metadata(row) -> dict:
    path = Path(row.run_dir) / "metrics.json"
    if not path.is_file():
        raise ValueError(f"missing run metrics for {row.run_id}: {path}")
    payload = json.loads(path.read_text())
    performance = payload.get("performance") or {}
    architecture = payload.get("architecture_signature") or {}
    return {"configuration_id": payload.get("configuration_id"),
            "split_sha256": payload.get("split_sha256"),
            "parameter_count": performance.get("parameter_count", architecture.get("parameter_count")),
            "training_seconds": performance.get("training_seconds"),
            "device": performance.get("device"), "best_epoch": payload.get("best_epoch"),
            "best_validation_rmse": payload.get("best_validation_rmse"),
            "evaluation_scope": payload.get("evaluation_scope")}


def aggregate(manifest_path: Path, out_dir: Path, bootstrap_draws: int = 10000,
              include_test: bool = False) -> None:
    manifest = pd.read_csv(manifest_path)
    validate_manifest(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    stage_marker = out_dir / "validation_stage.json"
    if include_test:
        if not stage_marker.is_file():
            raise ValueError("test aggregation requires a completed validation-only aggregation first")
        frozen = json.loads(stage_marker.read_text())
        if frozen.get("manifest_sha256") != manifest_hash or frozen.get("selection_rule") != "mean pooled validation RMSE across three predeclared seeds":
            raise ValueError("validation-stage manifest or frozen selection rule does not match")
        for row in manifest.itertuples():
            marker = Path(row.run_dir) / "validation_selection_frozen.json"
            if not marker.is_file():
                raise ValueError(f"missing validation selection freeze marker for {row.run_id}: {marker}")
    outputs = {}
    splits = ("validation", "test") if include_test else ("validation",)
    for split in splits:
        loaded = _load_predictions(manifest, split, required=True)
        rows, group_rows = [], []
        for (config, seed), pred in loaded.items():
            if "family_id" not in pred:
                raise ValueError("prediction files must include family_id for held-out-family reporting")
            fam = grouped_metrics(pred, "family_id")
            manifest_row = manifest[(manifest.configuration == config) & (manifest.seed.astype(int) == seed)].iloc[0]
            rows.append({"configuration": config, "seed": seed, **_run_metadata(manifest_row),
                         **{k:v for k,v in fam.items() if k != "groups"},
                         "macro_family_rmse": fam["macro_rmse"], "macro_family_mae": fam["macro_mae"]})
            gf = fam["groups"].copy()
            gf.insert(0, "seed", seed); gf.insert(0, "configuration", config); gf.insert(0, "grouping", "family")
            group_rows.extend(gf.to_dict("records"))
        frame = pd.DataFrame(rows)
        frame.to_csv(out_dir / f"{split}_run_metrics.csv", index=False)
        pd.DataFrame(group_rows).to_csv(out_dir / f"{split}_group_metrics.csv", index=False)
        outputs[split] = (frame, loaded)

    val, _ = outputs["validation"]
    summary = val.groupby("configuration", as_index=False).agg(
        mean_validation_pooled_rmse=("pooled_rmse", "mean"), seed_sd_validation_pooled_rmse=("pooled_rmse", "std"),
        mean_validation_pooled_mae=("pooled_mae", "mean"),
        mean_validation_macro_family_rmse=("macro_family_rmse", "mean"))
    summary["selection_rank"] = summary.mean_validation_pooled_rmse.rank(method="min").astype(int)
    summary.to_csv(out_dir / "selection_summary.csv", index=False)
    if not include_test:
        selected = str(summary.sort_values("mean_validation_pooled_rmse").iloc[0].configuration)
        stage_marker.write_text(json.dumps({"manifest_sha256": manifest_hash,
                                            "selection_rule": "mean pooled validation RMSE across three predeclared seeds",
                                            "selected_configuration": selected}, indent=2) + "\n")
        for row in manifest.itertuples():
            meta = _run_metadata(row)
            marker = {"status": "frozen", "seed": int(row.seed),
                      "configuration_id": meta["configuration_id"],
                      "split_sha256": meta["split_sha256"] or getattr(row, "split_sha256", None),
                      "selected_configuration": selected,
                      "selection_rule": "mean pooled validation RMSE across three predeclared seeds"}
            (Path(row.run_dir) / "validation_selection_frozen.json").write_text(json.dumps(marker, indent=2) + "\n")

    pair_rows = []
    pair_group_rows = []
    pair_group_deltas = {}
    for split, (frame, loaded) in outputs.items():
        for model, comparator in COMPARISONS:
            for seed in SEEDS:
                a, b = loaded[(model, seed)], loaded[(comparator, seed)]
                key = ["sample_id", "void_id"]
                ak, bk = a.set_index(key).sort_index(), b.set_index(key).sort_index()
                if not ak.index.equals(bk.index):
                    raise ValueError(f"paired {split} (sample_id, void_id) rows differ for {model} vs {comparator}, seed {seed}")
                am = _metric_frame(a, "family_id").set_index("group_id")
                bm = _metric_frame(b, "family_id").set_index("group_id")
                if set(am.index) != set(bm.index):
                    raise ValueError("paired family IDs differ")
                delta = (am.rmse - bm.rmse).sort_index().to_numpy()
                pair_group_rows.extend(
                    {"split": split, "model": model, "comparator": comparator,
                     "seed": seed, "family_id": str(group_id),
                     "rmse_delta": float(value)}
                    for group_id, value in zip(am.sort_index().index, delta)
                )
                lo, hi = grouped_bootstrap(delta, bootstrap_draws, 20261028 + seed % 997 + len(pair_rows))
                pair_rows.append({"split": split, "model": model, "comparator": comparator, "seed": seed,
                                  "mean_group_rmse_delta": float(delta.mean()), "ci95_low": lo, "ci95_high": hi,
                                  "group_count": len(delta), "interpretation": "descriptive only" if split == "test" else "validation comparison"})
                pair_group_deltas[(split, model, comparator, seed)] = pd.Series(delta, index=am.sort_index().index)
    for split in splits:
        for model, comparator in COMPARISONS:
            seed_series = [pair_group_deltas[(split, model, comparator, seed)] for seed in SEEDS]
            if not (seed_series[0].index.equals(seed_series[1].index) and seed_series[0].index.equals(seed_series[2].index)):
                raise ValueError(f"family IDs differ across seeds for {split} paired comparison")
            mean_by_group = pd.concat(seed_series, axis=1).mean(axis=1).to_numpy()
            lo, hi = grouped_bootstrap(mean_by_group, bootstrap_draws,
                                       20261028 + len(pair_rows))
            pair_rows.append({"split": split, "model": model, "comparator": comparator,
                              "seed": "mean_across_seeds", "mean_group_rmse_delta": float(mean_by_group.mean()),
                              "seed_sd_group_rmse_delta": float(pd.concat(seed_series, axis=1).std(axis=1, ddof=1).mean()),
                              "ci95_low": lo, "ci95_high": hi, "group_count": len(mean_by_group),
                              "interpretation": "descriptive only" if split == "test" else "validation comparison"})
    pairs = pd.DataFrame(pair_rows)
    pd.DataFrame(pair_group_rows).to_csv(out_dir / "paired_group_deltas.csv", index=False)
    pairs.to_csv(out_dir / "paired_comparison_summary.csv", index=False)
    pd.concat([outputs[s][0].assign(split=s) for s in splits], ignore_index=True).to_csv(out_dir / "run_metrics.csv", index=False)
    seed_summary = val.groupby("configuration", as_index=False).agg(seed_mean_rmse=("pooled_rmse", "mean"), seed_sd_rmse=("pooled_rmse", "std"))
    seed_summary.to_csv(out_dir / "seed_summary.csv", index=False)

    comparison = summary.copy()
    if "test" in outputs:
        test_summary = outputs["test"][0].groupby("configuration", as_index=False).agg(
            mean_test_pooled_rmse=("pooled_rmse", "mean"), mean_test_pooled_mae=("pooled_mae", "mean"),
            mean_test_macro_family_rmse=("macro_rmse", "mean"))
        comparison = comparison.merge(test_summary, on="configuration")
    comparison["parameter_count"] = [
        next((x for x in outputs["validation"][0].loc[outputs["validation"][0].configuration == c, "parameter_count"] if pd.notna(x)), np.nan)
        for c in comparison.configuration]
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)
    split_rows = []
    for split, (_, loaded) in outputs.items():
        reference = loaded[(CONFIGURATIONS[0], SEEDS[0])]
        split_rows.append({"split": split, "prediction_rows": len(reference),
                           "sample_count": reference.sample_id.nunique(), "void_count": reference.void_id.nunique(),
                           "family_count": reference.family_id.nunique()})
    pd.DataFrame(split_rows).to_csv(out_dir / "split_summary.csv", index=False)

    for split in splits:
        frame = outputs[split][0]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, config in enumerate(CONFIGURATIONS):
            vals = frame[frame.configuration == config].pooled_rmse.to_numpy(float)
            ax.scatter(np.full(len(vals), i), vals, label=config)
            ax.plot([i - .15, i + .15], [vals.mean()] * 2, color="black", lw=1)
        ax.set_xticks(range(3), CONFIGURATIONS, rotation=20, ha="right")
        ax.set_ylabel("Pooled RMSE"); ax.set_title(f"{split.capitalize()} RMSE by seed" + (" (descriptive only)" if split == "test" else ""))
        ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(out_dir / f"{split}_rmse.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot = pairs[pairs.split == "validation"]
    labels = [f"{r.model} vs {r.comparator}\nseed {r.seed}" for r in plot.itertuples()]
    x = np.arange(len(plot)); means = plot.mean_group_rmse_delta.to_numpy(float)
    ax.errorbar(x, means, yerr=[means - plot.ci95_low, plot.ci95_high - means], fmt="o", capsize=3)
    ax.axhline(0, color="black", lw=1); ax.set_xticks(x, labels, rotation=35, ha="right"); ax.set_ylabel("Family-level RMSE difference")
    ax.set_title("Paired validation differences; bootstrap resamples held-out families"); fig.tight_layout()
    fig.savefig(out_dir / "paired_group_deltas.png", dpi=180); plt.close(fig)

    if "test" in outputs:
        # Configuration choice and checkpoint are already frozen from validation.
        pred = outputs["test"][1][(str(summary.sort_values("mean_validation_pooled_rmse").iloc[0].configuration), SEEDS[0])]
        fig, ax = plt.subplots(figsize=(6, 5)); ax.scatter(pred.y_true, pred.y_pred, s=4, alpha=.2)
        low = min(pred.y_true.min(), pred.y_pred.min()); high = max(pred.y_true.max(), pred.y_pred.max())
        ax.plot([low, high], [low, high], color="red"); ax.set(xlabel="True", ylabel="Predicted", title="Test prediction diagnostic (descriptive only)")
        fig.tight_layout(); fig.savefig(out_dir / "prediction_diagnostics.png", dpi=180); plt.close(fig)
    report = (
        "# Confirmation study report\n\n"
        "## Validation-based selection\n\n"
        "Configurations are ranked by mean pooled validation RMSE across the three predeclared seeds. "
        "See `selection_summary.csv`; no test metric enters this rule.\n\n"
        "## Study specification\n\n"
        f"- Dataset path: {manifest.dataset_path.iloc[0] if 'dataset_path' in manifest else 'not recorded in manifest'}\n"
        f"- Dataset build fingerprint: {manifest.build_fingerprint.iloc[0] if 'build_fingerprint' in manifest else 'not recorded in manifest'}\n"
        f"- Existing family split path and SHA256: {manifest.split_path.iloc[0] if 'split_path' in manifest else 'not recorded in manifest'}; "
        f"{manifest.split_sha256.iloc[0] if 'split_sha256' in manifest else 'not recorded in manifest'}\n"
        f"- Grouping definition: {manifest.grouping_definition.iloc[0] if 'grouping_definition' in manifest else 'existing family-based split'}\n"
        f"- Seeds: {', '.join(map(str, SEEDS))}; configurations: {', '.join(CONFIGURATIONS)}\n"
        "- Selection rule: minimum mean pooled validation RMSE across three seeds.\n\n")
    if "test" in outputs:
        report += ("## Test-set descriptive confirmation\n\n"
                   "Test results are descriptive only and do not alter the selected configuration, features, hyperparameters, or split. "
                   "Paired intervals resample held-out family IDs, preserving within-family dependence. The existing family split is reused.\n\n")
    else:
        report += "Test evaluation is not included in this validation-only aggregation stage.\n\n"
    report += ("## Results files\n\n"
               "See `run_metrics.csv`, `validation_group_metrics.csv`, `test_group_metrics.csv` (after final evaluation), "
               "`paired_group_deltas.csv`, `seed_summary.csv`, `model_comparison.csv`, and `split_summary.csv`.\n")
    (out_dir / "confirmation_report.md").write_text(report)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--bootstrap-draws", type=int, default=10000)
    p.add_argument("--include-test", action="store_true", help="run only after validation aggregation and report rule are frozen")
    a = p.parse_args(); aggregate(a.manifest, a.out_dir, a.bootstrap_draws, include_test=a.include_test); return 0


if __name__ == "__main__":
    raise SystemExit(main())
