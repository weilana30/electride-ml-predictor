#!/usr/bin/env python3
"""Summarize the predeclared architecture matrix; never rank/select by test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ("mae", "rmse", "r2")
SPLIT_METRICS = ("validation", "test")


def read_run(row: dict) -> dict | None:
    path = Path(row["run_dir"]) / "metrics.json"
    if not path.exists():
        return None
    metrics = json.loads(path.read_text())
    out = {**row, "metrics_path": str(path), "status": metrics.get("status", "unknown")}
    for split in SPLIT_METRICS:
        values = metrics.get(split, {})
        for group in ("pooled", "macro_family"):
            for metric in METRICS:
                out[f"{split}_{group}_{metric}"] = values.get(group, {}).get(metric)
        out[f"{split}_family_count"] = values.get("macro_family", {}).get("family_count")
        out[f"{split}_per_family"] = values.get("per_family", {})
    out["best_epoch"] = metrics.get("best_epoch")
    out["best_iteration"] = metrics.get("best_iteration")
    out["best_validation_rmse"] = metrics.get("best_validation_rmse")
    out["training_seconds"] = metrics.get("performance", {}).get("training_seconds")
    out["peak_gpu_memory_bytes"] = metrics.get("performance", {}).get("peak_gpu_memory_bytes")
    out["parameter_count"] = metrics.get("performance", {}).get("parameter_count")
    out["training_mean_baseline"] = metrics.get("training_mean_baseline", {})
    out["evaluation_scope"] = metrics.get("evaluation_scope", "")
    return out


def savefig(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_report(manifest_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path).fillna("")
    complete = []
    missing = []
    for row in manifest.to_dict("records"):
        run = read_run(row)
        if run is None:
            missing.append({**row, "status": "not_started_or_metrics_missing"})
        else:
            complete.append(run)
    if not complete:
        raise SystemExit("No metrics.json files found under manifest run_dir paths")

    frame = pd.DataFrame(complete)
    detail_cols = [f"{split}_per_family" for split in SPLIT_METRICS] + ["training_mean_baseline"]
    frame.drop(columns=detail_cols, errors="ignore").to_csv(out_dir / "run_metrics.csv", index=False)
    pd.DataFrame(missing).to_csv(out_dir / "incomplete_runs.csv", index=False)

    # Expand per-family metrics and compute paired improvements against the
    # width-64, depth-2 hetero MPNN with the same seed on each fixed split.
    family_rows = []
    lookup = {(str(r["configuration"]), int(r["seed"])): r for r in complete}
    for run in complete:
        for family, values in run.get("test_per_family", {}).items():
            base = lookup.get(("hetero_w64_d2", int(run["seed"])))
            base_values = (base or {}).get("test_per_family", {}).get(family, {})
            family_rows.append({
                "run_id": run["run_id"], "configuration": run["configuration"],
                "model_type": run["model_type"], "seed": run["seed"],
                "family_id": family, "count": values.get("count"),
                "mae": values.get("mae"), "rmse": values.get("rmse"), "r2": values.get("r2"),
                "baseline_run_id": (base or {}).get("run_id", ""),
                "paired_mae_delta_vs_hetero_w64_d2": (
                    values.get("mae") - base_values.get("mae")
                    if values.get("mae") is not None and base_values.get("mae") is not None else np.nan
                ),
                "paired_rmse_delta_vs_hetero_w64_d2": (
                    values.get("rmse") - base_values.get("rmse")
                    if values.get("rmse") is not None and base_values.get("rmse") is not None else np.nan
                ),
            })
    family_df = pd.DataFrame(family_rows)
    family_df.to_csv(out_dir / "test_family_metrics_and_paired_deltas.csv", index=False)

    # Explicitly show all individual seeds. These are initialization variability
    # on the same fixed split, not independent test splits.
    configurations = list(manifest.configuration.drop_duplicates().astype(str))
    metrics = [
        ("validation_pooled_mae", "Validation pooled MAE"),
        ("test_pooled_mae", "Test pooled MAE"),
        ("validation_macro_family_mae", "Validation macro-family MAE"),
        ("test_macro_family_mae", "Test macro-family MAE"),
        ("validation_pooled_rmse", "Validation pooled RMSE"),
        ("test_pooled_rmse", "Test pooled RMSE"),
        ("validation_macro_family_rmse", "Validation macro-family RMSE"),
        ("test_macro_family_rmse", "Test macro-family RMSE"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    rng = np.random.default_rng(17)
    seed_labels = [str(x) for x in manifest.seed.drop_duplicates().tolist()]
    seed_colors = {label: plt.get_cmap("tab10")(i) for i, label in enumerate(seed_labels)}
    for ax, (column, title) in zip(axes.flat, metrics):
        for index, config in enumerate(configurations):
            subset = frame.loc[frame.configuration.astype(str) == config].dropna(subset=[column])
            vals = subset[column].astype(float).values
            if len(vals):
                jitter = rng.uniform(-0.08, 0.08, len(vals))
                for offset, (_, run) in zip(jitter, subset.iterrows()):
                    ax.scatter(index + offset, float(run[column]), s=28, alpha=0.8,
                               color=seed_colors.get(str(run.seed), "#3569a8"))
                ax.plot([index - 0.12, index + 0.12], [np.mean(vals)] * 2, color="#c44736", lw=2)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[-1]:
        ax.set_xticks(range(len(configurations)))
        ax.set_xticklabels(configurations, rotation=45, ha="right", fontsize=8)
    fig.suptitle("Predeclared architecture matrix: individual seeds and mean\n"
                 "Point colors identify seeds; 3-seed spread is initialization variability on one fixed split; test has 9 families",
                 fontsize=12)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=seed_colors[label], label=f"seed {label}")
               for label in seed_labels]
    handles.append(plt.Line2D([], [], color="#c44736", lw=2, label="mean across available runs"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.subplots_adjust(bottom=0.23, top=0.85)
    fig.tight_layout()
    savefig(fig, out_dir / "split_metric_seed_points")

    # Per-family differences are paired on family and seed. Negative deltas are
    # lower error than the existing width-64/depth-2 hetero MPNN for that seed.
    if not family_df.empty:
        grouped = family_df.groupby("configuration", as_index=False).agg(
            mean_paired_mae_delta=("paired_mae_delta_vs_hetero_w64_d2", "mean"),
            mean_paired_rmse_delta=("paired_rmse_delta_vs_hetero_w64_d2", "mean"),
        )
        grouped.to_csv(out_dir / "mean_paired_family_deltas.csv", index=False)
        plot_df = family_df[family_df.configuration.astype(str) != "hetero_w64_d2"]
        if not plot_df.empty:
            pivot = plot_df.pivot_table(index="configuration", columns="family_id",
                                        values="paired_mae_delta_vs_hetero_w64_d2", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(pivot.columns)), max(4, 0.55 * len(pivot.index))))
            vmax = float(np.nanmax(np.abs(pivot.values))) if np.isfinite(pivot.values).any() else 1.0
            image = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
            ax.set_title("Test per-family MAE change vs same-seed hetero MPNN\n"
                         "Negative is lower error; descriptive paired comparison on the fixed family split")
            fig.colorbar(image, ax=ax, label="MAE difference (ELF units)")
            fig.tight_layout()
            savefig(fig, out_dir / "paired_family_mae_delta")

    # Parity and residual distributions for each prespecified configuration;
    # all available seeds are overlaid, with deterministic point subsampling.
    pred_frames = []
    for row in manifest.to_dict("records"):
        path = Path(row["run_dir"]) / "test_predictions.parquet"
        if not path.exists():
            continue
        predictions = pd.read_parquet(path)
        predictions["run_id"] = row["run_id"]
        predictions["configuration"] = row["configuration"]
        predictions["seed"] = row["seed"]
        pred_frames.append(predictions)
    if pred_frames:
        predictions = pd.concat(pred_frames, ignore_index=True)
        predictions["residual"] = predictions.y_pred - predictions.y_true
        configs = [c for c in configurations if c in set(predictions.configuration.astype(str))]
        fig, axes = plt.subplots(len(configs), 2, figsize=(10, max(3.2 * len(configs), 6)), squeeze=False)
        rng = np.random.default_rng(23)
        for i, config in enumerate(configs):
            sub = predictions[predictions.configuration.astype(str) == config]
            if len(sub) > 12000:
                sub = sub.iloc[np.sort(rng.choice(len(sub), 12000, replace=False))]
            axes[i, 0].scatter(sub.y_true, sub.y_pred, s=3, alpha=0.12, color="#3569a8", rasterized=True)
            low = float(min(sub.y_true.min(), sub.y_pred.min()))
            high = float(max(sub.y_true.max(), sub.y_pred.max()))
            axes[i, 0].plot([low, high], [low, high], color="#c44736", lw=1)
            axes[i, 0].set_ylabel(config)
            axes[i, 1].scatter(sub.y_true, sub.residual, s=3, alpha=0.12, color="#3569a8", rasterized=True)
            axes[i, 1].axhline(0, color="#c44736", lw=1)
            axes[i, 0].grid(alpha=0.2)
            axes[i, 1].grid(alpha=0.2)
        axes[0, 0].set_title("Test parity (all available seeds)")
        axes[0, 1].set_title("Test residual vs true ELF")
        for ax in axes[:, 0]:
            ax.set_xlabel("True ELF at void center")
            ax.set_ylabel(ax.get_ylabel() or "Predicted ELF")
        for ax in axes[:, 1]:
            ax.set_xlabel("True ELF at void center")
            ax.set_ylabel("Prediction − truth")
        fig.suptitle("Descriptive held-out predictions for the fixed 9-family test split; not used for model selection")
        fig.tight_layout()
        savefig(fig, out_dir / "test_parity_residual")

    print(f"Wrote architecture comparison tables and figures to {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="CSV generated by generate_sweep.py")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.manifest, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
