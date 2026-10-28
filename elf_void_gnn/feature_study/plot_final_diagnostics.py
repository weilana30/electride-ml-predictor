#!/usr/bin/env python3
"""Plot feature redundancies and frozen test predictions from the final report."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


VARIANTS = {
    "baseline": ("Full baseline", "baseline"),
    "atomic_number_nearest_void_distances_edge_distances": ("Minimal", "minimal"),
    "atom_full_chemistry_and_periodicity": ("Atom-only", "atom-only"),
}


def short_feature(name: str) -> str:
    return (
        name.replace("void.", "Void: ")
        .replace("atom.", "Atom: ")
        .replace("_angstrom", "")
        .replace("nearest_atom_distance", "nearest atom distance")
        .replace("nearest_", "nearest ")
        .replace("atom_count_within_", "atom count ≤")
        .replace("atomic_number", "atomic number")
        .replace("atomic_radius", "atomic radius")
        .replace("is_d_block", "d-block flag")
        .replace("_distance", " distance")
        .replace("_", " ")
    )


def plot_redundancies(report_dir: Path) -> None:
    pairs = pd.read_csv(report_dir / "top_correlations.csv").head(15).iloc[::-1].copy()
    pairs["label"] = [
        f"{short_feature(a)}  ↔  {short_feature(b)}"
        for a, b in zip(pairs.feature_a, pairs.feature_b)
    ]
    fig, ax = plt.subplots(figsize=(11, 7.2))
    colors = ["#3b82f6" if value >= 0 else "#ef4444" for value in pairs.correlation]
    ax.barh(pairs.label, pairs.absolute_correlation, color=colors, alpha=0.88)
    for index, value in enumerate(pairs.correlation):
        ax.text(abs(value) + 0.008, index, f"ρ={value:.3f}", va="center", fontsize=9)
    ax.axvline(0.9, color="#6b7280", linestyle="--", linewidth=1)
    ax.set(xlim=(0, 1.08), xlabel="Absolute Spearman correlation across training structures")
    ax.set_title("Strongest feature redundancies", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(report_dir / "top_feature_redundancies.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(report_dir: Path, predictions_path: Path) -> None:
    predictions = pd.read_parquet(predictions_path)
    metrics = pd.read_csv(report_dir / "pooled_metrics.csv")
    metrics = metrics[metrics.split == "test"].set_index("variant")
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.3), sharex=True, sharey=True)
    density = None
    for ax, (variant, (title, metric_name)) in zip(axes, VARIANTS.items()):
        data = predictions[predictions.variant == variant]
        if data.empty:
            raise ValueError(f"no frozen test predictions for {variant}")
        density = ax.hexbin(
            data.y_true,
            data.y_pred,
            gridsize=85,
            extent=(-0.02, 0.90, -0.02, 0.90),
            mincnt=1,
            bins="log",
            cmap="viridis",
        )
        ax.plot([-0.02, 0.90], [-0.02, 0.90], color="white", linewidth=2.8)
        ax.plot([-0.02, 0.90], [-0.02, 0.90], color="#ef4444", linewidth=1.5)
        row = metrics.loc[metric_name]
        ax.set_title(title, fontweight="bold")
        ax.text(
            0.04,
            0.95,
            f"RMSE {row.rmse_mean:.4f} ± {row.rmse_sd:.4f}\n"
            f"MAE   {row.mae_mean:.4f} ± {row.mae_sd:.4f}\n"
            f"R²      {row.r2_mean:.3f} ± {row.r2_sd:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
        )
        ax.set(xlim=(-0.02, 0.90), ylim=(-0.02, 0.90), xlabel="Actual ELF at void center")
        ax.set_aspect("equal")
        ax.grid(alpha=0.12)
    axes[0].set_ylabel("Predicted ELF at void center")
    fig.suptitle(
        "Frozen test predictions for the three selected configurations",
        x=0.06,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.915,
        "Density includes all three seeds; the red diagonal is perfect prediction. Metrics are mean ± SD across seeds.",
        ha="left",
        fontsize=10,
        color="#4b5563",
    )
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.11, top=0.83, wspace=0.10)
    colorbar = fig.colorbar(density, ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label("Prediction density (log count)")
    fig.savefig(report_dir / "test_predicted_vs_actual.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    args = parser.parse_args()
    plot_redundancies(args.report_dir)
    plot_predictions(args.report_dir, args.test_predictions)


if __name__ == "__main__":
    main()
