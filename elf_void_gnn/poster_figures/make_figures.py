#!/usr/bin/env python3
"""Poster figures for the ELF void GNN from frozen test predictions.

Inputs are the feature-study final-test predictions (external, see
DATASET_PROVENANCE.md) and the committed architecture-test run metrics.
Outputs are a predicted-vs-true density plot for the atom-only model, a
test-RMSE comparison across models, and ``summary.json`` with every number
shown. Prediction files stay outside Git.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, HIGHLIGHT, NEUTRAL = "#e1e0d9", "#c3c2b7", "#2a78d6", "#b4b2a9"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ATOM_ONLY = "atom_full_chemistry_and_periodicity"
THRESHOLDS = (0.5, 0.7)
HERE = Path(__file__).resolve().parent


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out = {"n": int(len(y)), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
           "mae": float(np.mean(np.abs(y - p))),
           "r2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))}
    for t in THRESHOLDS:
        out[f"recall_elf_ge_{t}"] = float(np.mean(p[y >= t] >= t))
        out[f"precision_elf_ge_{t}"] = float(np.mean(y[p >= t] >= t)) if np.any(p >= t) else float("nan")
        out[f"share_true_elf_ge_{t}"] = float(np.mean(y >= t))
    return out


def mean_std(rows: list[dict[str, float]], key: str) -> list[float]:
    values = [r[key] for r in rows]
    return [float(np.mean(values)), float(np.std(values, ddof=1))]


def _style(plt) -> None:
    plt.rcParams.update({"font.size": 13, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def parity(frame: pd.DataFrame, seed_rows: list[dict[str, float]], seed: int, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    _style(plt)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    hb = ax.hexbin(frame["y_true"], frame["y_pred"], gridsize=70, extent=(0, 1, 0, 1), mincnt=1,
                   cmap=LinearSegmentedColormap.from_list("blue", SEQUENTIAL), norm=LogNorm(), linewidths=0)
    ax.plot([0, 1], [0, 1], color=INK_2, lw=1.2, ls=(0, (4, 3)))
    ax.text(0.97, 0.92, "perfect prediction", rotation=45, ha="right", va="top", color=INK_2, fontsize=10.5,
            rotation_mode="anchor", transform_rotates_text=True)
    ax.axvspan(0.7, 1.0, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(0.85, 0.04, "true ELF ≥ 0.7\n(rare; under-\npredicted)", ha="center", va="bottom",
            color=INK_2, fontsize=10.5)
    rmse, r2 = mean_std(seed_rows, "rmse"), mean_std(seed_rows, "r2")
    ax.text(0.03, 0.97, f"RMSE {rmse[0]:.3f} ± {rmse[1]:.3f}\nR² {r2[0]:.2f} ± {r2[1]:.2f}\n"
                        f"{len(frame):,} test voids, 9 unseen families",
            transform=ax.transAxes, va="top", fontsize=12, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("DFT ELF at void center")
    ax.set_ylabel("GNN-predicted ELF")
    ax.set_title("Atom-only heterogeneous GNN, test set", loc="left", fontsize=14)
    bar = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
    bar.set_label("Voids per hexagon", color=INK_2)
    bar.outline.set_visible(False)
    fig.text(0.01, 0.01, f"Plot shows seed {seed} (median RMSE); metrics are mean ± SD over 3 seeds.",
             color=INK_2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=300)
    plt.close(fig)


def comparison(models: list[dict], mean_baseline: float, path: Path) -> None:
    import matplotlib.pyplot as plt
    _style(plt)
    models = sorted(models, key=lambda m: m["test_rmse"][0], reverse=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    best = min(m["test_rmse"][0] for m in models)
    for i, m in enumerate(models):
        mean, sd = m["test_rmse"]
        color = HIGHLIGHT if mean == best else NEUTRAL
        ax.barh(i, mean, height=0.62, color=color)
        if sd:
            ax.errorbar(mean, i, xerr=sd, color=INK_2, capsize=3, lw=1.2)
        ax.text(mean + max(sd, 0) + 0.002, i, f"{mean:.3f}", va="center", fontsize=11.5,
                color=INK, fontweight="bold" if mean == best else "normal")
    ax.axvline(mean_baseline, color=MUTED, lw=1.2, ls=(0, (2, 3)))
    ax.text(mean_baseline - 0.002, len(models) - 1, "predict the\naverage ELF", color=INK_2, fontsize=10.5,
            ha="right", va="center")
    ax.set_yticks(range(len(models)), [m["label"] for m in models])
    ax.set_xlabel("Test RMSE in ELF (lower is better)")
    ax.set_xlim(0, mean_baseline * 1.18)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    fig.suptitle("Message passing between atoms and voids halves the error", x=0.01, ha="left", fontsize=14)
    fig.text(0.01, 0.01, "Mean ± SD over 3 seeds (tree: 1 run). Same family-held-out test split.",
             color=INK_2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feature-predictions", type=Path, required=True,
                        help="feature-study final-test test_predictions.parquet")
    parser.add_argument("--run-metrics", type=Path, default=HERE.parent / "architecture_test/results/run_metrics.csv")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()

    predictions = pd.read_parquet(args.feature_predictions)
    atom_only = predictions[predictions["variant"] == ATOM_ONLY]
    seed_rows = {int(s): metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
                 for s, g in atom_only.groupby("seed")}
    median_seed = sorted(seed_rows, key=lambda s: seed_rows[s]["rmse"])[len(seed_rows) // 2]
    rows = list(seed_rows.values())
    y = atom_only.loc[atom_only["seed"] == median_seed, "y_true"].to_numpy()
    mean_baseline = float(np.sqrt(np.mean((y - y.mean()) ** 2)))

    runs = pd.read_csv(args.run_metrics)
    labels = {"void_mlp_w64_d2": "Void-only MLP", "fixed_context_mlp_w64_d2": "Fixed-context MLP",
              "tree_hgb_context": "Gradient-boosted tree", "gated_mpnn_w64_d2": "Gated MPNN",
              "hetero_w128_d2": "Hetero GNN, all descriptors"}
    models = []
    for config, label in labels.items():
        sel = runs[runs["configuration"] == config]
        if sel.empty:
            raise ValueError(f"no runs for {config}")
        values = sel["test_pooled_rmse"].to_numpy()
        models.append({"label": label, "source": f"architecture_test/{config}", "runs": int(len(values)),
                       "test_rmse": [float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0]})
    models.append({"label": "Hetero GNN, atom-only", "source": "feature_study/atom-only", "runs": len(rows),
                   "test_rmse": mean_std(rows, "rmse")})

    args.output.mkdir(parents=True, exist_ok=True)
    parity(atom_only[atom_only["seed"] == median_seed], rows, median_seed, args.output / "gnn_parity_atom_only")
    comparison(models, mean_baseline, args.output / "gnn_model_comparison")
    summary = {"atom_only_test": {k: mean_std(rows, k) for k in rows[0] if k != "n"},
               "atom_only_test_voids": rows[0]["n"], "atom_only_seeds": seed_rows,
               "parity_plot_seed": median_seed,
               "predict_average_rmse": mean_baseline,
               "models": models,
               "notes": "Mean ± SD over seeds on the family-held-out test split. Recall/precision treat "
                        "ELF >= threshold as a positive. predict_average_rmse uses the test mean, "
                        "so it is slightly optimistic for a constant predictor."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    a = summary["atom_only_test"]
    print(f"atom-only test RMSE {a['rmse'][0]:.4f} ± {a['rmse'][1]:.4f}, R² {a['r2'][0]:.3f}; "
          f"recall ELF≥0.5 {a['recall_elf_ge_0.5'][0]:.2f}, ELF≥0.7 {a['recall_elf_ge_0.7'][0]:.2f}")


if __name__ == "__main__":
    main()
