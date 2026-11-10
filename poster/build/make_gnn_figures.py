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


INK, INK_2, MUTED = "#000000", "#333333", "#6F727B"
GRID, AXIS, HIGHLIGHT, NEUTRAL = "#E3E3E3", "#9D9795", "#000000", "#C4BFC0"
SEQUENTIAL = ["#E9DFCC", "#CDB997", "#AE9266", "#8E6F3E", "#6B5330", "#3D2F1B", "#000000"]
ATOM_ONLY = "atom_full_chemistry_and_periodicity"
THRESHOLDS = (0.5, 0.7)
HERE = Path(__file__).resolve().parents[2] / "elf_void_gnn" / "poster_figures"  # data lives next to this folder
POSTER_FIGURES = Path(__file__).resolve().parents[1] / "figures"


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
    plt.rcParams.update({"font.size": 18, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f4efe7", "savefig.facecolor": "#f4efe7"})


def parity(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    _style(plt)
    with plt.rc_context({"font.family": "Arial", "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(8.0, 7.0))
        ax.set_facecolor("#FFFFFF")
        hb = ax.hexbin(frame["y_true"], frame["y_pred"], gridsize=70, extent=(0, 1, 0, 1), mincnt=1,
                       cmap=LinearSegmentedColormap.from_list("blue", SEQUENTIAL), norm=LogNorm(), linewidths=0)
        # White reads over the dense core; past the data the same dashes turn dark for the pale background.
        ax.plot([0.02, 0.68], [0.02, 0.68], color="white", lw=2.2, ls=(0, (4, 2.5)))
        ax.plot([0.68, 1], [0.68, 1], color=INK_2, lw=2.2, ls=(0, (4, 2.5)))
        ax.text(0.91, 0.975, "perfect prediction", rotation=45, ha="right", va="bottom", color=INK_2, fontsize=25,
                rotation_mode="anchor", transform_rotates_text=True)
        ax.axvspan(0.7, 1.0, color="#E3E3E3", alpha=0.75, lw=0, zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax.set_xticks(ticks, [f"{t:.1f}" for t in ticks]); ax.set_yticks(ticks, [f"{t:.1f}" for t in ticks])
        ax.tick_params(labelsize=26, pad=10, length=6, color=AXIS, labelcolor=INK_2)
        ax.set_xlabel("Actual DFT ELF", fontsize=30, labelpad=10, color=INK, fontweight="bold")
        ax.set_ylabel("GNN Predicted ELF", fontsize=30, labelpad=10, color=INK, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(AXIS)
            spine.set_linewidth(1.2)
        bar = fig.colorbar(hb, ax=ax, shrink=0.943, pad=0.03)
        bar.ax.tick_params(labelsize=26, color=AXIS, labelcolor=INK_2)
        bar.set_label("Void count", color=INK, fontsize=26, labelpad=10)
        bar.outline.set_visible(False)
        for suffix in (".png", ".pdf"):
            fig.savefig(path.with_suffix(suffix), dpi=300, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)


def comparison(models: list[dict], mean_baseline: float, path: Path) -> None:
    import matplotlib.pyplot as plt
    _style(plt)
    # Poster palette: blue section color, paler tints for the baselines, orange accent.
    bar_colors = ["#8E6F3E", "#C4BFC0", "#D3CFCF", "#E1DEDE"]
    panel, grid, accent = "#F7F7F7", "#E3E3E3", "#000000"
    models = sorted(models, key=lambda m: m["test_rmse"][0])
    with plt.rc_context({"font.family": "Arial", "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(7.0, 6.2))
        ax.set_facecolor("#FFFFFF")
        for i, m in enumerate(models):
            mean = m["test_rmse"][0]
            selected = m.get("selected", False)
            ax.barh(i, mean, height=0.64, color=bar_colors[i], zorder=2)
            ax.text(mean - 0.004, i, f"{mean:.3f}", ha="right", va="center", fontsize=25, zorder=4,
                    color=("white" if selected else INK), fontweight="bold")
        ax.axvline(mean_baseline, color=accent, lw=3, ls=(0, (3.5, 2.2)), zorder=3)
        ax.set_yticks(range(len(models)), [m["label"].replace(" ", "\n", 1) for m in models])
        ax.tick_params(axis="y", length=0, labelcolor=INK, labelsize=23, pad=12)
        for label, m in zip(ax.get_yticklabels(), models):
            label.set_fontweight("bold" if m.get("selected", False) else "normal")
        ax.tick_params(axis="x", labelsize=22, pad=8, length=6, color=AXIS, labelcolor=INK_2)
        ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20], ["0.00", "0.05", "0.10", "0.15", "0.20"])
        ax.set_xlabel("Prediction error (RMSE)", fontsize=26, labelpad=10, color=("white" if selected else INK), fontweight="bold")
        ax.set_xlim(0, 0.20)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.grid(axis="x", color=grid, lw=1.2, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(AXIS)
            spine.set_linewidth(1.2)
        for suffix in (".png", ".pdf"):
            fig.savefig(path.with_suffix(suffix), dpi=300, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feature-predictions", type=Path, required=True,
                        help="feature-study final-test test_predictions.parquet")
    parser.add_argument("--run-metrics", type=Path, default=HERE.parent / "architecture_test/results/run_metrics.csv")
    parser.add_argument("--output", type=Path, default=POSTER_FIGURES)
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
    labels = {"void_mlp_w64_d2": "Void-only neural network", "fixed_context_mlp_w64_d2": "Averaged-atom neural network",
              "tree_hgb_context": "Boosted decision trees", "gated_mpnn_w64_d2": "Gated message-passing GNN",
              "hetero_w128_d2": "Atom–void GNN · all descriptors"}
    models = []
    for config, label in labels.items():
        sel = runs[runs["configuration"] == config]
        if sel.empty:
            raise ValueError(f"no runs for {config}")
        values = sel["test_pooled_rmse"].to_numpy()
        models.append({"label": label, "source": f"architecture_test/{config}", "runs": int(len(values)),
                       "test_rmse": [float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0]})
    legacy_atom_only = {"label": "Atom–void GNN · atomic features (feature study)",
                        "source": "feature_study/atom-only", "runs": len(rows),
                        "test_rmse": mean_std(rows, "rmse")}
    models.append(legacy_atom_only)
    # Direct, same-protocol confirmation values for the selected model and its
    # width-128 atom-only and all-descriptor comparators.
    confirm = pd.read_csv(HERE.parent / "confirmation_study/results/run_metrics.csv")
    confirm = confirm[confirm["split"] == "test"]
    confirmation_models = []
    for config, label, selected in [
        ("w64_d2_atom_only", "Atom–void graph network", True),
        ("w128_d2_atom_only", "Atom–void graph network (width 128)", False),
        ("w128_d2_baseline", "Atom–void graph network (all features)", False),
    ]:
        values = confirm.loc[confirm.configuration == config, "pooled_rmse"].to_numpy()
        confirmation_models.append({"label": label, "source": f"confirmation_study/{config}", "runs": len(values),
                                    "test_rmse": [float(values.mean()), float(values.std(ddof=1))], "selected": selected})

    args.output.mkdir(parents=True, exist_ok=True)
    parity(atom_only[atom_only["seed"] == median_seed], args.output / "gnn_parity_atom_only")
    # The poster chart compares ways of using atom information: the selected graph
    # model against the strongest non-graph baselines. All families stay in summary.json.
    shown = {"Boosted decision trees", "Averaged-atom neural network",
             "Void-only neural network"}
    selected_models = ([m for m in confirmation_models if m["label"] == "Atom–void graph network"]
                       + [m for m in models if m["label"] in shown])
    comparison(selected_models, mean_baseline, args.output / "gnn_model_comparison")
    summary = {"atom_only_test": {k: mean_std(rows, k) for k in rows[0] if k != "n"},
               "atom_only_test_voids": rows[0]["n"], "atom_only_seeds": seed_rows,
               "parity_plot_seed": median_seed,
               "predict_average_rmse": mean_baseline,
               "models": models,
               "confirmation_comparison": confirmation_models,
               "notes": "Mean ± SD over seeds on the family-held-out test split. Recall/precision treat "
                        "ELF >= threshold as a positive. predict_average_rmse uses the test mean, "
                        "so it is slightly optimistic for a constant predictor."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    a = summary["atom_only_test"]
    print(f"atom-only test RMSE {a['rmse'][0]:.4f} ± {a['rmse'][1]:.4f}, R² {a['r2'][0]:.3f}; "
          f"recall ELF≥0.5 {a['recall_elf_ge_0.5'][0]:.2f}, ELF≥0.7 {a['recall_elf_ge_0.7'][0]:.2f}")


if __name__ == "__main__":
    main()
