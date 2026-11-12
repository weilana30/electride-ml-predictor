"""Plots for poster_v4, drawn at their printed size in the v4 colors and fonts.

    python poster/build/make_v4_figures.py PATH/TO/test_predictions.parquet [OUT_DIR]

PATH is the confirmation-study prediction file for the selected model (width-64
atom-only GNN, seed 20261028), kept on Negishi; see make_parity_confirmation.py.
Without it, pass "-" to redraw only the bar chart and the connectivity flow.
Figure sizes are in poster inches, so a font size here is the printed size.
Writes OUT_DIR (default poster/figures/v4): PNG at 300 dpi, PDF, and numbers.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4_theme as T
from connectivity_flow_figure import connectivity_shares

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "elf_void_gnn/architecture_test/results_voidv2/poster_summary.json"
REGISTRY = ROOT / "elf_void_analysis/case_studies/registry_test"
BASELINES = ["Boosted decision trees", "Averaged-atom neural network", "Void-only neural network"]
RARE = 0.7  # ELF above this is rare in ELFNet and under-predicted


def save(fig, stem: Path) -> None:
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=300)
    plt.close(fig)


def comparison(stem: Path) -> dict:
    summary = json.loads(SUMMARY.read_text())
    models = ([m for m in summary["confirmation_comparison"] if m["label"] == "Atom–void graph network"]
              + [m for m in summary["models"] if m["label"] in BASELINES])
    models.sort(key=lambda m: m["test_rmse"][0])
    assert len(models) == 4 and models[0].get("selected")
    mean_rmse = summary["predict_average_rmse"]
    names = {"Atom–void graph network": "Atom–void\ngraph network", "Boosted decision trees": "Boosted\ndecision trees",
             "Averaged-atom neural network": "Averaged-atom\nneural network",
             "Void-only neural network": "Void-only\nneural network"}
    fig, ax = plt.subplots(figsize=(7.1, 6.9), layout="constrained")
    for i, m in enumerate(models):
        mean, sd = m["test_rmse"]
        ax.barh(i, mean, height=0.62, color=T.ELF if i == 0 else T.NEUTRAL_BARS[i - 1], zorder=2)
        if sd:  # the boosted trees were run once
            ax.errorbar(mean, i, xerr=sd, color=T.INK, lw=2.2, capsize=7, capthick=2.2, zorder=4)
        ax.text(0.005, i, f"{mean:.3f}", ha="left", va="center", fontsize=23, fontweight="bold",
                color="white" if i == 0 else T.INK, zorder=5)
    ax.axvline(mean_rmse, color=T.INK, lw=2.6, ls=(0, (3.5, 2.2)), zorder=3)
    ax.text(mean_rmse - 0.004, -0.56, f"always predict the mean: {mean_rmse:.3f}", ha="right", va="bottom",
            fontsize=18, color=T.INK_2)
    ax.set_yticks(range(4), [names[m["label"]] for m in models])
    ax.tick_params(axis="y", length=0, labelsize=20, labelcolor=T.INK, pad=10)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20], ["0", "0.05", "0.10", "0.15", "0.20"])
    ax.tick_params(axis="x", labelsize=19, labelcolor=T.INK_2, length=6, pad=6)
    ax.set_xlabel("Test RMSE in ELF (lower is better)", fontsize=21, fontweight="demibold", labelpad=10)
    ax.set_xlim(0, 0.205)
    ax.set_ylim(3.5, -0.95)
    ax.grid(axis="x", color=T.GRID, lw=1.4, zorder=0)
    ax.set_axisbelow(True)
    save(fig, stem)
    return {"models": [{"label": m["label"], "test_rmse": m["test_rmse"], "runs": m["runs"]} for m in models],
            "predict_average_rmse": mean_rmse}


def parity(predictions: Path, stem: Path) -> dict:
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    frame = pd.read_parquet(predictions, columns=["y_true", "y_pred"])
    assert len(frame) == 88680
    y, p = frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()
    numbers = {"n": int(len(y)), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
               "r2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
               "share_true_ge_rare": float(np.mean(y >= RARE)),
               "recall_ge_rare": float(np.mean(p[y >= RARE] >= RARE)), "rare_threshold": RARE}
    fig, ax = plt.subplots(figsize=(8.9, 6.9), layout="constrained")
    ax.axvspan(RARE, 1.0, color=T.GEO_TINT, lw=0, zorder=0)
    hb = ax.hexbin(y, p, gridsize=70, extent=(0, 1, 0, 1), mincnt=1, linewidths=0, norm=LogNorm(),
                   cmap=LinearSegmentedColormap.from_list("elf", T.ELF_RAMP), zorder=2)
    ax.plot([0, 1], [0, 1], color=T.INK, lw=2.2, ls=(0, (4, 2.5)), zorder=3)
    ax.text(0.80, 0.825, "perfect prediction", rotation=45, rotation_mode="anchor", transform_rotates_text=True,
            ha="center", va="bottom", fontsize=19, color=T.INK_2)
    ax.text(0.85, 0.05, f"ELF ≥ {RARE}:\n{100 * numbers['share_true_ge_rare']:.1f}% of voids,\nunder-predicted",
            ha="center", va="bottom", fontsize=19, color=T.GEO_DARK, fontweight="demibold", linespacing=1.15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.tick_params(labelsize=22, labelcolor=T.INK_2, length=6, pad=6)
    ax.set_xlabel("ELF from DFT", fontsize=24, fontweight="demibold", labelpad=8)
    ax.set_ylabel("ELF predicted by the GNN", fontsize=24, fontweight="demibold", labelpad=8)
    for spine in ax.spines.values():
        spine.set_visible(True)
    bar = fig.colorbar(hb, ax=ax, shrink=0.9, pad=0.025, aspect=24)
    bar.ax.tick_params(labelsize=20, labelcolor=T.INK_2, length=5)
    bar.set_label("Voids per bin", fontsize=21, color=T.INK, labelpad=8)
    bar.outline.set_visible(False)
    save(fig, stem)
    return numbers


def flow(stem: Path) -> dict:
    """Literature dimension (left) against the connectivity found across the probe sweep (right).

    Every class box has the same height, so a band's thickness is its share of that class.
    """
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as Curve
    counts = connectivity_shares(pd.read_csv(REGISTRY / "structures.csv"),
                                 pd.read_csv(REGISTRY / "probe_sweep.csv")).to_numpy()
    colors = {0: T.ATOM_LIGHT, 1: T.ATOM, 2: T.GEO}
    faded = {0: "#d3d9dd", 1: "#b4bcc1", 2: T.GEO_SOFT}
    on_band = {0: T.INK, 1: "#ffffff", 2: T.INK}
    names = {0: "0D", 1: "1D", 2: "2D"}
    box, gap, left, right, width, min_band = 100.0, 18.0, 0.122, 0.878, 0.03, 1.6
    left_totals, right_totals = counts.sum(axis=1), counts.sum(axis=0)

    def shares(raw, axis):
        out = np.where(counts > 0, np.maximum(raw, min_band), 0.0)
        return out / out.sum(axis=axis, keepdims=True) * box
    left_share = shares(counts / left_totals[:, None] * box, 1)
    right_share = shares(counts / np.maximum(right_totals[None, :], 1) * box, 0)
    hit = np.diag(counts) / left_totals
    top = np.arange(3) * (box + gap)
    middle = (left + right) / 2
    fig = plt.figure(figsize=(16.0, 7.3))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    left_next, right_next = top.astype(float), top.astype(float)
    for i, j in np.ndindex(counts.shape):
        if counts[i, j] == 0:
            continue
        a, b, na, nb = left_next[i], right_next[j], left_share[i, j], right_share[i, j]
        left_next[i] += na
        right_next[j] += nb
        band = Curve([(left, a), (middle, a), (middle, b), (right, b), (right, b + nb),
                      (middle, b + nb), (middle, a + na), (left, a + na), (left, a)],
                     [Curve.MOVETO, Curve.CURVE4, Curve.CURVE4, Curve.CURVE4, Curve.LINETO,
                      Curve.CURVE4, Curve.CURVE4, Curve.CURVE4, Curve.CLOSEPOLY])
        if i != j:
            ax.add_patch(PathPatch(band, fc=faded[i], ec="none", alpha=0.9, zorder=1))
            continue
        ax.add_patch(PathPatch(band, fc=colors[i], ec="none", zorder=2))
        ax.text(left + 0.3 * (right - left), a + na / 2 + (b + nb / 2 - a - na / 2) * 0.216,
                f"{100 * hit[i]:.0f}% agree", ha="center", va="center", fontsize=34, fontweight="bold",
                color=on_band[i], zorder=3)
    for i in range(3):
        centre = top[i] + box / 2
        for x, align, sign, total in ((left - width, "right", -1, left_totals[i]), (right, "left", 1, right_totals[i])):
            ax.add_patch(Rectangle((x, top[i]), width, box, fc=colors[i], ec="none"))
            edge = x - 0.012 if sign < 0 else x + width + 0.012
            ax.text(edge, centre - 14, names[i], ha=align, va="center", fontsize=42, fontweight="bold")
            ax.text(edge, centre + 32, f"n = {total}", ha=align, va="center", fontsize=24, color=T.INK_2)
    for x, align, heading in ((left - width, "left", "Literature label"),
                              (right + width, "right", "Geometry, over all probe sizes")):
        ax.text(x, -16, heading, ha=align, va="bottom", fontsize=31, fontweight="demibold")
    ax.set_xlim(0, 1)
    ax.set_ylim(top[-1] + box + 8, -64)
    save(fig, stem)
    return {"counts_literature_by_sweep": counts.tolist(), "agree": hit.tolist(),
            "share_0D_label_with_connected_space": float(1 - hit[0]),
            "n_0D_label_with_connected_space": int(left_totals[0] - counts[0, 0]), "n_0D": int(left_totals[0])}


def main() -> None:
    predictions = sys.argv[1] if len(sys.argv) > 1 else "-"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "figures" / "v4"
    out.mkdir(parents=True, exist_ok=True)
    T.use_matplotlib_fonts()
    numbers_path = out / "numbers.json"
    numbers = json.loads(numbers_path.read_text()) if numbers_path.exists() else {}
    numbers["comparison"] = comparison(out / "gnn_model_comparison")
    numbers["flow"] = flow(out / "connectivity_flow_sweep")
    if predictions != "-":
        numbers["parity"] = parity(Path(predictions), out / "gnn_parity_confirmation")
    numbers_path.write_text(json.dumps(numbers, indent=2) + "\n")
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
