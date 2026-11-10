"""Poster connectivity-flow figure (PNG at 300 dpi and PDF), in the Purdue poster colors.

A restyled copy of elf_void_analysis/registry_dimensionality_figures.py.
    python poster/build/connectivity_flow_figure.py RESULTS_DIR [OUT_STEM]
RESULTS_DIR is an output folder of elf_void_analysis.registry_dimensionality_test
(it holds structures.csv and probe_sweep.csv).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = {"balanced_accuracy": ("Balanced accuracy", "#2a78d6"),
          "exact_accuracy": ("Exact match", "#eb6834")}
SEQUENTIAL = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
PANELS = {"ionic": "Probe radius added to ionic radius (Å)",
          "none": "Probe radius from atom centers (Å)"}
TITLES = {"ionic": "Ionic radii + probe", "none": "Same size for every atom"}


def _style(plt) -> None:
    plt.rcParams.update({"font.size": 13, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def accuracy_vs_probe(summary: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt
    _style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (scheme, xlabel) in zip(axes, PANELS.items()):
        result = summary["schemes"][scheme]
        probes = [c["probe_angstrom"] for c in result["curve"]]
        for key, (label, color) in SERIES.items():
            ax.plot(probes, [100 * c[key] for c in result["curve"]], color=color, lw=2.5, label=label)
        for value, text in ((100 * summary["chance_balanced_accuracy"], "chance (balanced)"),
                            (100 * summary["all_0D_baseline_exact_accuracy"], "always guess 0D (exact)")):
            ax.axhline(value, color=MUTED, lw=1.2, ls=(0, (2, 3)))
            ax.text(probes[0], value + 1.5, text, color=INK_2, fontsize=11)
        chosen = result["default_probe_angstrom"]
        ax.axvline(chosen, color=INK_2, lw=1.2)
        ax.text(chosen, 97, f" chosen beforehand: {chosen:.1f} Å", color=INK_2, fontsize=11, va="top")
        ax.set_title(TITLES[scheme], loc="left", fontsize=14, color=INK)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Agreement with literature (%)")
    fig.suptitle(f"Void dimensionality vs literature on {summary['scored']} known electrides",
                 x=0.01, ha="left", fontsize=15)
    fig.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2, frameon=False,
               bbox_to_anchor=(0.99, 1.0))
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=300)
    plt.close(fig)


def confusion(summary: dict[str, Any], scheme: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    _style(plt)
    result = summary["schemes"][scheme]
    block = result["confusion_at_default"]
    counts = np.array(block["counts"])
    share = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("blue", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, j in np.ndindex(counts.shape):
        dark = share[i, j] > 0.55
        ax.text(j, i, f"{counts[i, j]}", ha="center", va="center", fontsize=14,
                color="white" if dark else INK, fontweight="bold" if j - 1 == i else "normal")
        if j - 1 == i:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.6))
    ax.set_xticks(range(counts.shape[1]), block["columns_predicted"])
    ax.set_yticks(range(counts.shape[0]), block["rows_literature"])
    ax.set_xlabel("Highest open dimension (geometry)")
    ax.set_ylabel("Literature dimension")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    d = result["default"]
    ax.set_title(f"{TITLES[scheme]}, {result['default_probe_angstrom']:.1f} Å probe: "
                 f"{100 * d['exact_accuracy']:.0f}% exact, {100 * d['balanced_accuracy']:.0f}% balanced",
                 loc="left", fontsize=12.5)
    fig.text(0.01, 0.01, "Shade = share of the row; boxed = correct", color=INK_2, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=300)
    plt.close(fig)


def connectivity_shares(structures: Any, sweep: Any) -> Any:
    """Counts of the connected stage seen in the ionic probe sweep, by literature class."""
    scored = structures[structures["scored"] & structures["literature_dimension"].isin([0, 1, 2])]
    ionic = sweep[sweep["radius_scheme"] == "ionic"]
    stages = ionic.groupby("mp_id")["highest_dimension"].agg(set)
    stage = scored["mp_id"].map(stages).map(lambda s: 1 if 1 in s else 2 if 2 in s else 0)
    return (stage.groupby(scored["literature_dimension"].astype(int)).value_counts()
            .unstack(fill_value=0).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0))


def connectivity_flow(structures: Any, sweep: Any, path: Path) -> None:
    """Flow from literature dimension to the connected stage found by the probe sweep.

    Correct flows (literature class = geometry class) are solid; mismatches are faded,
    and a remark under the chart names the class with the largest mismatch. Every number is computed from the counts.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as Curve
    from matplotlib.transforms import Bbox
    counts = connectivity_shares(structures, sweep).to_numpy()
    colors = {0: "#9D9795", 1: "#555960", 2: "#DAAA00"}
    labels = {0: "0D", 1: "1D", 2: "2D"}
    ink, gap, left, right, width = "#000000", 16, 0.34, 0.80, 0.035
    left_totals, right_totals = counts.sum(axis=1), counts.sum(axis=0)
    hit_rate = np.diag(counts) / np.maximum(left_totals, 1)
    left_top = np.r_[0, np.cumsum(left_totals + gap)[:-1]]
    right_top = np.r_[0, np.cumsum(right_totals + gap)[:-1]]
    off_diagonal = counts * (1 - np.eye(len(counts), dtype=int))
    worst = np.unravel_index(np.argmax(off_diagonal), counts.shape)
    with plt.rc_context({"font.family": "Arial", "pdf.fonttype": 42, "text.color": ink}):
        fig, ax = plt.subplots(figsize=(15.5, 9.0))
        ax.axis("off")
        left_next, right_next = left_top.astype(float), right_top.astype(float)
        middle = (left + right) / 2
        for i, j in np.ndindex(counts.shape):
            n = counts[i, j]
            if n == 0:
                continue
            a, b = left_next[i], right_next[j]
            left_next[i] += n
            right_next[j] += n
            points = [(left, a), (middle, a), (middle, b), (right, b), (right, b + n),
                      (middle, b + n), (middle, a + n), (left, a + n), (left, a)]
            codes = [Curve.MOVETO, Curve.CURVE4, Curve.CURVE4, Curve.CURVE4, Curve.LINETO,
                     Curve.CURVE4, Curve.CURVE4, Curve.CURVE4, Curve.CLOSEPOLY]
            correct = i == j
            # Thin mismatch bands get an outline so a single structure still shows.
            ax.add_patch(PathPatch(Curve(points, codes), fc=colors[i], alpha=0.95 if correct else 0.2,
                                   ec="none", zorder=2 if correct else 1))
            if not correct:  # a background-colored seam keeps neighboring faded bands apart
                ax.add_patch(PathPatch(Curve(points, codes), fc="none", ec="#fcf5e1", lw=1.5, zorder=1))
            if not correct and n <= 3:
                ax.add_patch(PathPatch(Curve(points, codes), fc="none", ec=colors[i], lw=2.2,
                                       alpha=0.8, zorder=1))
        for i in range(3):
            ax.add_patch(Rectangle((left - width, left_top[i]), width, left_totals[i], fc=colors[i], ec="none"))
            ax.text(left - width - 0.012, left_top[i] + left_totals[i] / 2,
                    f"{labels[i]}: {100 * hit_rate[i]:.0f}% identified correctly",
                    ha="right", va="center", fontsize=24, fontweight="bold")
            ax.add_patch(Rectangle((right, right_top[i]), width, right_totals[i], fc=colors[i], ec="none"))
            ax.text(right + width + 0.012, right_top[i] + right_totals[i] / 2,
                    f"{labels[i]}: {right_totals[i]}", ha="left", va="center",
                    fontsize=26, fontweight="bold")
        bottom = max(left_top[-1] + left_totals[-1], right_top[-1] + right_totals[-1])
        # Remark under the flows about the class with the largest mismatch.
        ax.text((left + right) / 2, bottom + 14,
                f"Many {labels[worst[0]]} electrides have connected open space",
                ha="center", va="top", fontsize=22, fontweight="bold", color="#3b3936")
        for x, heading in ((left - width / 2, "Actual (literature)"),
                           (right + width / 2, "Geometry (probe sweep)")):
            ax.text(x, -14, heading, ha="center", va="bottom", fontsize=24, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.text((left + right) / 2, -62, "Predicted vs actual void connectivity", ha="center",
                va="bottom", fontsize=28, fontweight="bold", color="#8e6f3e")
        ax.set_ylim(bottom + 50, -96)
        # Crop to the drawn content; the hidden axes would otherwise pad the sides.
        renderer = fig.canvas.get_renderer()
        drawn = Bbox.union([artist.get_window_extent(renderer) for artist in ax.texts + ax.patches])
        crop = drawn.transformed(fig.dpi_scale_trans.inverted()).padded(0.08)
        for suffix in (".png", ".pdf"):
            fig.savefig(path.with_suffix(suffix), dpi=300, bbox_inches=crop, facecolor="#fcf5e1")
        plt.close(fig)


def plot_all(summary: dict[str, Any], output: Path) -> None:
    accuracy_vs_probe(summary, output / "accuracy_vs_probe")
    for scheme, spec in summary["schemes"].items():
        probe = f"{spec['default_probe_angstrom']:.1f}".replace(".", "p")
        confusion(summary, scheme, output / f"confusion_{scheme}_{probe}")


if __name__ == "__main__":
    import sys
    import pandas as pd
    results = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "figures" / "connectivity_flow_sweep"
    connectivity_flow(pd.read_csv(results / "structures.csv"), pd.read_csv(results / "probe_sweep.csv"), out)
