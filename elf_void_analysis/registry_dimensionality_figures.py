"""Poster figures for the registry dimensionality test (PNG at 300 dpi and PDF)."""
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


def plot_all(summary: dict[str, Any], output: Path) -> None:
    accuracy_vs_probe(summary, output / "accuracy_vs_probe")
    for scheme, spec in summary["schemes"].items():
        probe = f"{spec['default_probe_angstrom']:.1f}".replace(".", "p")
        confusion(summary, scheme, output / f"confusion_{scheme}_{probe}")
