#!/usr/bin/env python3
"""Schematic diagrams for the poster, drawn at their printed size.

Each figure is sized in inches to match its slot on the 48 x 36 in poster,
so font sizes here are the printed point sizes. Run from anywhere:

    python poster/diagrams/make_diagrams.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
NAVY, BLUE, ORANGE = "#13294b", "#2a78d6", "#eb6834"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
CARD, OPEN = "#e8eff8", "#d8e6f6"
ATOM, ATOM_EDGE = "#7f8fa6", "#5b6b82"
plt.rcParams.update({"font.family": "Arial", "text.color": INK, "savefig.facecolor": "white"})


def save(fig, name: str) -> None:
    for suffix in (".png", ".pdf"):
        fig.savefig(HERE / f"{name}{suffix}", dpi=300)
    plt.close(fig)


def box(ax, x, y, w, h, text, *, fill=CARD, edge=NAVY, dashed=False, size=20, weight="normal", color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18",
                                fc=fill, ec=edge, lw=2.2, ls=(0, (5, 4)) if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color,
            fontweight=weight, linespacing=1.25)


def arrow(ax, start, end, color=NAVY):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=28, lw=2.6, color=color,
                                 shrinkA=2, shrinkB=2))


def badge(ax, x, y, text, color):
    ax.add_patch(Circle((x, y), 0.36, fc=color, ec="none", zorder=5))
    ax.text(x, y, text, ha="center", va="center", fontsize=15, color="white", fontweight="bold", zorder=6)


def pipeline() -> None:
    """Crystal -> voids -> (GNN ELF, void network dimension) -> combined label."""
    w, h = 15.4, 5.6
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    mid = h / 2
    box(ax, 0.15, mid - 1.0, 2.5, 2.0, "Crystal\nstructure\n(CIF / POSCAR)", size=19)
    box(ax, 3.25, mid - 1.0, 2.5, 2.0, "Candidate\nvoids", size=20)
    arrow(ax, (2.65, mid), (3.25, mid))
    top, bottom = mid + 0.75, mid - 2.35
    box(ax, 6.55, top, 4.6, 1.6, "Atom–void graph → GNN\nELF at each void", size=19)
    box(ax, 6.55, bottom, 4.6, 1.6, "Exact void network\n0D / 1D / 2D / 3D", size=19)
    badge(ax, 6.55, top + 1.6, "RQ1", BLUE)
    badge(ax, 6.55, bottom + 1.6, "RQ2", BLUE)
    arrow(ax, (5.75, mid + 0.3), (6.55, top + 0.8))
    arrow(ax, (5.75, mid - 0.3), (6.55, bottom + 0.8))
    box(ax, 12.0, mid - 1.15, 3.25, 2.3, "Electride type\nper void\nlocalized 0D or\nconnected 1D/2D",
        fill="#fdeee7", edge=ORANGE, dashed=True, size=17.5)
    badge(ax, 12.0, mid + 1.15, "RQ3", ORANGE)
    arrow(ax, (11.15, top + 0.8), (12.0, mid + 0.35), ORANGE)
    arrow(ax, (11.15, bottom + 0.8), (12.0, mid - 0.35), ORANGE)
    ax.text(13.62, mid - 1.5, "future work", ha="center", va="top", fontsize=15, color=ORANGE, style="italic")
    ax.text(8.85, top - 0.18, "is an electron there?", ha="center", va="top", fontsize=15, color=INK_2, style="italic")
    ax.text(8.85, bottom - 0.18, "is there room, and how is it connected?", ha="center", va="top",
            fontsize=15, color=INK_2, style="italic")
    save(fig, "pipeline")


def _channel(ax, y0: float, label: str, electrons: str) -> None:
    """Horizontal channel between two rows of atoms; necks at even x."""
    xs = np.arange(0, 9, 2.0)
    ax.add_patch(plt.Rectangle((-0.6, y0 - 0.95), 9.2, 1.9, fc=OPEN, ec="none", zorder=0))
    if electrons == "connected":
        x = np.linspace(-0.6, 8.6, 400)
        half = 0.2 + 0.14 * np.abs(np.sin(np.pi * x / 2))
        ax.fill_between(x, y0 - half, y0 + half, color=ORANGE, alpha=0.85, lw=0, zorder=1)
    else:
        for xc in np.arange(1, 8, 2.0):
            ax.add_patch(Ellipse((xc, y0), 1.1, 0.68, fc=ORANGE, alpha=0.85, ec="none", zorder=1))
    for xc in xs:
        for sign in (1, -1):
            ax.add_patch(Circle((xc, y0 + sign * 0.95), 0.62, fc=ATOM, ec=ATOM_EDGE, lw=1.5, zorder=2))
    ax.text(-0.6, y0 + 1.85, label, ha="left", va="center", fontsize=19, fontweight="bold", color=NAVY)
    neck = "ELF stays high at the necks" if electrons == "connected" else "ELF drops at the necks"
    ax.annotate(neck, xy=(4.0, y0), xytext=(9.0, y0), fontsize=16, color=INK_2, va="center",
                arrowprops={"arrowstyle": "-|>", "color": INK_2, "lw": 1.6, "shrinkB": 4})


def pipes_vs_electrons() -> None:
    """Same open channel, two electron pictures: connected vs separate pockets."""
    w, h = 13.8, 5.6
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0.01, 0.0, 0.98, 1.0))
    ax.set_xlim(-0.8, 14.0)
    ax.set_ylim(-4.6, 3.35)
    ax.set_aspect("equal")
    ax.axis("off")
    _channel(ax, 1.0, "Electrons fill the channel → delocalized 1D", "connected")
    _channel(ax, -2.6, "Electrons sit in separate pockets → localized 0D", "pockets")
    ax.text(-0.6, -4.55,"Same open space (geometry says 1D) in both rows; only ELF at the necks tells them apart.",
            ha="left", va="bottom", fontsize=15, color=INK_2, style="italic")
    save(fig, "pipes_vs_electrons")


def electride_concept() -> None:
    """Cations on a lattice with anionic electrons in the empty space between them."""
    w, h = 5.6, 4.4
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(-0.6, 5.0)
    ax.set_ylim(-0.6, 3.8)
    ax.set_aspect("equal")
    ax.axis("off")
    for i in range(5):
        for j in range(4):
            if (i + j) % 2 == 0:
                ax.add_patch(Circle((i * 1.1, j * 1.05), 0.42, fc=BLUE, ec=NAVY, lw=1.5))
                ax.text(i * 1.1, j * 1.05, "+", ha="center", va="center", fontsize=20, color="white",
                        fontweight="bold")
            else:
                ax.add_patch(Ellipse((i * 1.1, j * 1.05), 0.62, 0.52, fc=ORANGE, alpha=0.9, ec="none"))
                ax.text(i * 1.1, j * 1.05, r"e$^-$", ha="center", va="center", fontsize=14, color="white",
                        fontweight="bold")
    save(fig, "electride_concept")


if __name__ == "__main__":
    pipeline()
    pipes_vs_electrons()
    electride_concept()
    print("wrote", sorted(p.name for p in HERE.glob("*.png")))
