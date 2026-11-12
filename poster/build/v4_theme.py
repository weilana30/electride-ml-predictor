"""Colors and fonts shared by the v4 poster and its figures.

Two accent hues carry meaning everywhere on the poster:
  ELF_*  (slate blue)  predicted ELF / electrons   -> Tool 1
  GEO_*  (amber)       geometric open (void) space -> Tool 2
Everything else is a cool neutral.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

INK = "#16242c"        # text, header band
INK_2 = "#3b4a53"      # secondary text
MUTED = "#6b7a83"      # captions, tick labels
RULE = "#b9c3c9"       # card borders, axes
GRID = "#e3e8eb"
PAGE = "#e6ebee"       # page background behind the cards
CARD = "#ffffff"
SOFT = "#f2f5f6"       # inset panels inside a card

ELF = "#1f6e8c"
ELF_DARK = "#124a60"
ELF_TINT = "#dcebf1"
ELF_RAMP = ["#e4eff3", "#b5d3df", "#7fb2c6", "#4a90ab", "#1f6e8c", "#124a60", "#0a2a37"]

GEO = "#e8a917"
GEO_DARK = "#8a5d00"   # amber dark enough for text on white
GEO_TINT = "#fbefcc"
GEO_SOFT = "#f6dc95"

ATOM = "#4b5961"
ATOM_LIGHT = "#9aa6ad"
NEUTRAL_BARS = ["#aab4ba", "#c0c8cd", "#d5dbde"]

import os
ARIAL = os.environ.get("POSTER_FONT") == "arial"  # POSTER_FONT=arial redraws everything in Arial, for comparison
BODY = "Arial" if ARIAL else "Avenir Next"
HEAD = "Arial" if ARIAL else "Avenir Next Condensed"
_TTC = {BODY: "/System/Library/Fonts/Avenir Next.ttc", HEAD: "/System/Library/Fonts/Avenir Next Condensed.ttc"}
_cache: dict[tuple[str, int, bool], str] = {}


def font_file(family: str, weight: int = 400, italic: bool = False) -> str:
    """Path to a standalone .ttf for one face, extracted from the macOS .ttc on first use."""
    if ARIAL:
        return "/System/Library/Fonts/Supplemental/Arial" + (" Bold" if weight >= 600 else "") + (" Italic" if italic else "") + ".ttf"
    key = (family, weight, italic)
    if key not in _cache:
        from fontTools.ttLib import TTCollection
        out = Path(tempfile.gettempdir()) / "poster_v4_fonts"
        out.mkdir(exist_ok=True)
        for face in TTCollection(_TTC[family]).fonts:
            name = face["name"].getDebugName(4)
            k = (family, face["OS/2"].usWeightClass, "Italic" in name)
            path = out / (name.replace(" ", "") + ".ttf")
            if not path.exists():
                face.save(str(path))
            _cache[k] = str(path)
    return _cache[key]


def use_matplotlib_fonts() -> None:
    """Register Avenir Next with matplotlib and make it the default family."""
    import matplotlib
    from matplotlib import font_manager
    for weight in (400, 700) if ARIAL else (400, 500, 600, 700):
        font_manager.fontManager.addfont(font_file(BODY, weight))
        font_manager.fontManager.addfont(font_file(HEAD, weight))
    matplotlib.rcParams.update({
        "font.family": BODY, "pdf.fonttype": 42, "text.color": INK,
        "axes.edgecolor": RULE, "axes.labelcolor": INK, "xtick.color": RULE, "ytick.color": RULE,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": CARD, "savefig.facecolor": CARD})
