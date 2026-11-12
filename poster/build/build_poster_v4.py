"""Build poster/poster_v4.svg (36 x 48 in portrait) and export poster_v4.pdf.

    python poster/build/build_poster_v4.py [OUT.svg]

v4 is a redesign of poster_v3: numbered section cards with arrows between them,
a slate-blue / amber palette (v4_theme.py), Avenir Next type, and reworded text.
The four schematics are lifted from the hand-edited poster_v3.svg and recolored,
so poster_v3.svg must stay next to this build. Plots come from make_v4_figures.py.
Every number in the text is read from the result files named in SOURCES below.
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from pathlib import Path

from lxml import etree
from PIL import ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4_theme as T

POSTER = Path(__file__).resolve().parents[1]
ROOT = POSTER.parent
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = Path(_args[0]) if _args else POSTER / "poster_v4.svg"
INKSCAPE = "/Applications/Inkscape.app/Contents/MacOS/inkscape"
SVG, INK_NS = "http://www.w3.org/2000/svg", "http://www.inkscape.org/namespaces/inkscape"
PT = 96 / 72
W, H, M = 3456, 4608, 58

# ---------------------------------------------------------------- numbers
SOURCES = {"gnn": ROOT / "elf_void_gnn/architecture_test/results_voidv2/poster_summary.json",
           "confirm": ROOT / "elf_void_gnn/confirmation_study/results/run_metrics.csv",
           "registry": ROOT / "elf_void_analysis/case_studies/registry_test/summary.json",
           "figures": POSTER / "figures/v4/numbers.json"}
gnn = json.loads(SOURCES["gnn"].read_text())
reg = json.loads(SOURCES["registry"].read_text())
fig_numbers = json.loads(SOURCES["figures"].read_text())
selected = next(m for m in gnn["confirmation_comparison"] if m.get("selected"))
RMSE, RMSE_SD = selected["test_rmse"]
import pandas as pd
_c = pd.read_csv(SOURCES["confirm"])
R2 = _c[(_c.configuration == "w64_d2_atom_only") & (_c.split == "test")]["pooled_r2"].mean()
N_VOIDS = gnn["atom_only_test_voids"]
MEAN_RMSE = gnn["predict_average_rmse"]
VOID_ONLY = next(m for m in gnn["models"] if m["label"] == "Void-only neural network")["test_rmse"][0]
assert RMSE < MEAN_RMSE / 2
ionic = reg["schemes"]["ionic"]
BAL, EXACT = ionic["default"]["balanced_accuracy"], ionic["default"]["exact_accuracy"]
CHANCE, ALL0D = reg["chance_balanced_accuracy"], reg["all_0D_baseline_exact_accuracy"]
REACHED = ionic["literature_dimension_reached_at_some_probe"]
N_REG, N_EXP = reg["scored"], reg["experimentally_confirmed"]
flow = fig_numbers["flow"]
N0, N0_OPEN = flow["n_0D"], flow["n_0D_label_with_connected_space"]


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


# ---------------------------------------------------------------- text
_fonts: dict = {}


def _font(family: str, weight: int, italic: bool = False):
    key = (family, weight, italic)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(T.font_file(family, weight, italic), 200)
    return _fonts[key]


def _runs(s: str, weight: int):
    """Split '**bold**' and '~sub~' markup into (text, weight, is_subscript) runs."""
    out = []
    for tok in re.split(r"(\*\*.+?\*\*|~[^~]+~)", s):
        if not tok:
            continue
        if tok.startswith("**"):
            out.append((tok[2:-2], 700 if weight <= 600 else 900, False))
        elif tok.startswith("~"):
            out.append((tok[1:-1], weight, True))
        else:
            out.append((tok, weight, False))
    return out


def _wrap(s: str, px: float, width: float, family: str, weight: int, tracking: float = 0):
    """Greedy line breaking; returns a list of lines, each a list of (text, weight, sub) runs."""
    lines, cur, used = [], [], 0.0
    for text, w, sub in _runs(s, weight):
        f = _font(family, w)
        scale = px / 200 * (0.65 if sub else 1)
        for word in re.findall(r"\S+\s*|\s+", text):
            bare = f.getlength(word.rstrip()) * scale + tracking * len(word.rstrip())
            if cur and used + bare > width:
                lines.append(cur)
                cur, used = [], 0.0
                word = word.lstrip() or word
            full = f.getlength(word) * scale + tracking * len(word)
            if cur and cur[-1][1:] == (w, sub):
                cur[-1] = (cur[-1][0] + word, w, sub)
            else:
                cur.append((word, w, sub))
            used += full
    if cur:
        lines.append(cur)
    return lines


class Page:
    def __init__(self):
        self.layers: list[tuple[str, list[str]]] = []
        self.o: list[str] = []

    def layer(self, name: str) -> None:
        self.o = []
        self.layers.append((name, self.o))

    def text(self, x, y, w, s, pt, color=T.INK, weight=400, family=T.BODY, align="start", lh=1.28,
             tracking=0.0, italic=False) -> float:
        """Wrapped text whose first line's cap-top sits at y. Returns the y of the bottom."""
        px = pt * PT
        lines = _wrap(s, px, w, family, weight, tracking)
        ax = {"start": x, "center": x + w / 2, "end": x + w}[align]
        anchor = {"start": "start", "center": "middle", "end": "end"}[align]
        body = ""
        for i, line in enumerate(lines):
            line[-1] = (line[-1][0].rstrip(), *line[-1][1:])
            spans = ""
            for text, wt, sub in line:
                st = f"font-weight:{wt}" + (";baseline-shift:sub;font-size:65%" if sub else "")
                spans += f'<tspan style="{st}">{html.escape(text, quote=False)}</tspan>'
            body += f'<tspan x="{ax:.1f}" y="{y + px * 0.76 + i * px * lh:.1f}">{spans}</tspan>'
        style = (f"font-family:'{family}';font-size:{px:.2f}px;font-weight:{weight};fill:{color};text-anchor:{anchor}"
                 + (f";letter-spacing:{tracking:.2f}px" if tracking else "") + (";font-style:italic" if italic else ""))
        self.o.append(f'<text xml:space="preserve" style="{style}">{body}</text>')
        return y + (len(lines) - 1) * px * lh + px

    def rect(self, x, y, w, h, fill, r=0, stroke=None, sw=2.5, label="") -> None:
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        lab = f' inkscape:label="{label}"' if label else ""
        self.o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" ry="{r}" fill="{fill}"{st}{lab}/>')

    def image(self, href, x, y, w, h) -> None:
        self.o.append(f'<image xlink:href="{href}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" preserveAspectRatio="none"/>')

    def arrow(self, cx, y, down=True, w=110, h=40, color=T.INK_2) -> None:
        """Flow arrow in the gap between sections: a short stem and a head."""
        self.o.append(f'<path d="M{cx - 13:.1f} {y:.1f}h26v{h * .42:.1f}h{w / 2 - 13:.1f}L{cx:.1f} {y + h:.1f}L{cx - w / 2:.1f} {y + h * .42:.1f}h{w / 2 - 13:.1f}z" '
                      f'fill="{color}" inkscape:label="Flow arrow"/>')

    def card(self, x, y, w, h, number, title, color, on=T.CARD, tag="") -> float:
        """White section card with a colored header bar and a number badge. Returns content top."""
        r, hh = 16, 88
        self.rect(x, y, w, h, T.CARD, r, T.RULE, 2.5, f"Section {number} card")
        self.o.append(f'<path d="M{x} {y + hh}V{y + r}a{r} {r} 0 0 1 {r} -{r}H{x + w - r}a{r} {r} 0 0 1 {r} {r}V{y + hh}z" fill="{color}" inkscape:label="Section {number} header"/>')
        self.o.append(f'<circle cx="{x + 62}" cy="{y + hh / 2}" r="29" fill="{on}"/>')
        self.text(x + 32, y + hh / 2 - 21, 60, str(number), 32, color, 700, T.HEAD, "center")
        end = self.text(x + 112, y + hh / 2 - 21, w - 140, title, 33, on, 600, T.HEAD, tracking=1.2)
        if tag:
            self.text(x, y + hh / 2 - 14, w - 36, tag, 22, on, 500, T.BODY, "end", tracking=1.5)
        return y + hh

    def bullets(self, x, y, w, items, pt, color=T.INK, weight=400, dot=T.INK_2, gap=16, lh=1.26) -> float:
        px = pt * PT
        for item in items:
            self.o.append(f'<rect x="{x:.1f}" y="{y + px * .36:.1f}" width="{px * .3:.1f}" height="{px * .3:.1f}" fill="{dot}"/>')
            y = self.text(x + px * .75, y, w - px * .75, item, pt, color, weight, lh=lh) + gap
        return y - gap

    def stats(self, x, y, w, h, cells, color, tint) -> None:
        """A strip of big numbers with a caption under each."""
        self.rect(x, y, w, h, tint, 12, label="Stat strip")
        cw = w / len(cells)
        for i, (big, small) in enumerate(cells):
            cx = x + i * cw
            if i:
                self.o.append(f'<line x1="{cx:.1f}" y1="{y + 20}" x2="{cx:.1f}" y2="{y + h - 20}" stroke="{T.RULE}" stroke-width="2.5"/>')
            self.text(cx + 14, y + 22, cw - 28, big, 50, color, 700, T.HEAD, "center")
            self.text(cx + 14, y + 100, cw - 28, small, 21, T.INK_2, 500, T.BODY, "center", lh=1.2)

    def save(self, path: Path, defs: str) -> None:
        layers = "".join(f'<g id="layer-{i}" inkscape:groupmode="layer" inkscape:label="{name}">{"".join(o)}</g>'
                         for i, (name, o) in enumerate(self.layers))
        doc = (f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="{SVG}" xmlns:xlink="http://www.w3.org/1999/xlink" '
               f'xmlns:inkscape="{INK_NS}" xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd" '
               f'width="36in" height="48in" viewBox="0 0 {W} {H}" version="1.1"><title>Electride poster v4</title>'
               f'<sodipodi:namedview id="view" pagecolor="#ffffff" inkscape:document-units="in" units="in"/>'
               f'<defs id="poster-defs">{defs}</defs>'
               f'<g id="layer-bg" inkscape:groupmode="layer" inkscape:label="Page"><rect width="{W}" height="{H}" fill="{T.PAGE}"/></g>'
               f'{layers}</svg>')
        etree.fromstring(doc.encode())
        path.write_text(doc)


# ---------------------------------------------------------------- schematics from v3
RECOLOR = {"#daaa00": T.GEO, "#ebd99f": T.GEO_SOFT, "#f3dc8a": T.GEO_SOFT, "#f6e3a3": "#fbe7ad", "#9a7800": "#b07c00",
           "#6e5600": T.GEO_DARK, "#e9dfcc": T.GEO_TINT, "#fcf5e1": T.CARD, "#8e6f3e": T.GEO_DARK,
           "#555960": T.ATOM, "#9d9795": T.ATOM_LIGHT, "#6f727b": T.MUTED, "#c4bfc0": "#c3ccd1", "#d6d3d3": "#d5dbde",
           "#333333": T.INK_2, "#000000": T.INK, "#1c1c1e": "#1b262c", "#e9e9ea": "#eef1f3",
           "#2f5d8a": T.ELF, "#dfe8f2": T.ELF_TINT, "#e6e6e6": T.GRID, "#dcdcdc": T.GRID}
ELF_VALUE = {"#8e6f3e": T.ELF_DARK, "#daaa00": T.ELF, "#cdb997": "#8fbccd"}  # panel 3: color encodes predicted ELF
RETEXT = {"1  Crystal → voids": "1  Find candidate voids", "→ electrons connect (1D)": "Electrons connect: 1D",
          "→ isolated pockets (0D)": "Electrons stay apart: 0D", "It runs on through every cell": "The path repeats through every cell",
          "Every copy stays closed off": "Each pocket stays closed off"}


def load_v3():
    src = POSTER / "poster_v3.svg"
    root = etree.parse(str(src), etree.XMLParser(remove_comments=True)).getroot()
    ids = {e.get("id"): e for e in root.iter() if isinstance(e.tag, str)}
    boxes = {}
    for line in subprocess.run([INKSCAPE, "--query-all", str(src)], capture_output=True, text=True).stdout.splitlines():
        k, *v = line.split(",")
        boxes[k] = [float(a) for a in v]
    return ids, boxes


def restyle(el, elf_values=False) -> str:
    for t in el.iter(f"{{{SVG}}}text", f"{{{SVG}}}tspan"):
        if t.get("font-family"):
            t.set("font-family", T.BODY)
        if t.text and t.text.strip() in RETEXT:
            t.text = RETEXT[t.text.strip()]
        elif t.text and "→" in t.text:
            t.text = t.text.replace("→", "to")
    if elf_values:
        for p in el.iter(f"{{{SVG}}}path"):
            st = p.get("style") or ""
            m = re.match(r"fill: (#[0-9a-f]{6})(; stroke: #ffffff; stroke-width: 0\.9|$)", st)
            if m and m.group(1) in ELF_VALUE:
                p.set("style", st.replace(m.group(1), "%%" + m.group(1)[1:] + "%%", 1))
    s = etree.tostring(el, encoding="unicode")
    s = re.sub(r' xmlns(:\w+)?="[^"]+"', "", s)
    s = re.sub(r"font-family:\s*'?(DejaVu Sans|Arial)'?", f"font-family:'{T.BODY}'", s)
    s = re.sub(r"#[0-9a-fA-F]{6}\b", lambda m: RECOLOR.get(m.group(0).lower(), m.group(0)), s)
    return re.sub(r"%%([0-9a-f]{6})%%", lambda m: ELF_VALUE["#" + m.group(1)], s)


V3, BOX = load_v3()
BOX["elf"] = [48.0, 2002.0, 1640.0, 441.0]  # the three drawn panels; the queried box includes hidden padding


def enlarge_bn() -> None:
    """Larger labels in the section-5 schematic, with the legend laid out again to fit."""
    from PIL import ImageFont
    ids = {e.get("id"): e for e in V3["bn"].iter() if isinstance(e.tag, str)}
    for k in ("text242", "text243", "text256", "text257"):
        ids[k].set("font-size", "54")
    for k in ("text258", "text259"):
        ids[k].set("font-size", "50")
    LEG = 47
    font = ImageFont.truetype(T.font_file(T.BODY, 400), LEG)
    x = 40.0
    for sw, tx, label in (("rect259", "text260", "open channel (same geometry in both rows)"),
                          ("rect260", "text261", "electron-rich region (high ELF)"),
                          ("circle261", "text262", "atom")):
        if sw.startswith("circle"):
            ids[sw].set("cx", f"{x + 21:.1f}"), ids[sw].set("r", "21"), ids[sw].set("cy", "725")
        else:
            ids[sw].set("x", f"{x:.1f}"), ids[sw].set("width", "46"), ids[sw].set("height", "40"), ids[sw].set("y", "706")
        ids[tx].set("x", f"{x + 62:.1f}"), ids[tx].set("font-size", str(LEG)), ids[tx].set("y", "742")
        x += 62 + font.getlength(label) + 50


enlarge_bn()
BOX["bn"][2] = 2020.0
DEFS = "".join(restyle(d) for d in V3["poster-defs"] if isinstance(d.tag, str))


def schematic(p: Page, key: str, x: float, y: float, w: float | None = None, h: float | None = None, **kw) -> tuple[float, float]:
    """Place a v3 schematic group so its drawn box starts at (x, y); returns (width, height)."""
    bx, by, bw, bh = BOX[key]
    s = w / bw if w else h / bh
    p.o.append(f'<g inkscape:label="{key} (from v3)" transform="translate({x:.2f} {y:.2f}) scale({s:.5f}) translate({-bx:.2f} {-by:.2f})">'
               + restyle(V3[key], **kw) + "</g>")
    return bw * s, bh * s


# ---------------------------------------------------------------- layout
p = Page()
PAD = 42
FLOW_ASPECT = 7.3 / 16.0
HEAD_H, BG_Y, BG_H = 404, 444, 806
GOAL_Y, GOAL_H = BG_Y + BG_H + 44, 128
TOOL_Y = GOAL_Y + GOAL_H + 50
REF_H = 215
ROW3_H = 772
ROW3_Y = H - REF_H - 30 - ROW3_H
TOOL_H = ROW3_Y - 50 - TOOL_Y
CW = (W - 2 * M - 40) / 2
X1, X2 = M, M + CW + 40

# ---- header
p.layer("Title")
p.rect(0, 0, W, HEAD_H, T.INK, label="Header band")
p.rect(0, HEAD_H, W, 12, T.GEO, label="Header rule")
y = p.text(M, 52, 2700, "Predicting Interstitial ELF and Mapping Void Connectivity for Electride Screening", 78, "#ffffff", 700, T.HEAD, lh=1.08)
y = p.text(M, y + 20, 2700, "A graph neural network predicts the electron localization function (ELF) at interstitial voids; "
           "a geometric network labels how those voids connect", 27, "#c9d6dc", 400, lh=1.25)
p.text(M, HEAD_H - 62, 2700, "**Andrew Weiland**   ·   Advisor: Justin Andrews   ·   Department of Chemistry, Purdue University   ·   weilana@purdue.edu", 25, "#ffffff", 500)
LOGO = 300
p.image("diagrams/lab_logo.png", W - M - 250 - 285 - 70, (HEAD_H - LOGO) / 2, LOGO, LOGO)
p.rect(W - M - 250, 56, 250, 250, "#ffffff", 8, label="QR code backing")
p.image("diagrams/github_qr.png", W - M - 250, 56, 250, 250)
p.text(W - M - 250, 322, 250, "GitHub repository", 20, "#ffffff", 600, align="center", lh=1.15)

# ---- 1 background
p.layer("1 Background")
top = p.card(M, BG_Y, W - 2 * M, BG_H, 1, "BACKGROUND", T.INK_2, tag="WHAT IS AN ELECTRIDE?") + 30
ax, aw = M + PAD, 770
y = p.text(ax, top, aw, "Electrons that sit between the atoms", 34, T.INK, 600, T.HEAD)
y = p.text(ax, y + 14, aw, "In an electride, some valence electrons are not bound to any atom. They localize in the "
           "empty space (voids) of the crystal and act as the anions.", 26, T.INK_2) + 18
sw, sh = schematic(p, "electride", ax + (aw - 340) / 2, y, w=340)
p.text(ax, y + sh + 12, aw, "Schematic: cations (+) with electrons (e−) held in the voids between them.", 25, T.MUTED, align="center")

bx, bw = ax + aw + 70, 1480
p.o.append(f'<line x1="{bx - 35}" y1="{top}" x2="{bx - 35}" y2="{BG_Y + BG_H - 30}" stroke="{T.GRID}" stroke-width="3"/>')
y = p.text(bx, top, bw, "Those voids can be isolated or connected, which sets the dimensionality", 34, T.INK, 600, T.HEAD) + 16
CRYSTALS = [("0d", "**0D**  isolated cages", "Ca~6~Al~7~O~16~ (mayenite type)"),
            ("1d", "**1D**  channels", "Y~5~Si~3~"), ("2d", "**2D**  sheets", "Ca~2~N")]
cw = bw / 3
img = 490
for i, (k, label, formula) in enumerate(CRYSTALS):
    cx = bx + i * cw
    p.text(cx, y, cw, label, 28, T.INK, 500, align="center")
    p.image(f"diagrams/blender/crystals/electride_{k}_v2_transparent.png", cx + (cw - img) / 2, y + 44, img, img)
    p.text(cx, y + 44 + img + 4, cw, formula, 30, T.INK_2, align="center")

ex = bx + bw + 60
ew = W - M - PAD - ex
p.rect(ex, top - 6, ew + 18, BG_Y + BG_H - 24 - (top - 6), T.ELF_TINT, 12, label="ELF panel")
ix, iw = ex + 26, ew - 34
y = p.text(ix, top + 24, iw, "How electrides are identified: ELF", 38, T.ELF_DARK, 600, T.HEAD)
y = p.text(ix, y + 18, iw, "The electron localization function (ELF) runs from 0 to 1 and measures how localized "
           "electrons are compared with a uniform electron gas (0.5).", 28, T.INK_2, lh=1.3) + 46
stops = "".join(f'<stop offset="{i / (len(T.ELF_RAMP) - 1):.3f}" stop-color="{c}"/>' for i, c in enumerate(T.ELF_RAMP))
DEFS += f'<linearGradient id="elf-scale">{stops}</linearGradient>'
p.rect(ix, y, iw, 50, "url(#elf-scale)", 6, T.RULE, 1.5, "ELF scale")
for frac, lab, al in ((0, "0", "start"), (.5, "0.5  electron gas", "center"), (1, "1  localized", "end")):
    tx = ix + frac * iw
    p.o.append(f'<line x1="{min(max(tx, ix + 1), ix + iw - 1):.1f}" y1="{y + 50}" x2="{min(max(tx, ix + 1), ix + iw - 1):.1f}" y2="{y + 64}" stroke="{T.INK_2}" stroke-width="2.5"/>')
    p.text(tx - (0 if al == "start" else 200 if al == "center" else 400), y + 70, 400, lab, 24, T.INK_2, 500, align=al)
y = p.text(ix, y + 142, iw, "**An ELF maximum inside a void**, away from any nucleus, signals interstitial electrons.", 28, T.INK_2, lh=1.3) + 34
p.text(ix, y, iw, "**The bottleneck:** ELF needs a DFT calculation for every candidate crystal.", 28, T.INK_2, lh=1.3)

# ---- 2 goal
p.layer("2 Goal")
p.rect(M, GOAL_Y, W - 2 * M, GOAL_H, T.INK, 16, label="Goal band")
p.o.append(f'<circle cx="{M + 62}" cy="{GOAL_Y + GOAL_H / 2}" r="29" fill="{T.GEO}"/>')
p.text(M + 32, GOAL_Y + GOAL_H / 2 - 21, 60, "2", 31, T.INK, 700, T.HEAD, "center")
p.text(M + 112, GOAL_Y + GOAL_H / 2 - 21, 200, "GOAL", 31, T.GEO, 600, T.HEAD, tracking=1.2)
p.text(M + 250, GOAL_Y + 18, W - 2 * M - 250 - PAD, "Screen crystals for electrides **from structure alone**: predict ELF at each void without "
       "DFT, and map how the voids connect. We built two void-centered tools and tested each one separately.", 30, "#ffffff", 400, lh=1.3)

# ---- 3 tool 1
p.layer("3 Tool 1: predict ELF")
top = p.card(X1, TOOL_Y, CW, TOOL_H, 3, "TOOL 1  ·  PREDICT ELF AT EACH VOID", T.ELF, tag="GRAPH NEURAL NETWORK") + 28
x, w = X1 + PAD, CW - 2 * PAD
y = p.text(x, top, w, "Can a graph network predict the ELF at each candidate void from the surrounding atoms?", 35, T.INK, 600, lh=1.2)
y = p.text(x, y + 16, w, "Atoms and candidate voids are both nodes of one periodic graph, joined by distance edges. "
           "Message passing gathers each void’s atomic neighborhood and returns one ELF value at the void center.", 25.5, T.INK_2) + 30
p.rect(x, y, w, 441 * (w - 40) / 1640 + 40, T.SOFT, 12, label="Schematic panel")
ew_, eh_ = schematic(p, "elf", x + 20, y + 20, w=w - 40, elf_values=True)
y += eh_ + 40 + 26
p.stats(x, y, w, 172, [(f"{RMSE:.3f}", f"test RMSE (± {RMSE_SD:.3f}) on the 0–1 ELF scale"), (f"{R2:.2f}", "test R²"),
                       (f"{N_VOIDS:,}", "test voids in 5,000 crystals"), ("9", "structure families held out for the test")], T.ELF_DARK, T.ELF_TINT)
y += 172 + 30
FOOT = 25
foot_h = 2 * FOOT * PT * 1.3
call_h = 118
cw1, cw2, ch = 7.1 * 96 * .98, 8.9 * 96 * .98, 6.9 * 96 * .98
gap = w - cw1 - cw2
bottom = TOOL_Y + TOOL_H - 26 - foot_h - 20 - call_h - 14
ty = y + max(0, (bottom - y - 46 - ch) / 2)
print("tool 1 slack", round(bottom - y - 46 - ch))
p.text(x, ty, cw1, "Graph network vs. simpler models", 25, T.INK, 600, align="center")
p.text(x + cw1 + gap, ty, cw2, "Predicted vs. DFT ELF on the test set", 25, T.INK, 600, align="center")
p.image("figures/v4/gnn_model_comparison.png", x, ty + 46, cw1, ch)
p.image("figures/v4/gnn_parity_confirmation.png", x + cw1 + gap, ty + 46, cw2, ch)
p.rect(x, bottom + 14, w, call_h, T.ELF_TINT, 12, T.ELF, 3, "Key finding")
p.text(x + 30, bottom + 14 + 17, w - 60, f"**The atoms carry the signal.** A void’s own geometry does no better than predicting the mean "
       f"({VOID_ONLY:.3f} vs {MEAN_RMSE:.3f}); adding its atomic neighborhood cuts the error by more than half.", 25, T.INK, 400, align="center", lh=1.28)
p.text(x, TOOL_Y + TOOL_H - 26 - foot_h, w, "Data: ELFNet [2], 50,000 DFT structures (metal frameworks derived from compressed hydrides), split by structure family.", FOOT, T.MUTED, lh=1.3)

# ---- 4 tool 2
p.layer("4 Tool 2: map connectivity")
top = p.card(X2, TOOL_Y, CW, TOOL_H, 4, "TOOL 2  ·  MAP HOW THE VOIDS CONNECT", T.GEO, T.INK, tag="EXACT GEOMETRY") + 28
x = X2 + PAD
y = p.text(x, top, w, "Can geometry alone recover the dimensionality reported for electrides?", 35, T.INK, 600, lh=1.2)
y = p.text(x, y + 16, w, "Each atom blocks a sphere (ionic radius + probe radius). We follow the open space through the "
           "crystal and count how many directions it repeats in: none (0D), one (1D) or two (2D).", 25.5, T.INK_2) + 30
vw, vh = schematic(p, "voidmap", x, y, w=w)
y += vh + 26
# sub-panel: the probe-size sweep, set apart from the schematic above it
SP_TOP, SP_BOT, SP_HEAD, SP_PAD = y + 34, TOOL_Y + TOOL_H - 26, 100, 26
p.rect(x, SP_TOP, w, SP_BOT - SP_TOP, T.CARD, 14, T.GEO, 3.5, "Sweep panel")
p.o.append(f'<path d="M{x} {SP_TOP + SP_HEAD}V{SP_TOP + 14}a14 14 0 0 1 14 -14H{x + w - 14}a14 14 0 0 1 14 14V{SP_TOP + SP_HEAD}z" fill="{T.GEO_TINT}" inkscape:label="Sweep panel header"/>')
p.o.append(f'<line x1="{x}" y1="{SP_TOP + SP_HEAD}" x2="{x + w}" y2="{SP_TOP + SP_HEAD}" stroke="{T.GEO}" stroke-width="3.5"/>')
p.text(x + 34, SP_TOP + 18, w - 300, "Probe-size sweep", 40, T.INK, 700, T.HEAD)
p.text(x + 34, SP_TOP + 34, w - 68, "EXPLORATORY", 22, T.GEO_DARK, 700, align="end", tracking=2.4)
CALL2 = 140
cy = SP_BOT - SP_PAD - CALL2
fy = SP_TOP + SP_HEAD + 22
iw = w - 2 * SP_PAD
fh = iw * FLOW_ASPECT
avail = cy - 16 - fy
fw = iw if fh <= avail else iw * avail / fh
fh = min(fh, avail)
print("sweep chart scale", round(fw / iw, 3), "slack", round(avail - fh))
p.image("figures/v4/connectivity_flow_sweep.png", x + (w - fw) / 2, fy + (avail - fh) / 2, fw, fh)
p.rect(x + SP_PAD, cy, iw, CALL2, T.GEO_TINT, 12, T.GEO, 3, "Key finding")
p.text(x + SP_PAD + 34, cy + 20, iw - 68, f"**In {N0_OPEN} of {N0} electrides labeled 0D, the open space connects at some probe size.** "
       "The space forms a path, but the electrons may still sit in separate pockets.", 28, T.INK, 400, align="center", lh=1.28)

# ---- 5 combination
p.layer("5 Proposed combination")
C5W = 2230
top = p.card(M, ROW3_Y, C5W, ROW3_H, 5, "PROPOSED  ·  COMBINE THE TWO TOOLS", T.INK_2, tag="FUTURE WORK, NOT YET TESTED") + 28
x, w5 = M + PAD, C5W - 2 * PAD
y = p.text(x, top, w5, "Open space is not the same as electrons", 35, T.INK, 600)
y = p.text(x, y + 12, w5, "The same open channel can hold connected electrons (1D) or separate pockets (0D); only the ELF at its "
           "bottleneck tells them apart.", 24.5, T.INK_2) + 24
bh_avail = ROW3_Y + ROW3_H - 28 - y
sx = 0.0
bw_, bh_ = BOX["bn"][2], BOX["bn"][3]
scale = min(bh_avail / bh_, (w5 - 560) / bw_)
schematic(p, "bn", x, y + (bh_avail - bh_ * scale) / 2, w=bw_ * scale)
stx = x + bw_ * scale + 44
stw = M + C5W - PAD - stx
STEPS = [("TOOL 2  ·  GEOMETRY", "Find the channel and its narrowest bottleneck", T.GEO, T.GEO_DARK),
         ("TOOL 1  ·  GNN", "Predict the ELF at that bottleneck", T.ELF, T.ELF_DARK),
         ("COMBINED", "Structural vs. electronic dimensionality", T.INK, T.INK)]
sh_ = (bh_avail - 2 * 44) / 3
for i, (tag, body, bar, ink) in enumerate(STEPS):
    sy = y + i * (sh_ + 44)
    p.rect(stx, sy, stw, sh_, T.SOFT, 10, label="Step")
    p.rect(stx, sy, 12, sh_, bar, 0)
    p.text(stx + 34, sy + 18, stw - 50, tag, 19, ink, 700, tracking=2.2)
    p.text(stx + 34, sy + 54, stw - 50, body, 25, T.INK, 500, lh=1.22)
    if i < 2:
        p.arrow(stx + stw / 2, sy + sh_ + 6, w=70, h=32)

# ---- 6 takeaways
p.layer("6 Takeaways")
x6 = M + C5W + 40
w6 = W - M - x6
top = p.card(x6, ROW3_Y, w6, ROW3_H, 6, "TAKEAWAYS", T.INK_2) + 30
x, wi = x6 + PAD, w6 - 2 * PAD
y = p.bullets(x, top, wi, [
    f"**Void ELF is predictable from nearby atoms:** RMSE {RMSE:.3f}, R² {R2:.2f} on held-out structure families.",
    f"**Geometry finds the 1D and 2D electrides:** the sweep agrees with the literature label for {pct(flow['agree'][1])} of 1D and {pct(flow['agree'][2])} of 2D.",
    f"**Open space alone cannot call 0D:** {N0_OPEN} of {N0} electrides labeled 0D still have connected open space at some probe size.",
], 27.5, T.INK, 400, T.GEO, gap=24)
y += 26
p.o.append(f'<line x1="{x}" y1="{y}" x2="{x + wi}" y2="{y}" stroke="{T.RULE}" stroke-width="2.5"/>')
y = p.text(x, y + 26, wi, "NEXT STEPS", 21, T.ELF_DARK, 700, tracking=2.6) + 18
p.bullets(x, y, wi, ["Run DFT on known electrides and test the GNN on them.",
                     "Predict ELF at bottlenecks, not only at void centers.",
                     "Combine both tools into one crystal-level electride screen."], 26.5, T.INK_2, 400, T.ELF, gap=16)

# ---- references
p.layer("References")
ry = H - REF_H
p.rect(0, ry, W, REF_H, T.INK, label="References band")
p.text(M, ry + 24, W - 2 * M,
       "**References**  [1] Becke & Edgecombe, J. Chem. Phys. 92, 5397 (1990), ELF.  [2] Ellis, Ellis & Miao, J. Mater. Inf. 6, 45 (2026), "
       "doi:10.20517/jmi.2026.44, ELFNet.  [3] Registry labels: Magnetic electrides, J. Am. Chem. Soc. (2023), doi:10.1021/jacs.3c00284; "
       "Zhu, Frolov & Choudhary, Matter 1, 1293 (2019); Burton et al., Chem. Mater. 30, 7521 (2018).  [4] Methods: Shannon, Acta Cryst. A32, 751 (1976), "
       "ionic radii; Willems et al., Micropor. Mesopor. Mater. 149, 134 (2012), Zeo++; Gilmer et al., ICML (2017), message passing.  "
       "[5] Related work: Xiao, Bristowe & Mostofi, Theory and Discovery of Electrides, arXiv:2605.11724 (2026); "
       "Qin, Lv & Zhong, EAC-Net: Predicting Real-Space Charge Density via Equivariant Atomic Contributions, arXiv:2508.04052; "
       "Ryan, Lengyel & Shatruk, Crystal Structure Prediction via Deep Learning, J. Am. Chem. Soc. 140, 10158 (2018); "
       "Wang et al., Machine Learning-Accelerated Discovery of A~2~BC~2~ Ternary Electrides with Diverse Anionic Electron Densities, J. Am. Chem. Soc. 145, 26412 (2023).  "
       "**Acknowledgements**  [advisor, funding]; Purdue RCAC (Negishi).", 20, "#d5dee3", 400, lh=1.32)

p.save(OUT, DEFS)
print("flow figure scale", round(fw / w, 3))
print("wrote", OUT, "| tool cards", round(TOOL_H), "px tall")
if "--no-pdf" not in sys.argv:
    subprocess.run([INKSCAPE, str(OUT), "--export-type=pdf", f"--export-filename={OUT.with_suffix('.pdf')}"], check=True,
                   capture_output=True)
    print("wrote", OUT.with_suffix(".pdf"))
