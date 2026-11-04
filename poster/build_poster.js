// Build poster.pptx: one 48 x 36 in slide (4 ft wide, 3 ft tall).
//
//   npm install pptxgenjs@3.12.0   (once, anywhere on NODE_PATH)
//   node poster/build_poster.js
//
// Layout and text are documented in LAYOUT.md and CONTENT.md; keep them in
// sync with this file. Dashed white boxes labelled "TO DO" mark missing pieces
// (see TODO.md).
const path = require("path");
const pptxgen = require("pptxgenjs");

const HERE = __dirname;
const fig = (name) => path.join(HERE, "figures", name);
const dia = (name) => path.join(HERE, "diagrams", name);

const W = 48, H = 36;
const COLOR = {
  navy: "13294B", blue: "2A78D6", orange: "EB6834", ink: "1B1B1B", ink2: "52514E",
  card: "F1F5FA", white: "FFFFFF", todoEdge: "8A94A6", todoText: "5B6577", ice: "CADCFC",
};
const SIZE = { title: 80, subtitle: 38, authors: 28, header: 40, takeaway: 28, body: 24, caption: 18 };
const FONT = "Arial";

const pres = new pptxgen();
pres.defineLayout({ name: "POSTER_48x36", width: W, height: H });
pres.layout = "POSTER_48x36";
pres.title = "Electride void ELF and dimensionality poster";
pres.author = "Andrew Weiland";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
const slide = pres.addSlide();
slide.background = { color: COLOR.white };

let objectId = 0;
const name = (label) => `${label}-${++objectId}`;

function text(value, x, y, w, h, opts = {}) {
  slide.addText(value, {
    x, y, w, h, isTextBox: true, margin: 0, fontFace: FONT, color: COLOR.ink, fontSize: SIZE.body,
    valign: "top", objectName: name("text"), ...opts,
  });
}

// items: strings or arrays of [text, bold] runs; one bullet paragraph each.
function bullets(items, x, y, w, h, opts = {}) {
  const size = opts.fontSize || SIZE.body;
  const runs = [];
  items.forEach((item, i) => {
    const parts = typeof item === "string" ? [[item, false]] : item;
    parts.forEach(([t, bold], j) => {
      const last = j === parts.length - 1;
      runs.push({
        text: t,
        options: {
          bold, bullet: j === 0 ? { indent: size * 0.9 } : undefined,
          breakLine: last && i < items.length - 1, paraSpaceAfter: last ? size * 0.45 : 0,
        },
      });
    });
  });
  text(runs, x, y, w, h, { fontSize: size, ...opts });
}

function card(x, y, w, h, title, badge) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.3, fill: { color: COLOR.card }, line: { color: COLOR.card, width: 0 },
    objectName: name("card"),
  });
  let tx = x + 0.45;
  if (badge) {
    const color = badge === "RQ3" ? COLOR.orange : COLOR.blue;
    slide.addShape(pres.shapes.OVAL, {
      x: tx, y: y + 0.32, w: 1.2, h: 1.2, fill: { color }, line: { color, width: 0 }, objectName: name("badge"),
    });
    text(badge, tx, y + 0.32, 1.2, 1.2, {
      fontSize: 24, bold: true, color: COLOR.white, align: "center", valign: "middle",
    });
    tx += 1.5;
  }
  text(title, tx, y + 0.35, x + w - tx - 0.4, 1.15, {
    fontSize: SIZE.header, bold: true, color: COLOR.navy, valign: "middle",
  });
}

function image(file, x, y, w, h, alt) {
  slide.addImage({ path: file, x, y, w, h, altText: alt, objectName: name("image") });
}

function todo(label, x, y, w, h, opts = {}) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: opts.fill || COLOR.white },
    line: { color: opts.edge || COLOR.todoEdge, width: 2.5, dashType: "dash" }, objectName: name("todo"),
  });
  text(`TO DO: ${label}`, x + 0.2, y + 0.15, w - 0.4, h - 0.3, {
    fontSize: opts.fontSize || 20, italic: true, color: opts.color || COLOR.todoText, align: "center",
    valign: "middle",
  });
}

// ---------------------------------------------------------------- title band
slide.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: W, h: 4.9, fill: { color: COLOR.navy }, line: { color: COLOR.navy, width: 0 },
  objectName: name("title-band"),
});
text("Where Do the Extra Electrons Go?", 0.9, 0.45, 37, 1.45, {
  fontSize: SIZE.title, bold: true, color: COLOR.white,
});
text("Predicting interstitial electron localization and void dimensionality for electride screening",
  0.9, 1.95, 37, 0.8, { fontSize: SIZE.subtitle, color: COLOR.ice });
text([
  { text: "Andrew Weiland", options: { bold: true } },
  { text: "  ·  [Advisor name]  ·  [Department], Purdue University  ·  [email]" },
], 0.9, 3.15, 37, 0.7, { fontSize: SIZE.authors, color: COLOR.white });
todo("logo", 38.6, 0.7, 3.9, 3.5, { fill: COLOR.navy, edge: COLOR.ice, color: COLOR.ice });
todo("QR code (repo or contact)", 43.0, 0.7, 4.25, 3.5, { fill: COLOR.navy, edge: COLOR.ice, color: COLOR.ice });

// ----------------------------------------------------------------- columns
const TOP = 5.6, GAP = 0.6;
const C1 = { x: 0.75, w: 13.5 };
const C2 = { x: C1.x + C1.w + GAP, w: 16.2 };
const C3 = { x: C2.x + C2.w + GAP, w: 15.6 };
const pad = 0.5;

// Column 1 -------------------------------------------------- Background
{
  const { x, w } = C1, y = TOP, h = 13.9;
  card(x, y, w, h, "Background");
  bullets([
    [["Electrides", true], [" are crystals whose extra electrons sit in the empty space between atoms and act as anions.", false]],
    "How those electrons connect sets their properties: isolated 0D cages, 1D channels, or 2D layers.",
    [["ELF", true], [" (electron localization function, 0–1): a high value in a void marks a trapped electron, the electride fingerprint.", false]],
    "Finding them needs a DFT calculation per crystal: too slow to screen large databases.",
  ], x + pad, y + 1.65, w - 2 * pad, 4.3);
  image(dia("electride_concept.png"), x + pad, y + 6.05, 4.4, 3.46, "Cations with electrons in the spaces between them");
  text("Cations (+) with electrons (e⁻) in the empty space between them.", x + pad + 4.7, y + 6.3, w - 2 * pad - 4.7, 3.0,
    { fontSize: SIZE.caption, color: COLOR.ink2, italic: true, valign: "middle" });
  const vw = (w - 2 * pad - 2 * 0.3) / 3;
  [["0D cages", "mayenite"], ["1D channels", "Y5Si3"], ["2D layers", "Ca2N"]].forEach(([dim, mat], i) => {
    todo(`VESTA render: ${dim} (${mat}), open space at isosurface 0.5`, x + pad + i * (vw + 0.3), y + 9.8, vw, 3.65, { fontSize: 17 });
  });
}

// Column 1 -------------------------------------------------- Research questions
{
  const { x, w } = C1, y = TOP + 13.9 + GAP, h = 6.4;
  card(x, y, w, h, "Research questions");
  const qs = [
    ["RQ1", COLOR.blue, "Can a GNN predict ELF at candidate voids from the crystal structure alone?"],
    ["RQ2", COLOR.blue, "Can geometry alone recover an electride's dimensionality (0D/1D/2D)?"],
    ["RQ3", COLOR.orange, "How do the two combine into an electride screen?"],
  ];
  qs.forEach(([label, color, q], i) => {
    const yy = y + 1.75 + i * 1.5;
    slide.addShape(pres.shapes.OVAL, { x: x + pad, y: yy, w: 1.05, h: 1.05, fill: { color }, line: { color, width: 0 }, objectName: name("badge") });
    text(label, x + pad, yy, 1.05, 1.05, { fontSize: 20, bold: true, color: COLOR.white, align: "center", valign: "middle" });
    text(q, x + pad + 1.35, yy - 0.05, w - 2 * pad - 1.35, 1.2, { fontSize: SIZE.body, valign: "middle" });
  });
}

// Column 1 -------------------------------------------------- Data
{
  const { x, w } = C1, y = TOP + 13.9 + GAP + 6.4 + GAP, h = 4.4;
  card(x, y, w, h, "Data");
  bullets([
    [["50,000 DFT structures with ELF", true], [" (ELFNet): metal frameworks from high-pressure hydrides.", false]],
    "Split by family: 40,031 train · 4,969 validation · 5,000 test from 9 unseen families.",
    [["260 known electrides", true], [" with literature dimensionality for testing RQ2.", false]],
  ], x + pad, y + 1.6, w - 2 * pad, 2.7);
}

// Column 1 -------------------------------------------------- References
{
  const { x, w } = C1, y = TOP + 13.9 + GAP + 6.4 + GAP + 4.4 + GAP, h = H - 0.75 - (TOP + 13.9 + GAP + 6.4 + GAP + 4.4 + GAP);
  card(x, y, w, h, "References");
  todo("final reference list: ELF (Becke & Edgecombe 1990), ELFNet dataset, electride screens (Burton 2018, Zhu 2019, JACS 2023 magnetic electrides), Zeo++ (Willems 2012); acknowledgements (advisor, funding, Purdue RCAC Negishi)",
    x + pad, y + 1.55, w - 2 * pad, h - 1.95, { fontSize: 17 });
}

// Column 2 -------------------------------------------------- Approach
{
  const { x, w } = C2, y = TOP, h = 10.7;
  card(x, y, w, h, "Approach");
  image(dia("pipeline.png"), x + 0.4, y + 1.6, 15.4, 5.6, "Pipeline from crystal to voids, GNN ELF and void dimensionality");
  bullets([
    [["Voids:", true], [" Voronoi interstitials plus points farthest from atoms, expanded by symmetry.", false]],
    [["GNN:", true], [" atoms and voids are nodes joined by distance edges; message passing predicts ELF at each void.", false]],
    [["Void network:", true], [" each atom blocks a sphere (ionic radius + 1.0 Å probe); an exact power-diagram network counts how many directions the open space repeats through the crystal.", false]],
  ], x + pad, y + 7.35, w - 2 * pad, 3.2);
}

// Column 2 -------------------------------------------------- RQ1 results
{
  const { x, w } = C2, y = TOP + 10.7 + GAP, h = H - 0.75 - (TOP + 10.7 + GAP);
  card(x, y, w, h, "Can a GNN predict ELF at voids?", "RQ1");
  text("Learning from the surrounding atoms predicts void ELF: test RMSE 0.058, R² 0.83",
    x + pad, y + 1.75, w - 2 * pad, 1.0, { fontSize: SIZE.takeaway, bold: true, color: COLOR.blue });
  image(fig("gnn_model_comparison.png"), x + 1.1, y + 2.9, 14.0, 7.65, "Test RMSE for each model");
  image(fig("gnn_parity_atom_only.png"), x + pad, y + 10.75, 7.3, 6.84, "Predicted vs DFT ELF on the test set");
  bullets([
    "A void alone predicts nothing: 0.140, the same as guessing the average.",
    "Message passing beats fixed atom summaries (0.065 vs 0.077–0.085).",
    "Hand-made void descriptors hurt; atoms alone work best.",
    [["Limitation:", true], [" rare very-high-ELF voids (≥ 0.7) are under-predicted.", false]],
  ], x + pad + 7.6, y + 11.05, w - 2 * pad - 7.6, 6.5, { fontSize: 22 });
}

// Column 3 -------------------------------------------------- RQ2 results
{
  const { x, w } = C3, y = TOP, h = 15.4;
  card(x, y, w, h, "Can geometry alone find dimensionality?", "RQ2");
  text("On 260 known electrides: 57% balanced accuracy (chance 33%)",
    x + pad, y + 1.75, w - 2 * pad, 0.8, { fontSize: SIZE.takeaway, bold: true, color: COLOR.blue });
  image(fig("accuracy_vs_probe.png"), x + pad, y + 2.7, 14.6, 5.59, "Agreement with literature vs probe radius");
  image(fig("confusion_ionic_1p0.png"), x + pad, y + 8.45, 7.2, 5.24, "Literature vs predicted dimension");
  bullets([
    "Ionic radii beat one size for all atoms (57% vs 45%).",
    "The right dimension appears at some probe for 99%: one probe can't fit every material.",
    [["Main miss:", true], [" 50 of 167 0D electrides look 2D/3D. The space connects; the electrons don't.", false]],
  ], x + pad + 7.5, y + 8.6, w - 2 * pad - 7.5, 5.2, { fontSize: 22 });
  text("Literature labels come from computational screens (JACS 2023, Matter 2019, Chem. Mater. 2018), compiled with AI-assisted search; not fully hand-checked.",
    x + pad, y + 13.95, w - 2 * pad, 1.2, { fontSize: SIZE.caption, italic: true, color: COLOR.ink2 });
}

// Column 3 -------------------------------------------------- RQ3
{
  const { x, w } = C3, y = TOP + 15.4 + GAP, h = 7.4;
  card(x, y, w, h, "Putting them together (next step)", "RQ3");
  image(dia("pipes_vs_electrons.png"), x + 1.2, y + 1.65, 13.2, 5.36, "Connected electrons vs separate pockets in the same channel");
}

// Column 3 -------------------------------------------------- Conclusions
{
  const { x, w } = C3, y = TOP + 15.4 + GAP + 7.4 + GAP, h = H - 0.75 - (TOP + 15.4 + GAP + 7.4 + GAP);
  card(x, y, w, h, "Conclusions and next steps");
  const half = (w - 2 * pad - 0.5) / 2;
  bullets([
    "Void ELF can be predicted from structure alone (R² 0.83).",
    "Geometry finds electride dimensionality well above chance; its errors show where ELF is needed.",
  ], x + pad, y + 1.6, half, h - 1.9, { fontSize: 22 });
  bullets([
    "Run DFT on known electrides to test the GNN on real ones.",
    "Predict ELF at void necks → electronic dimensionality.",
    "Crystal-level electride score; test on unseen compositions.",
  ], x + pad + half + 0.5, y + 1.6, half, h - 1.9, { fontSize: 22 });
}

pres.writeFile({ fileName: path.join(HERE, "poster.pptx") }).then((f) => console.log("wrote", f));
