# Poster to-do list

Items marked **(on poster)** have a dashed TO DO box in `poster.pptx`.

## Must do

- [ ] **Author line (on poster):** replace `[Advisor name]`, `[Department]`, `[email]` in the title band.
- [ ] **VESTA renders × 3 (on poster, Background card):** 0D mayenite, 1D Y5Si3, 2D Ca2N.
  - Files: `elf_void_analysis/case_studies/power_diagram/{Ca6Al7O16_0D,Y5Si3_1D,Ca2N_2D}/ionic_1p0/open_space-filled.xsf`
  - Isosurface level 0.5; set atom radii to the ionic radii in each `report.json`
    (e.g. Ca 1.14 Å, N 1.32 Å) so the void doesn't look like it's inside atoms.
  - Same viewing direction style, background white, export PNG at ≥ 300 dpi, about 4 × 3.6 in.
- [ ] **References and acknowledgements (on poster):** final list in a consistent citation style:
  ELF (Becke & Edgecombe, J. Chem. Phys. 1990), ELFNet dataset (find the citation),
  Burton et al. Chem. Mater. 2018, Zhu et al. Matter 2019, magnetic electrides JACS 2023,
  Zeo++ (Willems et al. 2012). Acknowledge advisor, funding, Purdue RCAC (Negishi).
- [ ] **Logo (on poster):** university or lab logo, white version for the navy band.
- [ ] **Open `poster.pptx` in PowerPoint and check every text box fits.** The draft
  was checked only with macOS Quick Look, which can render fonts slightly differently.
- [ ] **Confirm the poster size and orientation** with the session organizers
  (built at the 48 × 36 in maximum, landscape).

## Should do

- [ ] **Confirmation study:** finish it with the agent prompt; if it picks a different
  best GNN, rerun `elf_void_gnn/poster_figures/make_figures.py` and update the RQ1 numbers.
- [ ] **QR code (on poster):** link to the repository or contact page.
- [ ] **Proofread** all text against `CONTENT.md`; check every number against its source file.
- [ ] **Print a test page** at 25% or one card at 100% to check readability.

## Nice to have

- [ ] Small method schematic for the void network (atoms as spheres + probe,
  open space repeating across the cell), if space allows in the Approach card.
- [ ] A one-line look at which kinds of 0D electrides get called 2D/3D (e.g. intermetallics).
- [ ] Spot-check the 9 experimentally confirmed electride labels against their papers.

## Done

- [x] Dimensionality test on 260 known electrides and its figures
- [x] GNN figures (model comparison, parity)
- [x] Pipeline, pipes-vs-electrons, and electride concept diagrams
- [x] Poster draft with all sections and figure slots
