# TODO

## Dataset: `void_dimension` label (low priority; not used by the GNN or the poster)

`build_dataset.py` computes a per-void `void_dimension` label with the grid
tool in `elf_void_analysis/label_void_dimensionality.py` and stores it in the
dataset. No training, validation, test or study code reads it; the GNN only
uses `targets/elf_at_center`.

Problems with the stored label:

- With the configured bare 1.0 Å cutoff from atom centers
  (`void_dimension_probe_radius_angstrom` in `dataset_config.json`), nearly
  all space is open, so every sampled ELFNet void is labeled 3D.
- The grid tool looks up each void at the grid point at the lower corner of
  its grid cell (`np.floor` at `label_void_dimensionality.py:161`), up to
  ~0.9 Å away, instead of at the void itself. Open voids can get -1 and
  blocked ones can get a dimension; it disagrees with the exact power-diagram
  method on 12–22% of test points.

Choose one:

- [ ] **Remove:** stop computing `void_dimension` / `void_dimension_valid`
  in `build_dataset.py` (saves build time), and update `schema.json`,
  `validate_dataset.py` and the README.
- [ ] **Fix:** relabel with `power_void_connectivity.py` at settings that
  matched the literature (2.0 Å uniform, or ionic radius + 1.0 Å), store
  the probe settings with the label, and consider keeping several probes.
