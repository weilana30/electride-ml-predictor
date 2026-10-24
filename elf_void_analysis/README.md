# ELFNet atom/void graph dataset

This package builds a model-ready dataset from the ELFNet POSCAR/ELFCAR pairs.
The graph has atom and candidate-void nodes, with periodic atom–atom,
void–atom, and void–void edges. Structure-derived graph arrays are model inputs.
ELF values and geometric accessible-space dimensionality are stored separately
under `targets/`.

## Workflow

```mermaid
flowchart LR
  A[ELFNet POSCAR + ELFCAR] --> B[build_manifest.py<br/>pair checks, IDs, hashes, grid metadata]
  B --> C[build_graph_records.py<br/>atoms, void candidates, node features, periodic edges]
  A --> D[label_voids_from_elf.py<br/>center ELF and local grid maxima]
  C --> J[label_void_dimensionality.py<br/>probe-accessible periodic void rank]
  C --> E[build_dataset.py<br/>join by sample ID, record failures]
  D --> E
  J --> E
  E --> F[write_hdf5_shards.py<br/>HDF5 graph and target arrays]
  E --> G[family split table<br/>same phase family stays together]
  F --> H[validate_dataset.py]
  B --> H
  G --> H
  H --> I[model loader / PyG]
```

| File | Responsibility | Output or contract |
|---|---|---|
| `build_manifest.py` | Index paired files, make stable sample/family IDs, parse POSCAR atom counts and ELFCAR grid dimensions, optionally hash source files | `manifest.parquet` |
| `build_graph_records.py` | Read POSCAR, run the existing candidate-void finder, expand each reduced candidate into unique symmetry-equivalent unit-cell nodes, create node features, and enumerate periodic atom–atom, void–atom and void–void edges | In-memory graph record; no ELFCAR targets |
| `label_voids_from_elf.py` | Check that POSCAR and ELFCAR structures match; interpolate ELF at each void and measure local grid maxima and offsets | Per-void target arrays |
| `label_void_dimensionality.py` | Threshold a periodic grid by nearest-atom clearance, label accessible components and compute independent winding rank | Per-void `void_dimension` and validity arrays plus definition metadata |
| `build_dataset.py` | Orchestrate the two extraction paths, join by sample, record failures, make family-safe splits | Complete version directory |
| `write_hdf5_shards.py` | Validate records and atomically write numeric arrays and UTF-8 metadata | `shards/graphs-*.h5` |
| `schema.json` | Describe manifest and HDF5 storage contracts | Copied into each built dataset |
| `dataset.py` | Load a stored graph and optionally convert it to PyG `HeteroData` | Model input; `targets/` stays separate from `x` |
| `validate_dataset.py` | Check HDF5 shapes, edges, targets, manifest/split membership and optionally source files | JSON validation report and optional void CSV |
| `dataset_config.json` | Store graph cutoffs, ELF label radii, dimensionality probe/grid spacing, split seed/fractions and shard size | Versioned build settings |

## Stored output

```text
gnn_void_dataset/v1/
├── config_used.json
├── schema.json
├── manifest.parquet       # one row per source sample directory, including failures
├── splits.parquet         # one row per successfully built sample
├── build_report.json      # counts, fingerprint, and per-sample failures
└── shards/
    ├── graphs-00000.h5
    └── graphs-00001.h5
```

Each HDF5 shard follows this layout:

```text
/samples/<sample_id>/
├── lattice_angstrom                 float [3, 3]
├── atoms/
│   ├── fractional_coords            float [n_atoms, 3]
│   ├── atomic_numbers                integer [n_atoms]
│   ├── features                      float [n_atoms, n_atom_features]
│   └── feature_validity              integer [n_atoms, n_validity_features]
├── voids/
│   ├── fractional_coords             float [n_voids, 3]
│   ├── features                      float [n_voids, n_void_features]
│   ├── method_multi_hot              integer [n_voids, n_methods]
│   ├── void_id                       UTF-8 [n_voids]
│   ├── orbit_id                      UTF-8 [n_voids]
│   ├── orbit_member_index            integer [n_voids]
│   └── orbit_multiplicity            integer [n_voids]
├── edges/
│   ├── atom_atom_index               integer [2, n_edges]
│   ├── atom_atom_distance_angstrom   float [n_edges]
│   ├── atom_atom_image_shift         integer [n_edges, 3]
│   ├── void_atom_index               integer [2, n_edges]
│   ├── void_atom_distance_angstrom   float [n_edges]
│   ├── void_atom_image_shift         integer [n_edges, 3]
│   ├── void_void_index               integer [2, n_edges]
│   ├── void_void_distance_angstrom   float [n_edges]
│   └── void_void_image_shift         integer [n_edges, 3]
├── targets/
│   ├── elf_at_center                 float [n_voids]
│   ├── elf_max_0.5A / 1A / 1.5A      float [n_voids] per radius
│   ├── offset_0.5A / 1A / 1.5A      float [n_voids, 3] per radius
│   ├── void_dimension               int8 [n_voids], rank 0–3 or -1
│   └── void_dimension_valid         uint8 [n_voids], 1 valid / 0 invalid
└── metadata/                         source paths, family, counts and settings
```

Edges are directed. Each `*_index` array has source indices in row zero and
destination indices in row one. The corresponding periodic `image_shift` is
added to the destination's fractional coordinate; all destination images
inside the configured cutoff are retained. Void–void edges store both
directions, exclude zero-shift self edges, and retain translated periodic self
images.

## Full build on Negishi

The local builder is serial. For the complete 50,000-sample dataset on Negishi,
use the single-node parallel workflow in
[`cluster/negishi/README.md`](cluster/negishi/README.md). It prepares one
manifest, builds independent chunks across one 128-core node, then merges them
and generates family-level splits globally.

## Build a bounded pilot

Use a new output path for each build. The builder refuses to write into a
nonempty directory, preventing old shards from being mistaken for current
outputs. Start with a small pilot, inspect its report, then choose a fresh
version output directory for the full run.

```bash
python -m pip install -r elf_void_analysis/requirements.txt
python -m elf_void_analysis.build_dataset \
  --limit 3 \
  --output elfnet_data/gnn_void_dataset/pilot-3
```

`--limit` limits the first sample directories in stable path order. Remove it
for a complete build, and provide a different `--output` path:

```bash
python -m elf_void_analysis.build_dataset \
  --output elfnet_data/gnn_void_dataset/v1
```

Validate the completed output (the validator checks every shard against the
manifest and split table):

```bash
python -m elf_void_analysis.validate_dataset \
  --shards 'elfnet_data/gnn_void_dataset/v1/shards/*.h5' \
  --manifest elfnet_data/gnn_void_dataset/v1/manifest.parquet \
  --splits elfnet_data/gnn_void_dataset/v1/splits.parquet \
  --schema elfnet_data/gnn_void_dataset/v1/schema.json \
  --source-root elfnet_data/extracted/dft-reference/dft-reference-elfs-50000-v1 \
  --require-full-coverage \
  --report-json elfnet_data/gnn_void_dataset/v1/validation.json
```

## Symmetry-expanded void nodes

Each symmetry-reduced candidate expands into its unique symmetry images inside
the unit cell before void–atom and void–void edges or per-void targets are
calculated. Thus each row in `voids/` and `targets/` corresponds to a physical
unique unit-cell site. `orbit_id` groups sites from one finder representative;
`orbit_member_index` indexes the lexically sorted wrapped member coordinates;
and `orbit_multiplicity` repeats the unique member count on every row.
`void_id` combines the representative and member indexes and is unique within
its sample. If symmetry analysis fails, each finder representative is retained
as a singleton orbit (member index 0, multiplicity 1), and the fallback status
is recorded in `metadata/settings/void_symmetry_expansion_status` as
`fallback_representatives_symmetry_analysis_failed`; successful expansion uses
`expanded`. The symmetry operation count and tolerance are also stored in the
settings metadata. All ELF and dimensionality labels use the expanded node
order, so each physical unit-cell site receives its own target.

## Current target interpretation and limits

The labels are ELFCAR `total` grid values. `elf_at_center` is periodic
trilinear interpolation at the candidate coordinate. A local maximum is the
largest raw grid sample inside a periodic Cartesian sphere of the configured
radius; its peak offset is also stored. The sphere is not masked against atoms,
so a local maximum can fall in an atomic basin. These are useful first
supervision targets, but should be inspected before interpreting them as
electron localization specifically at a void.

`void_dimension` is a geometric, probe-dependent accessible-space winding
rank. The default probe radius is 1.0 Å and grid spacing is approximately
0.5 Å along each lattice vector. Values 0–3 give the independent periodic
directions spanned by the accessible component containing a candidate center.
A center mapped to an inactive voxel receives `void_dimension=-1` and
`void_dimension_valid=0`. This describes geometric free space; it does not
claim ELF connectivity or electronic dimensionality. The probe radius, grid
spacing, and periodic six-neighbor connectivity are part of the label
definition and are saved in the build config and per-sample settings.
