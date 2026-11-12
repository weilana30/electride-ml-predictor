# ELFNet dataset provenance

## Source data

The raw ELFNet source tree contains paired VASP `POSCAR` and `ELFCAR` files:

```text
/scratch/negishi/USER/dft-reference-elfs-50000-v1/
```

This scratch location is temporary storage and must not be the only archive of the source data. The source tree is organized by ELFNet categories such as `2/` and `Mono/`, followed by parent phases and variants.

## Extraction workspace

The original Negishi extraction run used:

```text
/scratch/negishi/USER/electride-ml-predictor/elfnet-gnn-2026-10-23/
```

The workflow created a manifest, distributed extraction into independent chunks, merged the chunks, regenerated family-safe splits, and validated the result. Intermediate chunks and Slurm logs remain in this external run root.

## Persistent processed dataset

The processed dataset used by the GNNs is:

```text
/home/USER/elfnet-datasets/elfnet-gnn-2026-10-23/
```

It contains:

```text
manifest.parquet
splits.parquet
schema.json
config_used.json
build_report.json
validation.json
shards/graphs-*.h5
```

The final dataset contains 50,000 processed samples. Its split is:

| Split | Samples | Families |
| --- | ---: | ---: |
| Train | 40,031 | 37 |
| Validation | 4,969 | 9 |
| Test | 5,000 | 9 |

The split-table SHA256 is:

`cdb3fc87b80cb05ce1cfdd8b94267696cfb0bb9e810004512144b0c2f2aac7f4`

## Graph and label contract

Each sample contains atom nodes, candidate-void nodes, atom/atom, void/atom, atom/void, and void/void periodic relations, plus node features and edge distances. The model inputs are structure-derived. ELFCAR-derived quantities are labels only.

The primary architecture-study target is `elf_at_center`, the periodic trilinearly interpolated ELF value at each candidate void center. The dataset also stores local ELF maxima at 0.5, 1.0, and 1.5 Å, offsets to those maxima, and geometric void-network dimensionality.

The dimensionality label is geometric accessible-space winding rank; it is not an electronic or ELF dimensionality label.

## Validation and leakage controls

- POSCAR/ELFCAR pairs are checked for matching lattice, atom count, and atom ordering.
- Source manifests can include per-file hashes.
- HDF5 arrays, edges, targets, and metadata are validated before use.
- Related parent-phase families remain in one split.
- Training normalization is derived from training rows only.
- ELFCAR-derived targets are excluded from model inputs.

The split holds out named phase families, but related pressure or space-group variants may still occur across splits. It should not be interpreted as strict unseen-composition or unseen-pressure generalization.
