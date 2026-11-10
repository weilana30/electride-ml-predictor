#!/usr/bin/env python3
"""Write a copy of a built dataset with version-2 void features.

Version 1 void features counted each basis atom once (minimum image) and
padded missing nearest distances with 0, so they depended on the unit cell.
Version 2 (``build_graph_records.void_neighbor_features``) uses every periodic
atom image. Every stored void->atom edge within the 6 A cutoff is already a
periodic image, so the new features are computed from the stored edges without
reading ELFCARs. Everything else (graphs, targets, manifest, splits) is copied
unchanged, so the split SHA256 is preserved.

    python -m elf_void_analysis.recompute_void_features SOURCE_DATASET OUTPUT_DATASET
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

try:
    from .build_graph_records import VOID_FEATURE_DEFINITION, VOID_FEATURE_NAMES, void_neighbor_features
except ImportError:  # pragma: no cover - script execution
    from build_graph_records import VOID_FEATURE_DEFINITION, VOID_FEATURE_NAMES, void_neighbor_features


def patch_sample(group: h5py.Group) -> dict[str, float]:
    """Rewrite ``voids/features`` of one sample in place; return change stats."""
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in group["voids/feature_names"][...]]
    if names != VOID_FEATURE_NAMES:
        raise ValueError(f"{group.name}: unexpected void feature names {names}")
    settings = group["metadata/settings"].attrs
    cutoff = float(settings["atom_void_cutoff_angstrom"])
    old = group["voids/features"][...]
    new = void_neighbor_features(
        old.shape[0], group["edges/void_atom_index"][0], group["edges/void_atom_distance_angstrom"][...],
        group["voids/clearance_angstrom"][...], cutoff,
    ).astype(np.float32)
    if old.shape[0] and not np.allclose(new[:, :2], old[:, :2], atol=1e-4):
        raise ValueError(f"{group.name}: clearance/nearest slots disagree with stored edges")
    group["voids/features"][...] = new
    settings["void_neighbor_count_definition"] = VOID_FEATURE_DEFINITION
    return {"voids": int(old.shape[0]), "changed": int(np.any(np.abs(new - old) > 1e-5, axis=1).sum()),
            "old_zero_nearest": int(np.sum(old[:, 5:9] == 0.0)),
            "cutoff_filled": int(np.sum(new[:, 5:9] >= cutoff - 1e-6))}


def run(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"{output} exists; choose a new dataset directory")
    shutil.copytree(source, output)
    totals = {"samples": 0, "voids": 0, "changed": 0, "old_zero_nearest": 0, "cutoff_filled": 0}
    for shard in sorted((output / "shards").glob("*.h5")):
        with h5py.File(shard, "r+") as handle:
            handle.attrs["dataset_version"] = "v2-void-features"
            for sample in handle["samples"].values():
                for key, value in patch_sample(sample).items():
                    totals[key] += value
                totals["samples"] += 1
    split_sha = hashlib.sha256((output / "splits.parquet").read_bytes()).hexdigest()
    report = {"source_dataset": str(source), "void_feature_definition": VOID_FEATURE_DEFINITION,
              "split_sha256": split_sha, **totals}
    (output / "void_feature_patch.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
