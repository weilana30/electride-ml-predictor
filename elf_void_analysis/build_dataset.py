#!/usr/bin/env python3
"""Build the ELFNet atom/void graph dataset from paired POSCAR/ELFCAR files.

Each output shard is written atomically. Failed samples are retained in the
manifest and summarized in build_report.json. A new run requires a fresh output
directory so stale shards cannot be mixed into a dataset version.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .build_graph_records import build_graph_record
    from .build_manifest import build_manifest
    from .label_voids_from_elf import label_voids
    from .write_hdf5_shards import write_hdf5_shard
except ImportError:  # direct script invocation
    from build_graph_records import build_graph_record
    from build_manifest import build_manifest
    from label_voids_from_elf import label_voids
    from write_hdf5_shards import write_hdf5_shard


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError(
            "Parquet output requires a NumPy-compatible pyarrow installation. "
            f"Import failed: {exc}"
        ) from exc
    return pa, pq


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    pa, pq = _arrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _fingerprint(rows: list[dict[str, Any]], config: dict[str, Any]) -> str:
    payload = {
        "config": config,
        "samples": [
            (r["sample_id"], r.get("poscar_sha256"), r.get("elfcar_sha256"))
            for r in rows
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _assign_splits(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep each family intact and approximately balance sample counts."""
    fractions = config["split_fractions"]
    split_names = ("train", "validation", "test")
    if set(fractions) != set(split_names) or abs(sum(fractions.values()) - 1.0) > 1e-8:
        raise ValueError("split_fractions must define train/validation/test summing to 1")
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family_rows[str(row["family_id"])].append(row)
    seed = int(config.get("split_seed", 0))
    families = list(family_rows)
    random.Random(seed).shuffle(families)
    shuffle_rank = {family: index for index, family in enumerate(families)}
    total = sum(len(family_rows[f]) for f in families)
    targets = {name: total * fractions[name] for name in split_names}
    counts = {name: 0 for name in split_names}
    assignment: dict[str, str] = {}
    for family in sorted(families, key=lambda f: (-len(family_rows[f]), shuffle_rank[f])):
        size = len(family_rows[family])
        # Fill the split currently furthest below its requested sample count.
        chosen = max(split_names, key=lambda name: (targets[name] - counts[name], -split_names.index(name)))
        assignment[family] = chosen
        counts[chosen] += size
    return [
        {"sample_id": row["sample_id"], "family_id": row["family_id"],
         "split": assignment[str(row["family_id"])], "split_seed": seed}
        for row in rows
    ]


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _schema_for_config(config: dict[str, Any], schema_path: Path) -> dict[str, Any]:
    """Bind the storage schema's target names to the configured ELF radii."""
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    targets = {"elf_at_center": "periodic trilinear interpolation of ELFCAR total grid"}
    shapes = {"elf_at_center": ["n_voids"]}
    for radius_value in config["elf_local_max_radii_angstrom"]:
        radius = float(radius_value)
        key = f"{radius:g}A"
        targets[f"elf_max_{key}"] = f"raw local grid maximum within {radius:g} angstrom"
        targets[f"offset_{key}"] = f"Cartesian offset vector to local maximum within {radius:g} angstrom [n_voids,3]"
        shapes[f"elf_max_{key}"] = ["n_voids"]
        shapes[f"offset_{key}"] = ["n_voids", 3]
    schema["graph_storage"]["targets"]["per_void_arrays"] = targets
    schema["graph_storage"]["targets"]["shapes"] = shapes
    schema["graph_storage"]["targets"]["configured_local_max_radii_angstrom"] = [
        float(value) for value in config["elf_local_max_radii_angstrom"]
    ]
    return schema


def build(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Choose a new version/pilot directory."
        )
    output.mkdir(parents=True, exist_ok=True)
    source = args.source_root.expanduser().resolve()

    manifest = build_manifest(source, hash_files=not args.no_hashes, limit=args.limit)
    manifest_by_id = {str(row["sample_id"]): row for row in manifest}
    ready = [r for r in manifest if r["processing_status"] == "ready"]
    fingerprint = _fingerprint(ready, config)
    _write_parquet(output / "manifest.parquet", manifest)
    _json_write(output / "config_used.json", config)
    schema_path = Path(__file__).with_name("schema.json")
    _json_write(output / "schema.json", _schema_for_config(config, schema_path))

    radii = tuple(float(x) for x in config["elf_local_max_radii_angstrom"])
    shard_size = int(args.shard_size or config["shard_size"])
    if shard_size < 1:
        raise ValueError("shard size must be positive")
    shard_dir = output / "shards"
    shard_dir.mkdir()
    failures: list[dict[str, str]] = []
    successful: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    shard_index = 0

    def flush() -> None:
        nonlocal shard_index, shard_records
        if not shard_records:
            return
        write_hdf5_shard(
            shard_dir / f"graphs-{shard_index:05d}.h5", shard_records,
            build_fingerprint=fingerprint, shard_index=shard_index,
            source_root=source, schema_version="1.0",
            dataset_version=str(config["dataset_version"]), overwrite=False,
        )
        shard_index += 1
        shard_records = []

    for index, row in enumerate(ready, start=1):
        poscar = source / str(row["poscar_relative_path"])
        elfcar = source / str(row["elfcar_relative_path"])
        try:
            graph = build_graph_record(
                poscar, sample_id=row["sample_id"], family_id=row["family_id"],
                atom_cutoff_angstrom=float(config["atom_cutoff_angstrom"]),
                atom_void_cutoff_angstrom=float(config["atom_void_cutoff_angstrom"]),
                symprec=float(config["symprec"]),
            )
            graph["targets"] = label_voids(
                elfcar, graph["lattice_angstrom"],
                graph["voids"]["fractional_coords"], radii,
                poscar_path=poscar,
            )
            graph["source_poscar_relative_path"] = row["poscar_relative_path"]
            graph["source_elfcar_relative_path"] = row["elfcar_relative_path"]
            graph["grid_shape"] = row["grid_shape"]
            graph["source_poscar_sha256"] = row["poscar_sha256"]
            graph["source_elfcar_sha256"] = row["elfcar_sha256"]
            shard_records.append(graph)
            successful.append(row)
            manifest_by_id[str(row["sample_id"])]["processing_status"] = "processed"
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            manifest_by_id[str(row["sample_id"])]["processing_status"] = "failed_processing"
            manifest_by_id[str(row["sample_id"])]["failure_reason"] = message
            failures.append({"sample_id": row["sample_id"], "reason": message})
        if len(shard_records) >= shard_size:
            flush()
        if args.progress_every and index % args.progress_every == 0:
            print(f"processed {index}/{len(ready)} ready samples; {len(successful)} successful, {len(failures)} failed", flush=True)
    flush()

    final_manifest = list(manifest_by_id.values())
    _write_parquet(output / "manifest.parquet", final_manifest)
    splits = _assign_splits(successful, config)
    _write_parquet(output / "splits.parquet", splits)
    report = {
        "dataset_name": config["dataset_name"],
        "dataset_version": config["dataset_version"],
        "source_root": str(source), "output_root": str(output),
        "build_fingerprint": fingerprint,
        "manifest_count": len(final_manifest), "initially_ready_count": len(ready),
        "processed_count": len(successful), "failed_count": len(failures),
        "shard_count": shard_index, "shard_size": shard_size,
        "split_counts": {name: sum(x["split"] == name for x in splits) for name in ("train", "validation", "test")},
        "failures": failures,
    }
    _json_write(output / "build_report.json", report)
    return report


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    default_source = repo / "elfnet_data/extracted/dft-reference/dft-reference-elfs-50000-v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=default_source)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("dataset_config.json"))
    parser.add_argument("--limit", type=int, help="Bounded pilot run over first N sample directories")
    parser.add_argument("--shard-size", type=int, help="Override config shard size")
    parser.add_argument("--no-hashes", action="store_true", help="Skip source SHA-256 hashing")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    try:
        report = build(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
