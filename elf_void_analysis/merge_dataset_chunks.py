#!/usr/bin/env python3
"""Merge completed chunk builds into one validated dataset directory.

The array workers write independent HDF5 shards. This step combines their
manifests and status reports, creates one family-safe split table, and copies
shards into a single directory with a common build fingerprint.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import h5py

try:
    from .build_dataset import (
        _assign_splits,
        _fingerprint,
        _json_write,
        _read_parquet_rows,
        _schema_for_config,
        _write_parquet,
    )
except ImportError:  # direct script invocation
    from build_dataset import (
        _assign_splits,
        _fingerprint,
        _json_write,
        _read_parquet_rows,
        _schema_for_config,
        _write_parquet,
    )


def merge(
    manifest_path: Path,
    chunks_root: Path,
    output: Path,
    config_path: Path,
    source_root: Path,
    samples_per_task: int,
) -> dict[str, Any]:
    if samples_per_task < 1:
        raise ValueError("samples_per_task must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")

    manifest_rows = _read_parquet_rows(manifest_path)
    if not manifest_rows:
        raise ValueError("prepared manifest is empty")
    initially_ready_count = sum(row["processing_status"] == "ready" for row in manifest_rows)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task_count = math.ceil(len(manifest_rows) / samples_per_task)
    final_by_id = {str(row["sample_id"]): dict(row) for row in manifest_rows}
    seen_chunk_ids: set[str] = set()
    failures: list[dict[str, str]] = []
    shard_sources: list[Path] = []

    for task_index in range(task_count):
        task_dir = chunks_root / f"task-{task_index:05d}"
        chunk_manifest_path = task_dir / "manifest.parquet"
        report_path = task_dir / "build_report.json"
        shard_dir = task_dir / "shards"
        if not chunk_manifest_path.is_file() or not report_path.is_file() or not shard_dir.is_dir():
            raise FileNotFoundError(
                f"Task {task_index} is incomplete: expected manifest, report, and shards in {task_dir}"
            )
        expected_ids = {
            str(row["sample_id"])
            for row in manifest_rows[
                task_index * samples_per_task:(task_index + 1) * samples_per_task
            ]
        }
        chunk_rows = _read_parquet_rows(chunk_manifest_path)
        chunk_ids = {str(row["sample_id"]) for row in chunk_rows}
        if chunk_ids != expected_ids:
            raise ValueError(
                f"Task {task_index} manifest IDs do not match its expected input slice "
                f"(expected {len(expected_ids)}, got {len(chunk_ids)})"
            )
        overlap = seen_chunk_ids.intersection(chunk_ids)
        if overlap:
            raise ValueError(f"Samples were processed in more than one task: {sorted(overlap)[:5]}")
        seen_chunk_ids.update(chunk_ids)
        for row in chunk_rows:
            sample_id = str(row["sample_id"])
            final_by_id[sample_id] = dict(row)
        task_report = json.loads(report_path.read_text(encoding="utf-8"))
        failures.extend(task_report.get("failures", []))
        shard_sources.extend(sorted(shard_dir.glob("graphs-*.h5")))

    if seen_chunk_ids != set(final_by_id):
        raise ValueError("Array chunks do not cover every prepared manifest row")

    final_manifest = [final_by_id[str(row["sample_id"])] for row in manifest_rows]
    successful = [row for row in final_manifest if row["processing_status"] == "processed"]
    fingerprint = _fingerprint(successful, config)
    output.mkdir(parents=True, exist_ok=True)
    shard_dir = output / "shards"
    shard_dir.mkdir()
    seen_samples: set[str] = set()
    shard_count = 0
    record_count = 0
    for source_shard in shard_sources:
        destination = shard_dir / f"graphs-{shard_count:05d}.h5"
        shutil.copy2(source_shard, destination)
        with h5py.File(destination, "r+") as h5:
            sample_ids = set(h5["samples"].keys())
            if seen_samples.intersection(sample_ids):
                raise ValueError(f"Duplicate graph sample IDs found in {source_shard}")
            seen_samples.update(sample_ids)
            record_count += len(sample_ids)
            h5.attrs["build_fingerprint"] = fingerprint
            h5.attrs["shard_index"] = shard_count
            h5.attrs["source_root"] = str(source_root.resolve())
        shard_count += 1
    if seen_samples != {str(row["sample_id"]) for row in successful}:
        raise ValueError(
            "Merged HDF5 samples do not match processed rows in the manifest "
            f"({len(seen_samples)} groups vs {len(successful)} processed rows)"
        )

    schema_path = Path(__file__).with_name("schema.json")
    splits = _assign_splits(successful, config)
    _write_parquet(output / "manifest.parquet", final_manifest)
    _write_parquet(output / "splits.parquet", splits)
    _json_write(output / "config_used.json", config)
    _json_write(output / "schema.json", _schema_for_config(config, schema_path))
    report = {
        "dataset_name": config["dataset_name"],
        "dataset_version": config["dataset_version"],
        "source_root": str(source_root.resolve()),
        "output_root": str(output.resolve()),
        "build_fingerprint": fingerprint,
        "manifest_count": len(final_manifest),
        "initially_ready_count": initially_ready_count,
        "processed_count": len(successful),
        "failed_count": len(final_manifest) - len(successful),
        "shard_count": shard_count,
        "record_count": record_count,
        "shard_size": None,
        "split_counts": {name: sum(row["split"] == name for row in splits)
                         for name in ("train", "validation", "test")},
        "failures": failures,
        "array_tasks": task_count,
        "samples_per_task": samples_per_task,
    }
    _json_write(output / "build_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--chunks-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("dataset_config.json"))
    parser.add_argument("--samples-per-task", type=int, required=True)
    args = parser.parse_args()
    try:
        report = merge(
            args.manifest.expanduser().resolve(),
            args.chunks_root.expanduser().resolve(),
            args.output.expanduser().resolve(),
            args.config.expanduser().resolve(),
            args.source_root.expanduser().resolve(),
            args.samples_per_task,
        )
    except Exception as exc:
        parser.exit(2, f"ERROR: {type(exc).__name__}: {exc}\n")
    print(json.dumps(report, indent=2))
    # Per-sample failures are summarized in the report; a structurally complete
    # merge is still a successful merge and should proceed to dataset validation.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
