"""Write the nine predeclared validation-only confirmation jobs externally."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=Path(__file__).with_name("sweep_spec.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    spec = yaml.safe_load(args.spec.read_text())
    run_root = Path(spec["run_root"])
    dataset_root = Path(spec["dataset_root"])
    split_path = dataset_root / "splits.parquet"
    if not split_path.is_file():
        raise FileNotFoundError(f"Existing family split table not found: {split_path}")
    actual_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if actual_hash != spec["split_sha256"]:
        raise ValueError(f"Existing family split SHA256 mismatch: {actual_hash}")
    build_report = json.loads((dataset_root / "build_report.json").read_text())
    fingerprint = build_report["build_fingerprint"]
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for configuration in spec["configurations"]:
        for seed in spec["seeds"]:
            rows.append({
                "run_id": f"{configuration['id']}-seed{seed}",
                "configuration": configuration["id"],
                "configuration_id": configuration["id"],
                "seed": int(seed),
                "run_dir": str(run_root / f"{configuration['id']}-seed{seed}"),
                "dataset_path": str(dataset_root),
                "build_fingerprint": fingerprint,
                "split_path": str(split_path),
                "split_sha256": actual_hash,
                "grouping_definition": "existing family_id split; no family overlaps splits",
            })
    if len(rows) != 9 or len({row["run_dir"] for row in rows}) != 9:
        raise ValueError(f"Expected exactly nine unique run directories, got {len(rows)}")
    with args.manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
