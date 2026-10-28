#!/usr/bin/env python3
"""Generate immutable per-run YAML configs and a CSV submission manifest."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path(__file__).with_name("sweep_spec.yaml"))
    parser.add_argument("--config-root", type=Path, required=True,
                        help="Config directory outside the repository")
    parser.add_argument("--runs-root", type=Path, required=True,
                        help="Run directory root outside the repository")
    parser.add_argument("--manifest", type=Path,
                        help="Manifest path (defaults to config-root/manifest.csv)")
    args = parser.parse_args()

    spec = yaml.safe_load(args.spec.read_text())
    args.config_root.mkdir(parents=True, exist_ok=True)
    args.runs_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.config_root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in spec["configurations"]:
        for seed in config.get("seeds", spec["seeds"]):
            run_id = f"{config['id']}_s{seed}"
            run_dir = args.runs_root / run_id
            config_path = args.config_root / f"{run_id}.yaml"
            model = {
                "type": config["type"],
                "hidden_channels": int(config["hidden_channels"]),
                "message_layers": int(config["message_layers"]),
                **spec["model_defaults"],
            }
            payload = {
                "name": f"{spec['name']}-{run_id}",
                "seed": int(seed),
                "dataset_root": spec["dataset_root"],
                "run_root": str(run_dir),
                "model": model,
                "training": spec["training"],
            }
            config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
            rows.append({
                "run_id": run_id,
                "configuration": config["id"],
                "model_type": config["type"],
                "hidden_channels": config["hidden_channels"],
                "message_layers": config["message_layers"],
                "seed": seed,
                "config_path": str(config_path.resolve()),
                "run_dir": str(run_dir.resolve()),
                "dataset_root": spec["dataset_root"],
            })
    with manifest_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} run configs and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
