#!/usr/bin/env python3
"""Collect completed validation-only feature runs and make screening tables/figures.

This command deliberately refuses runs that evaluated test data. It validates
each run's saved dataset root and split counts against the caller-supplied split
file, records that file's fingerprint as a caller reference, and pairs variants
to same-seed baselines when available. Metrics do not contain an in-run split
fingerprint, so the saved path and counts cannot prove historical file identity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml


EXPECTED_ARCHITECTURE = {
    "type": "hetero_mpnn",
    "hidden_channels": 64,
    "message_layers": 2,
    "radial_channels": 8,
    "atom_input_width": 14,
    "void_input_width": 9,
    "edge_width": 8,
    "parameter_count": 80001,
}


def split_reference(split_path: Path) -> tuple[str, dict[str, dict[str, int]], dict[str, set[str]]]:
    """Read expected split counts/family IDs and fingerprint the caller's split file."""
    table = pq.read_table(split_path, columns=["sample_id", "family_id", "split"])
    rows = table.to_pylist()
    counts: dict[str, dict[str, int]] = {}
    families: dict[str, set[str]] = {}
    seen: set[str] = set()
    for row in rows:
        sample_id, family_id, split = map(str, (row["sample_id"], row["family_id"], row["split"]))
        if sample_id in seen:
            raise ValueError(f"duplicate sample ID in supplied split file: {sample_id}")
        seen.add(sample_id)
        counts.setdefault(split, {"samples": 0, "families": 0})["samples"] += 1
        families.setdefault(split, set()).add(family_id)
    for split, family_ids in families.items():
        counts[split]["families"] = len(family_ids)
    return hashlib.sha256(split_path.read_bytes()).hexdigest(), counts, families


def validation_exclusion_reason(
    run_dir: Path,
    metrics: dict,
    split_path: Path,
    expected_counts: dict[str, dict[str, int]],
    expected_families_by_split: dict[str, set[str]],
    expected_families: int,
) -> str | None:
    """Return why a run cannot enter a validation screen, or None if eligible."""
    if metrics.get("status") != "completed":
        return "run is not completed"
    if metrics.get("evaluation_mode") != "validation-only":
        return "not validation-only (test evaluation is prohibited in screening)"
    if metrics.get("test") is not None:
        return "contains test metrics"
    variant, seed = str(metrics.get("feature_variant", "")), str(metrics.get("seed", ""))
    if not variant or not seed:
        return "missing feature_variant or seed"

    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        return "missing saved config.yaml"
    try:
        config = yaml.safe_load(config_path.read_text())
        configured_root = Path(config["dataset_root"])
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        return f"invalid saved config.yaml: {exc}"
    if not configured_root.is_absolute():
        return "config.yaml dataset_root is not absolute; dataset provenance cannot be verified"
    if configured_root.resolve() != split_path.parent.resolve():
        return "config.yaml dataset_root does not match supplied splits.parquet parent"

    if metrics.get("split_counts") != expected_counts:
        return "metrics split_counts do not match supplied splits.parquet"
    validation = metrics.get("validation")
    if not isinstance(validation, dict) or not isinstance(validation.get("per_family"), dict):
        return "missing validation per-family metrics"
    family_metrics = validation["per_family"]
    expected_validation_families = expected_families_by_split.get("validation", set())
    if set(family_metrics) != expected_validation_families:
        return "validation family IDs do not match supplied splits.parquet"
    if len(family_metrics) != expected_families:
        return f"expected {expected_families} validation families"

    signature = metrics.get("architecture_signature")
    if not isinstance(signature, dict) or any(
        signature.get(key) != value for key, value in EXPECTED_ARCHITECTURE.items()
    ):
        return "architecture_signature differs from fixed HeteroMPNN architecture"
    model = config.get("model")
    expected_config = {
        "type": "hetero_mpnn",
        "hidden_channels": 64,
        "message_layers": 2,
        "radial_channels": 8,
        "dropout": 0.1,
    }
    if not isinstance(model, dict) or any(model.get(key) != value for key, value in expected_config.items()):
        return "config.yaml model differs from fixed HeteroMPNN architecture"
    return None

def feature_groups(variant: str) -> str:
    """Return coarse, stable groups used by paired summaries/interactions."""
    if variant == "baseline":
        return "baseline"
    groups: list[str] = []
    if variant.startswith("atom") or "atomic_number" in variant:
        groups.append("atom")
    if variant.startswith("void") or "void_" in variant or "void" in variant:
        groups.append("void")
    if "edge" in variant:
        groups.append("edge")
    if variant.startswith("no_"):
        groups.append("edge")
    return ";".join(dict.fromkeys(groups)) or "mixed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True,
                        help="The dataset's authoritative splits.parquet file")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-families", type=int, default=9)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    args = parser.parse_args()

    root, split_path = args.runs_root.resolve(), args.splits.resolve()
    if not root.is_dir():
        parser.error(f"runs root does not exist: {root}")
    if not split_path.is_file():
        parser.error(f"split file does not exist: {split_path}")
    try:
        split_id, expected_counts, expected_families_by_split = split_reference(split_path)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f"cannot validate supplied splits.parquet: {exc}")
    candidates = sorted(root.rglob("metrics.json"))
    if not candidates:
        parser.error(f"no metrics.json files found under {root}")

    records: list[dict[str, str]] = []
    excluded: list[tuple[str, str]] = []
    by_variant_seed: dict[tuple[str, str], str] = {}
    for metrics_path in candidates:
        run_dir = metrics_path.parent.resolve()
        try:
            run_id = run_dir.relative_to(root).as_posix()
        except ValueError:
            excluded.append((str(metrics_path), "outside runs root"))
            continue
        try:
            metrics = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            excluded.append((run_id, f"unreadable metrics: {exc}"))
            continue
        variant = str(metrics.get("feature_variant", ""))
        seed = str(metrics.get("seed", ""))
        reason = validation_exclusion_reason(
            run_dir, metrics, split_path, expected_counts,
            expected_families_by_split, args.expected_families,
        )
        if reason:
            excluded.append((run_id, reason))
            continue
        key = (variant, seed)
        if key in by_variant_seed:
            raise ValueError(f"duplicate completed run for variant={variant}, seed={seed}")
        by_variant_seed[key] = run_id
        records.append({
            "run_id": run_id,
            "variant": variant,
            "feature_groups": feature_groups(variant),
            "baseline": "",
            "split_id": split_id,
            "seed": seed,
        })

    if not records:
        parser.error("no completed validation-only runs passed screening checks")
    for row in records:
        if row["variant"] == "baseline":
            continue
        row["baseline"] = by_variant_seed.get(("baseline", row["seed"]), "")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "run_manifest.csv"
    columns = ("run_id", "variant", "feature_groups", "baseline", "split_id", "seed")
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    exclusions = args.out_dir / "excluded_runs.csv"
    with exclusions.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("run_id", "reason"))
        writer.writerows(excluded)

    command = [
        sys.executable, str(Path(__file__).with_name("aggregate_runs.py")),
        "--manifest", str(manifest), "--runs-root", str(root),
        "--out-dir", str(args.out_dir), "--split", "validation",
        "--expected-families", str(args.expected_families),
        "--bootstrap-draws", str(args.bootstrap_draws),
    ]
    subprocess.run(command, check=True)
    print(f"included {len(records)} validation-only runs; excluded {len(excluded)}")
    print(f"manifest: {manifest}")
    print(f"outputs: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
