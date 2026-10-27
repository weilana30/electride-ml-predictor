"""Evaluate frozen feature-study checkpoints on the sealed test split once.

The CSV manifest is the precommitted evaluation list. This command never fits
normalization or model weights and writes a single combined metrics file and
prediction table for all listed checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from torch_geometric.loader import DataLoader

from elf_void_gnn.data import ShardedGraphDataset, read_split_table
from elf_void_gnn.feature_study import FEATURE_VARIANTS
from elf_void_gnn.models import build_model
from elf_void_gnn.train import (
    FeatureVariantDataset,
    evaluate,
    evaluate_constant,
    seed_everything,
)


REQUIRED_COLUMNS = {"variant", "seed", "run_dir"}
EXPECTED_SEEDS = {20261023, 20261024, 20261025}
EXPECTED_SIGNATURE = {
    "type": "hetero_mpnn",
    "hidden_channels": 64,
    "message_layers": 2,
    "radial_channels": 8,
    "atom_input_width": 14,
    "void_input_width": 9,
    "edge_width": 8,
    "parameter_count": 80001,
}
EXPECTED_MODEL_CONFIG = {
    "hidden_channels": 64,
    "message_layers": 2,
    "radial_channels": 8,
    "dropout": 0.1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(
    path: Path,
    dataset_root: Path,
    split_rows: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"Manifest must contain columns {sorted(REQUIRED_COLUMNS)}")
        rows = list(reader)
    if len(rows) != 9:
        raise ValueError("Final evaluation requires 3 variants × 3 fixed seeds")
    seen: set[tuple[str, int]] = set()
    prepared = []
    for row in rows:
        variant = row["variant"].strip()
        if variant not in FEATURE_VARIANTS:
            raise ValueError(f"Unknown feature variant in manifest: {variant!r}")
        try:
            seed = int(row["seed"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid seed for {variant}: {row['seed']!r}") from exc
        key = (variant, seed)
        if key in seen:
            raise ValueError(f"Duplicate variant/seed entry in manifest: {key}")
        seen.add(key)
        run_path = Path(row["run_dir"].strip()).expanduser()
        if not run_path.is_absolute():
            raise ValueError(f"Manifest run_dir must be absolute: {row['run_dir']!r}")
        run_dir = run_path.resolve()
        if not run_dir.is_dir():
            raise ValueError(f"Run directory does not exist: {run_dir}")
        metrics_path = run_dir / "metrics.json"
        checkpoint_path = run_dir / "best_model.pt"
        config_path = run_dir / "config.yaml"
        if not all(path.is_file() for path in (metrics_path, checkpoint_path, config_path)):
            raise ValueError(f"Run lacks metrics.json, best_model.pt, or config.yaml: {run_dir}")
        metrics = json.loads(metrics_path.read_text())
        config = yaml.safe_load(config_path.read_text())
        if metrics.get("status") != "completed":
            raise ValueError(f"Source run is not completed: {run_dir}")
        if metrics.get("evaluation_mode") != "validation-only" or metrics.get("test") is not None:
            raise ValueError(f"Source run must be validation-only with no test metrics: {run_dir}")
        if metrics.get("feature_variant") != variant or metrics.get("seed") != seed:
            raise ValueError(f"Manifest variant/seed disagrees with source metrics: {run_dir}")
        if Path(config["dataset_root"]).resolve() != dataset_root.resolve():
            raise ValueError(f"Source run dataset root differs from requested dataset: {run_dir}")
        model_config = {
            key: value for key, value in config["model"].items() if key != "type"
        }
        if config["model"].get("type") != "hetero_mpnn" or model_config != EXPECTED_MODEL_CONFIG:
            raise ValueError(f"Source run model config differs from the fixed HeteroMPNN: {run_dir}")
        signature = metrics.get("architecture_signature", {})
        if signature != EXPECTED_SIGNATURE:
            raise ValueError(f"Source checkpoint does not use the fixed HeteroMPNN: {run_dir}")
        expected_counts = {
            name: {
                "samples": len(rows_for_split),
                "families": len({row["family_id"] for row in rows_for_split}),
            }
            for name, rows_for_split in split_rows.items()
        }
        if metrics.get("split_counts") != expected_counts:
            raise ValueError(f"Source run split counts differ from current split table: {run_dir}")
        expected_validation_families = {
            row["family_id"] for row in split_rows["validation"]
        }
        actual_validation_families = set(
            metrics.get("validation", {}).get("per_family", {})
        )
        if actual_validation_families != expected_validation_families:
            raise ValueError(f"Source run validation families differ from current split table: {run_dir}")
        prepared.append({
            "variant": variant,
            "seed": seed,
            "run_dir": run_dir,
            "metrics": metrics,
            "config": config,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": sha256_file(checkpoint_path),
        })
    variants = {entry["variant"] for entry in prepared}
    if not {"baseline", "atomic_number_nearest_void_distances_edge_distances"}.issubset(variants):
        raise ValueError("Frozen test manifest must include baseline and required minimal variant")
    if len(variants) != 3:
        raise ValueError("Frozen test manifest must contain exactly three feature variants")
    for variant in variants:
        seeds = {entry["seed"] for entry in prepared if entry["variant"] == variant}
        if seeds != EXPECTED_SEEDS:
            raise ValueError(
                f"Frozen test manifest has incorrect seeds for {variant}: {sorted(seeds)}"
            )
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Frozen CSV with variant, seed, run_dir columns")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Final test output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    split_rows = read_split_table(dataset_root)
    test_families = {row["family_id"] for row in split_rows["test"]}
    if len(split_rows["test"]) != 5000 or len(test_families) != 9:
        raise ValueError(
            "Expected the frozen dataset's 5,000 test samples across 9 families; "
            f"found {len(split_rows['test'])} samples across {len(test_families)} families"
        )
    checkpoints = read_manifest(args.manifest, dataset_root, split_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("A ROCm GPU is required for final evaluation")

    raw_test = ShardedGraphDataset(dataset_root, split_rows["test"])
    loader_options = {
        "batch_size": int(checkpoints[0]["config"]["training"]["batch_size"]),
        "num_workers": int(checkpoints[0]["config"]["training"]["num_workers"]),
        "persistent_workers": int(checkpoints[0]["config"]["training"]["num_workers"]) > 0,
    }
    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for entry in checkpoints:
        model_config = entry["config"]["model"]
        if model_config.get("type") != "hetero_mpnn":
            raise ValueError(f"Unsupported model type in {entry['run_dir']}")
        model = build_model("hetero_mpnn", {
            key: value for key, value in model_config.items() if key != "type"
        }).to(device)
        if sum(parameter.numel() for parameter in model.parameters()) != 80001:
            raise ValueError(f"Unexpected model size in {entry['run_dir']}")
        checkpoint = torch.load(entry["checkpoint_path"], map_location=device, weights_only=False)
        if checkpoint.get("model_type") != "hetero_mpnn":
            raise ValueError(f"Checkpoint architecture mismatch: {entry['run_dir']}")
        if checkpoint.get("seed") != entry["seed"]:
            raise ValueError(f"Checkpoint seed differs from manifest: {entry['run_dir']}")
        if checkpoint.get("model_config") != {
            key: value for key, value in model_config.items() if key != "type"
        }:
            raise ValueError(f"Checkpoint model config differs from saved config: {entry['run_dir']}")
        model.load_state_dict(checkpoint["model_state"])
        model.set_training_statistics(checkpoint["training_statistics"])
        dataset = FeatureVariantDataset(raw_test, entry["variant"])
        test_loader = DataLoader(dataset, shuffle=False, **loader_options)
        seed_everything(entry["seed"])
        metrics, predictions = evaluate(model, test_loader, device, collect_predictions=True)
        baseline = evaluate_constant(
            test_loader,
            float(checkpoint["training_statistics"]["target_mean"]),
            device,
        )
        for prediction in predictions:
            prediction["variant"] = entry["variant"]
            prediction["seed"] = entry["seed"]
        all_predictions.extend(predictions)
        all_metrics.append({
            "variant": entry["variant"],
            "seed": entry["seed"],
            "source_run_dir": str(entry["run_dir"]),
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "best_epoch": checkpoint.get("best_epoch"),
            "validation_rmse_selected_before_test": checkpoint.get("validation_rmse"),
            "test": metrics,
            "training_mean_baseline": baseline,
        })
        del model, checkpoint, dataset, test_loader

    result = {
        "status": "completed",
        "evaluation_scope": "Frozen checkpoints evaluated once on the existing held-out test split; no training or model selection performed.",
        "dataset_root": str(dataset_root),
        "splits_sha256": sha256_file(dataset_root / "splits.parquet"),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "test_split_counts": {
            "samples": len(split_rows["test"]),
            "families": len({row["family_id"] for row in split_rows["test"]}),
        },
        "checkpoints": all_metrics,
    }
    (out_dir / "final_test_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pq.write_table(pa.Table.from_pylist(all_predictions), out_dir / "test_predictions.parquet")
    print(f"evaluated {len(all_metrics)} frozen checkpoints on {len(split_rows['test'])} test structures")
    print(f"outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
