"""Train one predeclared confirmation-study model/seed run."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from torch_geometric.loader import DataLoader

from elf_void_gnn.data import (
    ShardShuffleSampler,
    ShardedGraphDataset,
    read_split_table,
)
from elf_void_gnn.metrics import evaluation_metrics
from elf_void_gnn.models import build_model
from elf_void_gnn.feature_study import FEATURE_VARIANTS, apply_feature_variant


SPLIT_NOTE = "Evaluation uses the existing validated family-safe dataset split."


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"elf_void_gnn.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def batch_metadata(value: Any, count: int, label: str) -> list[Any]:
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    if len(values) != count:
        raise ValueError(f"Batch {label} has {len(values)} entries for {count} graphs")
    return values


def graph_mean_squared_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    graph_index: torch.Tensor,
    graph_count: int,
    target_std: torch.Tensor,
) -> torch.Tensor | None:
    if targets.numel() == 0:
        return None
    squared = torch.square((predictions - targets) / target_std)
    sums = squared.new_zeros(graph_count)
    counts = squared.new_zeros(graph_count)
    sums.index_add_(0, graph_index, squared)
    counts.index_add_(0, graph_index, torch.ones_like(squared))
    valid = counts > 0
    if not torch.any(valid):
        return None
    return torch.mean(sums[valid] / counts[valid])


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    collect_predictions: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    actual: list[float] = []
    predicted: list[float] = []
    families: list[str] = []
    output_rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = batch.to(device)
        prediction = model(batch).detach().cpu().numpy()
        target = batch["void"].y.detach().cpu().numpy()
        void_ptr = batch["void"].ptr.detach().cpu().numpy()
        graph_count = int(batch.num_graphs)
        sample_ids = batch_metadata(batch.sample_id, graph_count, "sample_id")
        family_ids = batch_metadata(batch.family_id, graph_count, "family_id")
        void_id_groups = batch_metadata(batch["void"].void_id, graph_count, "void_id")
        for graph_index in range(graph_count):
            first, last = int(void_ptr[graph_index]), int(void_ptr[graph_index + 1])
            y_values = target[first:last]
            p_values = prediction[first:last]
            ids = list(void_id_groups[graph_index])
            if len(ids) != len(y_values):
                raise ValueError(f"Void ID count mismatch for {sample_ids[graph_index]}")
            actual.extend(float(value) for value in y_values)
            predicted.extend(float(value) for value in p_values)
            families.extend([str(family_ids[graph_index])] * len(y_values))
            if collect_predictions:
                output_rows.extend(
                    {
                        "sample_id": str(sample_ids[graph_index]),
                        "family_id": str(family_ids[graph_index]),
                        "void_id": str(void_id),
                        "y_true": float(y_value),
                        "y_pred": float(p_value),
                    }
                    for void_id, y_value, p_value in zip(ids, y_values, p_values)
                )
    return evaluation_metrics(actual, predicted, families), output_rows


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


class FeatureVariantDataset(torch.utils.data.Dataset):
    """Apply a fixed-shape feature adapter before PyG batches graphs."""
    def __init__(self, dataset, variant: str):
        self.dataset = dataset
        self.variant = variant

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        graph = self.dataset[index]
        topology = {relation: graph[relation].edge_index for relation in graph.edge_types}
        graph = apply_feature_variant(graph, self.variant, copy_graph=False)
        if graph["atom"].x.shape[1] != 14 or graph["void"].x.shape[1] != 9:
            raise RuntimeError(f"{self.variant} changed the fixed 14/9 node feature widths")
        for relation in graph.edge_types:
            store = graph[relation]
            if store.edge_index is not topology[relation]:
                raise RuntimeError(f"{self.variant} changed graph topology for {relation}")
            if store.edge_index.ndim != 2 or store.edge_index.shape[0] != 2:
                raise RuntimeError(f"{self.variant} changed graph topology for {relation}")
            edge_features = getattr(store, "edge_features", None)
            if edge_features is not None and edge_features.shape[-1] != 8:
                raise RuntimeError(f"{self.variant} must retain eight edge channels for {relation}")
        return graph


def transformed_training_statistics(dataset) -> dict[str, Any]:
    """Fit normalization moments on adapted training graphs only."""
    atom_sum = np.zeros(5, dtype=np.float64)
    atom_sq = np.zeros(5, dtype=np.float64)
    void_sum = np.zeros(9, dtype=np.float64)
    void_sq = np.zeros(9, dtype=np.float64)
    atom_count = void_count = target_count = 0
    target_sum = target_sq = 0.0
    for index in range(len(dataset)):
        graph = dataset[index]
        atoms = graph["atom"].x.detach().cpu().numpy().astype(np.float64)[:, :5]
        voids = graph["void"].x.detach().cpu().numpy().astype(np.float64)
        targets = graph["void"].y.detach().cpu().numpy().astype(np.float64)
        atom_sum += atoms.sum(axis=0)
        atom_sq += np.square(atoms).sum(axis=0)
        atom_count += len(atoms)
        if len(voids):
            void_sum += voids.sum(axis=0)
            void_sq += np.square(voids).sum(axis=0)
            void_count += len(voids)
            target_sum += float(targets.sum())
            target_sq += float(np.square(targets).sum())
            target_count += len(targets)
    if not atom_count or not void_count or not target_count:
        raise ValueError("Training split has no atom, void, or ELF target values")

    def moments(total, square_total, count):
        mean = total / count
        variance = np.maximum(square_total / count - np.square(mean), 1e-12)
        return mean.astype(np.float32).tolist(), np.sqrt(variance).astype(np.float32).tolist()
    atom_mean, atom_std = moments(atom_sum, atom_sq, atom_count)
    void_mean, void_std = moments(void_sum, void_sq, void_count)
    target_mean = target_sum / target_count
    target_std = float(np.sqrt(max(target_sq / target_count - target_mean**2, 1e-12)))
    return {"atom_mean": atom_mean, "atom_std": atom_std,
            "void_mean": void_mean, "void_std": void_std,
            "target_mean": float(target_mean), "target_std": target_std,
            "atom_count": atom_count, "void_count": void_count}


def build_split_datasets(dataset_root: Path, split_rows: dict[str, list[dict[str, Any]]], evaluation_mode: str):
    """Open only splits allowed by the selected evaluation mode."""
    allowed = {"train", "validation"} if evaluation_mode == "validation-only" else {"test"}
    return {
        split: ShardedGraphDataset(dataset_root, rows)
        for split, rows in split_rows.items()
        if split in allowed
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "sweep_spec.yaml")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--feature-variant", choices=sorted(FEATURE_VARIANTS))
    parser.add_argument("--evaluation-mode", choices=("validation-only", "final-test"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--configuration-id")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    args.evaluation_mode = args.evaluation_mode or config.get("default_evaluation_mode", "validation-only")
    if args.evaluation_mode not in {"validation-only", "final-test"}:
        parser.error("evaluation mode must be validation-only or final-test")
    if "configurations" in config:
        if not args.configuration_id:
            parser.error("--configuration-id is required when --config is a sweep specification")
        matches = [item for item in config["configurations"] if item["id"] == args.configuration_id]
        if len(matches) != 1:
            parser.error(f"Unknown or duplicate configuration id: {args.configuration_id}")
        selected = matches[0]
        config = {
            "dataset_root": config["dataset_root"], "run_root": config["run_root"],
            "split_sha256": config["split_sha256"], "seed": config["seeds"][0],
            "configuration_id": selected["id"], "feature_variant": selected["feature_variant"],
            "training": config["training"],
            "model": {**config["model_defaults"], "hidden_channels": selected["hidden_channels"]},
            "expected_parameters": selected["expected_parameters"],
        }
    elif args.configuration_id:
        parser.error("--configuration-id is only valid with a sweep specification")
    dataset_root = args.dataset_root or Path(config["dataset_root"])
    run_dir = args.run_dir or (Path(config["run_root"]) / f"{config.get('configuration_id', 'run')}-seed{args.seed or config['seed']}")
    final_test = args.evaluation_mode == "final-test"
    if final_test:
        if not run_dir.is_dir() or not (run_dir / "best_model.pt").is_file():
            raise FileNotFoundError("final-test requires an existing validation-selected run and best_model.pt")
        if (run_dir / "test_predictions.parquet").exists():
            raise FileExistsError("Test evaluation has already been written for this run")
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Run directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    logger = make_logger(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("A ROCm GPU is required for this run but is not available")
    seed = args.seed if args.seed is not None else int(config["seed"])
    seed_everything(seed)
    logger.info("device=%s torch=%s hip=%s", device, torch.__version__, torch.version.hip)
    if device.type == "cuda":
        logger.info("gpu=%s", torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats(device)

    split_path = dataset_root / "splits.parquet"
    split_sha256 = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if split_sha256 != config.get("split_sha256"):
        raise ValueError(f"Existing dataset split SHA256 mismatch: {split_sha256}")
    split_rows = read_split_table(dataset_root)
    if final_test:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError("final-test requires validation-only metrics.json")
        existing_metrics = json.loads(metrics_path.read_text())
        selection_path = run_dir / "validation_selection_frozen.json"
        if not selection_path.is_file():
            raise FileNotFoundError("final-test requires the validation selection freeze marker")
        selection = json.loads(selection_path.read_text())
        if (selection.get("status") != "frozen"
                or selection.get("selection_rule") != "mean pooled validation RMSE across three predeclared seeds"
                or selection.get("split_sha256") != split_sha256
                or selection.get("seed") != seed
                or selection.get("configuration_id") != config.get("configuration_id")
                or selection.get("selected_configuration") not in {
                    "w128_d2_baseline", "w128_d2_atom_only", "w64_d2_atom_only"
                }):
            raise ValueError("validation selection marker does not match the frozen study")
        if existing_metrics.get("evaluation_mode") != "validation-only" or existing_metrics.get("test") is not None:
            raise ValueError("final-test requires an unevaluated validation-only run")
        checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
        if checkpoint.get("evaluation_scope") != "validation-only-selected-checkpoint":
            raise ValueError("Checkpoint lacks validation-only selection provenance")
        expected = {
            "seed": int(args.seed if args.seed is not None else config["seed"]),
            "split_sha256": split_sha256,
            "configuration_id": config.get("configuration_id"),
            "feature_variant": args.feature_variant or config.get("feature_variant", "baseline"),
            "model_config": {key: value for key, value in config["model"].items() if key != "type"},
            "training_config": config["training"],
        }
        if checkpoint.get("model_type") != config["model"]["type"]:
            raise ValueError("Frozen model type does not match the validation checkpoint")
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"Frozen {key} mismatch: checkpoint={checkpoint.get(key)!r}, requested={value!r}")
        if existing_metrics.get("seed") != expected["seed"] or existing_metrics.get("configuration_id") != expected["configuration_id"]:
            raise ValueError("Validation metrics do not match the requested seed/configuration")
        if existing_metrics.get("best_epoch") != checkpoint.get("best_epoch") or existing_metrics.get("best_validation_rmse") != checkpoint.get("validation_rmse"):
            raise ValueError("Checkpoint does not match the validation-selected metrics")
        raw_datasets = {"test": ShardedGraphDataset(dataset_root, split_rows["test"])}
        variant = expected["feature_variant"]
        datasets = {split: FeatureVariantDataset(dataset, variant)
                    for split, dataset in raw_datasets.items()}
        model = build_model(checkpoint["model_type"], checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.set_training_statistics(checkpoint["training_statistics"])
        loader_options = {"batch_size": int(config["training"]["batch_size"]), "num_workers": int(config["training"]["num_workers"])}
        test_loader = DataLoader(datasets["test"], shuffle=False, persistent_workers=loader_options["num_workers"] > 0, **loader_options)
        test_metrics, predictions = evaluate(model, test_loader, device, collect_predictions=True)
        if not predictions or len({(row["sample_id"], row["void_id"]) for row in predictions}) != len(predictions):
            raise ValueError("Test predictions are empty or contain duplicate sample/void IDs")
        expected_samples = {row["sample_id"] for row in split_rows["test"]}
        if {row["sample_id"] for row in predictions} != expected_samples:
            raise ValueError("Test prediction sample IDs do not match the frozen test split")
        prediction_path = run_dir / "test_predictions.parquet"
        temporary_predictions = prediction_path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(predictions), temporary_predictions)
        os.replace(temporary_predictions, prediction_path)
        existing_metrics["test"] = test_metrics
        existing_metrics["test_evaluation"] = {"status": "completed", "split_sha256": split_sha256, "prediction_count": len(predictions)}
        atomic_json(metrics_path, existing_metrics)
        logger.info("final test evaluated once: rmse=%.6f predictions=%d", test_metrics["pooled"]["rmse"], len(predictions))
        return 0

    raw_datasets = build_split_datasets(dataset_root, split_rows, "validation-only")
    variant = args.feature_variant or config.get("feature_variant", "baseline")
    datasets = {split: FeatureVariantDataset(dataset, variant)
                for split, dataset in raw_datasets.items()}
    logger.info(
        "splits samples=%s families=%s",
        {name: len(rows) for name, rows in split_rows.items()},
        {
            name: len({row["family_id"] for row in rows})
            for name, rows in split_rows.items()
        },
    )
    stats = transformed_training_statistics(datasets["train"])
    model = build_model(config["model"]["type"], {
        key: value for key, value in config["model"].items() if key != "type"
    }).to(device)
    if config["model"]["type"] != "hetero_mpnn":
        raise ValueError("Feature study requires the unchanged hetero_mpnn architecture")
    if int(config["model"].get("radial_channels", 8)) != 8:
        raise ValueError("Feature study requires the fixed eight-channel edge representation")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_parameters = int(config.get("expected_parameters", {64: 80001, 128: 307457}.get(int(config["model"].get("hidden_channels", 64)))))
    if expected_parameters is None or parameter_count != expected_parameters:
        raise ValueError(f"Unexpected HeteroMPNN parameter count: {parameter_count} (expected {expected_parameters})")
    model.set_training_statistics(stats)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    train_sampler = ShardShuffleSampler(raw_datasets["train"], seed)
    num_workers = int(config["training"]["num_workers"])
    loader_options = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(datasets["train"], sampler=train_sampler, **loader_options)
    validation_loader = DataLoader(datasets["validation"], shuffle=False, **loader_options)
    test_loader = None

    history = []
    best_rmse = float("inf")
    best_epoch = 0
    patience = int(config["training"]["early_stopping_patience"])
    stale_epochs = 0
    start_training = time.perf_counter()
    total_train_graphs = 0
    total_train_voids = 0
    for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
        epoch_start = time.perf_counter()
        train_sampler.set_epoch(epoch - 1)
        model.train()
        loss_sum = 0.0
        batch_count = 0
        graph_count_seen = 0
        void_count_seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            prediction = model(batch)
            target = batch["void"].y
            graph_index = batch["void"].batch
            loss = graph_mean_squared_error(
                prediction,
                target,
                graph_index,
                int(batch.num_graphs),
                model.target_std,
            )
            if loss is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batch_count += 1
            graph_count_seen += int(batch.num_graphs)
            void_count_seen += int(target.numel())

        if batch_count == 0:
            raise RuntimeError("Training produced no batches with void targets")
        val_metrics, _ = evaluate(
            model, validation_loader, device, collect_predictions=False
        )
        val_rmse = float(val_metrics["pooled"]["rmse"])
        epoch_seconds = time.perf_counter() - epoch_start
        total_train_graphs += graph_count_seen
        total_train_voids += void_count_seen
        history.append({
            "epoch": epoch,
            "train_normalized_mse": loss_sum / batch_count,
            "validation_rmse": val_rmse,
            "seconds": epoch_seconds,
        })
        logger.info(
            "epoch=%d train_mse=%.6f val_rmse=%.6f seconds=%.1f",
            epoch, loss_sum / batch_count, val_rmse, epoch_seconds,
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch
            stale_epochs = 0
            checkpoint = {
                "model_type": config["model"]["type"],
                "model_config": {
                    key: value for key, value in config["model"].items() if key != "type"
                },
                "model_state": model.state_dict(),
                "training_statistics": stats,
                "seed": seed,
                "best_epoch": best_epoch,
                "validation_rmse": best_rmse,
                "evaluation_scope": "validation-only-selected-checkpoint",
                "split_sha256": split_sha256,
                "configuration_id": config.get("configuration_id"),
                "feature_variant": variant,
                "training_config": config["training"],
                "model_config": {key: value for key, value in config["model"].items() if key != "type"},
            }
            temporary = run_dir / "best_model.pt.tmp"
            torch.save(checkpoint, temporary)
            os.replace(temporary, run_dir / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.info("early stopping after %d stale epochs", stale_epochs)
                break

    training_seconds = time.perf_counter() - start_training
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation_metrics, validation_predictions = evaluate(
        model, validation_loader, device, collect_predictions=True
    )
    test_metrics, predictions = None, []
    training_mean = float(stats["target_mean"])
    train_mean_baseline = {
        "validation": evaluate_constant(validation_loader, training_mean, device),
        "test": evaluate_constant(test_loader, training_mean, device) if test_loader is not None else None,
    }
    pq.write_table(pa.Table.from_pylist(validation_predictions), run_dir / "validation_predictions.parquet")

    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda" else None
    )
    metrics = {
        "status": "completed",
        "task": "per-void ELF at center regression",
        "evaluation_scope": SPLIT_NOTE,
        "split_counts": {
            name: {
                "samples": len(rows),
                "families": len({row["family_id"] for row in rows}),
            }
            for name, rows in split_rows.items()
        },
        "seed": seed,
        "configuration_id": config.get("configuration_id", config["model"].get("id")),
        "feature_variant": variant,
        "evaluation_mode": args.evaluation_mode,
        "best_epoch": best_epoch,
        "best_validation_rmse": best_rmse,
        "model": config["model"],
        "architecture_signature": {
            "type": "hetero_mpnn",
            "hidden_channels": int(config["model"].get("hidden_channels", 64)),
            "message_layers": int(config["model"].get("message_layers", 2)),
            "radial_channels": int(config["model"].get("radial_channels", 8)),
            "atom_input_width": 14,
            "void_input_width": 9,
            "edge_width": 8,
            "parameter_count": parameter_count,
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "training_mean_baseline": train_mean_baseline,
        "performance": {
            "training_seconds": training_seconds,
            "train_graphs_seen": total_train_graphs,
            "train_voids_seen": total_train_voids,
            "graphs_per_second": total_train_graphs / max(training_seconds, 1e-9),
            "voids_per_second": total_train_voids / max(training_seconds, 1e-9),
            "peak_gpu_memory_bytes": peak_memory,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "history": history,
    }
    atomic_json(run_dir / "metrics.json", metrics)
    if test_metrics is not None:
        logger.info("test_rmse=%.6f test_mae=%.6f", test_metrics["pooled"]["rmse"], test_metrics["pooled"]["mae"])
    logger.info("artifacts=%s", run_dir)
    return 0


@torch.no_grad()
def evaluate_constant(loader: DataLoader, value: float, device: torch.device):
    actual: list[float] = []
    predicted: list[float] = []
    families: list[str] = []
    for batch in loader:
        target = batch["void"].y.detach().cpu().numpy()
        family_ids = batch_metadata(batch.family_id, int(batch.num_graphs), "family_id")
        void_ptr = batch["void"].ptr.detach().cpu().numpy()
        for graph_index, family_id in enumerate(family_ids):
            count = int(void_ptr[graph_index + 1] - void_ptr[graph_index])
            actual.extend(float(x) for x in target[void_ptr[graph_index]:void_ptr[graph_index + 1]])
            predicted.extend([value] * count)
            families.extend([str(family_id)] * count)
    return evaluation_metrics(actual, predicted, families)


if __name__ == "__main__":
    raise SystemExit(main())
