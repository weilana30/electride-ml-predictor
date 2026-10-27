"""Train and evaluate the initial ELF-center heterogeneous GNN."""
from __future__ import annotations

import argparse
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
    fit_training_statistics,
    read_split_table,
)
from elf_void_gnn.metrics import evaluation_metrics
from elf_void_gnn.models import build_model


SPLIT_NOTE = (
    "Evaluation uses the existing family-safe split by exact named phase folder. "
    "Related pressure or space-group variants may occur across splits; this is "
    "not an unseen-composition or unseen-pressure benchmark."
)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/elf_center_v1.yaml")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    dataset_root = args.dataset_root or Path(config["dataset_root"])
    run_dir = args.run_dir or Path(config["run_root"])
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run_dir / "config.yaml")
    logger = make_logger(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("A ROCm GPU is required for this run but is not available")
    seed = int(config["seed"])
    seed_everything(seed)
    logger.info("device=%s torch=%s hip=%s", device, torch.__version__, torch.version.hip)
    if device.type == "cuda":
        logger.info("gpu=%s", torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats(device)

    split_rows = read_split_table(dataset_root)
    datasets = {
        split: ShardedGraphDataset(dataset_root, rows)
        for split, rows in split_rows.items()
    }
    logger.info(
        "splits samples=%s families=%s",
        {name: len(rows) for name, rows in split_rows.items()},
        {
            name: len({row["family_id"] for row in rows})
            for name, rows in split_rows.items()
        },
    )
    stats = fit_training_statistics(datasets["train"])
    model = build_model(config["model"]["type"], {
        key: value for key, value in config["model"].items() if key != "type"
    }).to(device)
    model.set_training_statistics(stats)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    train_sampler = ShardShuffleSampler(datasets["train"], seed)
    num_workers = int(config["training"]["num_workers"])
    loader_options = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(datasets["train"], sampler=train_sampler, **loader_options)
    validation_loader = DataLoader(datasets["validation"], shuffle=False, **loader_options)
    test_loader = DataLoader(datasets["test"], shuffle=False, **loader_options)

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
    validation_metrics, _ = evaluate(
        model, validation_loader, device, collect_predictions=False
    )
    test_metrics, predictions = evaluate(
        model, test_loader, device, collect_predictions=True
    )
    training_mean = float(stats["target_mean"])
    train_mean_baseline = {
        "validation": evaluate_constant(validation_loader, training_mean, device),
        "test": evaluate_constant(test_loader, training_mean, device),
    }
    pq.write_table(pa.Table.from_pylist(predictions), run_dir / "test_predictions.parquet")

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
        "best_epoch": best_epoch,
        "best_validation_rmse": best_rmse,
        "model": config["model"],
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
