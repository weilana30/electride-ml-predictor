#!/usr/bin/env python3
"""Fit one fixed HistGradientBoosting regression baseline on per-void descriptors."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError as exc:  # pragma: no cover - environment-specific message
    raise SystemExit(
        "scikit-learn is required; install elf_void_gnn/architecture_test/requirements.txt in the training environment"
    ) from exc

from elf_void_gnn.data import ShardedGraphDataset, read_split_table
from elf_void_gnn.metrics import evaluation_metrics


def features_for_graph(graph):
    """Create input-only rows: void features, local atom context, graph context."""
    void_x = graph["void"].x.detach().cpu().numpy().astype(np.float32)
    atom_x = graph["atom"].x.detach().cpu().numpy().astype(np.float32)
    relation = graph[("void", "to", "atom")]
    edge_index = relation.edge_index.detach().cpu().numpy()
    distance = relation.edge_distance.detach().cpu().numpy().reshape(-1).astype(np.float32)
    n_voids = len(void_x)
    local_mean = np.zeros((n_voids, atom_x.shape[1]), dtype=np.float32)
    distance_summary = np.zeros((n_voids, 3), dtype=np.float32)
    if edge_index.size:
        void_index, atom_index = edge_index
        weights = np.exp(-np.maximum(distance, 0.0) / 2.0).astype(np.float32)
        weight_sum = np.zeros(n_voids, dtype=np.float32)
        weighted_distance = np.zeros(n_voids, dtype=np.float32)
        weighted_distance_sq = np.zeros(n_voids, dtype=np.float32)
        np.add.at(local_mean, void_index, atom_x[atom_index] * weights[:, None])
        np.add.at(weight_sum, void_index, weights)
        np.add.at(weighted_distance, void_index, distance * weights)
        np.add.at(weighted_distance_sq, void_index, np.square(distance) * weights)
        denom = np.maximum(weight_sum, 1e-8)
        local_mean /= denom[:, None]
        d_mean = weighted_distance / denom
        d_var = np.maximum(weighted_distance_sq / denom - np.square(d_mean), 0.0)
        distance_summary[:, 0] = d_mean
        distance_summary[:, 1] = np.sqrt(d_var)
        distance_summary[:, 2] = np.log1p(weight_sum)
    # Exactly the fixed_context_mlp descriptor contract: void(9), weighted
    # neighboring-atom mean(14), weighted distance mean/std, log weight sum.
    return np.concatenate((void_x, local_mean, distance_summary), axis=1)


def extract_split(dataset_root: Path, rows: list[dict[str, str]]):
    dataset = ShardedGraphDataset(dataset_root, rows)
    x_parts, y_parts, family_ids, sample_ids, void_ids = [], [], [], [], []
    for index in range(len(dataset)):
        graph = dataset[index]
        features = features_for_graph(graph)
        targets = graph["void"].y.detach().cpu().numpy().astype(np.float32)
        ids = list(graph["void"].void_id)
        if len(features) != len(targets) or len(ids) != len(targets):
            raise ValueError(f"Feature/target/void ID mismatch for {graph.sample_id}")
        x_parts.append(features)
        y_parts.append(targets)
        family_ids.extend([str(graph.family_id)] * len(targets))
        sample_ids.extend([str(graph.sample_id)] * len(targets))
        void_ids.extend(str(value) for value in ids)
    if not x_parts:
        raise ValueError("Split has no samples")
    return (np.concatenate(x_parts), np.concatenate(y_parts), family_ids, sample_ids, void_ids)


def metrics(y, prediction, families):
    return evaluation_metrics(y.tolist(), prediction.tolist(), families)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    start = time.perf_counter()
    split_rows = read_split_table(args.dataset_root)
    extracted = {name: extract_split(args.dataset_root, rows) for name, rows in split_rows.items()}
    x_train, y_train, _, _, _ = extracted["train"]
    train_sample_ids = extracted["train"][3]
    counts = {}
    for sample_id in train_sample_ids:
        counts[sample_id] = counts.get(sample_id, 0) + 1
    # The shared GNN trainer averages each graph's void loss before averaging
    # graphs in the batch. Match that per-material target weighting here.
    sample_weight = np.asarray([1.0 / counts[sample_id] for sample_id in train_sample_ids])
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.08, max_iter=200,
        max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=False, random_state=args.seed,
    )
    fit_start = time.perf_counter()
    model.fit(x_train, y_train, sample_weight=sample_weight)
    training_seconds = time.perf_counter() - fit_start
    output_rows = []
    metrics_by_split = {}
    for split in ("validation", "test"):
        x, y, families, samples, voids = extracted[split]
        predictions = model.predict(x).astype(np.float64)
        metrics_by_split[split] = metrics(y, predictions, families)
        if split == "test":
            output_rows = [
                {"sample_id": sid, "family_id": fid, "void_id": vid,
                 "y_true": float(target), "y_pred": float(pred)}
                for sid, fid, vid, target, pred in zip(samples, families, voids, y, predictions)
            ]
    training_mean = float(np.mean(y_train))
    baseline = {}
    for split in ("validation", "test"):
        _, y, families, _, _ = extracted[split]
        baseline[split] = metrics(y, np.full_like(y, training_mean), families)
    pq.write_table(pa.Table.from_pylist(output_rows), args.run_dir / "test_predictions.parquet")
    payload = {
        "status": "completed",
        "task": "per-void ELF at center regression",
        "evaluation_scope": (
            "Existing exact named phase-folder split; related pressure or space-group variants may cross splits. "
            "Not an unseen-composition or unseen-pressure benchmark."
        ),
        "split_counts": {
            name: {"samples": len(rows), "families": len({row["family_id"] for row in rows})}
            for name, rows in split_rows.items()
        },
        "seed": args.seed,
        "best_iteration": int(model.n_iter_),
        "best_validation_rmse": metrics_by_split["validation"]["pooled"]["rmse"],
        "model": {
            "type": "tree_baseline", "estimator": "HistGradientBoostingRegressor",
            "loss": "squared_error", "learning_rate": 0.08, "max_iter": 200,
            "max_leaf_nodes": 31, "l2_regularization": 1.0,
            "early_stopping": False, "feature_count": int(x_train.shape[1]),
            "descriptor_schema": "void(9)+weighted_local_atom_mean(14)+weighted_distance_mean/std+log1p_weight_sum(3); 26 columns; exactly matches fixed_context_mlp input information",
            "training_sample_weight": "inverse number of voids in each training graph; matches per-graph mean loss",
        },
        "validation": metrics_by_split["validation"],
        "test": metrics_by_split["test"],
        "training_mean_baseline": baseline,
        "performance": {
            "training_seconds": training_seconds,
            "extraction_and_total_seconds": time.perf_counter() - start,
            "train_voids": int(len(y_train)),
            "parameter_count": None,
            "peak_gpu_memory_bytes": None,
            "device": "cpu",
        },
    }
    temporary = args.run_dir / "metrics.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, args.run_dir / "metrics.json")
    print(f"Completed tree baseline in {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
