"""Split-aware HDF5 loading and training-only feature statistics."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import HeteroData

from elf_void_analysis.dataset import load_graph_record_from_group


SPLITS = ("train", "validation", "test")


def read_split_table(dataset_root: str | Path) -> dict[str, list[dict[str, str]]]:
    root = Path(dataset_root)
    report = json.loads((root / "validation.json").read_text())
    if not report.get("valid"):
        raise ValueError("The source dataset validation report is not valid")
    manifest_rows = pq.read_table(root / "manifest.parquet", columns=["sample_id", "family_id", "processing_status"]).to_pylist()
    expected = {str(row["sample_id"]): str(row["family_id"]) for row in manifest_rows if row["processing_status"] == "processed"}
    rows = pq.read_table(root / "splits.parquet").to_pylist()
    result = {split: [] for split in SPLITS}
    seen: dict[str, str] = {}
    split_families: dict[str, str] = {}
    family_split: dict[str, str] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        family_id = str(row["family_id"])
        split = str(row["split"])
        if split not in result:
            raise ValueError(f"Unknown split {split!r} for {sample_id}")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample ID in split table: {sample_id}")
        if family_id in family_split and family_split[family_id] != split:
            raise ValueError(f"Family {family_id} crosses {family_split[family_id]} and {split}")
        seen[sample_id] = split
        split_families[sample_id] = family_id
        family_split[family_id] = split
        result[split].append({"sample_id": sample_id, "family_id": family_id})
    if set(seen) != set(expected):
        missing = set(expected) - set(seen)
        extra = set(seen) - set(expected)
        raise ValueError(f"Split/sample manifest mismatch: {len(missing)} missing, {len(extra)} extra")
    for sample_id, family_id in expected.items():
        if split_families[sample_id] != family_id:
            raise ValueError(f"Family mismatch between manifest and split table for {sample_id}")
    if len(seen) != int(report["hdf5_sample_count"]):
        raise ValueError(
            f"Split table has {len(seen)} samples, dataset has {report['hdf5_sample_count']}"
        )
    return result


def graph_from_record(record: dict[str, Any]) -> HeteroData:
    """Convert one record while exposing only structural features and one label."""
    atoms = record["atoms"]
    voids = record["voids"]
    edges = record["edges"]
    atom_x = np.asarray(atoms["features"], dtype=np.float32)
    validity = atoms.get("feature_validity")
    if validity is not None:
        atom_x = np.concatenate((atom_x, np.asarray(validity, dtype=np.float32)), axis=1)

    graph = HeteroData()
    graph["atom"].x = torch.as_tensor(atom_x.copy())
    graph["void"].x = torch.as_tensor(
        np.asarray(voids["features"], dtype=np.float32).copy()
    )
    graph["void"].y = torch.as_tensor(
        np.asarray(record["targets"]["elf_at_center"], dtype=np.float32).copy()
    )
    graph["void"].void_id = [str(value.decode() if isinstance(value, bytes) else value)
                             for value in voids["void_id"]]

    relation_data = (
        (("atom", "to", "atom"), "atom_atom"),
        (("void", "to", "atom"), "void_atom"),
        (("void", "to", "void"), "void_void"),
    )
    for relation, key in relation_data:
        store = graph[relation]
        store.edge_index = torch.as_tensor(
            np.asarray(edges[f"{key}_index"], dtype=np.int64).copy()
        )
        store.edge_distance = torch.as_tensor(
            np.asarray(edges[f"{key}_distance_angstrom"], dtype=np.float32).copy()
        )
    void_atom = graph["void", "to", "atom"]
    reverse = graph["atom", "to", "void"]
    reverse.edge_index = void_atom.edge_index.flip(0)
    reverse.edge_distance = void_atom.edge_distance.clone()

    graph.sample_id = str(record["sample_id"])
    graph.family_id = str(record["metadata"]["family_id"])
    return graph


class ShardedGraphDataset(Dataset):
    """Read-only view across shards with one process-local HDF5 handle at a time."""

    def __init__(
        self,
        dataset_root: str | Path,
        split_rows: list[dict[str, str]],
    ):
        self.root = Path(dataset_root)
        wanted = {row["sample_id"]: row["family_id"] for row in split_rows}
        if len(wanted) != len(split_rows):
            raise ValueError("Duplicate sample IDs in requested split")
        shard_by_id: dict[str, Path] = {}
        for shard_path in sorted((self.root / "shards").glob("graphs-*.h5")):
            with h5py.File(shard_path, "r") as handle:
                for sample_id in handle["samples"]:
                    if sample_id in shard_by_id:
                        raise ValueError(f"Sample {sample_id} occurs in multiple shards")
                    shard_by_id[sample_id] = shard_path
        missing = set(wanted) - set(shard_by_id)
        if missing:
            raise ValueError(f"{len(missing)} split samples are absent from HDF5 shards")
        self.items = [
            (shard_by_id[row["sample_id"]], row["sample_id"], row["family_id"])
            for row in split_rows
        ]
        self.indices_by_shard: dict[Path, list[int]] = defaultdict(list)
        for index, (shard_path, _, _) in enumerate(self.items):
            self.indices_by_shard[shard_path].append(index)
        self._handle: h5py.File | None = None
        self._handle_path: Path | None = None

    def __len__(self) -> int:
        return len(self.items)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_handle_path"] = None
        return state

    def _open_shard(self, path: Path) -> h5py.File:
        if self._handle_path != path:
            if self._handle is not None:
                self._handle.close()
            self._handle = h5py.File(path, "r")
            self._handle_path = path
        return self._handle

    def __getitem__(self, index: int) -> HeteroData:
        shard_path, sample_id, family_id = self.items[index]
        handle = self._open_shard(shard_path)
        record = load_graph_record_from_group(handle[f"samples/{sample_id}"], sample_id)
        graph = graph_from_record(record)
        if graph.family_id != family_id:
            raise ValueError(f"Family mismatch for {sample_id}")
        return graph

    def iter_indices_by_shard(self) -> Iterator[list[int]]:
        for shard_path in sorted(self.indices_by_shard):
            yield self.indices_by_shard[shard_path]


class ShardShuffleSampler(Sampler[int]):
    """Shuffle shard order and records within shards to keep HDF5 access local."""

    def __init__(self, dataset: ShardedGraphDataset, seed: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        shard_groups = list(self.dataset.indices_by_shard.values())
        rng.shuffle(shard_groups)
        for indices in shard_groups:
            shuffled = list(indices)
            rng.shuffle(shuffled)
            yield from shuffled

    def __len__(self) -> int:
        return len(self.dataset)


def fit_training_statistics(dataset: ShardedGraphDataset) -> dict[str, Any]:
    """Compute feature and target moments from training samples only."""
    atom_sum = np.zeros(5, dtype=np.float64)
    atom_sq = np.zeros(5, dtype=np.float64)
    atom_count = 0
    void_sum = np.zeros(9, dtype=np.float64)
    void_sq = np.zeros(9, dtype=np.float64)
    void_count = 0
    target_sum = 0.0
    target_sq = 0.0
    target_count = 0

    for indices in dataset.iter_indices_by_shard():
        shard_path = dataset.items[indices[0]][0]
        with h5py.File(shard_path, "r") as handle:
            for index in indices:
                _, sample_id, _ = dataset.items[index]
                sample = handle[f"samples/{sample_id}"]
                atoms = sample["atoms"]
                atom_values = np.asarray(atoms["features"][...], dtype=np.float64)[:, :5]
                atom_sum += atom_values.sum(axis=0)
                atom_sq += np.square(atom_values).sum(axis=0)
                atom_count += len(atom_values)

                void_values = np.asarray(sample["voids/features"][...], dtype=np.float64)
                if len(void_values):
                    void_sum += void_values.sum(axis=0)
                    void_sq += np.square(void_values).sum(axis=0)
                    void_count += len(void_values)
                    targets = np.asarray(
                        sample["targets/elf_at_center"][...], dtype=np.float64
                    )
                    target_sum += float(targets.sum())
                    target_sq += float(np.square(targets).sum())
                    target_count += len(targets)

    if not atom_count or not void_count or not target_count:
        raise ValueError("Training split has no atom, void, or ELF target values")

    def moments(total: np.ndarray, square_total: np.ndarray, count: int):
        mean = total / count
        variance = np.maximum(square_total / count - np.square(mean), 1e-12)
        return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)

    atom_mean, atom_std = moments(atom_sum, atom_sq, atom_count)
    void_mean, void_std = moments(void_sum, void_sq, void_count)
    target_mean = target_sum / target_count
    target_std = float(np.sqrt(max(target_sq / target_count - target_mean**2, 1e-12)))
    return {
        "atom_mean": atom_mean.tolist(),
        "atom_std": atom_std.tolist(),
        "void_mean": void_mean.tolist(),
        "void_std": void_std.tolist(),
        "target_mean": float(target_mean),
        "target_std": target_std,
        "atom_count": atom_count,
        "void_count": void_count,
    }
