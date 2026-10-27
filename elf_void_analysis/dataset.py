"""Read-only loader for the ELFNet atom/void HDF5 shards.

Canonical layout (as written by the dataset builder)::

    samples/<sample_id>/
      lattice_angstrom
      atoms/{fractional_coords, atomic_numbers, features, feature_validity, ...}
      voids/{void_id, fractional_coords, features, ...}
      edges/{atom_atom_index, atom_atom_distance_angstrom,
             atom_atom_image_shift, void_atom_index,
             void_atom_distance_angstrom, void_atom_image_shift,
             void_void_index, void_void_distance_angstrom,
             void_void_image_shift}
      targets/<per-void label arrays>

The loader opens the HDF5 file for each read and never retains a handle across
calls, so it is safe to construct/use from data-loader worker processes. Target
arrays are returned separately from features and are only attached as labels
by the optional PyG conversion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


def _decode_text(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_attrs(attrs: h5py.AttributeManager) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in attrs.items():
        if key.endswith("__encoding"):
            continue
        encoding = attrs.get(f"{key}__encoding")
        if isinstance(encoding, bytes):
            encoding = encoding.decode("utf-8")
        value = _attr_value(value)
        if encoding == "json-null":
            value = None
        result[key] = value
    return result


def _read_group(group: h5py.Group, *, include_attrs: bool = False) -> dict[str, Any]:
    """Read a group recursively, copying arrays out of the HDF5 file."""
    values: dict[str, Any] = _read_attrs(group.attrs) if include_attrs else {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            values[key] = _read_group(item, include_attrs=include_attrs)
        else:
            value = item[()]
            encoding = item.attrs.get("encoding")
            if isinstance(encoding, bytes):
                encoding = encoding.decode("utf-8")
            if encoding == "json":
                if isinstance(value, np.ndarray) and value.size == 1:
                    value = value.reshape(-1)[0]
                value = json.loads(_decode_text(value))
            elif isinstance(value, bytes):
                value = value.decode("utf-8")
            elif isinstance(value, np.ndarray) and value.dtype.kind in {"S", "O"}:
                value = np.asarray([
                    _decode_text(x) for x in value.reshape(-1).tolist()
                ], dtype=object).reshape(value.shape)
            original_shape = item.attrs.get("original_shape")
            if original_shape is not None and isinstance(value, np.ndarray):
                value = value.reshape(tuple(int(dim) for dim in original_shape))
            values[key] = value
    return values


def _attr_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _check_record(record: dict[str, Any]) -> None:
    required = {"lattice_angstrom", "atoms", "voids", "edges", "targets"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"Graph record is missing groups/arrays: {sorted(missing)}")
    atoms, voids, edges, targets = (
        record["atoms"], record["voids"], record["edges"], record["targets"]
    )
    for group, key, width, label in (
        (atoms, "fractional_coords", 3, "atom coordinates"),
        (voids, "fractional_coords", 3, "void coordinates"),
    ):
        arr = np.asarray(group[key])
        if arr.ndim != 2 or arr.shape[1] != width:
            raise ValueError(f"Invalid {label} shape: {arr.shape}")
    n_atoms = len(atoms["fractional_coords"])
    n_voids = len(voids["fractional_coords"])
    atom_node_keys = {
        "fractional_coords", "atomic_numbers", "features", "feature_validity", "elements"
    }
    void_node_keys = {
        "void_id", "fractional_coords", "features", "method_multi_hot",
        "orbit_multiplicity", "clearance_angstrom", "methods"
    }
    for key, arr in atoms.items():
        if key in atom_node_keys and isinstance(arr, np.ndarray) and arr.ndim and len(arr) != n_atoms:
            raise ValueError(f"atoms/{key} has {len(arr)} rows; expected {n_atoms}")
    for key, arr in voids.items():
        if key in void_node_keys and isinstance(arr, np.ndarray) and arr.ndim and len(arr) != n_voids:
            raise ValueError(f"voids/{key} has {len(arr)} rows; expected {n_voids}")
    if np.asarray(record["lattice_angstrom"]).shape != (3, 3):
        raise ValueError("lattice_angstrom must have shape [3, 3]")

    edge_contracts = {
        "atom_atom": (n_atoms, n_atoms),
        "void_atom": (n_voids, n_atoms),
        "void_void": (n_voids, n_voids),
    }
    for relation, (n_source, n_destination) in edge_contracts.items():
        index = np.asarray(edges[f"{relation}_index"])
        if index.ndim != 2 or index.shape[0] != 2:
            raise ValueError(f"edges/{relation}_index must have shape [2, E]")
        if index.size and (
            index[0].min() < 0 or index[0].max() >= n_source
            or index[1].min() < 0 or index[1].max() >= n_destination
        ):
            raise ValueError(f"edges/{relation}_index contains out-of-range node indices")
        edge_count = index.shape[1]
        for suffix in ("distance_angstrom", "image_shift"):
            arr = np.asarray(edges[f"{relation}_{suffix}"])
            if len(arr) != edge_count:
                raise ValueError(f"edges/{relation}_{suffix} has wrong edge count")
        if np.asarray(edges[f"{relation}_image_shift"]).shape != (edge_count, 3):
            raise ValueError(f"edges/{relation}_image_shift must have shape [E, 3]")
    for name, labels in targets.items():
        arr = np.asarray(labels)
        if arr.ndim == 0 or len(arr) != n_voids:
            raise ValueError(f"targets/{name} must have first dimension n_voids={n_voids}")


def load_graph_record_from_group(group: h5py.Group, sample_id: str) -> dict[str, Any]:
    """Read and validate one sample from an already-open HDF5 group."""
    record: dict[str, Any] = {key: _read_group(group[key]) for key in (
        "atoms", "voids", "edges", "targets"
    )}
    if "lattice_angstrom" not in group:
        raise ValueError(f"{group.name} is missing lattice_angstrom")
    record["lattice_angstrom"] = group["lattice_angstrom"][()]
    record["sample_id"] = sample_id
    record["metadata"] = (
        _read_group(group["metadata"], include_attrs=True)
        if "metadata" in group else
        {key: _attr_value(value) for key, value in group.attrs.items()}
    )
    _check_record(record)
    return record


def load_graph_record(h5_path: str | Path, sample_id: str) -> dict[str, Any]:
    """Load and validate one graph record from a shard.

    All returned arrays are detached from the file. Targets stay separate from
    node features and can be attached as labels by downstream loaders.
    """
    group_key = f"samples/{sample_id}"
    with h5py.File(h5_path, "r") as handle:
        if group_key not in handle:
            raise KeyError(f"Sample {sample_id!r} not found in {h5_path}")
        return load_graph_record_from_group(handle[group_key], sample_id)


class HDF5GraphDataset:
    """Index and read structures from one HDF5 shard, opening per item read."""

    def __init__(self, h5_path: str | Path, sample_ids: Iterable[str] | None = None):
        self.h5_path = Path(h5_path)
        if sample_ids is None:
            with h5py.File(self.h5_path, "r") as handle:
                if "samples" not in handle:
                    raise ValueError(f"No samples group in {self.h5_path}")
                self.sample_ids = sorted(handle["samples"].keys())
        else:
            self.sample_ids = [str(sample_id) for sample_id in sample_ids]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return load_graph_record(self.h5_path, self.sample_ids[index])


def to_pyg_heterodata(record: dict[str, Any], *, y_key: str = "elf_at_center"):
    """Convert a loaded record to PyG ``HeteroData`` (PyG is optional).

    Input features are read only from ``atoms/features`` and ``voids/features``.
    ELFCAR-derived arrays remain clearly named node labels such as
    ``data['void'].target_elf_at_center`` and are never concatenated into ``x``.
    ``data['void'].y`` is additionally assigned when ``y_key`` exists. This
    flat tensor-attribute representation supports standard PyG batching.
    Reverse atom-to-void edges are derived by swapping
    the stored void-to-atom indices and negating its image shifts.
    """
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "to_pyg_heterodata requires torch and torch_geometric; the HDF5 "
            "loader itself only requires h5py and numpy"
        ) from exc

    _check_record(record)
    data = HeteroData()
    atoms, voids, edges = record["atoms"], record["voids"], record["edges"]
    data["atom"].x = torch.as_tensor(np.asarray(atoms["features"], dtype=np.float32))
    data["atom"].pos = torch.as_tensor(np.asarray(atoms["fractional_coords"], dtype=np.float32))
    data["atom"].atomic_number = torch.as_tensor(np.asarray(atoms["atomic_numbers"], dtype=np.int64))
    if "feature_validity" in atoms:
        data["atom"].feature_validity = torch.as_tensor(
            np.asarray(atoms["feature_validity"], dtype=np.uint8)
        )

    data["void"].x = torch.as_tensor(np.asarray(voids["features"], dtype=np.float32))
    data["void"].pos = torch.as_tensor(np.asarray(voids["fractional_coords"], dtype=np.float32))
    if "method_multi_hot" in voids:
        data["void"].method_multi_hot = torch.as_tensor(
            np.asarray(voids["method_multi_hot"], dtype=np.float32)
        )
    if "orbit_multiplicity" in voids:
        data["void"].orbit_multiplicity = torch.as_tensor(
            np.asarray(voids["orbit_multiplicity"], dtype=np.int64)
        )

    data["atom", "to", "atom"].edge_index = torch.as_tensor(
        np.asarray(edges["atom_atom_index"], dtype=np.int64)
    )
    atom_rel = data["atom", "to", "atom"]
    atom_rel.edge_distance = torch.as_tensor(
        np.asarray(edges["atom_atom_distance_angstrom"], dtype=np.float32)
    )
    atom_rel.edge_shift = torch.as_tensor(
        np.asarray(edges["atom_atom_image_shift"], dtype=np.int64)
    )

    void_void_rel = data["void", "to", "void"]
    void_void_rel.edge_index = torch.as_tensor(
        np.asarray(edges["void_void_index"], dtype=np.int64)
    )
    void_void_rel.edge_distance = torch.as_tensor(
        np.asarray(edges["void_void_distance_angstrom"], dtype=np.float32)
    )
    void_void_rel.edge_shift = torch.as_tensor(
        np.asarray(edges["void_void_image_shift"], dtype=np.int64)
    )

    void_index = np.asarray(edges["void_atom_index"], dtype=np.int64)
    void_distance = np.asarray(edges["void_atom_distance_angstrom"], dtype=np.float32)
    void_shift = np.asarray(edges["void_atom_image_shift"], dtype=np.int64)
    relation = data["void", "to", "atom"]
    relation.edge_index = torch.as_tensor(void_index)
    relation.edge_distance = torch.as_tensor(void_distance)
    relation.edge_shift = torch.as_tensor(void_shift)
    reverse = data["atom", "to", "void"]
    reverse.edge_index = torch.as_tensor(void_index[::-1].copy())
    reverse.edge_distance = torch.as_tensor(void_distance.copy())
    reverse.edge_shift = torch.as_tensor(-void_shift)

    targets = {
        key: torch.as_tensor(np.asarray(value))
        for key, value in record["targets"].items()
    }
    for key, value in targets.items():
        data["void"][f"target_{key}"] = value
    if y_key in targets:
        data["void"].y = targets[y_key]
    data.lattice = torch.as_tensor(np.asarray(record["lattice_angstrom"], dtype=np.float32))
    data.sample_id = record.get("sample_id", "")
    return data


__all__ = ["HDF5GraphDataset", "load_graph_record", "load_graph_record_from_group", "to_pyg_heterodata"]
