#!/usr/bin/env python3
"""Write validated atom/void graph records and per-void labels to HDF5 shards.

The public API accepts mappings shaped like ``build_graph_record`` with a
``targets`` mapping from ``label_voids``. No Python objects or pickle payloads
are stored: numeric data are native HDF5 datasets, strings are UTF-8, and
irregular JSON-compatible metadata are explicitly JSON-encoded UTF-8 datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _sample_id(record: Mapping[str, Any]) -> str:
    value = record.get("sample_id")
    if value is None or not str(value).strip():
        raise ValueError("graph record is missing a nonempty sample_id")
    return str(value)


def _void_count(record: Mapping[str, Any]) -> int:
    if not isinstance(record.get("voids"), Mapping):
        raise ValueError("graph record must contain a voids mapping")
    voids = record["voids"]
    coords = voids.get("fractional_coords")
    if coords is None:
        coords = voids.get("frac_coords")
    if coords is None:
        raise ValueError("voids must contain fractional_coords (or frac_coords)")
    array = np.asarray(coords)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"void fractional coordinates must have shape (N, 3), got {array.shape}")
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError("void fractional coordinates must be finite numeric data")
    return int(array.shape[0])


def validate_record(record: Mapping[str, Any]) -> int:
    """Validate graph group presence and that all labels are per-void arrays."""
    _sample_id(record)
    for key in ("atoms", "voids", "edges"):
        if not isinstance(record.get(key), Mapping):
            raise ValueError(f"graph record must contain a {key!r} mapping")
    lattice = np.asarray(record.get("lattice_angstrom"))
    if lattice.shape != (3, 3) or lattice.dtype.kind not in "biuf" or not np.isfinite(lattice).all():
        raise ValueError("lattice_angstrom must be a finite 3x3 array")
    n_voids = _void_count(record)
    atom_coords = record["atoms"].get("fractional_coords", record["atoms"].get("frac_coords"))
    if atom_coords is None:
        raise ValueError("atoms must contain fractional_coords")
    atom_coords = np.asarray(atom_coords)
    if (atom_coords.ndim != 2 or atom_coords.shape[1] != 3
            or atom_coords.dtype.kind not in "biuf" or not np.isfinite(atom_coords).all()):
        raise ValueError("atom fractional coordinates must be finite with shape (N, 3)")
    n_atoms = int(atom_coords.shape[0])
    for node_name, count in (("atoms", n_atoms), ("voids", n_voids)):
        node = record[node_name]
        row_fields = ("atomic_numbers", "orbit_multiplicity", "clearance_angstrom")
        for field in row_fields:
            if field in node:
                values = np.asarray(node[field])
                if values.ndim == 0 or values.shape[0] != count:
                    raise ValueError(f"{node_name}/{field} first dimension must be {count}")
                if values.dtype.kind not in "biuf" or not np.isfinite(values).all():
                    raise ValueError(f"{node_name}/{field} must be finite numeric data")
        for feature_key in ("features", "feature_validity", "method_multi_hot"):
            if feature_key in node:
                values = np.asarray(node[feature_key])
                if values.ndim == 0 or values.shape[0] != count:
                    raise ValueError(f"{node_name}/{feature_key} first dimension must be {count}")
                if values.dtype.kind not in "biuf" or not np.isfinite(values).all():
                    raise ValueError(f"{node_name}/{feature_key} must be finite numeric data")
        for key, value in node.items():
            if node_name == "voids" and key == "methods":
                # Redundant ragged labels; method_multi_hot is the stored form.
                continue
            if isinstance(value, (np.ndarray, list, tuple)):
                try:
                    array = np.asarray(value)
                except (TypeError, ValueError):
                    # Ragged descriptive metadata is not a numeric model input.
                    continue
                if array.dtype.kind in "fc" and not np.isfinite(array).all():
                    raise ValueError(f"{node_name}/{key} contains NaN or infinity")
    edges = record["edges"]
    relations = {
        "atom_atom": (n_atoms, n_atoms),
        "void_atom": (n_voids, n_atoms),
    }
    for relation, (source_count, destination_count) in relations.items():
        index_key = f"{relation}_index"
        if index_key not in edges:
            raise ValueError(f"edges is missing {index_key}")
        indices = np.asarray(edges[index_key])
        if indices.ndim != 2 or indices.shape[0] != 2 or indices.dtype.kind not in "iu":
            raise ValueError(f"edges/{index_key} must be integer with shape (2, E)")
        edge_count = int(indices.shape[1])
        if edge_count and (
            np.any(indices[0] < 0) or np.any(indices[0] >= source_count)
            or np.any(indices[1] < 0) or np.any(indices[1] >= destination_count)
        ):
            raise ValueError(f"edges/{index_key} contains out-of-range node indices")
        for suffix, expected_shape in (("distance_angstrom", (edge_count,)),
                                       ("image_shift", (edge_count, 3))):
            key = f"{relation}_{suffix}"
            if key not in edges:
                raise ValueError(f"edges is missing parallel field {key}")
            values = np.asarray(edges[key])
            if values.shape != expected_shape:
                raise ValueError(f"edges/{key} must have shape {expected_shape}")
            if values.dtype.kind not in "biuf" or not np.isfinite(values).all():
                raise ValueError(f"edges/{key} must be finite numeric data")
    for key, value in edges.items():
        array = np.asarray(value)
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"edges/{key} contains NaN or infinity")
    targets = record.get("targets")
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("graph record must contain a nonempty targets mapping")
    for name, target in targets.items():
        array = np.asarray(target)
        if array.ndim == 0 or array.shape[0] != n_voids:
            raise ValueError(
                f"target {name!r} first dimension must equal void count {n_voids}; "
                f"got shape {array.shape}"
            )
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError(f"target {name!r} must be finite numeric data")
    return n_voids


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _write_string_dataset(group: h5py.Group, name: str, values: list[str]) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    data = np.asarray(values, dtype=object)
    group.create_dataset(name, data=data, dtype=dtype)


def _write_value(group: h5py.Group, name: str, value: Any) -> None:
    """Recursively write primitive values without object/pickle datasets."""
    if not name or "/" in name:
        raise ValueError(f"HDF5 field names must be nonempty path components; got {name!r}")
    if isinstance(value, Mapping):
        child = group.create_group(name)
        for key, item in value.items():
            _write_value(child, str(key), item)
        return

    if value is None:
        group.attrs[name] = "null"
        group.attrs[f"{name}__encoding"] = "json-null"
        return
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        group.attrs[name] = value
        return
    if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
        group.attrs[name] = value
        return

    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (list, tuple)):
        # Flat string sequences are native UTF-8 datasets. Ragged JSON data
        # are encoded explicitly and can be recovered by read_hdf5_value().
        if all(isinstance(x, str) for x in value):
            _write_string_dataset(group, name, list(value))
            return
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            array = np.asarray(value, dtype=object)
        if array.dtype.kind == "U":
            data = array.astype(object)
            group.create_dataset(name, data=data, dtype=h5py.string_dtype("utf-8"))
            return
        if array.dtype.kind == "O":
            encoded = json.dumps(value, default=_json_default, allow_nan=False)
            _write_string_dataset(group, name, [encoded])
            group[name].attrs["encoding"] = "json"
            return
    else:
        raise TypeError(f"unsupported value at {group.name}/{name}: {type(value).__name__}")

    if array.dtype.kind in "OUS":
        if array.dtype.kind == "U":
            group.create_dataset(name, data=array.astype(object), dtype=h5py.string_dtype("utf-8"))
            return
        if array.dtype.kind == "O" and all(isinstance(x, str) for x in array.reshape(-1)):
            string_array = array.astype(str)
            group.create_dataset(name, data=string_array.astype(object), dtype=h5py.string_dtype("utf-8"))
            return
        encoded = json.dumps(array.tolist(), default=_json_default, allow_nan=False)
        dataset = group.create_dataset(
            name, data=np.asarray([encoded], dtype=object), dtype=h5py.string_dtype("utf-8")
        )
        dataset.attrs["encoding"] = "json"
        dataset.attrs["original_shape"] = array.shape
        return
    if array.dtype.kind not in "biufc":
        raise TypeError(f"unsupported array dtype {array.dtype} at {group.name}/{name}")
    kwargs: dict[str, Any] = {}
    if array.size >= 256 and array.ndim > 0:
        kwargs.update(compression="gzip", compression_opts=4, shuffle=True)
    group.create_dataset(name, data=array, **kwargs)


def read_hdf5_value(node: h5py.Group | h5py.Dataset) -> Any:
    """Decode a value written by this module, including JSON-encoded fields."""
    if isinstance(node, h5py.Group):
        result = {key: read_hdf5_value(value) for key, value in node.items()}
        for key, value in node.attrs.items():
            if key.endswith("__encoding"):
                continue
            encoding = node.attrs.get(f"{key}__encoding")
            result[key] = None if encoding == "json-null" else _decode_h5_scalar(value)
        return result
    if node.attrs.get("encoding") == "json":
        raw = node.asstr()[0]
        return json.loads(raw)
    value = node[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in "SO":
        return node.asstr()[()]
    return value


def _decode_h5_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_record(samples: h5py.Group, record: Mapping[str, Any]) -> None:
    sample_id = _sample_id(record)
    if sample_id in samples:
        raise ValueError(f"duplicate sample_id in one shard: {sample_id}")
    n_voids = validate_record(record)
    sample = samples.create_group(sample_id)
    sample.attrs["sample_id"] = sample_id
    sample.attrs["void_count"] = n_voids
    sample.attrs["schema_version"] = str(record.get("schema_version", "1.0"))

    targets = record["targets"]
    metadata = sample.create_group("metadata")
    for name, value in record.items():
        if name in {"sample_id", "schema_version", "targets"}:
            continue
        if name in {"atoms", "voids", "edges"}:
            if name == "voids":
                # method_multi_hot carries the model input; omit redundant
                # ragged method-name lists to keep HDF5 representation simple.
                value = {key: item for key, item in value.items() if key != "methods"}
            _write_value(sample, name, value)
        elif name == "lattice_angstrom":
            _write_value(sample, name, value)
        else:
            _write_value(metadata, name, value)
    target_group = sample.create_group("targets")
    target_group.attrs["semantics"] = "per-void ELFCAR-derived labels; model inputs must exclude this group"
    for name, value in targets.items():
        _write_value(target_group, str(name), value)


def _batches(records: Iterable[Mapping[str, Any]], size: int):
    batch: list[Mapping[str, Any]] = []
    for record in records:
        batch.append(record)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_shard(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    dataset_version: str = "v1",
    schema_version: str = "1.0",
    shard_index: int = 0,
    build_fingerprint: str | None = None,
    source_root: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Atomically write a bounded iterable of records to exactly one shard."""
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists() and not overwrite:
        raise FileExistsError(final_path)
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.", suffix=".tmp", dir=final_path.parent
    )
    os.close(temp_fd)
    try:
        with h5py.File(temp_name, "w") as h5:
            h5.attrs["dataset_version"] = dataset_version
            h5.attrs["schema_version"] = schema_version
            h5.attrs["shard_index"] = shard_index
            if build_fingerprint is not None:
                h5.attrs["build_fingerprint"] = build_fingerprint
            if source_root is not None:
                h5.attrs["source_root"] = str(source_root)
            samples = h5.create_group("samples")
            seen: set[str] = set()
            count = 0
            for record in records:
                sample_id = _sample_id(record)
                if sample_id in seen:
                    raise ValueError(f"duplicate sample_id in shard: {sample_id}")
                _write_record(samples, record)
                seen.add(sample_id)
                count += 1
            h5.attrs["record_count"] = count
            h5.flush()
        os.replace(temp_name, final_path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return final_path


def write_hdf5_shard(
    output_path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    build_fingerprint: str,
    shard_index: int,
    source_root: str | Path | None = None,
    schema_version: str = "1.0",
    dataset_version: str = "v1",
    overwrite: bool = False,
) -> Path:
    """Orchestrator-facing single-shard writer API."""
    final_path = Path(output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    existing = sorted(final_path.parent.glob("graphs-*.h5"))
    if shard_index == 0:
        other_entries = [p for p in final_path.parent.iterdir() if p not in existing]
        if (existing or other_entries) and not overwrite:
            raise FileExistsError(
                f"refusing nonempty shard directory {final_path.parent}; "
                "use a new directory or explicitly set overwrite=True"
            )
        if overwrite:
            for shard in existing:
                shard.unlink()
    else:
        indexes = []
        for shard in existing:
            match = re.fullmatch(r"graphs-(\d{5})\.h5", shard.name)
            if match:
                indexes.append((int(match.group(1)), shard))
        actual = sorted(index for index, _ in indexes)
        expected = list(range(shard_index))
        if actual != expected:
            raise FileExistsError(
                f"cannot append shard {shard_index}: existing shard indexes {actual}, "
                f"expected contiguous prior indexes {expected}; start in a fresh directory"
            )
        for _, shard in indexes:
            with h5py.File(shard, "r") as h5:
                prior_fingerprint = h5.attrs.get("build_fingerprint")
                if isinstance(prior_fingerprint, bytes):
                    prior_fingerprint = prior_fingerprint.decode("utf-8")
                if prior_fingerprint != build_fingerprint:
                    raise ValueError(
                        f"build fingerprint mismatch in existing shard {shard}; "
                        "start shard 0 in a fresh output directory"
                    )
    return write_shard(
        output_path, records, dataset_version=dataset_version,
        schema_version=schema_version, shard_index=shard_index,
        build_fingerprint=build_fingerprint, source_root=source_root,
        overwrite=overwrite,
    )


def write_hdf5_shards(
    records: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    shard_size: int = 512,
    dataset_version: str = "v1",
    schema_version: str = "1.0",
    overwrite: bool = False,
    build_fingerprint: str = "unspecified",
) -> list[Path]:
    """Write records in numbered shards and return the final shard paths.

    Each shard is first written to a temporary file in the destination
    directory, flushed and closed, then atomically replaced into its final
    path. On a record error the incomplete temporary file is removed.
    """
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    seen: set[str] = set()
    for shard_index, batch in enumerate(_batches(records, shard_size)):
        final_path = out / f"graphs-{shard_index:05d}.h5"
        for record in batch:
            sample_id = _sample_id(record)
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id across shards: {sample_id}")
            seen.add(sample_id)
        paths.append(write_hdf5_shard(
            final_path, batch, build_fingerprint=build_fingerprint,
            dataset_version=dataset_version, schema_version=schema_version,
            shard_index=shard_index, overwrite=overwrite,
        ))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records_jsonl", type=Path,
                        help="JSONL, one validated graph record with targets per line")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--schema-version", default="1.0")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace existing shard files; default refuses them")
    args = parser.parse_args()

    def records():
        with args.records_jsonl.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc

    paths = write_hdf5_shards(
        records(), args.output_dir, shard_size=args.shard_size,
        dataset_version=args.dataset_version, schema_version=args.schema_version,
        overwrite=args.overwrite,
    )
    sample_count = 0
    for path in paths:
        with h5py.File(path, "r") as h5:
            sample_count += len(h5["samples"])
    print(json.dumps({"shards": [str(path) for path in paths],
                      "sample_count": sample_count}, indent=2))


if __name__ == "__main__":
    main()
