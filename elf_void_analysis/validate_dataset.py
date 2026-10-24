#!/usr/bin/env python3
"""Validate model-ready atom/void HDF5 shards against the Parquet manifest.

The validator reads stored graph/label arrays only; it never regenerates
features or ELF targets. A void-level CSV is provided for quick plotting and
inspection of a bounded sample of structures.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _attr(group: h5py.Group, *names: str) -> Any:
    for name in names:
        if name in group.attrs:
            return group.attrs[name]
    return None


def _metadata_attr(sample: h5py.Group, *names: str) -> Any:
    """Read metadata written as attributes under metadata/ by the shard writer."""
    metadata = sample.get("metadata")
    if isinstance(metadata, h5py.Group):
        value = _attr(metadata, *names)
        if value is not None:
            return value
        for name in names:
            if name in metadata and isinstance(metadata[name], h5py.Dataset):
                data = metadata[name][()]
                if isinstance(data, np.ndarray) and data.size == 1:
                    data = data.reshape(-1)[0]
                return data
    return _attr(sample, *names)


def _dataset(group: h5py.Group, path: str, aliases: Iterable[str] = ()) -> h5py.Dataset | None:
    for name in (path, *aliases):
        if name in group and isinstance(group[name], h5py.Dataset):
            return group[name]
    return None


def _require_dataset(group: h5py.Group, path: str, aliases: Iterable[str] = ()) -> h5py.Dataset:
    dataset = _dataset(group, path, aliases)
    if dataset is None:
        raise ValueError(f"missing required dataset {group.name}/{path}")
    return dataset


def _check_finite_numeric(group: h5py.Group, errors: list[str]) -> None:
    def visit(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        if np.issubdtype(obj.dtype, np.number):
            data = obj[...]
            if not np.isfinite(data).all():
                errors.append(f"{obj.name}: contains NaN or infinity")
    group.visititems(visit)


def _decode_list(values: Any) -> list[str]:
    return [_text(value) for value in np.asarray(values).reshape(-1)]


def _validate_sample(sample: h5py.Group, manifest_row: dict[str, Any] | None,
                     void_row_limit: int = 0,
                     schema: dict[str, Any] | None = None
                     ) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    if not all(name in sample for name in ("atoms", "voids", "edges", "targets")):
        missing = [name for name in ("atoms", "voids", "edges", "targets") if name not in sample]
        return {"sample_id": sample.name.rsplit("/", 1)[-1]}, [f"{sample.name}: missing groups {missing}"], []
    atoms, voids, edges, targets = (sample[name] for name in ("atoms", "voids", "edges", "targets"))
    sid = sample.name.rsplit("/", 1)[-1]
    family = _text(_metadata_attr(sample, "family_id", "structure_family_id") or "")
    lattice_ds = _dataset(sample, "lattice_angstrom", ("lattice_A",))
    if lattice_ds is None:
        errors.append(f"{sample.name}: missing lattice_angstrom")
        lattice = None
    else:
        lattice = np.asarray(lattice_ds[...], dtype=float)
        if lattice.shape != (3, 3):
            errors.append(f"{lattice_ds.name}: expected shape (3,3), got {lattice.shape}")
            lattice = None
        elif not np.isfinite(lattice).all() or abs(np.linalg.det(lattice)) < 1e-10:
            errors.append(f"{lattice_ds.name}: lattice must be finite and nonsingular")

    atom_frac_ds = _require_dataset(atoms, "fractional_coords", ("frac_coords",))
    atom_frac = np.asarray(atom_frac_ds[...], dtype=float)
    if atom_frac.ndim != 2 or atom_frac.shape[1:] != (3,):
        errors.append(f"{atom_frac_ds.name}: expected [n_atoms,3], got {atom_frac.shape}")
        n_atoms = -1
    else:
        n_atoms = int(len(atom_frac))
        if np.any(atom_frac < -1e-5) or np.any(atom_frac >= 1.0 + 1e-5):
            errors.append(f"{atom_frac_ds.name}: fractional coordinates outside [0,1)")
    atomic_numbers = _require_dataset(atoms, "atomic_numbers", ("atomic_number",))
    if atomic_numbers.ndim != 1 or (n_atoms >= 0 and len(atomic_numbers) != n_atoms):
        errors.append(f"{atomic_numbers.name}: expected one atomic number per atom")
    if not np.issubdtype(atomic_numbers.dtype, np.integer):
        errors.append(f"{atomic_numbers.name}: expected integer dtype")
    atom_features = _require_dataset(atoms, "features")
    if atom_features.ndim != 2 or (n_atoms >= 0 and atom_features.shape[0] != n_atoms):
        errors.append(f"{atom_features.name}: expected [n_atoms,n_features], got {atom_features.shape}")
    validity = _require_dataset(atoms, "feature_validity")
    validity_names = _require_dataset(atoms, "feature_validity_names")
    if validity.ndim != 2 or (n_atoms >= 0 and validity.shape[0] != n_atoms):
        errors.append(f"{validity.name}: expected [n_atoms,n_validity_features], got {validity.shape}")
    if validity.ndim == 2 and len(validity_names) != validity.shape[1]:
        errors.append(f"{validity_names.name}: name count does not match validity width")
    expected_validity = (schema or {}).get("graph_storage", {}).get("node_features", {}).get("atoms", {}).get("feature_validity_names_in_order")
    if expected_validity is not None and _decode_list(validity_names[...]) != list(expected_validity):
        errors.append(f"{validity_names.name}: feature-validity order differs from schema")
    for names_key, features_ds in (("feature_names", atom_features),):
        names = atoms.get(names_key)
        if isinstance(names, h5py.Dataset) and features_ds.ndim == 2 and len(names) != features_ds.shape[1]:
            errors.append(f"{names.name}: feature-name count does not match feature width")
        expected_names = (schema or {}).get("graph_storage", {}).get("node_features", {}).get("atoms", {}).get("feature_names_in_order")
        if expected_names is not None and isinstance(names, h5py.Dataset):
            if _decode_list(names[...]) != list(expected_names):
                errors.append(f"{names.name}: feature order differs from schema")

    void_frac_ds = _require_dataset(voids, "fractional_coords", ("frac_coords",))
    void_frac = np.asarray(void_frac_ds[...], dtype=float)
    if void_frac.ndim != 2 or void_frac.shape[1:] != (3,):
        errors.append(f"{void_frac_ds.name}: expected [n_voids,3], got {void_frac.shape}")
        n_voids = -1
    else:
        n_voids = int(len(void_frac))
        if np.any(void_frac < -1e-5) or np.any(void_frac >= 1.0 + 1e-5):
            errors.append(f"{void_frac_ds.name}: fractional coordinates outside [0,1)")
    void_features = _require_dataset(voids, "features")
    if void_features.ndim != 2 or (n_voids >= 0 and void_features.shape[0] != n_voids):
        errors.append(f"{void_features.name}: expected [n_voids,n_features], got {void_features.shape}")
    if "feature_names" in voids and isinstance(voids["feature_names"], h5py.Dataset):
        if void_features.ndim == 2 and len(voids["feature_names"]) != void_features.shape[1]:
            errors.append(f"{voids['feature_names'].name}: feature-name count does not match feature width")
        expected_names = (schema or {}).get("graph_storage", {}).get("node_features", {}).get("voids", {}).get("feature_names_in_order")
        if expected_names is not None and _decode_list(voids["feature_names"][...]) != list(expected_names):
            errors.append(f"{voids['feature_names'].name}: feature order differs from schema")
    multi_hot = _require_dataset(voids, "method_multi_hot")
    method_names = _require_dataset(voids, "method_names")
    if multi_hot.ndim != 2 or multi_hot.shape[0] != n_voids:
        errors.append(f"{multi_hot.name}: expected [n_voids,n_methods]")
    if multi_hot.ndim == 2 and len(method_names) != multi_hot.shape[1]:
        errors.append(f"{method_names.name}: names must match method_multi_hot columns")
    expected_methods = (schema or {}).get("graph_storage", {}).get("node_features", {}).get("voids", {}).get("method_names_in_order")
    if expected_methods is not None and _decode_list(method_names[...]) != list(expected_methods):
        errors.append(f"{method_names.name}: method multi-hot order differs from schema")
    orbit_multiplicity = _require_dataset(voids, "orbit_multiplicity")
    if orbit_multiplicity.shape != (n_voids,):
        errors.append(f"{orbit_multiplicity.name}: expected one value per void")
    void_ids_ds = _dataset(voids, "void_id", ("void_ids",))
    if void_ids_ds is not None:
        void_ids = _decode_list(void_ids_ds[...])
        if len(void_ids) != n_voids or len(set(void_ids)) != len(void_ids):
            errors.append(f"{void_ids_ds.name}: IDs must be unique and count must equal n_voids")
    else:
        void_ids = [str(i) for i in range(max(0, n_voids))]

    # Any target dataset is a per-void label: scalar targets have shape [Nv],
    # vector targets [Nv, ...]. The first axis must always index void nodes.
    if len(targets) == 0:
        errors.append(f"{targets.name}: no target datasets")
    expected_targets = (schema or {}).get("graph_storage", {}).get("targets", {}).get("per_void_arrays")
    if expected_targets is not None and set(targets.keys()) != set(expected_targets):
        errors.append(
            f"{targets.name}: target names differ from schema; "
            f"expected {sorted(expected_targets)}, found {sorted(targets.keys())}"
        )
    expected_target_shapes = (schema or {}).get("graph_storage", {}).get("targets", {}).get("shapes", {})
    for name, obj in targets.items():
        if not isinstance(obj, h5py.Dataset):
            errors.append(f"{obj.name}: targets group may contain datasets only")
        elif obj.ndim == 0 or (n_voids >= 0 and obj.shape[0] != n_voids):
            errors.append(f"{obj.name}: first dimension must equal n_voids={n_voids}, got {obj.shape}")
        elif name in expected_target_shapes and n_voids >= 0:
            expected_shape = tuple(n_voids if dim == "n_voids" else int(dim)
                                   for dim in expected_target_shapes[name])
            if obj.shape != expected_shape:
                errors.append(f"{obj.name}: expected schema shape {expected_shape}, got {obj.shape}")

    edge_counts: dict[str, int] = {}
    edge_specs = (
        ("atom_atom", n_atoms, n_atoms, "atom_atom_index", "atom_atom_distance_angstrom", "atom_atom_image_shift"),
        ("void_atom", n_voids, n_atoms, "void_atom_index", "void_atom_distance_angstrom", "void_atom_image_shift"),
    )
    for edge_name, src_count, dst_count, index_name, dist_name, shift_name in edge_specs:
        idx_ds = _require_dataset(edges, index_name, (index_name.replace("atom_atom", "atom_atom"),))
        dist_ds = _require_dataset(edges, dist_name, (dist_name.replace("_angstrom", "_A"),))
        shift_ds = _require_dataset(edges, shift_name)
        idx = np.asarray(idx_ds[...])
        dist = np.asarray(dist_ds[...], dtype=float)
        shifts = np.asarray(shift_ds[...])
        if idx.ndim != 2 or idx.shape[0] != 2:
            errors.append(f"{idx_ds.name}: expected shape [2,n_edges], got {idx.shape}")
            edge_counts[edge_name] = -1
            continue
        count = int(idx.shape[1])
        edge_counts[edge_name] = count
        if not np.issubdtype(idx.dtype, np.integer):
            errors.append(f"{idx_ds.name}: indices must have integer dtype")
        if idx.shape[1] != len(dist) or dist.ndim != 1:
            errors.append(f"{dist_ds.name}: edge count must match index count")
        if shifts.shape != (count, 3):
            errors.append(f"{shift_ds.name}: expected [n_edges,3], got {shifts.shape}")
        if not np.issubdtype(shifts.dtype, np.integer):
            errors.append(f"{shift_ds.name}: image shifts must have integer dtype")
        if src_count >= 0 and count and (np.any(idx[0] < 0) or np.any(idx[0] >= src_count)):
            errors.append(f"{idx_ds.name}: source index out of bounds for {src_count} nodes")
        if dst_count >= 0 and count and (np.any(idx[1] < 0) or np.any(idx[1] >= dst_count)):
            errors.append(f"{idx_ds.name}: destination index out of bounds for {dst_count} nodes")
        if len(dist) and np.any(dist < 0):
            errors.append(f"{dist_ds.name}: distances must be nonnegative")
        # The contract says image_shift translates the destination coordinate.
        # Recompute distances to catch transposed edge arrays or shift signs.
        if lattice is not None and idx.shape == (2, count) and shifts.shape == (count, 3) and count:
            src_coords = atom_frac if edge_name == "atom_atom" else void_frac
            dst_coords = atom_frac
            if src_coords.ndim == 2 and dst_coords.ndim == 2 and np.max(idx) < len(src_coords) + len(dst_coords):
                valid_indices = ((idx[0] >= 0) & (idx[0] < len(src_coords)) &
                                 (idx[1] >= 0) & (idx[1] < len(dst_coords)))
                if valid_indices.all():
                    delta = dst_coords[idx[1]] + shifts - src_coords[idx[0]]
                    recomputed = np.linalg.norm(delta @ lattice, axis=1)
                    if len(dist) == count and not np.allclose(recomputed, dist, rtol=1e-4, atol=1e-4):
                        errors.append(f"{dist_ds.name}: distances disagree with destination + image_shift convention")

    _check_finite_numeric(sample, errors)
    stored_atom_count = _metadata_attr(sample, "n_atoms")
    stored_void_count = _attr(sample, "n_voids", "void_count")
    counts = sample.get("metadata/counts")
    if isinstance(counts, h5py.Group):
        if stored_atom_count is None:
            stored_atom_count = _attr(counts, "n_atoms")
        if stored_void_count is None:
            stored_void_count = _attr(counts, "n_voids")
    if stored_atom_count is not None and n_atoms >= 0 and int(stored_atom_count) != n_atoms:
        errors.append(f"{sample.name}: n_atoms attr disagrees with atom arrays")
    if stored_void_count is not None and n_voids >= 0 and int(stored_void_count) != n_voids:
        errors.append(f"{sample.name}: n_voids attr disagrees with void arrays")
    if manifest_row is not None:
        if manifest_row.get("family_id") is not None and family and _text(manifest_row["family_id"]) != family:
            errors.append(f"{sample.name}: family_id differs from manifest")
        if manifest_row.get("atom_count") is not None and n_atoms >= 0:
            if int(manifest_row["atom_count"]) != n_atoms:
                errors.append(f"{sample.name}: atom count differs from manifest")
        expected_grid = manifest_row.get("grid_shape")
        actual_grid = _metadata_attr(sample, "grid_shape")
        if expected_grid is not None and actual_grid is not None:
            if not np.array_equal(np.asarray(expected_grid, dtype=int), np.asarray(actual_grid, dtype=int)):
                errors.append(f"{sample.name}: ELFCAR grid shape differs from manifest")
        for attr_name, row_name in (("source_poscar_relative_path", "poscar_relative_path"),
                                    ("source_elfcar_relative_path", "elfcar_relative_path")):
            actual = _metadata_attr(sample, attr_name)
            expected = manifest_row.get(row_name)
            if actual is not None and expected is not None and _text(actual) != _text(expected):
                errors.append(f"{sample.name}: {attr_name} differs from manifest")

    summary = {
        "sample_id": sid,
        "family_id": family,
        "n_atoms": n_atoms,
        "n_voids": n_voids,
        "n_atom_atom_edges": edge_counts.get("atom_atom"),
        "n_void_atom_edges": edge_counts.get("void_atom"),
        "grid_shape": np.asarray(_metadata_attr(sample, "grid_shape")).tolist() if _metadata_attr(sample, "grid_shape") is not None else None,
        "target_names": sorted(targets.keys()),
        "source_poscar": _text(_metadata_attr(sample, "source_poscar_relative_path", "poscar_relative_path", "source_poscar") or "") or None,
        "source_elfcar": _text(_metadata_attr(sample, "source_elfcar_relative_path", "elfcar_relative_path", "source_elfcar") or "") or None,
    }
    rows = []
    center_ds = _dataset(targets, "elf_at_center", ("elf_center",))
    clearance_ds = _dataset(voids, "clearance_angstrom", ("clearance_A",))
    center = np.asarray(center_ds[...]) if center_ds is not None else None
    clearance = np.asarray(clearance_ds[...]) if clearance_ds is not None else None
    for i in range(min(max(0, n_voids), max(0, void_row_limit))):
        row = {"sample_id": sid, "family_id": family, "void_id": void_ids[i],
               "frac_x": float(void_frac[i, 0]), "frac_y": float(void_frac[i, 1]), "frac_z": float(void_frac[i, 2])}
        if clearance is not None and i < len(clearance):
            row["clearance_A"] = float(clearance[i])
        if center is not None and center.ndim == 1 and i < len(center):
            row["elf_at_center"] = float(center[i])
        rows.append(row)
    return summary, errors, rows


def validate_dataset(
    shard_paths: Iterable[str | Path],
    manifest_path: str | Path,
    *,
    splits_path: str | Path | None = None,
    schema_path: str | Path | None = None,
    require_full_coverage: bool = False,
    source_root: str | Path | None = None,
    sample_preview_limit: int = 25,
) -> dict[str, Any]:
    """Validate shards and their manifest/split metadata without recomputing labels."""
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Parquet validation requires pyarrow") from exc
    manifest_table = pq.read_table(manifest_path)
    manifest_records = manifest_table.to_pylist()
    manifest_columns = set(manifest_table.column_names)
    if not {"sample_id", "family_id"}.issubset(manifest_columns):
        raise ValueError("manifest must contain sample_id and family_id columns")
    manifest_ids = [str(row["sample_id"]) for row in manifest_records]
    if len(manifest_ids) != len(set(manifest_ids)):
        seen: set[str] = set()
        dupes = []
        for sample_id in manifest_ids:
            if sample_id in seen and sample_id not in dupes:
                dupes.append(sample_id)
            seen.add(sample_id)
            if len(dupes) >= 10:
                break
        raise ValueError(f"manifest sample_id values are not unique: {dupes}")
    manifest_rows = {str(row["sample_id"]): row for row in manifest_records}
    schema = None
    if schema_path:
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        expected_groups = set(schema.get("graph_storage", {}).get("required_groups", [])) | {"targets"}
        contract_groups = {"atoms", "voids", "edges", "targets"}
        if not contract_groups.issubset(expected_groups):
            raise ValueError(
                "schema graph_storage.required_groups must include atoms, voids, edges, and targets"
            )
    split_rows: dict[str, dict[str, Any]] = {}
    if splits_path:
        split_table = pq.read_table(splits_path)
        split_records = split_table.to_pylist()
        required = {"sample_id", "family_id", "split"}
        if not required.issubset(split_table.column_names):
            raise ValueError(f"splits table missing columns {sorted(required - set(split_table.column_names))}")
        split_ids = [str(row["sample_id"]) for row in split_records]
        if len(split_ids) != len(set(split_ids)):
            raise ValueError("splits sample_id values are not unique")
        family_splits: dict[str, set[str]] = {}
        for row in split_records:
            sid, family, split = str(row["sample_id"]), str(row["family_id"]), str(row["split"])
            if sid in manifest_rows and str(manifest_rows[sid].get("family_id")) != family:
                raise ValueError(f"split family_id disagrees with manifest for {sid}")
            family_splits.setdefault(family, set()).add(split)
            split_rows[sid] = row
        inconsistent = {family: sorted(values) for family, values in family_splits.items() if len(values) > 1}
        if inconsistent:
            raise ValueError(f"family groups span multiple splits: {dict(list(inconsistent.items())[:10])}")

    source_base = Path(source_root).expanduser().resolve() if source_root else None

    all_samples: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    void_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for raw_path in shard_paths:
        path = Path(raw_path)
        with h5py.File(path, "r") as handle:
            if schema is not None:
                expected_version = schema.get("dataset_version")
                stored_version = _attr(handle, "dataset_version")
                if stored_version is not None and expected_version is not None and _text(stored_version) != str(expected_version):
                    errors.append(f"{path}: dataset_version attr differs from schema")
            if "samples" not in handle:
                errors.append(f"{path}: missing /samples group")
                continue
            for sid in handle["samples"]:
                if sid in all_samples:
                    errors.append(f"duplicate sample_id across shards: {sid}")
                    continue
                sample = handle["samples"][sid]
                if not isinstance(sample, h5py.Group):
                    errors.append(f"/samples/{sid} must be a group")
                    continue
                attr_sid = _attr(sample, "sample_id")
                if attr_sid is not None and _text(attr_sid) != sid:
                    errors.append(f"{path}:/samples/{sid}: sample_id attr differs from group key")
                if _metadata_attr(sample, "family_id", "structure_family_id") is None:
                    errors.append(f"{path}:/samples/{sid}: missing family_id attr")
                if sid not in manifest_rows:
                    errors.append(f"sample {sid} absent from manifest")
                try:
                    remaining_preview = max(0, sample_preview_limit * 20 - len(void_rows))
                    summary, sample_errors, rows = _validate_sample(
                        sample, manifest_rows.get(sid), void_row_limit=remaining_preview,
                        schema=schema,
                    )
                except Exception as exc:
                    errors.append(f"{path}:/samples/{sid}: validation could not complete: {exc}")
                    continue
                all_samples[sid] = summary
                summaries.append(summary)
                void_rows.extend(rows)
                errors.extend(f"{path}:{message}" for message in sample_errors)
                if splits_path and sid not in split_rows:
                    errors.append(f"sample {sid} absent from splits table")
                if source_base and sid in manifest_rows:
                    row = manifest_rows[sid]
                    for field in ("poscar_relative_path", "elfcar_relative_path"):
                        rel = row.get(field)
                        attr_names = (("source_poscar_relative_path", "poscar_relative_path", "source_poscar")
                                      if field.startswith("poscar") else
                                      ("source_elfcar_relative_path", "elfcar_relative_path", "source_elfcar"))
                        stored = _metadata_attr(sample, *attr_names)
                        if rel is not None and stored is not None:
                            stored_path = Path(_text(stored))
                            expected_path = (source_base / str(rel)).resolve()
                            resolved_stored = (stored_path if stored_path.is_absolute() else source_base / stored_path).resolve()
                            if resolved_stored != expected_path:
                                errors.append(f"{path}:/samples/{sid}: stored source path {stored} differs from manifest {rel}")
                    poscar_rel = row.get("poscar_relative_path")
                    elfcar_rel = row.get("elfcar_relative_path")
                    if poscar_rel is not None and elfcar_rel is not None:
                        poscar_path, elfcar_path = source_base / str(poscar_rel), source_base / str(elfcar_rel)
                        if not poscar_path.is_file() or not elfcar_path.is_file():
                            errors.append(f"{sid}: source POSCAR or ELFCAR missing under {source_base}")
                        else:
                            try:
                                from pymatgen.core import Structure
                                try:
                                    from .build_manifest import parse_elfcar_grid
                                except ImportError:  # direct-script invocation
                                    from build_manifest import parse_elfcar_grid

                                structure = Structure.from_file(str(poscar_path))
                                if int(summary["n_atoms"]) != len(structure):
                                    errors.append(f"{sid}: graph atom count differs from source POSCAR")
                                lattice_ds = sample.get("lattice_angstrom")
                                if lattice_ds is not None and not np.allclose(
                                    np.asarray(lattice_ds[...]), structure.lattice.matrix, rtol=0, atol=1e-3
                                ):
                                    errors.append(f"{sid}: graph lattice differs from source POSCAR")
                                grid_shape = parse_elfcar_grid(elfcar_path, len(structure))
                                manifest_grid = row.get("grid_shape")
                                if manifest_grid is not None and not np.array_equal(
                                    np.asarray(grid_shape), np.asarray(manifest_grid, dtype=int)
                                ):
                                    errors.append(f"{sid}: manifest grid shape differs from source ELFCAR")
                                stored_grid = _metadata_attr(sample, "grid_shape")
                                if stored_grid is not None and not np.array_equal(
                                    np.asarray(grid_shape), np.asarray(stored_grid, dtype=int)
                                ):
                                    errors.append(f"{sid}: stored grid shape differs from source ELFCAR")
                            except Exception as exc:
                                errors.append(f"{sid}: source metadata check failed: {exc}")
    ready_ids = {
        str(row["sample_id"])
        for row in manifest_records
        if row.get("processing_status", "ready") in {"ready", "processed"}
    }
    missing = sorted(ready_ids - set(all_samples))
    if require_full_coverage and missing:
        errors.append(f"HDF5 shards missing {len(missing)} ready manifest samples; examples: {missing[:10]}")
    report = {
        "valid": not errors,
        "manifest_path": str(Path(manifest_path).resolve()),
        "shard_paths": [str(Path(p).resolve()) for p in shard_paths],
        "manifest_sample_count": len(manifest_rows),
        "schema_path": str(Path(schema_path).resolve()) if schema_path else None,
        "ready_manifest_sample_count": len(ready_ids),
        "hdf5_sample_count": len(all_samples),
        "missing_ready_sample_count": len(missing),
        "missing_ready_sample_examples": missing[:20],
        "errors": errors,
        "sample_preview": summaries[:sample_preview_limit],
        "void_preview_count": min(len(void_rows), sample_preview_limit * 20),
    }
    report["_void_rows"] = void_rows[:sample_preview_limit * 20]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", required=True, help="HDF5 shard paths or glob patterns")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest Parquet path")
    parser.add_argument("--schema", type=Path, help="Optional JSON storage schema to check required groups against")
    parser.add_argument("--splits", type=Path, help="Optional family split Parquet path")
    parser.add_argument("--source-root", type=Path, help="Optional root for checking manifest source files")
    parser.add_argument("--require-full-coverage", action="store_true", help="Require every ready manifest sample in shards")
    parser.add_argument("--report-json", type=Path, help="Write validation report JSON")
    parser.add_argument("--void-summary-csv", type=Path, help="Write first 500 voids as plotting-ready CSV")
    parser.add_argument("--preview-limit", type=int, default=25, help="Maximum sample summaries in JSON and 20x this many void CSV rows")
    args = parser.parse_args()
    paths = sorted({Path(p) for pattern in args.shards for p in glob.glob(pattern)})
    if not paths:
        parser.error("no HDF5 shards matched --shards")
    try:
        report = validate_dataset(paths, args.manifest, splits_path=args.splits, schema_path=args.schema,
                                  require_full_coverage=args.require_full_coverage,
                                  source_root=args.source_root,
                                  sample_preview_limit=args.preview_limit)
    except Exception as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    void_rows = report.pop("_void_rows")
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.void_summary_csv:
        args.void_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["sample_id", "family_id", "void_id", "frac_x", "frac_y", "frac_z", "clearance_A", "elf_at_center"]
        with args.void_summary_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(void_rows)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
