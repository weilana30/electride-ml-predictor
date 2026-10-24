#!/usr/bin/env python3
"""Generate per-void ELF targets from a paired POSCAR and ELFCAR.

ELF is dimensionless. Coordinates in the output are fractional; distances and
peak offsets are in angstrom. Local maxima are maxima over ELFCAR grid samples
    inside periodic Cartesian spheres, not continuous/interpolated maxima.
    They are unconstrained by atom distance and may select an atomic basin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Elfcar


DEFAULT_RADII_A = (0.5, 1.0, 1.5)
SCHEMA_VERSION = "elf-void-targets-v1"


def _load_grid(elfcar_path: str | Path) -> tuple[np.ndarray, Structure]:
    elf = Elfcar.from_file(str(elfcar_path))
    grid = np.asarray(elf.data.get("total"))
    if grid.ndim != 3 or any(n <= 0 for n in grid.shape):
        raise ValueError(f"ELFCAR total grid must be nonempty 3D, got {grid.shape}")
    if not np.issubdtype(grid.dtype, np.number) or not np.isfinite(grid).all():
        raise ValueError("ELFCAR total grid contains non-numeric or non-finite values")
    return grid, elf.structure


def _validate_pair(
    poscar_path: str | Path,
    elf_structure: Structure,
    lattice_atol_A: float = 1e-3,
    position_atol_A: float = 1e-3,
) -> Structure:
    structure = Structure.from_file(str(poscar_path))
    if not np.allclose(structure.lattice.matrix, elf_structure.lattice.matrix, rtol=0, atol=lattice_atol_A):
        raise ValueError("POSCAR and ELFCAR lattice matrices do not match within 0.001 Å")
    if len(structure) != len(elf_structure):
        raise ValueError(f"POSCAR/ELFCAR atom counts differ ({len(structure)} vs {len(elf_structure)})")
    # POSCAR ordering is significant for a paired VASP calculation.
    if [str(s.specie) for s in structure] != [str(s.specie) for s in elf_structure]:
        raise ValueError("POSCAR and ELFCAR species/order differ")
    for i, (poscar_site, elf_site) in enumerate(zip(structure, elf_structure)):
        # get_distance_and_image performs a periodic minimum-image comparison;
        # coordinates differing by an integer lattice translation therefore
        # still identify the same atom.
        distance = structure.lattice.get_distance_and_image(
            poscar_site.frac_coords, elf_site.frac_coords
        )[0]
        if distance > position_atol_A:
            raise ValueError(
                f"POSCAR/ELFCAR atom {i} differs by {distance:.6g} Å "
                f"(tolerance {position_atol_A:g} Å)"
            )
    return structure


def _normalize_candidates(candidates: Any) -> list[dict[str, Any]]:
    """Accept coordinate lists, void records, or builder graph void arrays."""
    if isinstance(candidates, dict):
        candidates = candidates.get("voids", candidates.get("void_nodes", candidates))
    if isinstance(candidates, dict):
        # build_graph_records.py stores node attributes as parallel arrays.
        coords = candidates.get("fractional_coords", candidates.get("frac_coords"))
        if coords is not None:
            coords = np.asarray(coords, dtype=float)
            if coords.ndim != 2 or coords.shape[1] != 3:
                raise ValueError("Graph void fractional_coords must have shape [n_voids, 3]")
            ids = candidates.get("void_id", list(range(len(coords))))
            if len(ids) != len(coords):
                raise ValueError("Graph void_id and fractional_coords lengths differ")
            candidates = [{"void_id": ids[i], "frac_coords": coords[i]} for i in range(len(coords))]
    if candidates is None:
        raise ValueError("No candidates found; provide a list or graph record with 'voids'")
    normalized = []
    for i, item in enumerate(candidates):
        if isinstance(item, dict):
            frac = item.get("frac_coords", item.get("frac", item.get("fractional_coords")))
            void_id = item.get("void_id", item.get("id", i))
        else:
            frac, void_id = item, i
        arr = np.asarray(frac, dtype=float)
        if arr.shape != (3,) or not np.isfinite(arr).all():
            raise ValueError(f"Candidate {i} must have three finite fractional coordinates")
        normalized.append({"void_id": str(void_id), "frac_coords": arr % 1.0})
    return normalized


def _trilinear(grid: np.ndarray, frac: np.ndarray) -> float:
    """Periodic trilinear interpolation, consistent with VASP grid indexing."""
    shape = np.asarray(grid.shape, dtype=int)
    pos = (frac % 1.0) * shape
    lower = np.floor(pos).astype(int)
    weight = pos - lower
    result = 0.0
    for off in np.ndindex(2, 2, 2):
        off_arr = np.asarray(off, dtype=int)
        idx = tuple((lower + off_arr) % shape)
        result += float(grid[idx]) * float(np.prod(np.where(off_arr, weight, 1.0 - weight)))
    return result


def _local_grid_max(grid: np.ndarray, frac: np.ndarray, structure: Structure, radius_a: float) -> dict[str, Any]:
    if not np.isfinite(radius_a) or radius_a < 0:
        raise ValueError("Local-maximum radii must be finite and nonnegative")
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    inv_lattice = np.linalg.inv(lattice)
    shape = np.asarray(grid.shape, dtype=int)
    # A Cartesian sphere of radius r is contained in this fractional box.
    frac_extent = radius_a * np.linalg.norm(inv_lattice, axis=0)
    center_grid = (frac % 1.0) * shape
    half_width = np.ceil(frac_extent * shape).astype(int) + 1
    axes = [np.arange(int(np.floor(center_grid[d])) - half_width[d],
                       int(np.floor(center_grid[d])) + half_width[d] + 1) for d in range(3)]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    wrapped = mesh % shape
    frac_points = mesh / shape
    delta = frac_points - frac
    # ``mesh`` contains unwrapped periodic images around the query coordinate,
    # so direct Cartesian distance is exact even for strongly skewed cells.
    distances = np.linalg.norm(delta @ lattice, axis=1)
    inside = distances <= radius_a + 1e-10
    if not np.any(inside):
        # A radius-zero sphere can miss all grid samples. Use nearest periodic
        # grid point as a well-defined fallback and make that explicit.
        nearest_unwrapped = np.rint(center_grid).astype(int)
        wrapped = (nearest_unwrapped % shape).reshape(1, 3)
        frac_points = (nearest_unwrapped / shape).reshape(1, 3)
        distances = np.linalg.norm((frac_points - frac) @ lattice, axis=1)
        inside = np.ones(1, dtype=bool)
        fallback = True
    else:
        fallback = False
    wrapped, frac_points, distances = wrapped[inside], frac_points[inside], distances[inside]
    vals = grid[wrapped[:, 0], wrapped[:, 1], wrapped[:, 2]]
    max_index = int(np.argmax(vals))
    peak_frac = frac_points[max_index] % 1.0
    return {
        "elf_max_grid": float(vals[max_index]),
        "peak_frac_coords": peak_frac.tolist(),
        "peak_offset_A": float(distances[max_index]),
        "peak_offset_vector_A": ((frac_points[max_index] - frac) @ lattice).astype(float).tolist(),
        "grid_samples_in_radius": int(len(vals)),
        "nearest_grid_fallback": fallback,
    }


def label_voids_from_elf(
    poscar_path: str | Path,
    elfcar_path: str | Path,
    candidates: Any,
    radii_A: Iterable[float] = DEFAULT_RADII_A,
) -> dict[str, Any]:
    """Return per-void ELF targets for one paired structure.

    ``candidates`` may be a sequence of fractional coordinate triples, a
    sequence of records with ``frac_coords``/``frac`` and optional IDs, or a
    graph record whose ``voids`` mapping contains parallel ``fractional_coords``
    and optional ``void_id`` arrays.
    """
    radii = tuple(float(r) for r in radii_A)
    if not radii or any(not np.isfinite(r) or r < 0 for r in radii):
        raise ValueError("Provide one or more finite, nonnegative radii in Å")
    grid, elf_structure = _load_grid(elfcar_path)
    structure = _validate_pair(poscar_path, elf_structure)
    voids = _normalize_candidates(candidates)
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    output_voids = []
    for item in voids:
        frac = item["frac_coords"]
        output_voids.append({
            "void_id": item["void_id"],
            "frac_coords": frac.tolist(),
            "elf_at_center": _trilinear(grid, frac),
            "local_maxima": {
                f"{radius:g}_A": _local_grid_max(grid, frac, structure, radius)
                for radius in radii
            },
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "units": {"elf": "dimensionless", "frac_coords": "fractional", "peak_offset_A": "angstrom"},
        "label_definitions": {
            "elf_at_center": "periodic trilinear interpolation of ELFCAR total grid at candidate coordinate",
            "elf_max_grid": "unconstrained maximum raw ELFCAR grid sample within periodic Cartesian sphere; may lie in an atomic basin",
            "peak_offset_A": "periodic Cartesian distance from candidate coordinate to maximum grid sample",
            "radii_A": list(radii),
        },
        "source": {"poscar": str(poscar_path), "elfcar": str(elfcar_path)},
        "lattice_A": lattice.tolist(),
        "grid_shape": list(grid.shape),
        "void_count": len(output_voids),
        "voids": output_voids,
    }


def label_voids(
    elfcar_path: str | Path,
    lattice: Any,
    void_frac_coords: Any,
    search_radii_A: Iterable[float] = (0.5, 1.0),
    *,
    poscar_path: str | Path,
) -> dict[str, np.ndarray]:
    """Stable array-oriented API for the dataset writer.

    Returns NumPy arrays keyed by ``elf_at_center``, ``elf_max_<radius>A``,
    and ``offset_<radius>A`` (signed Cartesian Å vectors). Invalid input raises
    ``ValueError``; validity masks are therefore unnecessary. The supplied
    lattice is in Å and must match ELFCAR. ``poscar_path`` is required so atom
    positions and species ordering are checked too; matching only the lattice
    would allow a mismatched structure to be labeled.
    """
    grid, elf_structure = _load_grid(elfcar_path)
    structure = _validate_pair(poscar_path, elf_structure)
    lattice_arr = np.asarray(lattice, dtype=float)
    if lattice_arr.shape != (3, 3) or not np.isfinite(lattice_arr).all() or abs(np.linalg.det(lattice_arr)) < 1e-10:
        raise ValueError("lattice must be a finite, nonsingular 3x3 matrix in Å")
    if not np.allclose(lattice_arr, elf_structure.lattice.matrix, rtol=0, atol=1e-3):
        raise ValueError("Supplied lattice and ELFCAR lattice do not match within 0.001 Å")
    voids = _normalize_candidates(void_frac_coords)
    coords = np.asarray([v["frac_coords"] for v in voids], dtype=float).reshape(-1, 3)
    radii = tuple(float(r) for r in search_radii_A)
    if not radii or any(not np.isfinite(r) or r < 0 for r in radii):
        raise ValueError("search_radii_A must contain finite, nonnegative radii")
    out: dict[str, np.ndarray] = {}
    out["elf_at_center"] = np.asarray([_trilinear(grid, f) for f in coords], dtype=np.float32)
    for radius in radii:
        key = f"{radius:g}A"
        peaks = [_local_grid_max(grid, f, structure, radius) for f in coords]
        out[f"elf_max_{key}"] = np.asarray([p["elf_max_grid"] for p in peaks], dtype=np.float32)
        out[f"offset_{key}"] = np.asarray([p["peak_offset_vector_A"] for p in peaks], dtype=np.float32).reshape(-1, 3)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("poscar", type=Path)
    parser.add_argument("elfcar", type=Path)
    parser.add_argument("candidates_json", type=Path, help="JSON list of fractional coordinates/void records, or graph record")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path (default: stdout)")
    parser.add_argument("--radii-A", type=float, nargs="+", default=list(DEFAULT_RADII_A), help="Local grid-maximum search radii in Å; default: 0.5 1.0 1.5")
    args = parser.parse_args()
    candidates = json.loads(args.candidates_json.read_text())
    result = label_voids_from_elf(args.poscar, args.elfcar, candidates, args.radii_A)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
