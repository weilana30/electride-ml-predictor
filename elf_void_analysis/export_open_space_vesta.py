#!/usr/bin/env python3
"""Export the 1 Å-probe accessible-space mask as VESTA-readable XSF files.

The scalar grid is 1 where a spherical probe center can fit without coming
within ``probe_radius`` of any atom and 0 otherwise. XSF stores the structure
and scalar field together; use an isosurface at 0.5 in VESTA to display the
accessible volume.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Structure

try:
    from .label_void_dimensionality import (
        _accessible_voxels,
        _component_winding_rank,
        _grid_shape,
    )
except ImportError:  # direct script invocation
    from label_void_dimensionality import (
        _accessible_voxels,
        _component_winding_rank,
        _grid_shape,
    )


def _safe_name(path: Path) -> str:
    text = "__".join(path.parts[-4:-1]) or path.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def write_xsf(
    path: Path,
    lattice: np.ndarray,
    atomic_numbers: np.ndarray,
    cart_coords: np.ndarray,
    active: np.ndarray,
) -> None:
    """Write structure and a periodic accessible-space mask to XSF."""
    # XSF general grids include both ends of the cell. The analyzer's grid is
    # periodic and omits the duplicate end planes, so copy index 0 to each end.
    extended = np.concatenate((active, active[:1, :, :]), axis=0)
    extended = np.concatenate((extended, extended[:, :1, :]), axis=1)
    extended = np.concatenate((extended, extended[:, :, :1]), axis=2)
    nx, ny, nz = extended.shape

    with path.open("w", encoding="utf-8") as out:
        out.write("CRYSTAL\nPRIMVEC\n")
        for vector in lattice:
            out.write("  " + "  ".join(f"{x:.12g}" for x in vector) + "\n")
        out.write(f"PRIMCOORD\n{len(atomic_numbers)} 1\n")
        for number, xyz in zip(atomic_numbers, cart_coords):
            out.write(f"{int(number)}  " + "  ".join(f"{x:.12g}" for x in xyz) + "\n")
        out.write("BEGIN_BLOCK_DATAGRID_3D\naccessible_space\n")
        out.write("BEGIN_DATAGRID_3D_accessible_space\n")
        out.write(f"{nx} {ny} {nz}\n0 0 0\n")
        for vector in lattice:
            out.write("  " + "  ".join(f"{x:.12g}" for x in vector) + "\n")
        # XSF's grid serialization is column-major: the first index varies
        # fastest. Explicit Fortran order keeps values aligned with grid axes.
        flat = extended.astype(np.uint8, copy=False).ravel(order="F")
        for start in range(0, len(flat), 8):
            out.write(" ".join(str(int(x)) for x in flat[start:start + 8]) + "\n")
        out.write("END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D\n")


def export_one(
    poscar: Path,
    output: Path,
    *,
    probe_radius_A: float,
    grid_spacing_A: float,
    chunk_size: int,
) -> dict[str, Any]:
    structure = Structure.from_file(str(poscar))
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    atom_frac = np.asarray(structure.frac_coords, dtype=float) % 1.0
    shape = _grid_shape(lattice, grid_spacing_A)
    active = _accessible_voxels(
        lattice, atom_frac, shape, probe_radius_A, chunk_size
    )
    component_ids, ranks = _component_winding_rank(active)
    xsf = output / f"{_safe_name(poscar)}.xsf"
    write_xsf(
        xsf,
        lattice,
        np.asarray([site.specie.Z for site in structure], dtype=int),
        np.asarray(structure.cart_coords, dtype=float),
        active,
    )
    return {
        "source_poscar": str(poscar),
        "xsf_file": xsf.name,
        "formula": structure.composition.reduced_formula,
        "space_group": structure.get_space_group_info(symprec=0.01)[0],
        "atom_count": len(structure),
        "cell_volume_angstrom3": float(structure.volume),
        "probe_radius_angstrom": float(probe_radius_A),
        "grid_spacing_angstrom": float(grid_spacing_A),
        "grid_shape": list(shape),
        "accessible_voxel_count": int(active.sum()),
        "accessible_fraction": float(active.mean()),
        "component_count": len(ranks),
        "component_winding_ranks": ranks,
        "void_dimensionality_definition": "geometric periodic accessible-space winding rank",
        "xsf_grid_shape": [int(x + 1) for x in shape],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("poscar", nargs="+", type=Path, help="One or more POSCAR files")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for XSF files")
    parser.add_argument("--probe-radius", type=float, default=1.0, help="Probe radius in Å (default: 1.0)")
    parser.add_argument("--grid-spacing", type=float, default=0.5, help="Approximate grid spacing in Å (default: 0.5)")
    parser.add_argument("--chunk-size", type=int, default=65536)
    args = parser.parse_args()
    if args.probe_radius <= 0 or args.grid_spacing <= 0 or args.chunk_size <= 0:
        parser.error("probe radius, grid spacing, and chunk size must be positive")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = [export_one(
        path.expanduser().resolve(), output,
        probe_radius_A=args.probe_radius,
        grid_spacing_A=args.grid_spacing,
        chunk_size=args.chunk_size,
    ) for path in args.poscar]
    (output / "samples.json").write_text(
        json.dumps({
            "description": "VESTA XSF volumes of geometric space accessible to a spherical probe.",
            "mask_values": {"0": "blocked: closer than probe radius to an atom", "1": "accessible to probe center"},
            "display_in_vesta": "Create/show the grid isosurface at 0.5 to display all accessible regions.",
            "caveat": "This is geometric free space, not ELF/electron density or an electronic electride-dimensionality label.",
            "samples": rows,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "samples.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} XSF files and sample metadata to {output}")


if __name__ == "__main__":
    main()
