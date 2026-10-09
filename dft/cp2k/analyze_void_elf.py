#!/usr/bin/env python3
"""Find ELF cube maxima near structural void candidates (structure remains input)."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903


def read_cube(path: Path):
    with path.open(encoding="utf-8") as handle:
        handle.readline()
        handle.readline()
        atomline = handle.readline().split()
        natoms = abs(int(atomline[0]))
        origin = np.asarray([float(x) for x in atomline[1:4]], dtype=float)
        shape, steps = [], []
        for _ in range(3):
            fields = handle.readline().split()
            shape.append(abs(int(fields[0])))
            steps.append([float(x) for x in fields[1:4]])
        for _ in range(natoms):
            handle.readline()
        values = np.fromfile(handle, sep=" ", dtype=float)
    if values.size != math.prod(shape):
        raise ValueError(f"{path}: expected {math.prod(shape)} cube values, found {values.size}")
    return origin, np.asarray(steps), np.asarray(shape), values.reshape(shape)


def maxima_near_void(cube_path: Path, candidate_path: Path, radius_A: float):
    origin, steps, shape, values = read_cube(cube_path)
    cell_bohr = steps * shape[:, None]
    cell_A = cell_bohr * BOHR_TO_ANGSTROM
    inv_cell = np.linalg.inv(cell_A)
    index_grid = np.indices(values.shape, dtype=float).reshape(3, -1).T
    cart_grid_A = (origin + index_grid @ steps) * BOHR_TO_ANGSTROM
    with candidate_path.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    out = []
    for site in candidates:
        center = np.asarray([float(site[k]) for k in ("cart_x_A", "cart_y_A", "cart_z_A")])
        delta = (cart_grid_A - center) @ inv_cell
        # The neighboring periodic images make the radius test correct for
        # triclinic cells and voids close to a unit-cell boundary.
        distance2 = np.full(len(delta), np.inf)
        for shift in itertools.product((-1, 0, 1), repeat=3):
            image = (delta + shift) @ cell_A
            distance2 = np.minimum(distance2, np.einsum("ij,ij->i", image, image))
        mask = distance2 <= radius_A**2
        if not np.any(mask):
            raise ValueError(f"No cube grid points within {radius_A:g} A of structural site {site['site']}")
        flat = np.flatnonzero(mask)
        idx = flat[np.argmax(values.ravel()[flat])]
        ijk = np.asarray(np.unravel_index(idx, values.shape), dtype=int)
        xyz = (origin + ijk @ steps) * BOHR_TO_ANGSTROM
        out.append({"site": int(site["site"]), "clearance_radius_A": float(site["clearance_radius_A"]),
                    "void_coordinate_A": center, "elf_maximum": float(values.ravel()[idx]),
                    "elf_maximum_coordinate_A": xyz, "search_radius_A": radius_A})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf_cube", type=Path)
    parser.add_argument("void_candidates_csv", type=Path)
    parser.add_argument("--radius", type=float, default=1.5, help="Local ELF maximum search radius in angstrom")
    parser.add_argument("--site", type=int, help="Analyze only this structural candidate index")
    parser.add_argument("--output", type=Path, help="Optional CSV report path")
    args = parser.parse_args()
    if args.radius <= 0:
        parser.error("--radius must be positive")
    results = maxima_near_void(args.elf_cube, args.void_candidates_csv, args.radius)
    if args.site is not None:
        results = [row for row in results if row["site"] == args.site]
        if not results:
            parser.error(f"site {args.site} was not found in {args.void_candidates_csv}")
    rows = []
    for item in results:
        center, peak = item["void_coordinate_A"], item["elf_maximum_coordinate_A"]
        row = {key: value for key, value in item.items() if key not in ("void_coordinate_A", "elf_maximum_coordinate_A")}
        row.update(dict(zip(("void_x_A", "void_y_A", "void_z_A"), center)))
        row.update(dict(zip(("peak_x_A", "peak_y_A", "peak_z_A"), peak)))
        rows.append(row)
    outpath = args.output or args.elf_cube.with_suffix(".void_elf.csv")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"site {row['site']}: ELF max={row['elf_maximum']:.6f} within {args.radius:g} A; void=({row['void_x_A']:.3f}, {row['void_y_A']:.3f}, {row['void_z_A']:.3f}) A")
    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()
