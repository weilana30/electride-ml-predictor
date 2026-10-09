#!/usr/bin/env python3
"""Find periodic, structure-only interstitial cavity candidates by Voronoi geometry.

Input is a CIF. Candidate coordinates are Voronoi vertices in the periodic
crystal; they are not assigned electronic meaning. The script reports one
representative per crystallographic symmetry orbit and the nearest-atom
clearance radius for each representative.

Dependencies: numpy, scipy, spglib (see requirements.txt).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import spglib
from scipy.spatial import Voronoi, QhullError


@dataclass(frozen=True)
class Atom:
    symbol: str
    frac: tuple[float, float, float]


def _clean(value: str) -> str:
    return re.sub(r"(?<=\d)\([^)]*\)$", "", value.strip("'\""))


def _symbol(value: str) -> str:
    match = re.match(r"([A-Z][a-z]?)", value.strip("'\""))
    if not match:
        raise ValueError(f"Cannot read element symbol from CIF value {value!r}")
    return match.group(1)


def read_cif(path: Path):
    """Read the unit cell and atom sites from a conventional fractional CIF."""
    lines = path.read_text(encoding="utf-8").splitlines()
    scalar: dict[str, str] = {}
    atoms: list[Atom] = []
    i = 0
    wanted = {"_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"}
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        tokens = shlex.split(line, comments=False)
        if len(tokens) >= 2 and tokens[0].startswith("_"):
            scalar[tokens[0].lower()] = tokens[1]
            i += 1
            continue
        if line.lower() != "loop_":
            i += 1
            continue
        i += 1
        headers = []
        while i < len(lines) and lines[i].strip().startswith("_"):
            headers.append(lines[i].strip().split()[0].lower())
            i += 1
        rows = []
        while i < len(lines):
            rowline = lines[i].strip()
            if not rowline or rowline.startswith("#"):
                i += 1
                continue
            if rowline.lower() == "loop_" or rowline.startswith("_") or rowline.startswith("data_"):
                break
            row = shlex.split(rowline, comments=False)
            if len(row) < len(headers):
                raise ValueError(f"Incomplete CIF loop row at line {i + 1}")
            rows.append(row[:len(headers)])
            i += 1
        if wanted.issubset(headers):
            ix = {key: headers.index(key) for key in headers}
            for row in rows:
                occ = float(_clean(row[ix["_atom_site_occupancy"]])) if "_atom_site_occupancy" in ix else 1.0
                if occ > 0:
                    atoms.append(Atom(_symbol(row[ix["_atom_site_type_symbol"]]), tuple(float(_clean(row[ix[k]])) % 1.0 for k in ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"))))
        # The terminating line is intentionally handled again by the loop.
    needed = ("_cell_length_a", "_cell_length_b", "_cell_length_c", "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")
    missing = [key for key in needed if key not in scalar]
    if missing or not atoms:
        raise ValueError(f"CIF is missing cell fields/sites: {missing or 'atom-site loop'}")
    a, b, c = (float(_clean(scalar[k])) for k in needed[:3])
    alpha, beta, gamma = (math.radians(float(_clean(scalar[k]))) for k in needed[3:])
    avec = (a, 0.0, 0.0)
    bvec = (b * math.cos(gamma), b * math.sin(gamma), 0.0)
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
    cz2 = c*c - cx*cx - cy*cy
    if cz2 <= 0:
        raise ValueError("Invalid CIF cell geometry")
    # spglib and the conversion below use lattice vectors as rows.
    lattice = np.array([avec, bvec, (cx, cy, math.sqrt(cz2))], dtype=float)
    return lattice, atoms


def _cart(frac: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    return frac @ lattice


def _periodic_distance(frac_a: np.ndarray, frac_b: np.ndarray, lattice: np.ndarray) -> float:
    delta = frac_a - frac_b
    # Search adjacent images; this is exact for the cells in this dataset and
    # handles non-orthogonal cells without fractional-coordinate shortcuts.
    best = float("inf")
    for shift in itertools.product((-1, 0, 1), repeat=3):
        best = min(best, float(np.linalg.norm((delta + shift) @ lattice)))
    return best


def clearance(frac: np.ndarray, atom_fracs: np.ndarray, lattice: np.ndarray) -> tuple[float, int, float]:
    distances = [(_periodic_distance(frac, atom, lattice), idx) for idx, atom in enumerate(atom_fracs)]
    radius, nearest = min(distances)
    return radius, nearest, radius


def find_candidates(lattice: np.ndarray, atoms: list[Atom], symprec: float = 1e-3, min_radius: float = 0.7):
    atom_fracs = np.array([atom.frac for atom in atoms], dtype=float)
    species = {symbol: i + 1 for i, symbol in enumerate(sorted({a.symbol for a in atoms}))}
    numbers = [species[a.symbol] for a in atoms]
    symmetry = spglib.get_symmetry((lattice, atom_fracs, numbers), symprec=symprec)
    if symmetry is None:
        raise ValueError("spglib could not determine crystal symmetry")

    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float)
    tiled_frac = np.concatenate([atom_fracs + shift for shift in shifts], axis=0)
    tiled_cart = _cart(tiled_frac, lattice)
    try:
        vor = Voronoi(tiled_cart)
    except QhullError as exc:
        raise ValueError(f"Periodic Voronoi tessellation failed: {exc}") from exc

    candidates = []
    for vertex in vor.vertices:
        raw_frac = np.linalg.solve(lattice.T, vertex).astype(float)
        # Only the vertices of the central unit cell are physically relevant.
        # Wrapping vertices from the finite outer image cloud would turn
        # boundary artifacts into apparent sites inside the central cell.
        if np.any(raw_frac < -1e-8) or np.any(raw_frac >= 1.0 - 1e-8):
            continue
        frac = raw_frac % 1.0
        frac[np.isclose(frac, 1.0, atol=1e-8)] = 0.0
        radius, nearest_idx, _ = clearance(frac, atom_fracs, lattice)
        if radius + 1e-8 >= min_radius:
            candidates.append((frac, radius, nearest_idx))
    if not candidates:
        return []

    # Remove numerical duplicates from neighboring periodic tiles before
    # symmetry grouping.
    unique = []
    coord_tol = max(symprec * 2, 1e-4)
    for candidate in sorted(candidates, key=lambda item: (-item[1], tuple(item[0]))):
        if not any(_periodic_distance(candidate[0], prior[0], lattice) < coord_tol for prior in unique):
            unique.append(candidate)

    rotations = symmetry["rotations"]
    translations = symmetry["translations"]
    representatives = []
    for candidate in unique:
        frac, radius, nearest_idx = candidate
        equivalent = False
        for rotation, translation in zip(rotations, translations):
            mapped = (rotation @ frac + translation) % 1.0
            if any(_periodic_distance(mapped, prior[0], lattice) < coord_tol for prior in representatives):
                equivalent = True
                break
        if equivalent:
            continue
        representatives.append(candidate)

    out = []
    for index, (frac, radius, nearest_idx) in enumerate(representatives, start=1):
        nearest_distances = []
        for atom_idx, atom in enumerate(atoms):
            for shift in itertools.product((-1, 0, 1), repeat=3):
                image_frac = np.asarray(atom.frac) + shift
                distance = float(np.linalg.norm((frac - image_frac) @ lattice))
                nearest_distances.append((distance, atom_idx, atom.symbol, shift))
        nearest_distances.sort()
        shell_distance = nearest_distances[0][0]
        shell = [item for item in nearest_distances if item[0] <= shell_distance + 1e-4][:8]
        orbit_points = []
        for rotation, translation in zip(rotations, translations):
            mapped = (rotation @ frac + translation) % 1.0
            if not any(_periodic_distance(mapped, prior, lattice) < coord_tol for prior in orbit_points):
                orbit_points.append(mapped)
        out.append({
            "site": index,
            "frac": frac,
            "cart": _cart(frac, lattice),
            "clearance_radius_A": radius,
            "nearest_species": atoms[nearest_idx].symbol,
            "nearest_distance_A": radius,
            "nearest_shell": ";".join(f"{sym}{tuple(shift)}:{dist:.3f}" for dist, _, sym, shift in shell),
            "multiplicity": len(orbit_points),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cif", type=Path)
    parser.add_argument("--output", type=Path, help="CSV output (default: <cif>.voids.csv)")
    parser.add_argument("--min-radius", type=float, default=0.7, help="Minimum nearest-atom clearance in angstrom (default: 0.7)")
    parser.add_argument("--symprec", type=float, default=1e-3, help="Symmetry tolerance in angstrom (default: 0.001)")
    args = parser.parse_args()
    lattice, atoms = read_cif(args.cif)
    sites = find_candidates(lattice, atoms, args.symprec, args.min_radius)
    output = args.output or args.cif.with_suffix(".voids.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fields = ["site", "frac_x", "frac_y", "frac_z", "cart_x_A", "cart_y_A", "cart_z_A", "clearance_radius_A", "nearest_species", "nearest_distance_A", "nearest_shell", "multiplicity"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for site in sites:
            row = {key: site[key] for key in ("site", "clearance_radius_A", "nearest_species", "nearest_distance_A", "nearest_shell", "multiplicity")}
            row.update(dict(zip(("frac_x", "frac_y", "frac_z"), site["frac"])))
            row.update(dict(zip(("cart_x_A", "cart_y_A", "cart_z_A"), site["cart"])))
            writer.writerow(row)
    print(f"{args.cif}: {len(atoms)} atoms; {len(sites)} symmetry-distinct candidates; wrote {output}")
    for site in sites:
        f = site["frac"]
        c = site["cart"]
        print(f"  site {site['site']:2d}: frac=({f[0]:.5f}, {f[1]:.5f}, {f[2]:.5f}) cart=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}) A; r_clear={site['clearance_radius_A']:.3f} A; nearest={site['nearest_shell']}")


if __name__ == "__main__":
    main()
