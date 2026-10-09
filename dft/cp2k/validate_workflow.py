#!/usr/bin/env python3
"""Check structural candidates and generated CP2K inputs without running DFT."""

from __future__ import annotations

import csv
import itertools
import re
from pathlib import Path

import numpy as np

from find_voids import _periodic_distance, find_candidates, read_cif
from generate_workflow import CP2K_ROOT, MATERIALS, PSEUDO, ROOT, frac_positions, read_qe_structure

SOURCE_CIFS = {"Ba2As": "mp-2768", "Ba3CrN3": "mp-12905", "BaSn3": "mp-7759",
               "Ca2N": "mp-2686", "Cu2SiSe3": "mp-15896", "MgS": "mp-1315"}
MEMORY_GB = {"cutoff/300": 16, "cutoff/400": 16, "cutoff/500": 20, "cutoff/600": 24,
             "kpoints/3": 16, "kpoints/5": 20, "kpoints/7": 24, "kpoints/9": 32, "basis/tzvp": 24}


def near_images(frac, atoms, lattice):
    distances = []
    for atom in atoms:
        for shift in itertools.product((-1, 0, 1), repeat=3):
            delta = frac - (np.asarray(atom.frac) + shift)
            distances.append((float(np.linalg.norm(delta @ lattice)), atom.symbol))
    return sorted(distances)


def validate_material(material):
    source = ROOT / "dft" / material / "PBE" / "relax" / "pbe_vc-relax.in"
    lattice, cart_atoms = read_qe_structure(source)
    atom_sites = frac_positions(lattice, cart_atoms)
    cif_lattice, cif_atoms = read_cif(ROOT / "data" / "cifs" / f"{SOURCE_CIFS[material]}.cif")
    assert len(atom_sites) == len(cif_atoms), f"{material}: QE input/CIF atom count mismatch"
    assert np.allclose(lattice, cif_lattice, atol=1e-7), f"{material}: QE input/CIF lattice mismatch"
    for atom in cif_atoms:
        d = [_periodic_distance(np.asarray(atom.frac), np.asarray(other.frac), lattice)
             for other in atom_sites if other.symbol == atom.symbol]
        assert d and min(d) < 1e-5, f"{material}: CIF atom not represented in QE positions: {atom}"

    sites = find_candidates(lattice, atom_sites)
    assert sites, f"{material}: no structural void candidates found"
    for site in sites:
        frac = site["frac"]
        assert np.all(frac >= 0) and np.all(frac < 1), f"{material}: candidate outside unit cell: {frac}"
        neighbors = near_images(frac, atom_sites, lattice)
        radius = neighbors[0][0]
        assert abs(radius - site["clearance_radius_A"]) < 1e-6, f"{material}: incorrect candidate clearance"
        assert sum(abs(d - radius) < 1e-4 for d, _ in neighbors) >= 4, f"{material}: candidate is not a periodic Voronoi vertex"
        assert np.allclose(site["cart"], frac @ lattice, atol=1e-7), f"{material}: Cartesian/fractional coordinates disagree"

    structure = CP2K_ROOT / material / "structure"
    xyz = (structure / "structure.xyz").read_text().splitlines()
    assert int(xyz[0]) == len(cart_atoms) and len(xyz) == len(cart_atoms) + 2
    with (structure / "void_candidates.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(sites)
    for group in ("cutoff", "kpoints"):
        paths = sorted((CP2K_ROOT / material / group).glob("*/cp2k.inp"))
        assert len(paths) == 4, f"{material}: expected four {group} inputs, found {len(paths)}"
        for path in paths:
            text = path.read_text()
            assert "PBE" in text and "GTH-PBE-q" in text
            assert "&ELF_CUBE" in text and "STRIDE 1 1 1" in text
            assert "SMEAR" not in text.upper(), f"{path}: smearing was enabled"
            coord_text = re.search(r"&COORD\s*(.*?)\s*&END COORD", text, re.S).group(1)
            coord_rows = [line for line in coord_text.splitlines() if line.strip()]
            assert len(coord_rows) == len(cart_atoms), f"{path}: coordinate count mismatch"
            assert "--time=03:00:00" in (path.parent / "submit.slurm").read_text()
            assert f"--mem={MEMORY_GB[f'{group}/{path.parent.name}']}G" in (path.parent / "submit.slurm").read_text()
    basis_input = CP2K_ROOT / material / "basis" / "tzvp" / "cp2k.inp"
    basis_text = basis_input.read_text()
    assert "BASIS_SET_FILE_NAME BASIS_MOLOPT_UCL" in basis_text
    assert all(f"BASIS_SET TZVP-MOLOPT-SR-GTH-q{PSEUDO[symbol][1].rsplit('q', 1)[1]}" in basis_text for symbol in {a.symbol for a in atom_sites})
    assert "SMEAR" not in basis_text.upper()
    assert "--time=03:00:00" in (basis_input.parent / "submit.slurm").read_text()
    assert f"--mem={MEMORY_GB['basis/tzvp']}G" in (basis_input.parent / "submit.slurm").read_text()
    print(f"PASS {material}: {len(atom_sites)} QE/CIF atoms agree; {len(sites)} valid Voronoi candidates; 9 fixed-geometry CP2K inputs checked")


def main():
    for material in MATERIALS:
        validate_material(material)
    print("PASS all structural and generated-input checks (CP2K runtime not tested on this workstation)")


if __name__ == "__main__":
    main()
