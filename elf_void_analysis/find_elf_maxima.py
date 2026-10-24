#!/usr/bin/env python3
"""Sample ELFCAR values at the void-marker coordinates in a VESTA CIF."""

import argparse

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Elfcar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cif", help="CIF containing Void_* markers")
    parser.add_argument("elfcar", help="Matching VASP ELFCAR file")
    parser.add_argument("--void", help="Only analyze one marker, for example Void_1")
    args = parser.parse_args()

    structure = Structure.from_file(args.cif)
    elfcar = Elfcar.from_file(args.elfcar)
    if not np.allclose(structure.lattice.matrix, elfcar.structure.lattice.matrix, atol=1e-3):
        parser.error("CIF and ELFCAR lattice parameters do not match")

    all_markers = [site for site in structure if site.label.startswith("Void_")]
    if not all_markers:
        parser.error(f"No Void_* markers found in {args.cif}")
    markers = all_markers
    if args.void:
        markers = [site for site in markers if site.label == args.void]
        if not markers:
            parser.error(f"No void marker labeled {args.void!r} in {args.cif}")

    values = np.asarray(elfcar.data["total"])
    shape = np.asarray(values.shape)

    for marker in markers:
        frac = marker.frac_coords % 1.0
        grid_position = frac * shape
        lower = np.floor(grid_position).astype(int)
        weight = grid_position - lower
        elf_value = 0.0
        for offset in np.ndindex(2, 2, 2):
            offset = np.asarray(offset)
            index = tuple((lower + offset) % shape)
            corner_weight = np.prod(np.where(offset, weight, 1.0 - weight))
            elf_value += corner_weight * values[index]

        cart = structure.lattice.get_cartesian_coords(frac)
        print(
            f"{marker.label}: ELF at void center={elf_value:.6f}; "
            f"fractional=({frac[0]:.5f}, {frac[1]:.5f}, {frac[2]:.5f}); "
            f"Cartesian=({cart[0]:.5f}, {cart[1]:.5f}, {cart[2]:.5f}) angstrom"
        )


if __name__ == "__main__":
    main()
