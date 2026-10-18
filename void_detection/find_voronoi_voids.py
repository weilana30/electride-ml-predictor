"""Print Voronoi interstitial candidates from a POSCAR or CIF structure."""

import argparse

from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
from pymatgen.core import Structure

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "structure_file",
    help="Path to a POSCAR, CIF, or other pymatgen-readable structure",
)
args = parser.parse_args()

structure = Structure.from_file(args.structure_file)
generator = VoronoiInterstitialGenerator()
voids = list(generator.generate(structure, insert_species=["H"]))

print(f"Found {len(voids)} Voronoi candidate sites")
for number, void in enumerate(voids, start=1):
    coords = ", ".join(f"{value:.5f}" for value in void.site.frac_coords)
    clearance = min(void.site.distance(atom) for atom in structure)
    print(
        f"{number:>3}: frac=({coords}), nearest_atom={clearance:.3f} A, "
        f"multiplicity={void.multiplicity}"
    )
