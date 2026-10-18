"""Find candidate voids in a periodic crystal structure."""

import argparse
from math import ceil

import numpy as np
from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
from pymatgen.core import Structure
from scipy.ndimage import maximum_filter


def find_voronoi_voids(structure):
    """Find candidate interstitial sites with Voronoi analysis."""
    return [
        {
            "coords": v.site.frac_coords,
            "distance": min(v.site.distance(atom) for atom in structure),
            "method": "Voronoi",
        }
        for v in VoronoiInterstitialGenerator().generate(structure, insert_species=["H"])
    ]


def find_furthest_points(structure):
    """Find grid points that are local maxima in distance from the atoms."""
    shape = tuple(max(1, ceil(length / 0.5)) for length in structure.lattice.abc)
    axes = [np.arange(n) / n for n in shape]
    frac = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    cart = structure.lattice.get_cartesian_coords(frac.reshape(-1, 3))
    distance = structure.lattice.get_all_distances(cart, structure.cart_coords).min(axis=1).reshape(shape)
    peaks = (distance == maximum_filter(distance, size=3, mode="wrap")) & (distance >= 1.5)
    return [
        {"coords": i / np.asarray(shape), "distance": distance[tuple(i)], "method": "distance"}
        for i in np.argwhere(peaks)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure_file", help="POSCAR, CIF, or another pymatgen-readable structure")
    structure = Structure.from_file(parser.parse_args().structure_file)

    voids = []
    for candidate in find_voronoi_voids(structure) + find_furthest_points(structure):
        for existing in voids:
            if structure.lattice.get_distance_and_image(existing["coords"], candidate["coords"])[0] < 0.5:
                existing["method"] += "+" + candidate["method"]
                break
        else:
            voids.append(candidate)

    print(f"Found {len(voids)} candidate voids")
    for i, void in enumerate(voids, 1):
        xyz = ", ".join(f"{x:.3f}" for x in void["coords"])
        print(f"{i}: ({xyz})  nearest atom={void['distance']:.2f} A  [{void['method']}]")


if __name__ == "__main__":
    main()
