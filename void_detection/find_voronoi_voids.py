"""Find candidate voids in a periodic crystal structure."""

import argparse
from math import ceil

import numpy as np
from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize


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
    distance = structure.lattice.get_all_distances(
        frac.reshape(-1, 3), structure.frac_coords
    ).min(axis=1).reshape(shape)
    peaks = (distance == maximum_filter(distance, size=3, mode="wrap")) & (distance >= 1.5)
    return [
        {"coords": i / np.asarray(shape), "distance": distance[tuple(i)], "method": "distance"}
        for i in np.argwhere(peaks)
    ]


def refine_void_center(structure, fractional_coords):
    """Move a candidate to the nearby maximum of its distance from all atoms."""
    lattice = structure.lattice
    start_cart = lattice.get_cartesian_coords(fractional_coords)

    def negative_clearance(cart_coords):
        frac_coords = lattice.get_fractional_coords(cart_coords) % 1.0
        distances = lattice.get_all_distances(
            np.asarray(frac_coords).reshape(1, 3), structure.frac_coords
        )
        return -float(distances.min())

    step = 0.1
    initial_simplex = np.vstack(
        [start_cart, *(start_cart + step * np.eye(3))]
    )
    result = minimize(
        negative_clearance,
        start_cart,
        method="Nelder-Mead",
        options={
            "initial_simplex": initial_simplex,
            "xatol": 1e-4,
            "fatol": 1e-6,
            "maxiter": 500,
        },
    )
    refined_coords = lattice.get_fractional_coords(result.x) % 1.0
    return refined_coords, -float(result.fun)


def find_candidate_voids(structure, symprec=0.01, merge_tolerance=0.2):
    """Return refined, symmetry-unique void centers as JSON-friendly dictionaries."""
    symmetry = SpacegroupAnalyzer(structure, symprec=symprec).get_symmetry_operations(
        cartesian=False
    )

    candidates = find_voronoi_voids(structure) + find_furthest_points(structure)
    refined_candidates = []
    for candidate in candidates:
        coords, clearance = refine_void_center(structure, candidate["coords"])
        refined_candidates.append(
            {"coords": coords, "distance": clearance, "method": candidate["method"]}
        )

    voids = []
    for candidate in refined_candidates:
        coords = np.asarray(candidate["coords"], dtype=float) % 1.0
        for existing in voids:
            equivalent = any(
                structure.lattice.get_distance_and_image(
                    operation.operate(coords) % 1.0, existing["fractional_coords"]
                )[0]
                < merge_tolerance
                for operation in symmetry
            )
            if equivalent:
                existing["methods"].add(candidate["method"])
                existing["clearance_angstrom"] = max(
                    existing["clearance_angstrom"], float(candidate["distance"])
                )
                break
        else:
            voids.append(
                {
                    "fractional_coords": coords.tolist(),
                    "clearance_angstrom": float(candidate["distance"]),
                    "methods": {candidate["method"]},
                }
            )

    for void in voids:
        void["methods"] = sorted(void["methods"])
    return voids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure_file", help="POSCAR, CIF, or another pymatgen-readable structure")
    structure = Structure.from_file(parser.parse_args().structure_file)
    voids = find_candidate_voids(structure)

    print(f"Found {len(voids)} symmetry-unique candidate voids")
    for i, void in enumerate(voids, 1):
        xyz = ", ".join(f"{x:.3f}" for x in void["fractional_coords"])
        method = "+".join(void["methods"])
        print(
            f"{i}: ({xyz})  clearance={void['clearance_angstrom']:.2f} A  [{method}]"
        )


if __name__ == "__main__":
    main()
