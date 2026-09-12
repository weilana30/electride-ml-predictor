"""Build graph inputs for the A2BC2 ELFmax dataset."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Composition, Structure


CUTOFF = 6.0  # Angstrom
NUM_DISTANCE_CENTERS = 40
OCTAHEDRAL_FRACTIONAL_COORDS = np.array([0.0, 0.0, 0.5])


NODE_FEATURE_NAMES = [
    "atomic_number",
    "electronegativity",
    "atomic_radius",
    "atomic_mass",
    "period",
    "group",
    "mendeleev_number",
    "melting_point",
    "boiling_point",
    "solid_density",
    "is_metal",
    "is_transition_metal",
    "is_post_transition_metal",
    "is_metalloid",
    "block_s",
    "block_p",
    "block_d",
    "block_f",
]


def gaussian_distance_expansion(distances, centers, width):
    return np.exp(-((distances - centers) ** 2) / width**2)


def structure_to_graph(structure):
    """Create a graph dictionary with elemental node and distance edge features."""
    edge_sources = []
    edge_targets = []
    edge_distances = []
    edge_images = []

    for source, neighbors in enumerate(structure.get_all_neighbors(r=CUTOFF)):
        for neighbor in neighbors:
            edge_sources.append(source)
            edge_targets.append(neighbor.index)
            edge_distances.append(neighbor.nn_distance)
            edge_images.append(neighbor.image)

    # Add one ghost node at the consistent A2BC2 octahedral interstitial site.
    ghost_index = len(structure)
    ghost_cart_coords = structure.lattice.get_cartesian_coords(
        OCTAHEDRAL_FRACTIONAL_COORDS
    )
    ghost_neighbors = structure.get_sites_in_sphere(
        ghost_cart_coords, CUTOFF, include_index=True
    )
    nearest_six = sorted(ghost_neighbors, key=lambda neighbor: neighbor.nn_distance)[:6]
    nearest_distances = np.array(
        [neighbor.nn_distance for neighbor in nearest_six], dtype=float
    )
    void_features = np.array(
        [
            1.0,
            1.0,
            nearest_distances.min(),
            nearest_distances.mean(),
        ],
        dtype=float,
    )

    for neighbor in ghost_neighbors:
        edge_sources.extend([ghost_index, neighbor.index])
        edge_targets.extend([neighbor.index, ghost_index])
        edge_distances.extend([neighbor.nn_distance, neighbor.nn_distance])
        image = np.asarray(neighbor.image, dtype=np.int64)
        edge_images.extend([image, -image])

    edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)
    edge_distances = np.asarray(edge_distances, dtype=float).reshape(-1, 1)
    edge_images = np.asarray(edge_images, dtype=np.int64)

    node_features = []
    for site in structure:
        element = site.specie
        node_features.append(
            [
                element.Z,
                element.X if element.X is not None else 0.0,
                float(element.atomic_radius) if element.atomic_radius is not None else 0.0,
                float(element.atomic_mass),
                element.row,
                element.group if element.group is not None else 0.0,
                element.mendeleev_no,
                float(element.melting_point) if element.melting_point is not None else 0.0,
                float(element.boiling_point) if element.boiling_point is not None else 0.0,
                float(element.density_of_solid)
                if element.density_of_solid is not None
                else 0.0,
                float(element.is_metal),
                float(element.is_transition_metal),
                float(element.is_post_transition_metal),
                float(element.is_metalloid),
                float(element.block == "s"),
                float(element.block == "p"),
                float(element.block == "d"),
                float(element.block == "f"),
            ]
        )
    atom_features = np.asarray(node_features, dtype=float)
    node_features = np.vstack(
        [atom_features, np.zeros((1, atom_features.shape[1]), dtype=float)]
    )
    all_void_features = np.zeros((ghost_index + 1, 4), dtype=float)
    all_void_features[ghost_index] = void_features

    centers = np.linspace(0, CUTOFF, NUM_DISTANCE_CENTERS)
    width = centers[1] - centers[0]
    edge_distance_features = gaussian_distance_expansion(
        edge_distances, centers, width
    )

    return {
        "formula": structure.composition.reduced_formula,
        "num_nodes": ghost_index + 1,
        "num_atoms": len(structure),
        "ghost_index": ghost_index,
        "lattice": structure.lattice.matrix,
        "frac_coords": structure.frac_coords,
        "cart_coords": structure.cart_coords,
        "ghost_frac_coords": OCTAHEDRAL_FRACTIONAL_COORDS,
        "ghost_cart_coords": ghost_cart_coords,
        "atom_features": atom_features,
        "node_features": node_features,
        "void_features": all_void_features,
        "edge_index": edge_index,
        "edge_distances": edge_distances,
        "edge_distance_features": edge_distance_features,
        "edge_images": edge_images,
    }


def load_labels(data_path):
    data = pd.read_csv(data_path)
    required_columns = {"chemical_formula", "elf_max"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {data_path.name}: {names}")

    labels = data[["chemical_formula", "elf_max"]].copy()
    labels.columns = ["formula", "ELFmax"]

    if labels.isna().any().any():
        raise ValueError("formula and ELFmax must not contain missing values")
    if labels["formula"].duplicated().any():
        duplicates = labels.loc[labels["formula"].duplicated(), "formula"].tolist()
        raise ValueError(f"Duplicate formulas in {data_path.name}: {duplicates}")

    labels["ELFmax"] = pd.to_numeric(labels["ELFmax"])
    labels["cif_filename"] = labels["formula"] + ".cif"
    return labels


def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data(1).csv"
    source_cif_dir = base_dir / "structures" / "elfcar"
    output_cif_dir = base_dir / "cifs"
    processed_dir = base_dir / "processed"

    labels = load_labels(data_path)
    source_paths = [source_cif_dir / name for name in labels["cif_filename"]]
    missing_cifs = [path.name for path in source_paths if not path.is_file()]
    if missing_cifs:
        raise FileNotFoundError(f"Missing CIF files: {missing_cifs}")

    graphs = []
    for formula, source_path in zip(labels["formula"], source_paths):
        structure = Structure.from_file(source_path)
        graph = structure_to_graph(structure)
        expected_composition = Composition(formula).reduced_composition
        actual_composition = structure.composition.reduced_composition
        if actual_composition != expected_composition:
            raise ValueError(
                f"Composition mismatch for {source_path.name}: "
                f"CSV has {formula}, CIF has {structure.composition.formula}"
            )
        graphs.append(graph)

    processed_dir.mkdir(exist_ok=True)
    # NumPy arrays match the notebook; load this file with weights_only=False.
    torch.save(graphs, processed_dir / "graphs.pt")

    output_cif_dir.mkdir(exist_ok=True)
    for source_path in source_paths:
        shutil.copy2(source_path, output_cif_dir / source_path.name)

    labels.to_csv(base_dir / "labels.csv", index=False)

    print(f"Created {len(graphs)} graphs in {processed_dir / 'graphs.pt'}")
    print(f"Copied {len(source_paths)} CIFs to {output_cif_dir}")
    print(f"Created {len(labels)} label rows in {base_dir / 'labels.csv'}")


if __name__ == "__main__":
    main()
