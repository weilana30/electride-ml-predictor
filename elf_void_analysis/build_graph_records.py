#!/usr/bin/env python3
"""Build a model-ready periodic atom/void graph record from one POSCAR.

The record is JSON-serializable (arrays are represented as nested lists) so it
can be inspected independently and converted to HDF5 by the dataset writer.
ELFCAR is deliberately not read here: this module creates inputs only.

Edges are stored as directed source-to-destination relations. Periodic image
enumeration retains every destination image inside the edge cutoff (not just
the nearest image). ``image_shift`` is the integer translation applied to the
destination fractional coordinate. Atom and void node indices are zero-based
and refer to the corresponding coordinate arrays. Distances are in Angstrom;
coordinates are fractional. ``void_atom_*`` edges are void-source to atom-
destination. A consumer that wants atom-to-void message passing should reverse
each edge and negate its image shift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Element, Structure

try:  # Support both ``python -m elf_void_analysis...`` and direct script use.
    from void_detection.find_voronoi_voids import find_candidate_voids
except ModuleNotFoundError:  # pragma: no cover - exercised in direct invocation
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from void_detection.find_voronoi_voids import find_candidate_voids


ATOM_FEATURE_NAMES = [
    "atomic_number", "electronegativity", "atomic_radius_angstrom",
    "period", "group", "is_s_block", "is_p_block", "is_d_block", "is_f_block",
]
ATOM_FEATURE_VALIDITY_NAMES = [
    "electronegativity_valid", "atomic_radius_valid", "period_valid",
    "group_valid", "block_valid",
]
VOID_FEATURE_NAMES = [
    "clearance_angstrom", "nearest_atom_distance_angstrom",
    "atom_count_within_3A", "atom_count_within_4A", "atom_count_within_5A",
    "nearest_1_distance_angstrom", "nearest_2_distance_angstrom",
    "nearest_3_distance_angstrom", "nearest_4_distance_angstrom",
]
METHOD_NAMES = ("Voronoi", "distance")


def _value_and_validity(value: Any) -> tuple[float, float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0, 0.0
    if not np.isfinite(value):
        return 0.0, 0.0
    return value, 1.0


def _atom_features(element: Element) -> tuple[list[float], list[float]]:
    block = str(getattr(element, "block", "") or "").lower()
    electronegativity, x_valid = _value_and_validity(getattr(element, "X", None))
    radius, radius_valid = _value_and_validity(getattr(element, "atomic_radius", None))
    period, period_valid = _value_and_validity(getattr(element, "row", None))
    group, group_valid = _value_and_validity(getattr(element, "group", None))
    block_valid = float(block in {"s", "p", "d", "f"})
    features = [
        float(element.Z), electronegativity, radius, period, group,
        float(block == "s"), float(block == "p"), float(block == "d"),
        float(block == "f"),
    ]
    return features, [x_valid, radius_valid, period_valid, group_valid, block_valid]


def _periodic_edges(
    lattice, source_frac: np.ndarray, destination_frac: np.ndarray, cutoff: float,
    *, same_node_graph: bool = False,
) -> tuple[list[list[int]], list[float], list[list[int]]]:
    """Return directed edges within cutoff, retaining all periodic images.

    Enumerating images matters for small primitive cells: minimum-image-only
    logic loses multiple bonds to the same basis atom and periodic self-neighbor
    edges that GNNs need to represent coordination correctly.
    """
    indices: list[list[int]] = []
    distances: list[float] = []
    shifts: list[list[int]] = []
    reciprocal_lengths = np.linalg.norm(lattice.reciprocal_lattice.matrix, axis=1)
    for source, src in enumerate(source_frac):
        for destination, dst in enumerate(destination_frac):
            delta = np.asarray(dst) - np.asarray(src)
            # |delta_i + image_i| <= cutoff * |b_i| is a necessary bound.
            lo = np.ceil(-delta - cutoff * reciprocal_lengths).astype(int)
            hi = np.floor(-delta + cutoff * reciprocal_lengths).astype(int)
            for image_tuple in np.ndindex(*(hi - lo + 1)):
                image = lo + np.asarray(image_tuple, dtype=int)
                if same_node_graph and source == destination and np.all(image == 0):
                    continue
                vector = lattice.get_cartesian_coords(delta + image)
                distance = float(np.linalg.norm(vector))
                if distance <= cutoff + 1e-8:
                    indices.append([source, destination])
                    distances.append(distance)
                    shifts.append(image.tolist())
    return indices, distances, shifts


def _orbit_multiplicity(structure: Structure, frac: np.ndarray, symprec: float) -> int:
    """Count distinct symmetry images using the finder tolerance (0.2 A).

    Returns 1 if symmetry analysis fails. This is metadata/weighting information,
    not a node feature, since it depends on the chosen symmetry tolerance.
    """
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        ops = SpacegroupAnalyzer(structure, symprec=symprec).get_symmetry_operations(
            cartesian=False
        )
        unique: list[np.ndarray] = []
        for op in ops:
            image = np.asarray(op.operate(frac), dtype=float) % 1.0
            if not any(
                structure.lattice.get_distance_and_image(image, prior)[0] < 0.2
                for prior in unique
            ):
                unique.append(image)
        return max(1, len(unique))
    except Exception:
        return 1


def build_graph_record(
    structure_path: str | Path,
    *,
    sample_id: str | None = None,
    family_id: str | None = None,
    atom_cutoff_angstrom: float = 5.0,
    atom_void_cutoff_angstrom: float = 6.0,
    symprec: float = 0.01,
) -> dict[str, Any]:
    """Create one crystal's graph arrays and structural metadata from POSCAR."""
    path = Path(structure_path)
    structure = Structure.from_file(str(path))
    atom_frac = np.asarray(structure.frac_coords, dtype=float) % 1.0
    atom_elements = [str(site.specie.symbol) for site in structure]
    atom_feature_values, atom_feature_validity = zip(
        *[_atom_features(site.specie) for site in structure]
    )

    # Keep the established finder unchanged; its symmetry-reduced candidate
    # sites and method/clearance fields are carried into this record.
    candidates = find_candidate_voids(structure, symprec=symprec)
    void_frac = np.asarray(
        [v["fractional_coords"] for v in candidates], dtype=float
    ).reshape((-1, 3))
    clearances = np.asarray([v["clearance_angstrom"] for v in candidates], dtype=float)

    aa_idx, aa_dist, aa_shift = _periodic_edges(
        structure.lattice, atom_frac, atom_frac, atom_cutoff_angstrom,
        same_node_graph=True,
    )
    # Build void -> atom relations; the first index addresses voids and the
    # second addresses atoms. Keep this direction explicit in the key name.
    va_idx, va_dist, va_shift = _periodic_edges(
        structure.lattice, void_frac, atom_frac, atom_void_cutoff_angstrom
    )

    # These descriptors count distinct basis-site atoms using minimum-image
    # distances. They intentionally do not count translated images separately.
    if len(void_frac):
        void_atom_dist = structure.lattice.get_all_distances(void_frac, atom_frac)
    else:
        void_atom_dist = np.empty((0, len(atom_frac)), dtype=float)
    void_features = []
    multiplicities = []
    methods = []
    for i, candidate in enumerate(candidates):
        ds = np.sort(void_atom_dist[i])
        padded = np.pad(ds[:4], (0, max(0, 4 - len(ds))), constant_values=0.0)
        void_features.append([
            float(clearances[i]), float(ds[0]) if len(ds) else 0.0,
            float(np.count_nonzero(ds <= 3.0)),
            float(np.count_nonzero(ds <= 4.0)),
            float(np.count_nonzero(ds <= 5.0)),
            *[float(x) for x in padded],
        ])
        methods.append([int(name in candidate["methods"]) for name in METHOD_NAMES])
        multiplicities.append(_orbit_multiplicity(structure, void_frac[i], symprec))

    return {
        "sample_id": sample_id or path.parent.name or path.stem,
        "source_poscar": str(path),
        "family_id": family_id,
        "schema_version": "1.0",
        "units": {"coordinates": "fractional", "distance": "angstrom"},
        "edge_convention": (
            "directed source->destination; all periodic destination images within "
            "cutoff are retained; image_shift is added to destination fractional "
            "coordinates; void_atom_* is void->atom and reverse relation is derived "
            "by swapping endpoints and negating image_shift"
        ),
        "lattice_angstrom": np.asarray(structure.lattice.matrix, dtype=np.float32),
        "atoms": {
            "fractional_coords": atom_frac.astype(np.float32),
            "elements": atom_elements,
            "atomic_numbers": np.asarray([int(site.specie.Z) for site in structure], dtype=np.int16),
            "feature_names": ATOM_FEATURE_NAMES,
            "features": np.asarray(atom_feature_values, dtype=np.float32),
            "feature_validity_names": ATOM_FEATURE_VALIDITY_NAMES,
            "feature_validity": np.asarray(atom_feature_validity, dtype=np.uint8),
        },
        "voids": {
            "void_id": [f"{sample_id or path.parent.name or path.stem}:void:{i:04d}"
                        for i in range(len(void_frac))],
            "fractional_coords": void_frac.astype(np.float32),
            "feature_names": VOID_FEATURE_NAMES,
            "features": np.asarray(void_features, dtype=np.float32).reshape(-1, len(VOID_FEATURE_NAMES)),
            "methods": [candidate["methods"] for candidate in candidates],
            "method_names": list(METHOD_NAMES),
            "method_multi_hot": np.asarray(methods, dtype=np.uint8).reshape(-1, len(METHOD_NAMES)),
            "orbit_multiplicity": np.asarray(multiplicities, dtype=np.int16),
            "clearance_angstrom": clearances.astype(np.float32),
        },
        "edges": {
            "atom_atom_index": np.asarray(aa_idx, dtype=np.int32).T.reshape(2, -1),
            "atom_atom_distance_angstrom": np.asarray(aa_dist, dtype=np.float32),
            "atom_atom_image_shift": np.asarray(aa_shift, dtype=np.int16).reshape(-1, 3),
            "void_atom_index": np.asarray(va_idx, dtype=np.int32).T.reshape(2, -1),
            "void_atom_distance_angstrom": np.asarray(va_dist, dtype=np.float32),
            "void_atom_image_shift": np.asarray(va_shift, dtype=np.int16).reshape(-1, 3),
        },
        "counts": {
            "n_atoms": len(atom_frac), "n_voids": len(void_frac),
            "n_atom_atom_edges": len(aa_idx), "n_void_atom_edges": len(va_idx),
        },
        "settings": {
            "atom_cutoff_angstrom": atom_cutoff_angstrom,
            "atom_void_cutoff_angstrom": atom_void_cutoff_angstrom,
            "symprec": symprec,
            "void_finder": "void_detection.find_voronoi_voids.find_candidate_voids",
            "void_neighbor_count_definition": (
                "counts distinct atom basis sites using minimum-image distance; "
                "periodic images of the same atom are not counted separately"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("poscar", help="POSCAR or pymatgen-readable structure file")
    parser.add_argument("--sample-id")
    parser.add_argument("--output", "-o", help="JSON output path (default: stdout)")
    parser.add_argument("--atom-cutoff", type=float, default=5.0)
    parser.add_argument("--atom-void-cutoff", type=float, default=6.0)
    parser.add_argument("--symprec", type=float, default=0.01)
    args = parser.parse_args()
    record = build_graph_record(
        args.poscar, sample_id=args.sample_id, atom_cutoff_angstrom=args.atom_cutoff,
        atom_void_cutoff_angstrom=args.atom_void_cutoff, symprec=args.symprec,
    )
    def json_ready(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: json_ready(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_ready(item) for item in value]
        return value

    rendered = json.dumps(json_ready(record), indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
