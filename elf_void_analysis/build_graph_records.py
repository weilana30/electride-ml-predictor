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
each edge and negate its image shift. ``void_void_*`` edges are directed
void-source to void-destination and include periodic images, with only the
zero-shift self-edge excluded.
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
VOID_SYMMETRY_TOLERANCE_ANGSTROM = 0.2
VOID_COUNT_RADII_ANGSTROM = (3.0, 4.0, 5.0)
VOID_NEAREST_COUNT = 4
# Version 2 counts every periodic atom image (version 1 counted each basis
# atom once by minimum image and padded missing nearest distances with 0).
VOID_FEATURE_DEFINITION = (
    "v2: counts and nearest distances over all periodic atom images from the "
    "void-atom edges; nearest distances beyond the edge cutoff are set to the cutoff"
)


def void_neighbor_features(
    n_voids: int, void_index: np.ndarray, distances: np.ndarray,
    clearances: np.ndarray, cutoff: float,
) -> np.ndarray:
    """Per-void geometry features from periodic void->atom edges.

    ``void_index`` and ``distances`` describe every atom image within
    ``cutoff`` of each void, so counts and nearest distances do not depend on
    the chosen unit cell. A nearest-neighbour slot with no atom image inside
    the cutoff is set to the cutoff (a lower bound on the true distance), never
    to 0. The count radii must not exceed the edge cutoff.
    """
    if max(VOID_COUNT_RADII_ANGSTROM) > cutoff + 1e-8:
        raise ValueError("void count radii exceed the void-atom edge cutoff")
    void_index = np.asarray(void_index, dtype=int).reshape(-1)
    distances = np.asarray(distances, dtype=float).reshape(-1)
    rows = []
    for i in range(int(n_voids)):
        ds = np.sort(distances[void_index == i])
        nearest = np.full(VOID_NEAREST_COUNT, float(cutoff))
        nearest[: min(len(ds), VOID_NEAREST_COUNT)] = ds[:VOID_NEAREST_COUNT]
        rows.append([
            float(clearances[i]), float(nearest[0]),
            *[float(np.count_nonzero(ds <= r + 1e-8)) for r in VOID_COUNT_RADII_ANGSTROM],
            *[float(x) for x in nearest],
        ])
    return np.asarray(rows, dtype=float).reshape(-1, len(VOID_FEATURE_NAMES))


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
    # Crystallographic reciprocal vectors (no 2*pi): |b_i| = 1 / d_i.
    reciprocal_lengths = np.linalg.norm(
        lattice.reciprocal_lattice_crystallographic.matrix, axis=1
    )
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


def _fractional_sort_key(frac: np.ndarray) -> tuple[float, float, float]:
    """Stable ordering for wrapped fractional coordinates."""
    return tuple(float(value) for value in np.asarray(frac) % 1.0)


def _expand_candidate_orbits(
    structure: Structure,
    candidates: list[dict[str, Any]],
    *,
    symprec: float,
    tolerance_angstrom: float = VOID_SYMMETRY_TOLERANCE_ANGSTROM,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand detector representatives to unique explicit unit-cell void nodes.

    The detector symmetry-reduces candidates. Applying the structure's symmetry
    operations restores their distinct positions inside the unit cell, which
    cannot be represented by translational edge image shifts. Near-duplicate
    positions are merged using the same 0.2 A tolerance as the void detector.
    Candidate and member ordering is coordinate-sorted for stable IDs.

    On symmetry-analysis failure, retain each representative as a singleton
    orbit and return an explicit fallback status so incomplete neighborhoods
    are visible in record metadata.
    """
    if not np.isfinite(tolerance_angstrom) or tolerance_angstrom <= 0:
        raise ValueError("void symmetry tolerance must be finite and positive")
    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (
            _fractional_sort_key(candidate["fractional_coords"]),
            float(candidate.get("clearance_angstrom", 0.0)),
            tuple(sorted(str(x) for x in candidate.get("methods", []))),
        ),
    )
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        operations = SpacegroupAnalyzer(
            structure, symprec=symprec
        ).get_symmetry_operations(cartesian=False)
        if not operations:
            raise ValueError("symmetry analyzer returned no operations")
        status = "expanded"
        operation_count = len(operations)
    except Exception:
        operations = []
        status = "fallback_representatives_symmetry_analysis_failed"
        operation_count = 0

    # A union-find lets us defensively merge candidate representatives whose
    # expanded symmetry orbits overlap within the position tolerance.
    parent = list(range(len(sorted_candidates)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    nodes: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(sorted_candidates):
        representative = np.asarray(candidate["fractional_coords"], dtype=float) % 1.0
        if operations:
            orbit_positions = [
                np.asarray(operation.operate(representative), dtype=float) % 1.0
                for operation in operations
            ]
        else:
            orbit_positions = [representative]

        unique_positions: list[np.ndarray] = []
        for position in orbit_positions:
            if not any(
                structure.lattice.get_distance_and_image(position, prior)[0]
                < tolerance_angstrom
                for prior in unique_positions
            ):
                unique_positions.append(position)

        for position in unique_positions:
            duplicate = next((
                node for node in nodes
                if structure.lattice.get_distance_and_image(
                    position, node["fractional_coords"]
                )[0] < tolerance_angstrom
            ), None)
            if duplicate is None:
                nodes.append({
                    "fractional_coords": position,
                    "candidate_indices": {candidate_index},
                })
            else:
                prior_candidate = min(duplicate["candidate_indices"])
                duplicate["candidate_indices"].add(candidate_index)
                union(candidate_index, prior_candidate)

    # Resolve all representative groups after overlap unions, and gather
    # candidate-level method/clearance metadata for the resulting orbit.
    groups: dict[int, list[dict[str, Any]]] = {}
    for node in nodes:
        root = find(min(node["candidate_indices"]))
        groups.setdefault(root, []).append(node)

    expanded: list[dict[str, Any]] = []
    for root, group_nodes in sorted(groups.items()):
        # Sort the coordinates at their stored float32 precision so the member
        # indexes remain consistent with the HDF5 coordinates checked later.
        for node in group_nodes:
            node["fractional_coords"] = (
                np.asarray(node["fractional_coords"], dtype=np.float32).astype(float) % 1.0
            )
        group_nodes.sort(key=lambda node: _fractional_sort_key(node["fractional_coords"]))
        candidate_indices = sorted({
            index for node in group_nodes for index in node["candidate_indices"]
        })
        methods = sorted({
            str(method)
            for index in candidate_indices
            for method in sorted_candidates[index].get("methods", [])
        })
        clearance = max(
            float(sorted_candidates[index]["clearance_angstrom"])
            for index in candidate_indices
        )
        orbit_id = f"candidate:{root:04d}"
        multiplicity = len(group_nodes)
        for member_index, node in enumerate(group_nodes):
            expanded.append({
                "fractional_coords": np.asarray(node["fractional_coords"], dtype=float),
                "clearance_angstrom": clearance,
                "methods": methods,
                "orbit_id": orbit_id,
                "orbit_member_index": member_index,
                "orbit_multiplicity": multiplicity,
            })

    return expanded, {
        "status": status,
        "operation_count": operation_count,
        "tolerance_angstrom": float(tolerance_angstrom),
    }


def build_graph_record(
    structure_path: str | Path,
    *,
    sample_id: str | None = None,
    family_id: str | None = None,
    atom_cutoff_angstrom: float = 5.0,
    atom_void_cutoff_angstrom: float = 6.0,
    void_void_cutoff_angstrom: float = 5.0,
    symprec: float = 0.01,
    void_symmetry_tolerance_angstrom: float = VOID_SYMMETRY_TOLERANCE_ANGSTROM,
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
    detected_candidates = find_candidate_voids(structure, symprec=symprec)
    candidates, symmetry_expansion = _expand_candidate_orbits(
        structure, detected_candidates, symprec=symprec,
        tolerance_angstrom=void_symmetry_tolerance_angstrom,
    )
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
    vv_idx, vv_dist, vv_shift = _periodic_edges(
        structure.lattice, void_frac, void_frac, void_void_cutoff_angstrom,
        same_node_graph=True,
    )

    # Counts and nearest distances use every periodic atom image (the
    # void->atom edges), so they do not depend on the chosen unit cell.
    void_features = void_neighbor_features(
        len(candidates),
        np.asarray(va_idx, dtype=int).reshape(-1, 2)[:, 0],
        np.asarray(va_dist, dtype=float),
        clearances, atom_void_cutoff_angstrom,
    )
    multiplicities = []
    methods = []
    for candidate in candidates:
        methods.append([int(name in candidate["methods"]) for name in METHOD_NAMES])
        multiplicities.append(int(candidate["orbit_multiplicity"]))

    resolved_sample_id = sample_id or path.parent.name or path.stem

    return {
        "sample_id": resolved_sample_id,
        "source_poscar": str(path),
        "family_id": family_id,
        "schema_version": "1.0",
        "units": {"coordinates": "fractional", "distance": "angstrom"},
        "edge_convention": (
            "directed source->destination; all periodic destination images within "
            "cutoff are retained; image_shift is added to destination fractional "
            "coordinates; void_atom_* is void->atom and reverse relation is derived "
            "by swapping endpoints and negating image_shift; void_void_* is void->void "
            "and excludes only the zero-shift self-edge"
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
            "void_id": [
                f"{resolved_sample_id}:{candidate['orbit_id']}:member:{candidate['orbit_member_index']:04d}"
                for candidate in candidates
            ],
            "orbit_id": [
                f"{resolved_sample_id}:{candidate['orbit_id']}"
                for candidate in candidates
            ],
            "orbit_member_index": np.asarray(
                [candidate["orbit_member_index"] for candidate in candidates],
                dtype=np.int16,
            ),
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
            "void_void_index": np.asarray(vv_idx, dtype=np.int32).T.reshape(2, -1),
            "void_void_distance_angstrom": np.asarray(vv_dist, dtype=np.float32),
            "void_void_image_shift": np.asarray(vv_shift, dtype=np.int16).reshape(-1, 3),
        },
        "counts": {
            "n_atoms": len(atom_frac), "n_voids": len(void_frac),
            "n_atom_atom_edges": len(aa_idx), "n_void_atom_edges": len(va_idx),
            "n_void_void_edges": len(vv_idx),
        },
        "settings": {
            "atom_cutoff_angstrom": atom_cutoff_angstrom,
            "atom_void_cutoff_angstrom": atom_void_cutoff_angstrom,
            "void_void_cutoff_angstrom": void_void_cutoff_angstrom,
            "symprec": symprec,
            "void_symmetry_expansion_status": symmetry_expansion["status"],
            "void_symmetry_tolerance_angstrom": symmetry_expansion["tolerance_angstrom"],
            "void_symmetry_operation_count": symmetry_expansion["operation_count"],
            "void_finder": "void_detection.find_voronoi_voids.find_candidate_voids",
            "void_neighbor_count_definition": VOID_FEATURE_DEFINITION,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("poscar", help="POSCAR or pymatgen-readable structure file")
    parser.add_argument("--sample-id")
    parser.add_argument("--output", "-o", help="JSON output path (default: stdout)")
    parser.add_argument("--atom-cutoff", type=float, default=5.0)
    parser.add_argument("--atom-void-cutoff", type=float, default=6.0)
    parser.add_argument("--void-void-cutoff", type=float, default=5.0)
    parser.add_argument("--symprec", type=float, default=0.01)
    parser.add_argument(
        "--void-symmetry-tolerance", type=float,
        default=VOID_SYMMETRY_TOLERANCE_ANGSTROM,
    )
    args = parser.parse_args()
    record = build_graph_record(
        args.poscar, sample_id=args.sample_id, atom_cutoff_angstrom=args.atom_cutoff,
        atom_void_cutoff_angstrom=args.atom_void_cutoff,
        void_void_cutoff_angstrom=args.void_void_cutoff, symprec=args.symprec,
        void_symmetry_tolerance_angstrom=args.void_symmetry_tolerance,
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
