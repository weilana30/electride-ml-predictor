"""Focused tests for explicit symmetry-expanded void graph nodes."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from pymatgen.core import Lattice, Structure

from elf_void_analysis.build_graph_records import _expand_candidate_orbits


class SymmetryOrbitExpansionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.structure = Structure(Lattice.cubic(4.0), ["Si"], [[0.0, 0.0, 0.0]])
        self.candidates = [{
            "fractional_coords": [0.123, 0.234, 0.345],
            "clearance_angstrom": 1.25,
            "methods": ["Voronoi"],
        }]

    def test_general_position_expands_to_unique_sorted_unit_cell_sites(self) -> None:
        expanded, metadata = _expand_candidate_orbits(
            self.structure, self.candidates, symprec=0.01
        )

        self.assertEqual(metadata["status"], "expanded")
        self.assertGreater(metadata["operation_count"], 1)
        self.assertGreater(len(expanded), 1)
        self.assertEqual(
            {node["orbit_multiplicity"] for node in expanded}, {len(expanded)}
        )
        self.assertEqual(
            [node["orbit_member_index"] for node in expanded],
            list(range(len(expanded))),
        )
        positions = [node["fractional_coords"] for node in expanded]
        self.assertEqual(positions, sorted(positions, key=lambda xyz: tuple(np.round(xyz, 10))))
        for i, position in enumerate(positions):
            for other in positions[:i]:
                self.assertGreaterEqual(
                    self.structure.lattice.get_distance_and_image(position, other)[0],
                    metadata["tolerance_angstrom"],
                )
        self.assertTrue(all(node["methods"] == ["Voronoi"] for node in expanded))
        self.assertTrue(all(node["clearance_angstrom"] == 1.25 for node in expanded))

    def test_symmetry_analysis_failure_is_marked_and_keeps_representative(self) -> None:
        with patch(
            "pymatgen.symmetry.analyzer.SpacegroupAnalyzer.get_symmetry_operations",
            side_effect=RuntimeError("simulated symmetry failure"),
        ):
            expanded, metadata = _expand_candidate_orbits(
                self.structure, self.candidates, symprec=0.01
            )

        self.assertEqual(
            metadata["status"], "fallback_representatives_symmetry_analysis_failed"
        )
        self.assertEqual(metadata["operation_count"], 0)
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0]["orbit_member_index"], 0)
        self.assertEqual(expanded[0]["orbit_multiplicity"], 1)

    def test_nonpositive_tolerance_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            _expand_candidate_orbits(
                self.structure, self.candidates, symprec=0.01,
                tolerance_angstrom=0.0,
            )


if __name__ == "__main__":
    unittest.main()
