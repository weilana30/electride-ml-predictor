"""Exactness and periodic regression checks for power-diagram void connectivity."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .power_void_connectivity import PowerSpace, element_radii, run


def dimensions(space: PowerSpace) -> list[int]:
    return [region["dimension"] for region in space.explore()["regions"]]


class PowerSpaceTests(unittest.TestCase):
    def test_simple_cubic_thresholds_match_geometry(self) -> None:
        # a = 4 Å: edge midpoints close at R = 2, face centers at 2*sqrt(2),
        # and the body center at 2*sqrt(3).
        for radius, expected in [(1.9, [3]), (2.7, [3]), (2.9, [0]), (3.5, [])]:
            with self.subTest(radius=radius):
                space = PowerSpace(np.eye(3) * 4, np.zeros((1, 3)), np.array([radius]))
                self.assertEqual(dimensions(space), expected)

    def test_degenerate_atom_walls_yield_exact_pocket_channel_and_sheet(self) -> None:
        points = np.array(list(np.ndindex(6, 6, 6)), dtype=int)
        for axes, expected in [((0, 1, 2), 0), ((1, 2), 1), ((2,), 2)]:
            with self.subTest(axes=axes):
                atoms = points[np.any(np.isin(points[:, axes], [0, 3]), axis=1)] / 6
                space = PowerSpace(np.eye(3) * 6, atoms, np.ones(len(atoms)))
                labels = dimensions(space)
                self.assertTrue(labels)
                self.assertEqual(set(labels), {expected})
                node, status = space.locate(np.array([.25, .25, .25]))
                self.assertEqual(status, "assigned")
                self.assertIn(node, space.membership)

    def test_power_vertices_are_equidistant_in_power(self) -> None:
        lattice = np.array([[5., 0., 0.], [1., 5., 0.], [0., .5, 6.]])
        atoms = np.array([[0., 0., 0.], [.5, .4, .3], [.2, .7, .6]])
        space = PowerSpace(lattice, atoms, np.array([1.2, 2.0, 1.6]))
        for f in space.node:
            sites = space.simplices[f]
            power = ((space.centers[sites] - space.vertex_xyz[f]) ** 2).sum(1) - space.weights[sites]
            self.assertLess(np.ptp(power), 1e-8)
            self.assertAlmostEqual(power[0], space.vertex_power[f], places=8)

    def test_larger_body_center_ball_isolates_edge_pockets(self) -> None:
        # CsCl-type cell, a = 4 Å. Equal 1.9 Å balls leave 3D space. A 2.5 Å
        # body-center ball covers face centers (2 Å away) but not the three
        # cube-edge midpoints (2 Å from corners, 2*sqrt(2) Å from the center).
        lattice = np.eye(3) * 4
        atoms = np.array([[0., 0., 0.], [.5, .5, .5]])
        self.assertEqual(dimensions(PowerSpace(lattice, atoms, np.array([1.9, 1.9]))), [3])
        self.assertEqual(dimensions(PowerSpace(lattice, atoms, np.array([1.9, 2.5]))), [0, 0, 0])

    def test_locate_rejects_covered_points(self) -> None:
        space = PowerSpace(np.eye(3) * 4, np.zeros((1, 3)), np.array([1.0]))
        space.explore()
        self.assertEqual(space.locate(np.zeros(3)), (None, "below_probe_cutoff"))
        node, status = space.locate(np.array([.5, .5, .5]))
        self.assertEqual(status, "assigned")


class RadiusAndReportTests(unittest.TestCase):
    def test_ionic_scheme_uses_anion_and_cation_shannon_radii(self) -> None:
        radii, notes = element_radii(["Ca", "Ca", "N"], "ionic")
        self.assertAlmostEqual(radii["Ca"], 1.14)
        self.assertAlmostEqual(radii["N"], 1.32)
        self.assertIn("N-3", notes["N"])
        self.assertEqual(element_radii(["Ca"], "none")[0], {"Ca": 0.0})
        with self.assertRaises(ValueError):
            element_radii(["Ca", "N"], "custom", {"Ca": 1.0})

    def test_report_labels_every_open_void_exactly(self) -> None:
        graph = {"lattice_angstrom": (np.eye(3) * 4).tolist(),
                 "atoms": {"fractional_coords": [[0., 0., 0.]], "atomic_numbers": [1]},
                 "voids": {"fractional_coords": [[.5, .5, .5], [.5, .5, 0.], [0., 0., 0.]],
                           "void_id": ["body", "face", "atom"]}}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "structure.cif"
            source.write_text("test source\n")
            with patch("elf_void_analysis.power_void_connectivity.build_graph_record", return_value=graph):
                report = run(source, Path(directory) / "out", probe_A=2.9)
            self.assertEqual(json.loads((Path(directory) / "out" / "report.json").read_text()), report)
        body, face, atom = report["voids"]
        self.assertEqual((body["connectivity_label"], body["label_status"]), ("0D", "exact"))
        self.assertEqual(face["assignment_status"], "below_probe_cutoff")
        self.assertEqual(atom["assignment_status"], "below_probe_cutoff")
        self.assertEqual(report["regions"][0]["void_ids"], ["body"])

    def test_xsf_export_draws_each_exact_region(self) -> None:
        points = np.array(list(np.ndindex(6, 6, 6)), dtype=int)
        atoms = points[np.any(np.isin(points[:, (1, 2)], [0, 3]), axis=1)] / 6
        graph = {"lattice_angstrom": (np.eye(3) * 6).tolist(),
                 "atoms": {"fractional_coords": atoms.tolist(), "atomic_numbers": [1] * len(atoms)},
                 "voids": {"fractional_coords": [[.25, .25, .25]], "void_id": ["channel"]}}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "structure.cif"
            source.write_text("test source\n")
            out = Path(directory) / "out"
            with patch("elf_void_analysis.power_void_connectivity.build_graph_record", return_value=graph):
                report = run(source, out, probe_A=1.0, export_spacing_A=0.25)
            self.assertEqual(len(report["regions"]), 4)
            self.assertEqual(report["unplaced_open_grid_points"], 0)
            for region in report["regions"]:
                self.assertEqual(region["connectivity_label"], "1D")
                self.assertGreater(region["export_open_samples"], 0)
                # Probe balls fill more space than their centers alone.
                self.assertGreater(region["export_filled_samples"], region["export_open_samples"])
                self.assertIn("BEGIN_DATAGRID_3D_accessible_space", (out / region["xsf_file"]).read_text())
            self.assertTrue((out / "open_space.xsf").is_file())


if __name__ == "__main__":
    unittest.main()
