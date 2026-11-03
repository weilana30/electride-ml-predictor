"""Geometric and periodic regression checks for adaptive void connectivity."""
from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .adaptive_void_connectivity import AdaptiveSpace, run
from .export_open_space_vesta import write_xsf


class AdaptiveBlockTests(unittest.TestCase):
    def test_certified_open_and_blocked_blocks_obey_distance_rule(self) -> None:
        lattice = np.array([[4., 0., 0.], [1., 4., 0.], [0., 0., 4.]])
        space = AdaptiveSpace(lattice, np.array([[0., 0., 0.]]), 1.0, 0.25)
        space.split((0, 0, 0, 0))
        for key in space.nodes[(0, 0, 0, 0)].children:
            status = space.classify(key)
            if status not in {"open", "blocked"}:
                continue
            lo, hi = space.bounds(key)
            for corner in np.ndindex(3, 3, 3):
                frac = lo + (hi-lo) * np.asarray(corner) / 2
                distance = space.distance(frac)
                if status == "open":
                    self.assertGreaterEqual(distance, space.probe_A - 1e-8)
                else:
                    self.assertLess(distance, space.probe_A + 1e-8)

    def test_face_neighbor_finds_finer_blocks_and_wraps_periodically(self) -> None:
        space = AdaptiveSpace(np.eye(3)*4, np.array([[0., 0., 0.]]), 0.1, 1.)
        space.split((0, 0, 0, 0))
        space.split((1, 1, 0, 0))
        neighbors, wrap = space.face_neighbors((1, 0, 0, 0), 0, 1)
        self.assertEqual(len(neighbors), 4)
        self.assertTrue(all(key[0] == 2 and key[1] == 2 for key in neighbors))
        np.testing.assert_array_equal(wrap, [0, 0, 0])
        periodic, wrap = space.face_neighbors((1, 0, 0, 0), 0, -1)
        self.assertEqual(len(periodic), 4)
        self.assertTrue(all(key[0] == 2 and key[1] == 3 for key in periodic))
        np.testing.assert_array_equal(wrap, [-1, 0, 0])

    def test_seeded_search_certifies_three_periodic_directions(self) -> None:
        space = AdaptiveSpace(np.eye(3)*4, np.array([[0., 0., 0.]]), 0.1, 1.)
        result = space.explore(np.array([[.5, .5, .5], [.6, .6, .6], [0., 0., 0.]]))
        self.assertEqual(result["assignments"][:2], [0, 0])
        self.assertIsNone(result["assignments"][2])
        self.assertTrue(result["invalid"][2])
        self.assertEqual(result["regions"][0]["certified_winding_rank"], 3)
        self.assertEqual(result["regions"][0]["dimension"], 3)

    def test_periodic_atom_walls_yield_pocket_channel_and_sheet(self) -> None:
        lattice = np.eye(3) * 6
        points = np.array(list(np.ndindex(6, 6, 6)), dtype=int)
        for axes, expected in [((0, 1, 2), 0), ((1, 2), 1), ((2,), 2)]:
            with self.subTest(axes=axes):
                wall = np.any(np.isin(points[:, axes], [0, 3]), axis=1)
                atoms = points[wall] / 6
                space = AdaptiveSpace(lattice, atoms, 1., .5)
                result = space.explore(np.array([[.25, .25, .25]]))
                region = result["regions"][0]
                self.assertEqual(region["certified_winding_rank"], expected)
                # Boundary leaves are unresolved at this finite cutoff; rank
                # is a lower bound, not an unjustified exact dimension.
                self.assertTrue(region["uncertain"])
                self.assertIsNone(region["dimension"])

    def test_uncertain_limit_is_not_mislabeled_zero_dimensional(self) -> None:
        # A one-block budget cannot decide the accessible component around a
        # seed close to an atom; its dimension must remain unknown.
        space = AdaptiveSpace(np.eye(3)*4, np.array([[0., 0., 0.]]), 1., .25, max_blocks=1)
        result = space.explore(np.array([[.5, .5, .5]]))
        self.assertEqual(result["assignments"], [None])
        self.assertEqual(result["regions"], [])
        self.assertTrue(space.budget_hit)

    def test_xsf_periodic_end_planes_and_reportable_raster(self) -> None:
        space = AdaptiveSpace(np.eye(3)*4, np.array([[0., 0., 0.]]), .1, 1.)
        result = space.explore(np.array([[.5, .5, .5]]))
        mask = space.raster(set(result["membership"]), 1.)
        self.assertEqual(mask.shape, (4, 4, 4))
        self.assertTrue(mask.any())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "region.xsf"
            write_xsf(path, space.lattice, np.array([1]), np.zeros((1, 3)), mask)
            content = path.read_text()
            self.assertIn("BEGIN_DATAGRID_3D_accessible_space", content)
            self.assertIn("5 5 5", content)  # duplicate periodic end planes

    def test_report_only_groups_voids_without_writing_grids(self) -> None:
        graph = {"lattice_angstrom": (np.eye(3)*4).tolist(),
                 "atoms": {"fractional_coords": [[0., 0., 0.]], "atomic_numbers": [1]},
                 "voids": {"fractional_coords": [[.5, .5, .5], [.6, .6, .6], [0., 0., 0.]],
                           "void_id": ["void-a", "void-b", "void-c"]}}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "structure.cif"
            source.write_text("test source\n")
            output = Path(directory) / "report-only"
            with patch("elf_void_analysis.adaptive_void_connectivity.build_graph_record", return_value=graph), \
                 patch.object(AdaptiveSpace, "raster", side_effect=AssertionError("report-only rasterized")):
                report = run(source, output, probe_A=.1, min_width_A=1., report_only=True)
            self.assertEqual([p.name for p in output.iterdir()], ["report.json"])
            self.assertEqual(json.loads((output / "report.json").read_text()), report)
            region = report["regions"][0]
            self.assertEqual(region["connectivity_label"], "3D")
            self.assertEqual(region["label_status"], "exact")
            self.assertEqual(region["void_ids"], ["void-a", "void-b"])
            self.assertIsNone(region["xsf_file"])
            self.assertEqual([v["region_uid"] for v in report["voids"][:2]],
                             [region["region_uid"]]*2)
            self.assertEqual(report["voids"][2]["assignment_status"], "below_probe_cutoff")
            self.assertIsNone(report["voids"][2]["region_uid"])
            with patch("elf_void_analysis.adaptive_void_connectivity.build_graph_record", return_value=graph):
                visual = run(source, Path(directory) / "with-xsf", probe_A=.1,
                             min_width_A=1., export_spacing_A=1.)
            self.assertEqual(visual["analysis_id"], report["analysis_id"])
            self.assertEqual([v["region_uid"] for v in visual["voids"]],
                             [v["region_uid"] for v in report["voids"]])
            self.assertTrue((Path(directory) / "with-xsf" / visual["regions"][0]["xsf_file"]).is_file())

    def test_uncertain_region_label_is_a_lower_bound_for_each_void(self) -> None:
        points = np.array(list(np.ndindex(6, 6, 6)), dtype=int)
        atoms = points[np.any(np.isin(points[:, (1, 2)], [0, 3]), axis=1)] / 6
        graph = {"lattice_angstrom": (np.eye(3)*6).tolist(),
                 "atoms": {"fractional_coords": atoms.tolist(), "atomic_numbers": [1]*len(atoms)},
                 "voids": {"fractional_coords": [[.25, .25, .25]], "void_id": ["channel-site"]}}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "structure.cif"
            source.write_text("test source\n")
            with patch("elf_void_analysis.adaptive_void_connectivity.build_graph_record", return_value=graph):
                report = run(source, Path(directory) / "report-only", probe_A=1.,
                             min_width_A=.5, report_only=True)
            region, void = report["regions"][0], report["voids"][0]
            self.assertEqual(region["connectivity_label"], "1D")
            self.assertEqual(region["label_status"], "lower_bound")
            self.assertIsNone(region["dimension"])
            self.assertEqual(void["region_uid"], region["region_uid"])
            self.assertEqual(void["connectivity_label"], "1D")
            self.assertEqual(void["label_status"], "lower_bound")


if __name__ == "__main__":
    unittest.main()
