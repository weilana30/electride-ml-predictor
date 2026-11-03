"""Geometric and periodic regression checks for adaptive void connectivity."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from .adaptive_void_connectivity import AdaptiveSpace
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


if __name__ == "__main__":
    unittest.main()
