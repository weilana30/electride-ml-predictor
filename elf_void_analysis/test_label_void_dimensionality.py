"""Synthetic regression checks for periodic void-dimensionality labels."""
from __future__ import annotations

import unittest

import numpy as np

from .label_void_dimensionality import _component_winding_rank, label_void_dimensionality


class PeriodicWindingTests(unittest.TestCase):
    def test_ranks_for_isolated_line_sheet_and_volume(self) -> None:
        n = 5
        masks = {
            "isolated": np.zeros((n, n, n), dtype=bool),
            "line": np.zeros((n, n, n), dtype=bool),
            "sheet": np.zeros((n, n, n), dtype=bool),
            "volume": np.ones((n, n, n), dtype=bool),
        }
        masks["isolated"][2, 2, 2] = True
        masks["line"][:, 2, 2] = True
        masks["sheet"][:, :, 2] = True
        expected = {"isolated": 0, "line": 1, "sheet": 2, "volume": 3}
        for name, mask in masks.items():
            with self.subTest(name=name):
                _, ranks = _component_winding_rank(mask)
                self.assertEqual(ranks, [expected[name]])

    def test_disconnected_periodic_components_have_independent_ranks(self) -> None:
        active = np.zeros((7, 7, 7), dtype=bool)
        active[:, 1, 1] = True  # winding line
        active[4, 4, 4] = True  # isolated voxel
        _, ranks = _component_winding_rank(active)
        self.assertCountEqual(ranks, [0, 1])


class VoidLabelTests(unittest.TestCase):
    def test_inactive_void_center_is_invalid(self) -> None:
        # The atom sits on a sampled grid point. Its own voxel is inaccessible;
        # the central point in the large cell lies in a 3D wrapping component.
        result = label_void_dimensionality(
            np.eye(3) * 8.0,
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            probe_radius_A=1.0,
            grid_spacing_A=1.0,
            chunk_size=64,
        )
        np.testing.assert_array_equal(result["void_dimension"], [-1, 3])
        np.testing.assert_array_equal(result["void_dimension_valid"], [0, 1])
        self.assertEqual(result["metadata"]["dimension_class_names"], ["0D", "1D", "2D", "3D"])

    def test_zero_voids_return_empty_per_void_arrays(self) -> None:
        result = label_void_dimensionality(
            np.eye(3) * 4.0,
            [[0.0, 0.0, 0.0]],
            np.empty((0, 3)),
            probe_radius_A=0.5,
            grid_spacing_A=2.0,
        )
        self.assertEqual(result["void_dimension"].shape, (0,))
        self.assertEqual(result["void_dimension_valid"].shape, (0,))

    def test_bad_lattice_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonsingular"):
            label_void_dimensionality(
                np.zeros((3, 3)), [[0.0, 0.0, 0.0]], [[0.5, 0.5, 0.5]]
            )


if __name__ == "__main__":
    unittest.main()
