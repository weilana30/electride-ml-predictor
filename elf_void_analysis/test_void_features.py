"""Void descriptors must not depend on the chosen unit cell."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from pymatgen.core import Lattice, Structure

from elf_void_analysis.build_graph_records import _periodic_edges, void_neighbor_features
from elf_void_analysis.recompute_void_features import patch_sample


def _features(structure: Structure, void_cart: np.ndarray, cutoff: float = 6.0) -> np.ndarray:
    void_frac = structure.lattice.get_fractional_coords(void_cart)[None, :] % 1.0
    idx, dist, _ = _periodic_edges(structure.lattice, void_frac, structure.frac_coords % 1.0, cutoff)
    clearance = np.array([min(dist)])
    return void_neighbor_features(1, np.asarray(idx)[:, 0], np.asarray(dist), clearance, cutoff)[0]


class VoidFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = Structure(Lattice.hexagonal(3.1, 4.9), ["Y", "Rb"],
                              [[1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]])
        self.void = self.cell.lattice.get_cartesian_coords([0.0, 0.0, 0.5])

    def test_supercell_gives_identical_features(self) -> None:
        small = _features(self.cell, self.void)
        for scaling in ([2, 1, 1], [2, 2, 3]):
            big = self.cell.copy()
            big.make_supercell(scaling)
            np.testing.assert_allclose(_features(big, self.void), small, atol=1e-6)

    def test_counts_include_periodic_images_and_no_zero_padding(self) -> None:
        features = _features(self.cell, self.void)
        self.assertGreater(features[4], len(self.cell))  # more than 2 atoms within 5 A
        self.assertTrue(np.all(features[5:9] > 0.5))
        self.assertTrue(np.all(np.diff(features[5:9]) >= 0))
        self.assertAlmostEqual(features[1], features[5])

    def test_missing_neighbours_are_set_to_cutoff(self) -> None:
        features = void_neighbor_features(1, np.array([0, 0]), np.array([2.0, 3.5]), np.array([2.0]), 6.0)[0]
        np.testing.assert_allclose(features, [2.0, 2.0, 1, 2, 2, 2.0, 3.5, 6.0, 6.0])

    def test_dataset_patch_rewrites_features_from_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with h5py.File(Path(tmp) / "s.h5", "w") as handle:
                group = handle.create_group("sample")
                group["voids/feature_names"] = np.array(
                    [b"clearance_angstrom", b"nearest_atom_distance_angstrom", b"atom_count_within_3A",
                     b"atom_count_within_4A", b"atom_count_within_5A", b"nearest_1_distance_angstrom",
                     b"nearest_2_distance_angstrom", b"nearest_3_distance_angstrom",
                     b"nearest_4_distance_angstrom"])
                group["voids/features"] = np.array([[2.0, 2.0, 1, 2, 2, 2.0, 3.5, 0, 0]], np.float32)
                group["voids/clearance_angstrom"] = np.array([2.0], np.float32)
                group["edges/void_atom_index"] = np.array([[0, 0, 0], [0, 1, 1]], np.int32)
                group["edges/void_atom_distance_angstrom"] = np.array([2.0, 3.5, 4.5], np.float32)
                group.create_group("metadata/settings").attrs["atom_void_cutoff_angstrom"] = 6.0
                stats = patch_sample(group)
                np.testing.assert_allclose(group["voids/features"][0], [2.0, 2.0, 1, 2, 3, 2.0, 3.5, 4.5, 6.0])
        self.assertEqual(stats["changed"], 1)


if __name__ == "__main__":
    unittest.main()
