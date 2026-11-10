"""Scoring and test-set rules for the registry dimensionality test."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from .registry_dimensionality_figures import connectivity_shares
from .registry_dimensionality_test import half_split_tuning, load_test_set, score


class ScoreTests(unittest.TestCase):
    def test_balanced_accuracy_averages_0d_to_2d_recall(self) -> None:
        literature = np.array([0, 0, 0, 0, 1, 1, 2, 3])
        predicted = np.array([0, 0, 0, 3, 1, 0, -1, 3])
        result = score(literature, predicted)
        self.assertAlmostEqual(result["exact_accuracy"], 5 / 8)
        self.assertEqual(result["recall"], {"0D": 0.75, "1D": 0.5, "2D": 0.0, "3D": 1.0})
        self.assertAlmostEqual(result["balanced_accuracy"], (0.75 + 0.5 + 0.0) / 3)

    def test_half_split_tuning_finds_the_one_correct_probe(self) -> None:
        literature = np.array([0, 1, 2] * 10)
        sweep = np.stack([np.full(30, -1), literature, np.full(30, 3)], axis=1)
        result = half_split_tuning(literature, sweep, repeats=20)
        self.assertEqual(result["held_out_exact_accuracy_mean"], 1.0)


class ConnectivityTests(unittest.TestCase):
    def test_channels_win_ties_and_unscored_or_3d_rows_are_left_out(self) -> None:
        structures = pd.DataFrame({"mp_id": ["a", "b", "c", "d", "e"],
                                   "literature_dimension": [0, 0, 2, 3, 1],
                                   "scored": [True, True, True, True, False]})
        dims = {"a": [3, 0], "b": [3, 2, 1, 0], "c": [3, 2, 0], "d": [3, 0], "e": [3, 1, 0]}
        sweep = pd.DataFrame([{"mp_id": m, "radius_scheme": "ionic", "highest_dimension": d}
                              for m, values in dims.items() for d in values])
        self.assertEqual(connectivity_shares(structures, sweep).to_numpy().tolist(),
                         [[1, 1, 0], [0, 0, 0], [0, 0, 1]])


class LoadTests(unittest.TestCase):
    def test_conflicting_calibration_unresolved_and_missing_cif_rows(self) -> None:
        rows = [("ELD-1", "A", "0D", "mp-1", "literature_exact"),
                ("ELD-2", "B", "0D", "mp-2", "literature_exact"),
                ("ELD-3", "B", "1D", "mp-2", "literature_exact"),
                ("ELD-4", "Ca2N", "2D", "mp-2686", "literature_exact"),
                ("ELD-5", "C", "2D", "mp-3", "unresolved"),
                ("ELD-6", "D", "1D", "mp-4", "literature_exact"),
                ("ELD-7", "E", "mixed", "mp-5", "literature_exact")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mp_id in ("mp-1", "mp-2", "mp-2686", "mp-3", "mp-5"):
                (root / f"{mp_id}.cif").write_text("")
            registry = pd.DataFrame(rows, columns=["material_id", "normalized_formula", "electride_dimensionality",
                                                   "mp_id", "structure_identity_status"])
            registry["electride_status"] = "theoretically_predicted"
            registry["primary_reference"] = "ref"
            registry.to_csv(root / "registry.csv", index=False)
            table = load_test_set(root / "registry.csv", root).set_index("mp_id")
        self.assertEqual(sorted(table.index), ["mp-1", "mp-2", "mp-2686"])
        self.assertEqual(table.loc["mp-2", "literature_labels"], "0D;1D")
        self.assertEqual(table["scored"].to_dict(), {"mp-1": True, "mp-2": False, "mp-2686": False})
        self.assertEqual(table.loc["mp-1", "literature_dimension"], 0)


if __name__ == "__main__":
    unittest.main()
