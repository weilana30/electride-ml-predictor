import unittest

from elf_void_gnn.feature_study.aggregate_runs import interaction_rows


class InteractionAggregationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"run_id": "base", "variant": "baseline", "seed": "7", "baseline": "", "split_id": "s"},
            {"run_id": "atom", "variant": "atom_ablate_chemistry", "seed": "7", "baseline": "base", "split_id": "s"},
            {"run_id": "void", "variant": "void_ablate_coordination", "seed": "7", "baseline": "base", "split_id": "s"},
            {"run_id": "joint", "variant": "atom_chemistry_void_coordination_removed", "seed": "7", "baseline": "base", "split_id": "s"},
        ]
        values = {
            "base": (1.0, 1.0),
            "atom": (1.2, 1.1),
            "void": (1.3, 1.4),
            "joint": (1.8, 1.7),
        }
        self.loaded = {
            run_id: ({}, {
                "family-a": {"rmse": pair[0]},
                "family-b": {"rmse": pair[1]},
            }, {})
            for run_id, pair in values.items()
        }
        self.byid = {row["run_id"]: row for row in self.rows}

    def test_difference_in_differences_pairs_exact_named_cells(self):
        effects = interaction_rows(self.rows, self.loaded, self.byid, draws=200)
        self.assertEqual(len(effects), 1)
        self.assertEqual(effects[0]["single_a"], "atom")
        self.assertEqual(effects[0]["single_b"], "void")
        self.assertAlmostEqual(effects[0]["macro_rmse_interaction"], 0.25)

    def test_missing_cell_is_rejected(self):
        rows = [row for row in self.rows if row["variant"] != "void_ablate_coordination"]
        with self.assertRaisesRegex(ValueError, "missing cells"):
            interaction_rows(rows, self.loaded, self.byid, draws=100)

    def test_ambiguous_variant_and_seed_is_rejected(self):
        rows = self.rows + [dict(self.rows[1], run_id="other-atom")]
        with self.assertRaisesRegex(ValueError, "duplicate run"):
            interaction_rows(rows, self.loaded, self.byid, draws=100)


if __name__ == "__main__":
    unittest.main()
