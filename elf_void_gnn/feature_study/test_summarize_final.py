"""Synthetic checks for the final summary's read-only metric adapters."""
import json
import tempfile
import unittest
from pathlib import Path

from elf_void_gnn.feature_study.summarize_final import read_test, read_validation, summarize


def metric(value):
    return {"pooled": {"rmse": value, "mae": value / 2, "r2": 0.5},
            "per_family": {"f1": {"rmse": value, "mae": value / 2, "r2": 0.5}}}


class FinalSummaryInputs(unittest.TestCase):
    def test_validation_and_frozen_test_metric_schemas(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for seed, value in ((11, 1.0), (12, 2.0), (13, 3.0)):
                (root / str(seed)).mkdir()
                (root / str(seed) / "metrics.json").write_text(json.dumps({
                    "feature_variant": "baseline", "seed": seed,
                    "validation": metric(value),
                }))
            validation = read_validation([root])
            rows = summarize(validation, "validation")
            self.assertEqual(rows[0]["n_seeds"], 3)
            self.assertEqual(rows[0]["rmse_mean"], 2.0)
            final_path = root / "final.json"
            final_path.write_text(json.dumps({"checkpoints": [{
                "variant": "baseline", "seed": 11, "test": metric(4.0)
            }]}))
            self.assertEqual(read_test(final_path)[("baseline", "11")]["pooled"]["rmse"], 4.0)


if __name__ == "__main__":
    unittest.main()
