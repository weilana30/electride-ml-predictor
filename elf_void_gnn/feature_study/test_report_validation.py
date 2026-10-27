"""Focused provenance checks for validation report discovery."""
from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from elf_void_gnn.feature_study.report_validation import (
    EXPECTED_ARCHITECTURE,
    split_reference,
    validation_exclusion_reason,
)


class ReportProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        split_path = self.dataset / "splits.parquet"
        rows = [
            {"sample_id": "tr1", "family_id": "train-a", "split": "train"},
            {"sample_id": "tr2", "family_id": "train-b", "split": "train"},
            {"sample_id": "va1", "family_id": "val-a", "split": "validation"},
            {"sample_id": "va2", "family_id": "val-b", "split": "validation"},
            {"sample_id": "te1", "family_id": "test-a", "split": "test"},
            {"sample_id": "te2", "family_id": "test-b", "split": "test"},
        ]
        pq.write_table(pa.Table.from_pylist(rows), split_path)
        self.split_path = split_path
        self.split_id, self.counts, self.families = split_reference(split_path)
        self.run_dir.joinpath("config.yaml").write_text(
            f"""dataset_root: {self.dataset}\nmodel:\n  type: hetero_mpnn\n  hidden_channels: 64\n  message_layers: 2\n  radial_channels: 8\n  dropout: 0.1\n"""
        )
        self.metrics = {
            "status": "completed",
            "evaluation_mode": "validation-only",
            "test": None,
            "feature_variant": "baseline",
            "seed": 20261023,
            "split_counts": deepcopy(self.counts),
            "validation": {"per_family": {key: {} for key in self.families["validation"]}},
            "architecture_signature": dict(EXPECTED_ARCHITECTURE),
        }

    def tearDown(self):
        self.temp.cleanup()

    def reason(self):
        return validation_exclusion_reason(
            self.run_dir, self.metrics, self.split_path, self.counts, self.families, 2
        )

    def test_matching_run_provenance_and_full_architecture_is_accepted(self):
        self.assertTrue(self.split_id)
        self.assertIsNone(self.reason())

    def test_split_count_mismatch_is_rejected(self):
        self.metrics["split_counts"]["train"]["samples"] += 1
        self.assertIn("split_counts", self.reason())

    def test_wrong_dataset_path_is_rejected(self):
        self.run_dir.joinpath("config.yaml").write_text(
            "dataset_root: /another/dataset\nmodel:\n  type: hetero_mpnn\n"
            "  hidden_channels: 64\n  message_layers: 2\n"
            "  radial_channels: 8\n  dropout: 0.1\n"
        )
        self.assertIn("dataset_root", self.reason())

    def test_each_architecture_signature_field_is_checked(self):
        for key in EXPECTED_ARCHITECTURE:
            with self.subTest(key=key):
                self.metrics["architecture_signature"] = dict(EXPECTED_ARCHITECTURE)
                self.metrics["architecture_signature"][key] = -1
                self.assertIn("architecture_signature", self.reason())

    def test_test_metrics_are_refused(self):
        self.metrics["test"] = {"pooled": {"rmse": 1.0}}
        self.assertIn("test metrics", self.reason())


if __name__ == "__main__":
    unittest.main()
