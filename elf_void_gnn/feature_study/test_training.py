import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import HeteroData

from elf_void_gnn import train


class FeatureStudyTrainingTests(unittest.TestCase):
    def test_validation_only_never_constructs_test_dataset(self):
        rows = {name: [] for name in ("train", "validation", "test")}
        with patch.object(train, "ShardedGraphDataset", side_effect=lambda root, data: data) as factory:
            datasets = train.build_split_datasets("unused", rows, "validation-only")
        self.assertEqual(set(datasets), {"train", "validation"})
        self.assertEqual(factory.call_count, 2)

    def test_feature_variant_statistics_use_transformed_training_features_and_targets(self):
        graph = HeteroData()
        graph["atom"].x = torch.arange(28, dtype=torch.float32).reshape(2, 14)
        graph["void"].x = torch.arange(18, dtype=torch.float32).reshape(2, 9)
        graph["void"].y = torch.tensor([2.0, 4.0])

        adapted = train.FeatureVariantDataset([graph], "atomic_number_only")
        stats = train.transformed_training_statistics(adapted)

        self.assertEqual(stats["atom_mean"][0], 7.0)
        self.assertEqual(stats["atom_mean"][1:5], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(stats["void_mean"], [0.0] * 9)
        self.assertEqual(stats["target_mean"], 3.0)
        self.assertAlmostEqual(stats["target_std"], 1.0)


if __name__ == "__main__":
    unittest.main()
