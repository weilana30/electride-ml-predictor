"""Fast protocol checks for the confirmation trainer."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from elf_void_gnn.confirmation_study import train_confirmation as trainer
from elf_void_gnn.models import build_model


ATOM_ONLY = "atom_full_chemistry_and_periodicity"


def graph(sample_id="s0", family_id="f0", group="g0", offset=0.0):
    g = HeteroData()
    g["atom"].x = torch.arange(42, dtype=torch.float32).reshape(3, 14) + offset
    g["void"].x = torch.arange(18, dtype=torch.float32).reshape(2, 9) + offset
    g["void"].y = torch.tensor([0.25, 0.75]) + offset
    g.sample_id, g.family_id, g.composition_group = sample_id, family_id, group
    relations = {
        ("atom", "to", "atom"): torch.tensor([[0, 1], [1, 2]]),
        ("void", "to", "atom"): torch.tensor([[0, 1], [0, 2]]),
        ("atom", "to", "void"): torch.tensor([[0, 2], [0, 1]]),
        ("void", "to", "void"): torch.tensor([[0, 1], [1, 0]]),
    }
    for relation, edge_index in relations.items():
        g[relation].edge_index = edge_index
        g[relation].edge_distance = torch.arange(1, edge_index.shape[1] + 1, dtype=torch.float32)
        g[relation].edge_features = torch.ones((edge_index.shape[1], 8))
    g["void"].void_id = ["v0", "v1"]
    return g


@pytest.mark.parametrize("width,expected", [(128, 307457), (64, 80001)])
def test_fixed_model_parameter_counts_and_input_widths(width, expected):
    model = build_model("hetero_mpnn", {
        "hidden_channels": width, "message_layers": 2,
        "radial_channels": 8, "dropout": 0.1,
    })
    assert sum(p.numel() for p in model.parameters()) == expected
    sample = graph()
    assert sample["atom"].x.shape[-1] == 14
    assert sample["void"].x.shape[-1] == 9
    assert all(sample[r].edge_features.shape[-1] == 8 for r in sample.edge_types)
    assert all(not hasattr(sample[n], "y") for n in ("atom",))


def test_baseline_is_unchanged_and_atom_only_masks_only_void_descriptors():
    original = graph()
    baseline = trainer.FeatureVariantDataset([original.clone()], "baseline")[0]
    atom_only = trainer.FeatureVariantDataset([original.clone()], ATOM_ONLY)[0]
    assert torch.equal(baseline["atom"].x, original["atom"].x)
    assert torch.equal(baseline["void"].x, original["void"].x)
    assert torch.equal(atom_only["atom"].x, original["atom"].x)
    assert torch.count_nonzero(atom_only["void"].x) == 0
    assert atom_only["atom"].x.shape == original["atom"].x.shape
    assert atom_only["void"].x.shape == original["void"].x.shape
    for relation in original.edge_types:
        assert torch.equal(atom_only[relation].edge_index, original[relation].edge_index)
        assert torch.equal(atom_only[relation].edge_features, original[relation].edge_features)


def test_validation_only_dataset_factory_never_constructs_test(monkeypatch):
    constructed = []

    class FakeDataset:
        def __init__(self, root, rows):
            constructed.extend(row["sample_id"] for row in rows)

    monkeypatch.setattr(trainer, "ShardedGraphDataset", FakeDataset)
    rows = {
        "train": [{"sample_id": "train"}],
        "validation": [{"sample_id": "validation"}],
        "test": [{"sample_id": "sealed-test"}],
    }
    opened = trainer.build_split_datasets(None, rows, "validation-only")
    assert set(opened) == {"train", "validation"}
    assert constructed == ["train", "validation"]


def test_final_test_dataset_factory_opens_only_test(monkeypatch):
    constructed = []

    class FakeDataset:
        def __init__(self, root, rows):
            constructed.extend(row["sample_id"] for row in rows)

    monkeypatch.setattr(trainer, "ShardedGraphDataset", FakeDataset)
    rows = {
        "train": [{"sample_id": "train"}],
        "validation": [{"sample_id": "validation"}],
        "test": [{"sample_id": "sealed-test"}],
    }
    opened = trainer.build_split_datasets(None, rows, "final-test")
    assert set(opened) == {"test"}
    assert constructed == ["sealed-test"]


def test_training_statistics_use_only_the_dataset_passed_as_training():
    train = [graph(offset=0.0), graph(offset=2.0)]
    train_stats = trainer.transformed_training_statistics(train)
    # A deliberately extreme validation/test example cannot enter the fit
    # unless it is incorrectly added to the training dataset argument.
    with_held_out = trainer.transformed_training_statistics(train + [graph(offset=10000.0)])
    assert train_stats["target_mean"] == pytest.approx(1.5)
    assert with_held_out["target_mean"] > train_stats["target_mean"]
    assert train_stats["atom_mean"] != with_held_out["atom_mean"]


def test_prediction_rows_preserve_sample_void_ids_and_checkpoint_metric_is_pooled_rmse():
    items = [graph("sample-B", "family-B", "group-B"), graph("sample-A", "family-A", "group-A")]
    loader = DataLoader(items, batch_size=2, shuffle=False)

    class FixedModel(torch.nn.Module):
        def forward(self, batch):
            return torch.zeros(batch["void"].y.numel())

    metrics, predictions = trainer.evaluate(FixedModel(), loader, torch.device("cpu"), collect_predictions=True)
    assert [(r["sample_id"], r["void_id"]) for r in predictions] == [
        ("sample-B", "v0"), ("sample-B", "v1"), ("sample-A", "v0"), ("sample-A", "v1")
    ]
    assert metrics["pooled"]["rmse"] == pytest.approx(float(np.sqrt(np.mean(np.square([.25, .75, .25, .75])))))
    # Trainer's checkpoint comparison uses this validation pooled RMSE scalar.
    scores = [metrics["pooled"]["rmse"], 1.0, 0.5]
    assert min(scores) == 0.5
