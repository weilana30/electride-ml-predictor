import numpy as np
import pandas as pd
import pytest

from elf_void_gnn.confirmation_study.report import (
    CONFIGURATIONS, SEEDS, grouped_bootstrap, grouped_metrics, validate_manifest,
    validate_prediction_identity,
)


def test_requires_all_nine_unique_configuration_seed_pairs():
    rows = [{"run_id": f"{c}-{s}", "configuration": c, "seed": s, "run_dir": "."}
            for c in CONFIGURATIONS for s in SEEDS]
    validate_manifest(pd.DataFrame(rows))
    with pytest.raises(ValueError, match="duplicate"):
        validate_manifest(pd.DataFrame(rows + [rows[0]]))
    with pytest.raises(ValueError, match="exactly nine"):
        validate_manifest(pd.DataFrame(rows[:-1]))


def test_macro_metrics_weight_groups_equally_not_rows():
    pred = pd.DataFrame({"g": ["small", "large", "large"],
                         "y_true": [0., 0., 0.], "y_pred": [0., 2., 2.]})
    got = grouped_metrics(pred, "g")
    assert got["pooled_rmse"] == pytest.approx(np.sqrt(8 / 3))
    assert got["macro_rmse"] == pytest.approx(1.)
    assert got["groups"].set_index("group_id").loc["large", "rmse"] == 2.


def test_family_metrics_need_no_composition_group_column():
    pred = pd.DataFrame({"family_id": ["f1", "f2", "f2"],
                         "y_true": [0., 0., 0.], "y_pred": [0., 2., 2.]})
    got = grouped_metrics(pred, "family_id")
    assert got["macro_rmse"] == pytest.approx(1.)
    assert got["group_count"] == 2


def test_bootstrap_uses_group_values_and_is_deterministic():
    # A single large group contributes one group-level observation, not its row count.
    values = np.array([0., 10.])
    first = grouped_bootstrap(values, draws=500, seed=31)
    assert first == grouped_bootstrap(values, draws=500, seed=31)
    assert first[0] <= 5 <= first[1]


def test_prediction_identity_allows_multiple_voids_per_sample():
    rows = pd.DataFrame({"sample_id": ["s1", "s1", "s2"], "void_id": [0, 1, 0]})
    validate_prediction_identity(rows, "run", "validation")
    with pytest.raises(ValueError, match="sample_id, void_id"):
        validate_prediction_identity(pd.concat([rows, rows.iloc[[0]]]), "run", "validation")


def test_prediction_identity_requires_void_id():
    with pytest.raises(ValueError, match="sample_id, void_id"):
        validate_prediction_identity(pd.DataFrame({"sample_id": ["s1"]}), "run", "test")
