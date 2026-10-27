"""Compact regression metrics for pooled and family-level analysis."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def regression_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(actual), dtype=np.float64)
    p = np.asarray(list(predicted), dtype=np.float64)
    if y.size == 0 or p.shape != y.shape:
        raise ValueError("Metrics require equally sized, non-empty target and prediction arrays")
    residual = y - p
    mse = float(np.mean(np.square(residual)))
    denominator = float(np.sum(np.square(y - np.mean(y))))
    r2 = 1.0 - float(np.sum(np.square(residual))) / denominator if denominator > 0 else (
        1.0 if mse == 0 else 0.0
    )
    return {
        "count": int(y.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2),
    }


def grouped_metrics(
    actual: Iterable[float],
    predicted: Iterable[float],
    group_ids: Iterable[str],
) -> dict[str, Any]:
    y = np.asarray(list(actual), dtype=np.float64)
    p = np.asarray(list(predicted), dtype=np.float64)
    groups = list(group_ids)
    if len(groups) != len(y) or p.shape != y.shape:
        raise ValueError("Group IDs, targets, and predictions must have matching lengths")
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[str(group)].append(index)
    details = {}
    for group, indices in sorted(by_group.items()):
        details[group] = regression_metrics(y[indices], p[indices])
    macro = {
        metric: float(np.mean([values[metric] for values in details.values()]))
        for metric in ("mae", "rmse", "r2")
    }
    macro["family_count"] = len(details)
    return {"macro_family": macro, "per_family": details}


def evaluation_metrics(
    actual: Iterable[float],
    predicted: Iterable[float],
    family_ids: Iterable[str],
) -> dict[str, Any]:
    y = list(actual)
    p = list(predicted)
    groups = list(family_ids)
    return {
        "pooled": regression_metrics(y, p),
        **grouped_metrics(y, p, groups),
    }
