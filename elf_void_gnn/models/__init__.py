"""Model registry used by the shared training and evaluation pipeline."""
from __future__ import annotations

from typing import Any, Mapping

from .hetero_mpnn import HeteroMPNN
from .variants import FixedContextMLP, GatedMPNN, VoidMLP


MODEL_REGISTRY = {
    "hetero_mpnn": HeteroMPNN,
    "void_mlp": VoidMLP,
    "fixed_context_mlp": FixedContextMLP,
    "gated_mpnn": GatedMPNN,
}


def build_model(name: str, config: Mapping[str, Any]):
    try:
        model_type = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}") from exc
    return model_type(**dict(config))


__all__ = ["FixedContextMLP", "GatedMPNN", "HeteroMPNN", "MODEL_REGISTRY", "VoidMLP", "build_model"]
