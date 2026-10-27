"""Utilities for controlled feature ablation studies."""

from .adapters import FEATURE_VARIANTS, apply_feature_variant

__all__ = ["FEATURE_VARIANTS", "apply_feature_variant"]
