"""Additional per-void regressors sharing the ELF-center graph contract."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .hetero_mpnn import RELATIONS, RadialDistance


class _NormalizedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("atom_mean", torch.zeros(5))
        self.register_buffer("atom_std", torch.ones(5))
        self.register_buffer("void_mean", torch.zeros(9))
        self.register_buffer("void_std", torch.ones(9))
        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_std", torch.tensor(1.0))

    def set_training_statistics(self, stats: Mapping[str, Any]) -> None:
        self.atom_mean.copy_(torch.as_tensor(stats["atom_mean"], device=self.atom_mean.device))
        self.atom_std.copy_(torch.as_tensor(stats["atom_std"], device=self.atom_std.device))
        self.void_mean.copy_(torch.as_tensor(stats["void_mean"], device=self.void_mean.device))
        self.void_std.copy_(torch.as_tensor(stats["void_std"], device=self.void_std.device))
        self.target_mean.fill_(float(stats["target_mean"]))
        self.target_std.fill_(float(stats["target_std"]))

    def _void_features(self, graph: Any) -> torch.Tensor:
        return (graph["void"].x - self.void_mean) / self.void_std

    def _atom_features(self, graph: Any) -> torch.Tensor:
        x = graph["atom"].x
        return torch.cat(((x[:, :5] - self.atom_mean) / self.atom_std, x[:, 5:]), dim=-1)

    def _raw(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * self.target_std + self.target_mean


class VoidMLP(_NormalizedModel):
    """Per-void MLP baseline; uses void structural features only."""

    def __init__(self, hidden_channels: int = 64, message_layers: int = 2,
                 dropout: float = 0.1, radial_channels: int = 8):
        super().__init__()
        del radial_channels
        layers: list[nn.Module] = []
        width = 9
        for _ in range(max(1, message_layers)):
            layers.extend((nn.Linear(width, hidden_channels), nn.ReLU(), nn.Dropout(dropout)))
            width = hidden_channels
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, graph: Any) -> torch.Tensor:
        return self._raw(self.network(self._void_features(graph)).squeeze(-1))


class FixedContextMLP(_NormalizedModel):
    """MLP over void features and fixed distance-weighted neighboring atom summaries."""

    def __init__(self, hidden_channels: int = 64, message_layers: int = 2,
                 dropout: float = 0.1, radial_channels: int = 8):
        super().__init__()
        del radial_channels
        layers: list[nn.Module] = []
        width = 9 + 14 + 3  # void; weighted atom-feature mean; distance mean/std/count
        for _ in range(max(1, message_layers)):
            layers.extend((nn.Linear(width, hidden_channels), nn.ReLU(), nn.Dropout(dropout)))
            width = hidden_channels
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, graph: Any) -> torch.Tensor:
        void = self._void_features(graph)
        atom = self._atom_features(graph)
        relation = graph[("void", "to", "atom")]
        edge_index = relation.edge_index
        distances = relation.edge_distance
        count = void.shape[0]
        sums = atom.new_zeros((count, atom.shape[1]))
        weights = atom.new_zeros((count, 1))
        dsum = atom.new_zeros((count, 1))
        d2sum = atom.new_zeros((count, 1))
        if edge_index.numel():
            src, dst = edge_index
            weight = torch.exp(-distances.reshape(-1, 1).clamp_min(0) / 2.0)
            sums.index_add_(0, src, atom[dst] * weight)
            weights.index_add_(0, src, weight)
            dsum.index_add_(0, src, distances.reshape(-1, 1) * weight)
            d2sum.index_add_(0, src, distances.reshape(-1, 1).square() * weight)
        denom = weights.clamp_min(1e-8)
        context = sums / denom
        dmean = dsum / denom
        dvar = (d2sum / denom - dmean.square()).clamp_min(0)
        log_weight_sum = torch.log1p(weights)
        features = torch.cat((void, context, dmean, dvar.sqrt(), log_weight_sum), dim=-1)
        return self._raw(self.network(features).squeeze(-1))


class _GatedLayer(nn.Module):
    def __init__(self, hidden: int, radial_channels: int, dropout: float):
        super().__init__()
        self.radial = RadialDistance(radial_channels, maximum=6.0)
        self.message = nn.ModuleDict({
            name: nn.Linear(hidden, hidden, bias=False) for name in RELATIONS
        })
        self.gate = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(radial_channels, hidden), nn.Sigmoid())
            for name in RELATIONS
        })
        self.update = nn.ModuleDict({
            "atom": nn.GRUCell(hidden, hidden), "void": nn.GRUCell(hidden, hidden)
        })
        self.norm = nn.ModuleDict({"atom": nn.LayerNorm(hidden), "void": nn.LayerNorm(hidden)})
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: dict[str, torch.Tensor], graph: Any) -> dict[str, torch.Tensor]:
        aggregate = {kind: torch.zeros_like(value) for kind, value in states.items()}
        for name, (edge_type, source_type, destination_type) in RELATIONS.items():
            relation = graph[edge_type]
            if relation.edge_index.numel() == 0:
                continue
            src, dst = relation.edge_index
            radial = self.radial(relation.edge_distance)
            msg = self.message[name](states[source_type][src]) * self.gate[name](radial)
            total = torch.zeros_like(aggregate[destination_type])
            degree = msg.new_zeros((aggregate[destination_type].shape[0], 1))
            total.index_add_(0, dst, msg)
            degree.index_add_(0, dst, msg.new_ones((len(dst), 1)))
            aggregate[destination_type] += total / degree.clamp_min(1.0)
        updated = {}
        for kind in states:
            candidate = self.update[kind](aggregate[kind], states[kind])
            updated[kind] = self.norm[kind](states[kind] + self.dropout(candidate))
        return updated


class GatedMPNN(_NormalizedModel):
    """Distance-gated relation-specific message passing with GRU updates."""

    def __init__(self, hidden_channels: int = 64, message_layers: int = 2,
                 dropout: float = 0.1, radial_channels: int = 8):
        super().__init__()
        self.atom_encoder = nn.Sequential(nn.Linear(14, hidden_channels), nn.ReLU(),
                                          nn.Linear(hidden_channels, hidden_channels))
        self.void_encoder = nn.Sequential(nn.Linear(9, hidden_channels), nn.ReLU(),
                                          nn.Linear(hidden_channels, hidden_channels))
        self.layers = nn.ModuleList([
            _GatedLayer(hidden_channels, radial_channels, dropout)
            for _ in range(max(1, message_layers))
        ])
        self.head = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_channels, 1))

    def forward(self, graph: Any) -> torch.Tensor:
        states = {"atom": self.atom_encoder(self._atom_features(graph)),
                  "void": self.void_encoder(self._void_features(graph))}
        for layer in self.layers:
            states = layer(states, graph)
        return self._raw(self.head(states["void"]).squeeze(-1))
