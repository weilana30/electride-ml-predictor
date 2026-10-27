"""Two-layer distance-aware message passing for atom/void heterogeneous graphs."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


RELATIONS = {
    "atom_to_atom": (("atom", "to", "atom"), "atom", "atom"),
    "void_to_atom": (("void", "to", "atom"), "void", "atom"),
    "atom_to_void": (("atom", "to", "void"), "atom", "void"),
    "void_to_void": (("void", "to", "void"), "void", "void"),
}


class RadialDistance(nn.Module):
    def __init__(self, channels: int, maximum: float):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, maximum, channels))
        self.width = maximum / max(channels - 1, 1)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        delta = distances.reshape(-1, 1) - self.centers.reshape(1, -1)
        return torch.exp(-torch.square(delta / self.width))


class HeteroMessageLayer(nn.Module):
    def __init__(self, hidden: int, radial_channels: int, dropout: float, update_atoms: bool = True):
        super().__init__()
        self.update_atoms = update_atoms
        active_relations = {name: value for name, value in RELATIONS.items() if update_atoms or value[2] == "void"}
        self.radial = RadialDistance(radial_channels, maximum=6.0)
        self.messages = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden + radial_channels, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            for name in active_relations
        })
        self.self_updates = nn.ModuleDict({
            **({"atom": nn.Linear(hidden, hidden)} if update_atoms else {}),
            "void": nn.Linear(hidden, hidden),
        })
        self.norms = nn.ModuleDict({
            **({"atom": nn.LayerNorm(hidden)} if update_atoms else {}),
            "void": nn.LayerNorm(hidden),
        })
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        atom_state: torch.Tensor,
        void_state: torch.Tensor,
        graph: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = {"atom": atom_state, "void": void_state}
        aggregated = {
            "atom": atom_state.new_zeros(atom_state.shape),
            "void": void_state.new_zeros(void_state.shape),
        }
        for name, (edge_type, source_type, destination_type) in RELATIONS.items():
            relation = graph[edge_type]
            edge_index = relation.edge_index
            if name not in self.messages or edge_index.numel() == 0:
                continue
            source_index, destination_index = edge_index
            radial = self.radial(relation.edge_distance)
            messages = self.messages[name](
                torch.cat((states[source_type][source_index], radial), dim=-1)
            )
            destination_count = states[destination_type].shape[0]
            relation_sum = messages.new_zeros((destination_count, messages.shape[-1]))
            relation_sum.index_add_(0, destination_index, messages)
            degrees = messages.new_zeros((destination_count, 1))
            degrees.index_add_(
                0, destination_index, messages.new_ones((len(destination_index), 1))
            )
            aggregated[destination_type] += relation_sum / degrees.clamp_min(1.0)

        updated = {}
        for node_type, state in states.items():
            if node_type not in self.self_updates:
                updated[node_type] = state
                continue
            message = self.self_updates[node_type](state) + aggregated[node_type]
            updated[node_type] = self.norms[node_type](
                state + self.dropout(torch.relu(message))
            )
        return updated["atom"], updated["void"]


class HeteroMPNN(nn.Module):
    """Predict raw ELF-at-center values for every void node in a batch."""

    def __init__(
        self,
        *,
        hidden_channels: int = 64,
        message_layers: int = 2,
        radial_channels: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.atom_encoder = nn.Sequential(
            nn.Linear(14, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)
        )
        self.void_encoder = nn.Sequential(
            nn.Linear(9, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)
        )
        self.layers = nn.ModuleList([
            HeteroMessageLayer(hidden_channels, radial_channels, dropout, update_atoms=i < message_layers - 1)
            for i in range(message_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
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

    def forward(self, graph: Any) -> torch.Tensor:
        atom_x = graph["atom"].x
        atom_continuous = (atom_x[:, :5] - self.atom_mean) / self.atom_std
        atom_x = torch.cat((atom_continuous, atom_x[:, 5:]), dim=-1)
        void_x = (graph["void"].x - self.void_mean) / self.void_std
        atom_state = self.atom_encoder(atom_x)
        void_state = self.void_encoder(void_x)
        for layer in self.layers:
            atom_state, void_state = layer(atom_state, void_state, graph)
        normalized = self.head(void_state).squeeze(-1)
        return normalized * self.target_std + self.target_mean
