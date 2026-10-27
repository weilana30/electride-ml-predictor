"""Fixed-shape, leakage-safe feature variants for controlled model studies."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch


# Columns: atom [Z, X, radius, period, group, s/p/d/f, validity x5]; void
# [clearance, nearest atom distance, counts <=3/4/5A, nearest four distances].
# Only chemistry and geometry derived from structures are used.
FEATURE_VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {"mode": "baseline", "description": "Unmodified current graph inputs."},
    "atomic_number_nearest_void_distances_edge_distances": {
        "mode": "minimal", "description": "Z, four nearest void-to-atom distances (deduplicated), and edge distances."
    },
    "atomic_number_only": {"mode": "mask", "atom_columns": (0,), "void_columns": ()},
    "atom_chemistry_only": {"mode": "mask", "atom_columns": (0, 1, 2, 3, 4), "void_columns": ()},
    "atom_periodic_table_groups": {"mode": "mask", "atom_columns": (5, 6, 7, 8), "void_columns": ()},
    "atom_validity_only": {"mode": "mask", "atom_columns": tuple(range(9, 14)), "void_columns": ()},
    "atom_full_chemistry_and_periodicity": {
        "mode": "mask", "atom_columns": tuple(range(14)), "void_columns": (),
        "description": "All stored atom chemistry and periodic-table channels."
    },
    "void_nearest_distance_only": {"mode": "mask", "atom_columns": (), "void_columns": (1,)},
    "void_clearance_only": {"mode": "mask", "atom_columns": (), "void_columns": (0,)},
    "void_coordination_counts_only": {"mode": "mask", "atom_columns": (), "void_columns": (2, 3, 4)},
    "void_nearest_distances_only": {"mode": "mask", "atom_columns": (), "void_columns": (5, 6, 7, 8)},
    "void_full_geometry": {
        "mode": "mask", "atom_columns": (), "void_columns": tuple(range(9)),
        "description": "All stored clearance, neighbor counts, and nearest-distance geometry."
    },
    "atomic_number_void_geometry": {
        "mode": "mask", "atom_columns": (0,), "void_columns": tuple(range(9))
    },
    "edge_distances_only": {
        "mode": "mask", "atom_columns": (), "void_columns": (),
        "description": "Topology+distance diagnostic: zero all node values, retain four edge sets and their distances."
    },
    "no_edge_distance": {
        "mode": "ablation", "atom_columns": tuple(range(14)), "void_columns": tuple(range(9)),
        "zero_edge_relations": ("all",),
        "description": "Keep all node features and topology; zero the 8-channel distance encoding on every relation."
    },
    "no_atom_atom_distance": {
        "mode": "ablation", "atom_columns": tuple(range(14)), "void_columns": tuple(range(9)),
        "zero_edge_relations": ("atom_to_atom",),
        "description": "Keep nodes/topology; remove atom-atom edge-distance signal."
    },
    "no_void_atom_distance": {
        "mode": "ablation", "atom_columns": tuple(range(14)), "void_columns": tuple(range(9)),
        "zero_edge_relations": ("void_to_atom", "atom_to_void"),
        "description": "Keep nodes/topology; remove void-atom and reverse edge-distance signal together."
    },
    "no_void_void_distance": {
        "mode": "ablation", "atom_columns": tuple(range(14)), "void_columns": tuple(range(9)),
        "zero_edge_relations": ("void_to_void",),
        "description": "Keep nodes/topology; remove void-void edge-distance signal."
    },
    "void_only_no_edge_distance": {
        "mode": "ablation", "atom_columns": (), "void_columns": tuple(range(9)),
        "zero_edge_relations": ("all",),
        "description": "All void features, zero atom features, and zero edge-distance channels."
    },
    "atom_only_no_edge_distance": {
        "mode": "ablation", "atom_columns": tuple(range(14)), "void_columns": (),
        "zero_edge_relations": ("all",),
        "description": "All atom features, zero void features, and zero edge-distance channels."
    },
    "atom_derived_coordination": {
        "mode": "atom_geometry",
        "description": "Preserve Z; replace other atom channels with shell counts, nearest bond distances, and nearest void distances."
    },
    "void_derived_neighborhood": {
        "mode": "void_geometry",
        "description": "Preserve clearance; replace remaining channels with void-network shell counts and nearest void-neighbor distances."
    },
    "void_derived_environment": {
        "mode": "void_environment",
        "description": "Replace all void channels with five void-atom coordination shells, mean/spread of nearest four distances, and mean/spread of neighboring atomic numbers."
    },
    "richer_edge_context": {
        "mode": "edge_context",
        "description": "Preserve node features; encode bond distance plus source/destination relation degrees and smooth distance scales."
    },
}


def _atom_geometry(graph: Any, atom_x: torch.Tensor) -> torch.Tensor:
    """Make 13 fixed structural channels per atom from current edge topology."""
    device, dtype = atom_x.device, atom_x.dtype
    geometry = atom_x.new_zeros((atom_x.shape[0], 13))
    aa = graph[("atom", "to", "atom")]
    if aa.edge_index.numel():
        src, dst = aa.edge_index
        distance = aa.edge_distance.to(device=device, dtype=dtype).reshape(-1)
        # Coordination counts per destination at five physically useful shells.
        for column, cutoff in enumerate((1.5, 2.0, 2.5, 3.0, 4.0)):
            selected = distance <= cutoff
            geometry[:, column].index_add_(0, dst[selected], torch.ones_like(distance[selected]))
        for node in range(len(atom_x)):
            incident = distance[dst == node]
            if incident.numel():
                nearest = incident.sort().values[:4]
                geometry[node, 5:5 + len(nearest)] = nearest
    va = graph[("void", "to", "atom")]
    if va.edge_index.numel():
        atom_index = va.edge_index[1]
        distance = va.edge_distance.to(device=device, dtype=dtype).reshape(-1)
        for node in range(len(atom_x)):
            nearest = distance[atom_index == node].sort().values[:4]
            if nearest.numel():
                geometry[node, 9:9 + len(nearest)] = nearest
    return torch.cat((atom_x[:, :1], geometry), dim=1)


def _edge_context(graph: Any, relation: Any) -> torch.Tensor:
    store = graph[relation]
    distance = store.edge_distance.reshape(-1).clamp_min(1e-3)
    source, destination = store.edge_index
    src_type, _, dst_type = relation
    src_count = int(graph[src_type].x.shape[0])
    dst_count = int(graph[dst_type].x.shape[0])
    src_degree = distance.new_zeros(src_count)
    dst_degree = distance.new_zeros(dst_count)
    if len(source):
        src_degree.index_add_(0, source, torch.ones_like(distance))
        dst_degree.index_add_(0, destination, torch.ones_like(distance))
    scales = distance.new_tensor([0.75, 1.5, 3.0, 6.0])
    return torch.cat((
        (distance / 6.0)[:, None], (1.0 / distance)[:, None],
        (src_degree[source] / max(dst_count, 1))[:, None],
        (dst_degree[destination] / max(src_count, 1))[:, None],
        torch.exp(-distance[:, None] / scales[None, :]),
    ), dim=-1)


def _void_geometry(graph: Any, void_x: torch.Tensor) -> torch.Tensor:
    """Keep clearance and derive eight channels from the void-void graph."""
    geometry = void_x.new_zeros((void_x.shape[0], 8))
    relation = graph[("void", "to", "void")]
    if relation.edge_index.numel():
        source, destination = relation.edge_index
        distance = relation.edge_distance.to(device=void_x.device, dtype=void_x.dtype).reshape(-1)
        # Directed neighbor shell populations, then the four nearest neighbors.
        for column, cutoff in enumerate((1.5, 2.0, 3.0, 4.0)):
            selected = distance <= cutoff
            geometry[:, column].index_add_(0, destination[selected], torch.ones_like(distance[selected]))
        for node in range(len(void_x)):
            nearest = distance[destination == node].sort().values[:4]
            if nearest.numel():
                geometry[node, 4:4 + len(nearest)] = nearest
    return torch.cat((void_x[:, :1], geometry), dim=1)


def _void_environment(graph: Any, void_x: torch.Tensor) -> torch.Tensor:
    """Nine local environment values from void-atom edges and atomic numbers."""
    features = void_x.new_zeros((void_x.shape[0], 9))
    relation = graph[("void", "to", "atom")]
    if not relation.edge_index.numel():
        return features
    void_index, atom_index = relation.edge_index
    distances = relation.edge_distance.to(device=void_x.device, dtype=void_x.dtype).reshape(-1)
    atomic_numbers = graph["atom"].x[:, 0]
    for node in range(len(void_x)):
        selected = void_index == node
        local_distances = distances[selected]
        local_z = atomic_numbers[atom_index[selected]]
        for column, cutoff in enumerate((2.0, 2.5, 3.0, 4.0, 5.0)):
            features[node, column] = (local_distances <= cutoff).sum()
        nearest = local_distances.sort().values[:4]
        if nearest.numel():
            features[node, 5] = nearest.mean()
            features[node, 6] = nearest.std(unbiased=False)
        if local_z.numel():
            features[node, 7] = local_z.mean()
            features[node, 8] = local_z.std(unbiased=False)
    return features


def apply_feature_variant(graph: Any, name: str, *, copy_graph: bool = False) -> Any:
    """Apply a named fixed-shape variant, optionally to a deep graph copy.

    `richer_edge_context` augments each existing edge with eight structural
    channels. Existing graph topology, relation ordering and model widths stay
    unchanged. The stored graph does not include periodic displacement vectors,
    so directional bond-orientation features cannot be reconstructed here.
    """
    if name not in FEATURE_VARIANTS:
        raise KeyError(f"Unknown feature variant {name!r}; choose from {sorted(FEATURE_VARIANTS)}")
    if copy_graph:
        graph = deepcopy(graph)
    config = FEATURE_VARIANTS[name]
    mode = config["mode"]
    if mode == "baseline":
        return graph
    if mode == "ablation":
        for node_type, key in (("atom", "atom_columns"), ("void", "void_columns")):
            values = graph[node_type].x
            keep = torch.zeros(values.shape[1], dtype=torch.bool, device=values.device)
            columns = config[key]
            if columns:
                keep[list(columns)] = True
            graph[node_type].x = values * keep.to(dtype=values.dtype)[None, :]
        zero_relations = config["zero_edge_relations"]
        for relation in graph.edge_types:
            # Distances are inputs to a fixed 8-channel radial transform. Zero
            # the transform itself so the edge index/topology remains intact.
            if "all" in zero_relations or relation[1] == "to" and (
                (relation[0] == "atom" and relation[2] == "atom" and "atom_to_atom" in zero_relations)
                or (relation[0] == "void" and relation[2] == "void" and "void_to_void" in zero_relations)
                or (relation[0] == "void" and relation[2] == "atom" and "void_to_atom" in zero_relations)
                or (relation[0] == "atom" and relation[2] == "void" and "atom_to_void" in zero_relations)
            ):
                store = graph[relation]
                store.edge_features = store.edge_distance.new_zeros((store.edge_distance.numel(), 8))
        return graph
    if mode == "minimal":
        graph["atom"].x = graph["atom"].x * torch.nn.functional.one_hot(
            torch.zeros(1, dtype=torch.long, device=graph["atom"].x.device), num_classes=14
        ).to(graph["atom"].x.dtype)
        keep = torch.zeros(9, dtype=torch.bool, device=graph["void"].x.device)
        keep[[5, 6, 7, 8]] = True
        graph["void"].x = graph["void"].x * keep.to(graph["void"].x.dtype)[None, :]
        return graph
    if mode == "atom_geometry":
        graph["atom"].x = _atom_geometry(graph, graph["atom"].x)
        return graph
    if mode == "void_geometry":
        graph["void"].x = _void_geometry(graph, graph["void"].x)
        return graph
    if mode == "void_environment":
        graph["void"].x = _void_environment(graph, graph["void"].x)
        return graph
    if mode == "edge_context":
        for relation in graph.edge_types:
            graph[relation].edge_features = _edge_context(graph, relation)
        return graph
    for node_type, key in (("atom", "atom_columns"), ("void", "void_columns")):
        values = graph[node_type].x
        keep = torch.zeros(values.shape[1], dtype=torch.bool, device=values.device)
        columns = config.get(key, ())
        if columns:
            if max(columns) >= values.shape[1]:
                raise ValueError(f"{node_type} feature width {values.shape[1]} is smaller than required by {name}")
            keep[list(columns)] = True
        graph[node_type].x = values * keep.to(dtype=values.dtype)[None, :]
    return graph
