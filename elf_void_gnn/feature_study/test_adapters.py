import unittest

import torch
from torch_geometric.data import HeteroData

from elf_void_gnn.feature_study import FEATURE_VARIANTS, apply_feature_variant


class FeatureAdapterTests(unittest.TestCase):
    def graph(self):
        graph = HeteroData()
        graph["atom"].x = torch.arange(42, dtype=torch.float32).reshape(3, 14)
        graph["void"].x = torch.arange(18, dtype=torch.float32).reshape(2, 9)
        graph["void"].y = torch.tensor([99.0, 100.0])
        for relation in (("atom", "to", "atom"), ("void", "to", "atom"),
                         ("atom", "to", "void"), ("void", "to", "void")):
            graph[relation].edge_index = torch.tensor([[0], [0]])
            graph[relation].edge_distance = torch.tensor([1.25])
        graph.sample_id = "sample"
        return graph

    def test_variants_preserve_shapes_edges_and_supervision(self):
        graph = self.graph()
        for name in FEATURE_VARIANTS:
            varied = apply_feature_variant(graph, name, copy_graph=True)
            self.assertEqual(tuple(varied["atom"].x.shape), (3, 14))
            self.assertEqual(tuple(varied["void"].x.shape), (2, 9))
            self.assertEqual(tuple(varied["atom", "to", "atom"].edge_distance.shape), (1,))
            self.assertTrue(torch.equal(varied["void"].y, graph["void"].y))

    def test_minimal_keeps_z_four_nearest_distances_and_edge_distances(self):
        graph = apply_feature_variant(self.graph(), "atomic_number_nearest_void_distances_edge_distances")
        self.assertTrue(torch.equal(graph["atom"].x[:, 0], torch.tensor([0.0, 14.0, 28.0])))
        self.assertTrue(torch.count_nonzero(graph["atom"].x[:, 1:]) == 0)
        self.assertTrue(torch.equal(graph["void"].x[:, 5:9], torch.tensor([[5., 6., 7., 8.], [14., 15., 16., 17.]])))
        keep = torch.zeros(9, dtype=torch.bool); keep[5:9] = True
        self.assertTrue(torch.count_nonzero(graph["void"].x[:, ~keep]) == 0)
        for relation in (("atom", "to", "atom"), ("void", "to", "atom"),
                         ("atom", "to", "void"), ("void", "to", "void")):
            self.assertEqual(graph[relation].edge_distance.item(), 1.25)
        self.assertFalse(hasattr(graph["atom"], "y"))

    def test_richer_variants_add_derived_structure_without_masking_inputs(self):
        original = self.graph()
        edge_graph = apply_feature_variant(original, "richer_edge_context", copy_graph=True)
        self.assertTrue(torch.equal(edge_graph["atom"].x, original["atom"].x))
        for relation in edge_graph.edge_types:
            self.assertEqual(tuple(edge_graph[relation].edge_features.shape), (1, 8))
        atom_graph = apply_feature_variant(original, "atom_derived_coordination", copy_graph=True)
        self.assertTrue(torch.equal(atom_graph["atom"].x[:, 0], original["atom"].x[:, 0]))
        self.assertEqual(tuple(atom_graph["atom"].x.shape), (3, 14))
        void_graph = apply_feature_variant(original, "void_derived_neighborhood", copy_graph=True)
        self.assertTrue(torch.equal(void_graph["void"].x[:, 0], original["void"].x[:, 0]))
        self.assertEqual(tuple(void_graph["void"].x.shape), (2, 9))
        environment = apply_feature_variant(original, "void_derived_environment", copy_graph=True)
        self.assertEqual(tuple(environment["void"].x.shape), (2, 9))
        self.assertFalse(torch.equal(environment["void"].x, original["void"].x))

    def test_distance_ablations_keep_topology_and_zero_fixed_width_channels(self):
        graph = self.graph()
        original_edges = {relation: graph[relation].edge_index.clone() for relation in graph.edge_types}
        for name in ("no_edge_distance", "no_atom_atom_distance", "no_void_atom_distance", "no_void_void_distance"):
            varied = apply_feature_variant(graph, name, copy_graph=True)
            for relation in varied.edge_types:
                self.assertTrue(torch.equal(varied[relation].edge_index, original_edges[relation]))
            if name == "no_edge_distance":
                for relation in varied.edge_types:
                    self.assertEqual(tuple(varied[relation].edge_features.shape), (1, 8))
                    self.assertEqual(torch.count_nonzero(varied[relation].edge_features).item(), 0)
            if name == "no_void_atom_distance":
                self.assertEqual(torch.count_nonzero(varied["void", "to", "atom"].edge_features).item(), 0)
                self.assertEqual(torch.count_nonzero(varied["atom", "to", "void"].edge_features).item(), 0)
                self.assertTrue(torch.equal(varied["atom", "to", "atom"].edge_distance, graph["atom", "to", "atom"].edge_distance))

    def test_node_group_by_edge_distance_interaction_cells(self):
        graph = self.graph()
        original_edges = {relation: graph[relation].edge_index.clone() for relation in graph.edge_types}
        for name, retained_nodes in (
            ("atom_only_no_edge_distance", "atom"),
            ("void_only_no_edge_distance", "void"),
        ):
            varied = apply_feature_variant(graph, name, copy_graph=True)
            other_nodes = "void" if retained_nodes == "atom" else "atom"
            self.assertTrue(torch.equal(varied[retained_nodes].x, graph[retained_nodes].x))
            self.assertEqual(torch.count_nonzero(varied[other_nodes].x).item(), 0)
            for relation in varied.edge_types:
                self.assertTrue(torch.equal(varied[relation].edge_index, original_edges[relation]))
                self.assertEqual(tuple(varied[relation].edge_features.shape), (1, 8))
                self.assertEqual(torch.count_nonzero(varied[relation].edge_features).item(), 0)


if __name__ == "__main__":
    unittest.main()
