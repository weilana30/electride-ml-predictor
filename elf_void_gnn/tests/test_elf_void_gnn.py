import unittest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from elf_void_gnn.models import build_model
from elf_void_gnn.data import graph_from_record
torch.set_num_threads(1)

class ModelTests(unittest.TestCase):
    def graph(self, empty=False):
        g=HeteroData(); g['atom'].x=torch.randn(3,14); g['void'].x=torch.randn(2,9); g['void'].y=torch.randn(2)
        g['atom','to','atom'].edge_index=torch.empty((2,0),dtype=torch.long) if empty else torch.tensor([[0,1,1],[1,0,2]])
        g['atom','to','atom'].edge_distance=torch.ones(g['atom','to','atom'].edge_index.shape[1])
        g['void','to','atom'].edge_index=torch.tensor([[0,1],[0,2]]); g['void','to','atom'].edge_distance=torch.tensor([1.,2.])
        g['atom','to','void'].edge_index=torch.tensor([[0,2],[0,1]]); g['atom','to','void'].edge_distance=torch.tensor([1.,2.])
        g['void','to','void'].edge_index=torch.empty((2,0),dtype=torch.long) if empty else torch.tensor([[0,1],[1,0]])
        g['void','to','void'].edge_distance=torch.ones(g['void','to','void'].edge_index.shape[1])
        return g
    def test_batch_forward_backward_and_empty_relations(self):
        batch=next(iter(DataLoader([self.graph(),self.graph(True)],batch_size=2)))
        model=build_model('hetero_mpnn', {'hidden_channels':64,'message_layers':2,'radial_channels':8,'dropout':0.1})
        pred=model(batch); self.assertEqual(pred.shape,(4,)); self.assertTrue(torch.isfinite(pred).all())
        loss=(pred-batch['void'].y).square().mean(); loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters() if p.requires_grad))
        opt=torch.optim.Adam(model.parameters(),lr=1e-3); opt.step()
    def test_targets_are_supervision_only(self):
        g=self.graph(); self.assertEqual(set(g.node_types),{'atom','void'}); self.assertFalse(hasattr(g['atom'],'y'))
        self.assertEqual(g['void'].y.numel(),2)
    def test_adapter_feature_shapes_edge_directions_and_distances(self):
        record={
            'atoms': {'features':torch.ones(2,5).numpy(), 'feature_validity':torch.ones(2,5).numpy()},
            'voids': {'features':torch.ones(1,9).numpy(), 'void_id':['v0']},
            'edges': {
                'atom_atom_index':torch.tensor([[0],[1]]).numpy(), 'atom_atom_distance_angstrom':[1.5],
                'void_atom_index':torch.tensor([[0],[1]]).numpy(), 'void_atom_distance_angstrom':[2.5],
                'void_void_index':torch.empty((2,0),dtype=torch.long).numpy(), 'void_void_distance_angstrom':[],
            },
            'targets': {'elf_at_center':[0.7], 'other_target':[99.]},
            'metadata': {'family_id':'fam'}, 'sample_id':'sample'
        }
        g=graph_from_record(record)
        self.assertEqual(tuple(g['atom'].x.shape),(2,10)); self.assertEqual(tuple(g['void'].x.shape),(1,9))
        self.assertTrue(torch.equal(g['atom','to','void'].edge_index,torch.tensor([[1],[0]])))
        self.assertEqual(float(g['atom','to','void'].edge_distance[0]),2.5)
        self.assertFalse(hasattr(g['void'],'other_target'))
    def test_tiny_subset_overfit(self):
        torch.manual_seed(4)
        graph=self.graph(); graph['void'].y=torch.tensor([0.2,0.2])
        batch=next(iter(DataLoader([graph],batch_size=1)))
        model=build_model('hetero_mpnn', {'hidden_channels':16,'message_layers':2,'radial_channels':4,'dropout':0.0})
        optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
        initial=(model(batch)-batch['void'].y).square().mean().item()
        for _ in range(35):
            optimizer.zero_grad(); loss=(model(batch)-batch['void'].y).square().mean(); loss.backward(); optimizer.step()
        final=(model(batch)-batch['void'].y).square().mean().item()
        self.assertLess(final,initial*0.25)

if __name__=='__main__': unittest.main()
