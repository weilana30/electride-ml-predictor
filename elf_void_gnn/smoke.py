"""Small real-data GPU smoke run; writes no model artifacts."""
from __future__ import annotations
import argparse
import torch
from torch_geometric.loader import DataLoader
from elf_void_gnn.data import ShardedGraphDataset, read_split_table, fit_training_statistics
from elf_void_gnn.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='elf_void_gnn/configs/elf_center_v1.yaml')
    args = parser.parse_args()
    import yaml
    with open(args.config) as f: cfg = yaml.safe_load(f)
    if not torch.cuda.is_available(): raise RuntimeError('ROCm GPU is unavailable')
    device = torch.device('cuda')
    print(f'GPU={torch.cuda.get_device_name(0)} HIP={torch.version.hip}', flush=True)
    splits = read_split_table(cfg['dataset_root'])
    train_data = ShardedGraphDataset(cfg['dataset_root'], splits['train'])
    stats = fit_training_statistics(train_data)
    dataset = ShardedGraphDataset(cfg['dataset_root'], splits['train'][:16])
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False))).to(device)
    model_cfg = cfg['model']
    model = build_model(model_cfg['type'], {k: v for k, v in model_cfg.items() if k != 'type'}).to(device)
    model.set_training_statistics(stats)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = (pred - batch['void'].y).square().mean()
        if not torch.isfinite(pred).all() or not torch.isfinite(loss): raise RuntimeError('non-finite prediction/loss')
        loss.backward(); opt.step()
        print(f'step={step+1} graphs={batch.num_graphs} voids={pred.numel()} loss={loss.item():.6f}', flush=True)
    print('GPU smoke succeeded')

if __name__ == '__main__': main()
