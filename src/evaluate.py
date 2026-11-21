"""Evaluate a checkpoint on the CIFAR-10 test set.

Usage:
    python -m src.evaluate --run-name baseline
    python -m src.evaluate --ckpt outputs/baseline/best.pth --out-dir outputs/baseline

Prints overall + per-class accuracy, saves the confusion matrix
and numbers into the run's output directory.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from src.dataset import CLASSES, get_loaders
from src.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-name', default='baseline')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    ckpt_path = args.ckpt or f'./outputs/{args.run_name}/best.pth'
    out_dir = args.out_dir or f'./outputs/{args.run_name}'
    torch.set_num_threads(4)
    device = torch.device('cpu')
    _, _, test_loader = get_loaders()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_model(ckpt.get('model', 'simplecnn')).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    conf = torch.zeros(10, 10, dtype=torch.int64)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x.to(device)).argmax(1).cpu()
            for t, p in zip(y, pred):
                conf[t, p] += 1
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    per_class = (conf.diag() / conf.sum(1)).tolist()
    print(f'[{args.run_name}] test accuracy: {acc:.4f}  '
          f'(best val acc {ckpt["val_acc"]:.4f}, epoch {ckpt["epoch"]})')
    for name, a in zip(CLASSES, per_class):
        print(f'  {name:10s} {a:.4f}')

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'test_results.json'), 'w') as f:
        json.dump({'test_accuracy': acc, 'best_val_accuracy': ckpt['val_acc'],
                   'best_epoch': ckpt['epoch'],
                   'per_class_accuracy': dict(zip(CLASSES, per_class))}, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(conf.numpy(), cmap='Blues')
    ax.set_xticks(range(10), CLASSES, rotation=45, ha='right')
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion matrix ({args.run_name}, test acc {acc:.3f})')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, int(conf[i, j]), ha='center', va='center', fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150)
    print(f'[{args.run_name}] saved {out_dir}/confusion_matrix.png')


if __name__ == '__main__':
    main()
