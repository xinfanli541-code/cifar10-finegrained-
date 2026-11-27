"""Quantitative Grad-CAM position comparison: baseline vs guided model.

For every cat/dog image in the 10k test set, compute batched layer4
Grad-CAM for the TRUE class (same batched recipe as cam_precompute.py:
sum of true-class logits per batch, one backward per batch), then
measure where the CAM mass sits on the raw (ReLU, pre-normalisation)
map, averaged per class:

  - top12_frac : mass in rows 0-11 (upper ~37%, where head/ears usually are)
  - top_half_frac : mass in rows 0-15
  - center_frac : mass in the central 16x16 block (rows 8-23, cols 8-23)
  - top25_mass_frac : fraction of total mass carried by the top 25%
    highest-valued pixels (concentration: 0.25 = perfectly flat)

This is a descriptive proxy only: on 32x32 CIFAR images "face/ears"
cannot be segmented exactly, but a systematic upward shift of CAM
mass (higher top12_frac) is consistent with the model looking more
at the upper part of the animal, and a drop in concentration means
it spreads attention over more regions instead of one torso blob.

Usage:
    python -m src.cam_shift --ckpt <path> --out analysis/cam_shift_<tag>.json
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.analyze import load_model
from src.dataset import get_loaders

CAT, DOG = 3, 5


def batched_cam(model, x, y, device, acts, grads):
    """Batched Grad-CAM (true class) on the raw ReLU map, shape (B,32,32)."""
    model.zero_grad()
    out = model(x.to(device))
    score = out[torch.arange(len(y)), y.to(device)].sum()
    score.backward()
    weights = grads['v'].mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts['v']).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(32, 32), mode='bilinear', align_corners=False)
    return cam[:, 0].detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data-root', default='./data')
    ap.add_argument('--batch-size', type=int, default=128)
    args = ap.parse_args()

    torch.set_num_threads(4)
    device = torch.device('cpu')
    model, ckpt = load_model(args.ckpt, device)
    print(f'ckpt: {args.ckpt} (val acc {ckpt.get("val_acc")})', flush=True)

    _, _, test_loader = get_loaders(args.data_root)
    acts, grads = {}, {}

    def fwd_hook(m, inp, out):
        acts['v'] = out.detach()

    def bwd_hook(m, gin, gout):
        grads['v'] = gout[0].detach()

    h1 = model.layer4.register_forward_hook(fwd_hook)
    h2 = model.layer4.register_full_backward_hook(bwd_hook)

    stats = {CAT: [], DOG: []}  # per-image (top12, top_half, center, top25mass)
    n_seen = 0
    for bi, (x, y) in enumerate(test_loader):
        mask = (y == CAT) | (y == DOG)
        if not mask.any():
            continue
        xs, ys = x[mask], y[mask]
        cams = batched_cam(model, xs, ys, device, acts, grads)
        for cam, label in zip(cams, ys.tolist()):
            total = cam.sum() + 1e-8
            top12 = cam[:12, :].sum() / total
            top_half = cam[:16, :].sum() / total
            center = cam[8:24, 8:24].sum() / total
            flat = np.sort(cam.ravel())[::-1]
            k = max(1, int(0.25 * flat.size))
            top25 = flat[:k].sum() / total
            stats[label].append((float(top12), float(top_half),
                                 float(center), float(top25)))
        n_seen += len(ys)
        if (bi + 1) % 10 == 0:
            print(f'  batch {bi + 1}  cat/dog seen {n_seen}', flush=True)
    h1.remove()
    h2.remove()

    out = {'ckpt': args.ckpt, 'n_catdog_test': n_seen}
    for label, name in ((CAT, 'cat'), (DOG, 'dog')):
        arr = np.array(stats[label])
        out[name] = {
            'n': len(arr),
            'top12_frac': float(arr[:, 0].mean()),
            'top_half_frac': float(arr[:, 1].mean()),
            'center_frac': float(arr[:, 2].mean()),
            'top25_mass_frac': float(arr[:, 3].mean()),
        }
    both = np.array(stats[CAT] + stats[DOG])
    out['all'] = {
        'n': len(both),
        'top12_frac': float(both[:, 0].mean()),
        'top_half_frac': float(both[:, 1].mean()),
        'center_frac': float(both[:, 2].mean()),
        'top25_mass_frac': float(both[:, 3].mean()),
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
