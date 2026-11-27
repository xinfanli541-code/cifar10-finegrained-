"""Precompute baseline Grad-CAM maps for cat/dog training images.

Uses the baseline (aug-base) checkpoint to compute layer4 Grad-CAM
for the TRUE class of every cat/dog image in the 45k train split
(same split as the comparison experiments, seed 42). The maps are
the teacher signal for the guided run: train_guided.py erases the
top-CAM region of cat/dog images so the model cannot keep staring
at the torso the baseline attends to.

Usage:
    python -m src.cam_precompute --ckpt \
        ~/workspace/pytorch-image-classification/outputs/resnet18/best.pth \
        --out outputs/guided-cam/train_cam_catdog.npz

Output is a local-only .npz (NOT pushed to GitHub): idx (parquet row
index in the 50k train file), label, cam (N, 32, 32) float16,
min-max normalised per image to [0, 1]. Grad-CAM is computed in
eval mode, batched (sum of true-class logits per batch, one
backward per batch), so the whole pass takes ~15-25 min on CPU.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.analyze import load_model
from src.dataset import ParquetCIFAR10, plain_transform

CAT, DOG = 3, 5


def train_split_indices(seed=42, n_train=50000, n_keep=45000):
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_train, generator=gen)
    return indices[:n_keep].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='outputs/guided-cam/train_cam_catdog.npz')
    ap.add_argument('--data-root', default='./data')
    ap.add_argument('--batch-size', type=int, default=128)
    args = ap.parse_args()

    torch.set_num_threads(4)
    device = torch.device('cpu')
    model, ckpt = load_model(args.ckpt, device)
    print(f'baseline ckpt: {args.ckpt} (val acc {ckpt.get("val_acc")})', flush=True)

    train_p = os.path.join(args.data_root, 'parquet', 'train-00000-of-00001.parquet')
    full = ParquetCIFAR10(train_p, plain_transform)
    split = train_split_indices()
    catdog = [i for i in split if full.labels[i] in (CAT, DOG)]
    print(f'train split: {len(split)} images, cat/dog subset: {len(catdog)}', flush=True)
    subset = Subset(full, catdog)
    loader = DataLoader(subset, batch_size=args.batch_size,
                        shuffle=False, num_workers=2)

    acts, grads = {}, {}

    def fwd_hook(m, inp, out):
        acts['v'] = out.detach()

    def bwd_hook(m, gin, gout):
        grads['v'] = gout[0].detach()

    h1 = model.layer4.register_forward_hook(fwd_hook)
    h2 = model.layer4.register_full_backward_hook(bwd_hook)

    all_idx, all_label, all_cam = [], [], []
    for bi, (x, y) in enumerate(loader):
        model.zero_grad()
        out = model(x.to(device))
        score = out[torch.arange(len(y)), y.to(device)].sum()
        score.backward()
        a = acts['v']          # (B, 512, 4, 4)
        g = grads['v']         # (B, 512, 4, 4)
        w = g.mean(dim=(2, 3), keepdim=True)   # (B, 512, 1, 1)
        cam = (w * a).sum(dim=1, keepdim=True)  # (B, 1, 4, 4)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(32, 32), mode='bilinear',
                            align_corners=False)[:, 0].cpu().numpy()
        for k in range(len(y)):
            c = cam[k]
            cam[k] = (c - c.min()) / (c.max() - c.min() + 1e-8)
        all_cam.append(cam.astype(np.float16))
        all_label.extend(y.tolist())
        all_idx.extend([catdog[bi * args.batch_size + k] for k in range(len(y))])
        if (bi + 1) % 10 == 0:
            print(f'  batch {bi + 1}/{len(loader)}', flush=True)
    h1.remove()
    h2.remove()

    cam_arr = np.concatenate(all_cam, axis=0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out,
                        idx=np.array(all_idx, dtype=np.int32),
                        label=np.array(all_label, dtype=np.int8),
                        cam=cam_arr)
    stats = {'n_images': int(len(all_idx)),
             'n_cat': int(sum(1 for l in all_label if l == CAT)),
             'n_dog': int(sum(1 for l in all_label if l == DOG)),
             'cam_shape': list(cam_arr.shape),
             'cam_dtype': str(cam_arr.dtype)}
    with open(args.out.replace('.npz', '.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'saved {args.out}: {stats}', flush=True)


if __name__ == '__main__':
    main()
