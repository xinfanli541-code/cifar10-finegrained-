"""Guided training: ResNet18 on CIFAR-10 with Grad-CAM-guided masking.

Single-variable probe against the aug-base baseline (87.02%):
everything is identical to src/train.py --aug base (same 45k/5k
split seed 42, same base augmentation RandomCrop(32, padding=4) +
RandomHorizontalFlip, Adam lr 1e-3, wd 5e-4, CosineAnnealingLR,
batch 128, 8 epochs, seed 42, from scratch) EXCEPT one change:
for cat/dog training images, with probability --mask-prob the
region the BASELINE model attends to most is erased before
augmentation.

How the guidance works (the "look at the face" idea):
  * The baseline's Grad-CAM (layer4, true class) concentrates on
    the animal's torso, not its face/ears (see 对比实验报告.md §6).
  * src/cam_precompute.py computed that baseline CAM for every
    cat/dog image in the 45k train split and saved it as .npz.
  * Here, for a cat/dog sample, with prob --mask-prob (default
    0.5) the pixels whose CAM value is in the top --mask-frac
    (default 0.25) are painted over with the dataset mean colour
    BEFORE augmentation, so the crop/flip moves image and mask
    together and the erased region stays aligned with the torso.
  * The model therefore cannot rely on the torso cue on those
    samples and must learn from the remaining regions -- face,
    ears, legs, background context. On the other half of cat/dog
    samples (and all other classes) the image is untouched, so
    the torso cue is weakened, not removed entirely.

Checkpointing mirrors src/train.py: checkpoint.pth every 100
batches and at every epoch end, resuming mid-epoch, so a runtime
restart loses at most ~100 batches.

Usage:
    python -m src.train_guided --run-name guided \
        --cam-npz outputs/guided-cam/train_cam_catdog.npz --epochs 8
"""

import argparse
import io
import json
import os
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from src.dataset import (AUG_TRANSFORMS, CIFAR10_MEAN, ParquetCIFAR10,
                         plain_transform)
from src.model import build_model, count_parameters
from src.train import CKPT_EVERY_BATCHES, evaluate, save_ckpt

CAT, DOG = 3, 5
MEAN_RGB = tuple(int(round(m * 255)) for m in CIFAR10_MEAN)


class GuidedTrainSet(Dataset):
    """Train split with CAM-guided masking for cat/dog samples.

    Loads the same parquet blobs as ParquetCIFAR10. Masking is
    applied on the plain PIL image (mean-colour fill) before the
    base augmentation, so crop/flip keep mask and content aligned.
    """

    def __init__(self, parquet_path, indices, cam_map,
                 mask_prob=0.5, mask_frac=0.25):
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path)
        self.labels = table.column('label').to_pylist()
        img_col = table.column('img').to_pylist()
        self.blobs = [x['bytes'] if isinstance(x, dict) else x for x in img_col]
        self.indices = indices
        self.cam_map = cam_map
        self.mask_prob = mask_prob
        self.mask_frac = mask_frac
        self.base_aug = AUG_TRANSFORMS['base']
        # NOTE: with num_workers>0, __getitem__ runs in workers, so
        # per-epoch masked counts cannot be read back in the main
        # process; the masking rate is fixed by mask_prob (0.5 of
        # cat/dog samples, ~10% of all samples) and not counted here.

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = Image.open(io.BytesIO(self.blobs[idx])).convert('RGB')
        y = self.labels[idx]
        cam = self.cam_map.get(idx)
        if cam is not None and y in (CAT, DOG):
            if random.random() < self.mask_prob:
                thr = np.percentile(cam, 100 * (1 - self.mask_frac))
                arr = np.array(img)
                arr[cam >= thr] = MEAN_RGB
                img = Image.fromarray(arr)
        return self.base_aug(img), y


def load_cam_map(npz_path):
    d = np.load(npz_path)
    cam_arr = d['cam']          # load the array ONCE (NpzFile re-reads
    idx_arr = d['idx']          # and re-decompresses on every d['cam'])
    cam_map = {int(i): cam_arr[k].astype(np.float32)
               for k, i in enumerate(idx_arr.tolist())}
    print(f'loaded CAM map: {len(cam_map)} cat/dog images from {npz_path}',
          flush=True)
    return cam_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-name', default='guided')
    ap.add_argument('--cam-npz', required=True)
    ap.add_argument('--mask-prob', type=float, default=0.5)
    ap.add_argument('--mask-frac', type=float, default=0.25)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--data-root', default='./data')
    args = ap.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device('cpu')
    out_dir = os.path.join('./outputs', args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'best.pth')
    resume_path = os.path.join(out_dir, 'checkpoint.pth')

    train_p = os.path.join(args.data_root, 'parquet',
                           'train-00000-of-00001.parquet')
    gen = torch.Generator().manual_seed(42)
    split = torch.randperm(50000, generator=gen)
    train_idx = split[:45000].tolist()
    val_idx = split[45000:].tolist()
    cam_map = load_cam_map(args.cam_npz)
    train_set = GuidedTrainSet(train_p, train_idx, cam_map,
                               args.mask_prob, args.mask_frac)
    val_set = Subset(ParquetCIFAR10(train_p, plain_transform), val_idx)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)
    print(f'run {args.run_name}: guided masking mask_prob={args.mask_prob} '
          f'mask_frac={args.mask_frac} (CAM top fraction erased, mean fill); '
          f'train {len(train_set)} val {len(val_set)}', flush=True)

    model = build_model('resnet18').to(device)
    print(f'run {args.run_name}: model resnet18 params: '
          f'{count_parameters(model):,} aug=base epochs={args.epochs} '
          f'lr={args.lr}', flush=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    start_epoch = 1
    resume_batch = 0
    ep_loss_sum, ep_correct, ep_total = 0.0, 0, 0
    if os.path.exists(resume_path):
        # The checkpoint lives in this run's own directory, so any
        # checkpoint found here belongs to the guided run; resume it
        # whenever it is not already past the final epoch.
        ck = torch.load(resume_path, map_location='cpu')
        if ck.get('epoch', 1) > args.epochs:
            print(f'[{args.run_name}] already finished '
                  f'(best val acc: {ck["best_acc"]:.4f}); nothing to do.')
            return
        model.load_state_dict(ck['state_dict'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        history = ck['history']
        best_acc = ck['best_acc']
        start_epoch = ck['epoch']
        resume_batch = ck.get('batch_idx', 0)
        ep_loss_sum = ck.get('ep_loss_sum', 0.0)
        ep_correct = ck.get('ep_correct', 0)
        ep_total = ck.get('ep_total', 0)
        print(f'[{args.run_name}] resumed at epoch {start_epoch} '
              f'batch {resume_batch} (best val acc so far: {best_acc:.4f})',
              flush=True)
    n_batches = len(train_loader)
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        if resume_batch == 0:
            ep_loss_sum, ep_correct, ep_total = 0.0, 0, 0
        skip = resume_batch
        resume_batch = 0
        for bi, (x, y) in enumerate(train_loader):
            if bi < skip:
                continue
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            ep_loss_sum += loss.item() * y.size(0)
            ep_correct += (out.argmax(1) == y).sum().item()
            ep_total += y.size(0)
            if (bi + 1) % CKPT_EVERY_BATCHES == 0:
                save_ckpt(resume_path, epoch, bi + 1, model, optimizer,
                          scheduler, history, best_acc, 'base',
                          ep_loss_sum, ep_correct, ep_total)
                print(f'[{args.run_name}] epoch {epoch} batch {bi + 1}/{n_batches} '
                      f'ckpt saved ({time.time() - t0:.0f}s)', flush=True)
        scheduler.step()
        train_loss = ep_loss_sum / max(ep_total, 1)
        train_acc = ep_correct / max(ep_total, 1)
        val_acc = evaluate(model, val_loader, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'val_acc': val_acc, 'model': 'resnet18',
                        'aug': 'base', 'guided': True}, ckpt_path)
        print(f'[{args.run_name}] epoch {epoch:2d}/{args.epochs}  loss {train_loss:.4f}  '
              f'train_acc {train_acc:.4f}  val_acc {val_acc:.4f}  '
              f'({time.time() - t0:.0f}s)', flush=True)
        save_ckpt(resume_path, epoch + 1, 0, model, optimizer,
                  scheduler, history, best_acc, 'base', 0.0, 0, 0)

    with open(os.path.join(out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history['train_loss'])
    ax1.set_title('Train loss')
    ax1.set_xlabel('Epoch')
    ax2.plot(history['train_acc'], label='train')
    ax2.plot(history['val_acc'], label='val')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'training_curves.png'), dpi=150)
    print(f'[{args.run_name}] done. best val acc: {best_acc:.4f}  checkpoint: {ckpt_path}')


if __name__ == '__main__':
    main()
