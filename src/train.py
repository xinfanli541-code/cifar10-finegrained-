"""Train ResNet18 on CIFAR-10 with a chosen augmentation variant.

Usage:
    python -m src.train --run-name aug-none --aug none --epochs 8
    python -m src.train --run-name aug-strong --aug strong --epochs 8

Same recipe as the prior resnet18 run (Adam lr 1e-3, wd 5e-4,
CosineAnnealingLR, batch 128, seed 42) except the augmentation
variant, so the groups are a single-variable comparison. The
aug-base group is the prior run itself (87.02%), not retrained.

Checkpointing: checkpoint.pth is saved every 100 batches *and* at
every epoch end, and resumes mid-epoch (batch index + partial
epoch accumulators are stored). This matters because a ResNet18
epoch takes ~35-45 min on this CPU and the runtime restarts
frequently; an epoch-end-only checkpoint would lose a whole epoch
per restart.
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_loaders
from src.model import build_model, count_parameters

CKPT_EVERY_BATCHES = 100


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def save_ckpt(path, epoch, batch_idx, model, optimizer, scheduler,
              history, best_acc, aug, ep_loss_sum, ep_correct, ep_total):
    torch.save({'epoch': epoch, 'batch_idx': batch_idx,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'history': history, 'best_acc': best_acc,
                'model': 'resnet18', 'aug': aug,
                'ep_loss_sum': ep_loss_sum, 'ep_correct': ep_correct,
                'ep_total': ep_total}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--aug', default='base', choices=['base', 'none', 'strong'])
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

    train_loader, val_loader, _ = get_loaders(
        args.data_root, args.batch_size, aug=args.aug)
    model = build_model('resnet18').to(device)
    print(f'run {args.run_name}: model resnet18 params: {count_parameters(model):,} '
          f'aug={args.aug} epochs={args.epochs} lr={args.lr}', flush=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    start_epoch = 1
    resume_batch = 0
    ep_loss_sum, ep_correct, ep_total = 0.0, 0, 0
    if os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location='cpu')
        if ck.get('aug') == args.aug and ck.get('epoch', 1) > args.epochs:
            print(f'[{args.run_name}] already finished '
                  f'(best val acc: {ck["best_acc"]:.4f}); nothing to do.')
            return
        if ck.get('aug') == args.aug and ck.get('epoch', 1) <= args.epochs:
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
        else:
            os.remove(resume_path)
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
                          scheduler, history, best_acc, args.aug,
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
                        'aug': args.aug}, ckpt_path)
        print(f'[{args.run_name}] epoch {epoch:2d}/{args.epochs}  loss {train_loss:.4f}  '
              f'train_acc {train_acc:.4f}  val_acc {val_acc:.4f}  '
              f'({time.time() - t0:.0f}s)', flush=True)
        save_ckpt(resume_path, epoch + 1, 0, model, optimizer,
                  scheduler, history, best_acc, args.aug, 0.0, 0, 0)

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
