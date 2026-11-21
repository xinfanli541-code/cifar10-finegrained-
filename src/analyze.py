"""Confusion analysis + Grad-CAM for the fine-grained hard-class project.

1. confusion: load a checkpoint, run the 10k test set, save the full
   confusion matrix (JSON + plot) and the top off-diagonal confusions.
2. gradcam: Grad-CAM on ResNet18 layer4, for chosen test samples
   (typically misclassified cat/dog/bird samples), saving
   original / heatmap / overlay images.

Usage:
    python -m src.analyze --ckpt <path> --out-dir analysis/resnet18-prior
    python -m src.analyze --ckpt outputs/aug-base/best.pth --out-dir analysis/aug-base --gradcam 12
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import CLASSES, get_loaders, test_transform
from src.model import build_model

MEAN = np.array([0.4914, 0.4822, 0.4465])
STD = np.array([0.2023, 0.1994, 0.2010])


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_model(ckpt.get('model', 'resnet18')).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt


def confusion_matrix(model, loader, device):
    conf = torch.zeros(10, 10, dtype=torch.int64)
    preds_all, labels_all = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            for t, p in zip(y, pred):
                conf[t, p] += 1
            preds_all.append(pred)
            labels_all.append(y)
    return conf, torch.cat(preds_all), torch.cat(labels_all)


def gradcam(model, img_tensor, target_class, device):
    """Grad-CAM for ResNet18 layer4. img_tensor: (1,3,32,32) normalized."""
    acts, grads = {}, {}

    def fwd_hook(m, inp, out):
        acts['v'] = out.detach()

    def bwd_hook(m, gin, gout):
        grads['v'] = gout[0].detach()

    h1 = model.layer4.register_forward_hook(fwd_hook)
    h2 = model.layer4.register_full_backward_hook(bwd_hook)
    model.zero_grad()
    out = model(img_tensor.to(device))
    score = out[0, target_class]
    score.backward()
    h1.remove()
    h2.remove()
    weights = grads['v'].mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts['v']).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(32, 32), mode='bilinear', align_corners=False)
    cam = cam[0, 0].cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, out.detach().cpu()


def denorm(tensor):
    arr = tensor.cpu().numpy().transpose(1, 2, 0)
    arr = (arr * STD + MEAN).clip(0, 1)
    return (arr * 255).astype(np.uint8)


def save_gradcam_fig(orig_uint8, cam, path, title):
    cmap = plt.get_cmap('jet')
    heat = (cmap(cam)[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.55 * orig_uint8.astype(float) + 0.45 * heat.astype(float)).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(orig_uint8)
    axes[0].set_title('original')
    axes[0].axis('off')
    axes[1].imshow(heat)
    axes[1].set_title('Grad-CAM')
    axes[1].axis('off')
    axes[2].imshow(overlay)
    axes[2].set_title('overlay')
    axes[2].axis('off')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--gradcam', type=int, default=12,
                    help='max Grad-CAM samples to draw')
    ap.add_argument('--skip-confusion', action='store_true',
                    help='reuse confusion.json in out-dir, only draw Grad-CAM')
    ap.add_argument('--data-root', default='./data')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device('cpu')
    model, ckpt = load_model(args.ckpt, device)
    if args.skip_confusion:
        with open(os.path.join(args.out_dir, 'confusion.json')) as f:
            saved = json.load(f)
        pairs = [(n, t, p) for n, t, p in saved['top_confusions']]
        acc = saved['test_accuracy']
        print(f'reusing confusion from {args.out_dir}/confusion.json '
              f'(test acc {acc:.4f})')
    else:
        _, _, test_loader = get_loaders(args.data_root)

        conf, preds, labels = confusion_matrix(model, test_loader, device)
        acc = (preds == labels).float().mean().item()
        per_class = (conf.diag() / conf.sum(1)).tolist()
        print(f'test accuracy: {acc:.4f}')
        for name, a in zip(CLASSES, per_class):
            print(f'  {name:10s} {a:.4f}')

        # top off-diagonal confusions
        pairs = []
        for i in range(10):
            for j in range(10):
                if i != j and conf[i, j] > 0:
                    pairs.append((int(conf[i, j]), CLASSES[i], CLASSES[j]))
        pairs.sort(reverse=True)
        print('top confusions (true -> predicted, count / 1000):')
        for n, t, p in pairs[:15]:
            print(f'  {t:10s} -> {p:10s} {n}')
        with open(os.path.join(args.out_dir, 'confusion.json'), 'w') as f:
            json.dump({'test_accuracy': acc,
                       'per_class_accuracy': dict(zip(CLASSES, per_class)),
                       'confusion_matrix': conf.tolist(),
                       'top_confusions': [[n, t, p] for n, t, p in pairs[:15]],
                       'ckpt': args.ckpt}, f, indent=2)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(conf.numpy(), cmap='Blues')
        ax.set_xticks(range(10), CLASSES, rotation=45, ha='right')
        ax.set_yticks(range(10), CLASSES)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion matrix (test acc {acc:.3f})')
        for i in range(10):
            for j in range(10):
                ax.text(j, i, int(conf[i, j]), ha='center', va='center', fontsize=8)
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, 'confusion_matrix.png'), dpi=150)
        plt.close(fig)

    # Grad-CAM: pick misclassified samples, prefer cat->dog, bird->*,
    # dog->cat, deer->horse etc. (top confusion pairs first)
    _, _, test_loader2 = get_loaders(args.data_root)
    # need raw dataset with plain transform for images; reuse loader items
    wanted = [(CLASSES.index(t), CLASSES.index(p)) for _, t, p in pairs[:6]]
    picked = []  # (true, pred, img_tensor, idx)
    counts = {}
    with torch.no_grad():
        for x, y in test_loader2:
            pred = model(x.to(device)).argmax(1).cpu()
            for k in range(len(y)):
                key = (int(y[k]), int(pred[k]))
                if key in wanted and counts.get(key, 0) < 2:
                    counts[key] = counts.get(key, 0) + 1
                    picked.append((int(y[k]), int(pred[k]), x[k].clone()))
                    if len(picked) >= args.gradcam:
                        break
            if len(picked) >= args.gradcam:
                break
    print(f'Grad-CAM samples picked: {len(picked)}')
    for i, (t, p, img) in enumerate(picked):
        cam, out = gradcam(model, img.unsqueeze(0), p, device)
        title = f'true={CLASSES[t]}  pred={CLASSES[p]}'
        path = os.path.join(args.out_dir, f'gradcam_{i:02d}_{CLASSES[t]}_as_{CLASSES[p]}.png')
        save_gradcam_fig(denorm(img), cam, path, title)
        print(f'  {path}  ({title})')
    with open(os.path.join(args.out_dir, 'gradcam_samples.json'), 'w') as f:
        json.dump([{'true': CLASSES[t], 'pred': CLASSES[p]} for t, p, _ in picked], f, indent=2)


if __name__ == '__main__':
    main()
