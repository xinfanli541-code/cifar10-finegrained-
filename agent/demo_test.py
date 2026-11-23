"""Agent 小规模实测：从 CIFAR-10 测试集取 120 张（每类 12 张，seed 42）。

用法（在仓库根目录）：
    python -m agent.demo_test [--out-dir agent/runs/demo_test] [--n-per-class 12]

输出：
- <out-dir>/summary.json：准确率、逐类准确率、top 混淆对、错判数（数字如实写，不美化）
- <out-dir>/misclassified_<true>_as_<pred>_<idx>.png：6 张典型错判三联图，
  优先猫->狗、狗->猫、猫->鸟，其余按混淆频次补齐

只推理、不训练。Grad-CAM 只给 6 张错判图算，120 张的准确率用批量推理。
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.explain import (  # noqa: E402
    cam_summary, explain, gradcam_for, load_agent_model, predict_top3,
    preprocess_pil,
)
from src.analyze import denorm, save_gradcam_fig  # noqa: E402
from src.dataset import CLASSES, ParquetCIFAR10, test_transform  # noqa: E402

SEED = 42
N_PER_CLASS = 12
# 优先展示的错判对（按 aug-base 已知混淆频次排序）
PRIORITY_PAIRS = [("cat", "dog"), ("dog", "cat"), ("cat", "bird"),
                  ("cat", "deer"), ("dog", "cat"), ("bird", "deer")]


def pick_indices(labels, n_per_class=12, seed=42):
    """每类取 n 张：用 seed 固定打乱该类下标后取前 n 张。返回按类、再按原下标排序的下标列表。"""
    rng = random.Random(seed)
    picked = []
    for cls_idx in range(len(CLASSES)):
        idxs = [i for i, y in enumerate(labels) if y == cls_idx]
        rng.shuffle(idxs)
        picked.extend(sorted(idxs[:n_per_class]))
    picked.sort()
    return picked


def main():
    ap = argparse.ArgumentParser(prog="agent.demo_test")
    ap.add_argument("--out-dir", default="agent/runs/demo_test")
    ap.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cpu")

    model, ckpt = load_agent_model(None, device)
    test_p = REPO_ROOT / args.data_root / "parquet" / "test-00000-of-00001.parquet"
    ds = ParquetCIFAR10(str(test_p), test_transform)
    labels = ds.labels
    idxs = pick_indices(labels, args.n_per_class, args.seed)
    print(f"抽样：{len(idxs)} 张（每类 {args.n_per_class} 张，seed {args.seed}）")

    # 批量推理求准确率
    preds, trues = [], []
    with torch.no_grad():
        for i in idxs:
            x, y = ds[i]
            with torch.no_grad():
                p = model(x.unsqueeze(0)).argmax(1).item()
            preds.append(p)
            trues.append(y)
    correct = sum(p == t for p, t in zip(preds, trues))
    acc = correct / len(idxs)
    per_class = {}
    for ci, name in enumerate(CLASSES):
        sel = [p == t for p, t in zip(preds, trues) if t == ci]
        per_class[name] = round(sum(sel) / len(sel), 4) if sel else None
    conf_pairs = Counter((CLASSES[t], CLASSES[p]) for t, p in zip(trues, preds) if t != p)
    top_conf = [[n, t, p] for (t, p), n in conf_pairs.most_common(10)]

    print(f"准确率：{acc:.4f}（{correct}/{len(idxs)}）")
    for name in CLASSES:
        print(f"  {name:10s} {per_class[name]:.4f}")
    print("top 混淆对（实际 -> 预测）：")
    for n, t, p in top_conf:
        print(f"  {t} -> {p}  {n}")

    # 6 张典型错判三联图：优先 PRIORITY_PAIRS，其余按频次补
    mis = [(idx, t, p) for idx, t, p in zip(idxs, trues, preds) if t != p]
    wanted = []
    for pair in PRIORITY_PAIRS:
        for m in mis:
            if (CLASSES[m[1]], CLASSES[m[2]]) == pair and m not in wanted:
                wanted.append(m)
                break
        if len(wanted) >= 6:
            break
    if len(wanted) < 6:
        for m in sorted(mis, key=lambda m: -conf_pairs[(CLASSES[m[1]], CLASSES[m[2]])]):
            if m not in wanted:
                wanted.append(m)
            if len(wanted) >= 6:
                break
    wanted = wanted[:6]

    saved = []
    for idx, t, p in wanted:
        x, y = ds[idx]
        # 原图：从 parquet 读原始 PIL（不经过 transform 的裁剪/翻转，test 本来就是 plain）
        import io
        raw = Image.open(io.BytesIO(ds.blobs[idx])).convert("RGB")
        tensor, orig = preprocess_pil(raw)
        cam, _ = gradcam_for(model, tensor, p, device)
        stats = cam_summary(cam)
        top3 = predict_top3(model, tensor, device)
        text = explain(CLASSES[p], top3, stats, true_label=CLASSES[t])
        fname = f"misclassified_{CLASSES[t]}_as_{CLASSES[p]}_{idx}.png"
        save_gradcam_fig(orig, cam, str(out_dir / fname),
                         f"true={CLASSES[t]}  pred={CLASSES[p]}")
        saved.append({"file": fname, "index": idx, "true": CLASSES[t],
                      "pred": CLASSES[p], "explanation": text,
                      "cam_stats": stats})
        print(f"  {fname}: {text}")

    summary = {
        "n_total": len(idxs), "n_per_class": args.n_per_class, "seed": args.seed,
        "accuracy": round(acc, 4), "n_correct": correct,
        "n_misclassified": len(idxs) - correct,
        "per_class_accuracy": per_class,
        "top_confusions": top_conf,
        "misclassified_examples": saved,
        "ckpt_epoch": ckpt.get("epoch"), "ckpt_val_acc": ckpt.get("val_acc"),
        "note": "120 张小样本实测，不是 10k 全测试集；数字如实记录，勿与 87.02% 全集准确率直接比较。",
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"汇总：{out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
