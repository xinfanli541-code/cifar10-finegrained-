"""Agent 命令行。

预测一张图：
    python -m agent.cli --image <图片路径> [--true cat] [--out-dir agent/runs/<时间戳>]
    --true 接受英文类名或中文名（如 cat / 猫）。只在给了 --true 且预测错时，
    才把记录追加到 agent/logs/misclassified.jsonl。

汇总错判日志：
    python -m agent.cli review [--log agent/logs/misclassified.jsonl]

在仓库根目录下运行。权重默认用 ../pytorch-image-classification/outputs/resnet18/best.pth，
可用 --ckpt 覆盖。
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.explain import (  # noqa: E402
    CN_NAMES, DEFAULT_CKPT, cam_summary, explain, gradcam_for,
    load_agent_model, predict_top3, preprocess_pil, resolve_class,
)

LOG_PATH = REPO_ROOT / "agent" / "logs" / "misclassified.jsonl"


def _now_stamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_predict(args):
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"图片不存在：{img_path}")
        return 1
    true_en = resolve_class(args.true)
    if args.true and true_en is None:
        print(f"无法识别的真实类别：{args.true}（可用：{', '.join(CN_NAMES)} 或中文名）")
        return 1

    device = torch.device("cpu")
    model, ckpt = load_agent_model(args.ckpt, device)
    img = Image.open(img_path)
    tensor, orig = preprocess_pil(img)
    top3 = predict_top3(model, tensor, device)
    pred = top3[0][0]
    cam, _logits = gradcam_for(model, tensor, pred, device)
    stats = cam_summary(cam)
    text = explain(pred, top3, stats, true_label=true_en)

    print(f"图片：{img_path.name}")
    print("Top3：")
    for name, p in top3:
        print(f"  {CN_NAMES[name]} ({name})  {p:.4f}")
    print(f"解释：{text}")
    print(f"CAM：质心=({stats['centroid_x']}, {stats['centroid_y']})，"
          f"最热格子={stats['hottest_cell']}（{stats['hottest_share']:.0%}），"
          f"中心占比={stats['center_share']:.0%}")

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "agent" / "runs" / _now_stamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    from src.analyze import save_gradcam_fig

    title = f"pred={pred}" + (f"  true={true_en}" if true_en else "")
    save_gradcam_fig(orig, cam, str(out_dir / "triptych.png"), title)
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump({"image": img_path.name, "true": true_en, "pred": pred,
                   "top3": [[n, round(p, 4)] for n, p in top3],
                   "cam_stats": stats, "explanation": text,
                   "ckpt": str(ckpt.get("epoch", "")), "time": _now_stamp()},
                  f, ensure_ascii=False, indent=2)
    print(f"三联图：{out_dir / 'triptych.png'}")

    if true_en is not None and pred != true_en:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {"time": datetime.now().isoformat(timespec="seconds"),
                  "image": img_path.name, "true": true_en, "pred": pred,
                  "top3": [[n, round(p, 4)] for n, p in top3],
                  "cam_centroid": [stats["centroid_x"], stats["centroid_y"]],
                  "cam_hottest_cell": stats["hottest_cell"]}
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"判错了，已记入错判日志：{LOG_PATH}")
    elif true_en is not None:
        print("判对了，不记入错判日志。")
    return 0


def run_review(args):
    log_path = Path(args.log) if args.log else LOG_PATH
    if not log_path.exists():
        print(f"错判日志还不存在：{log_path}（还没有判错的记录）")
        return 0
    records = [json.loads(line) for line in open(log_path, encoding="utf-8") if line.strip()]
    print(f"错判总数：{len(records)}")
    if not records:
        return 0
    pairs = Counter((r["true"], r["pred"]) for r in records)
    print("错判对数 top5（实际 -> 预测）：")
    for (t, p), n in pairs.most_common(5):
        print(f"  {CN_NAMES[t]} ({t}) -> {CN_NAMES[p]} ({p})  {n} 次")
    return 0


def main(argv=None):
    # review 是子命令（python -m agent.cli review）；其余为预测模式。
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "review":
        ap = argparse.ArgumentParser(prog="agent.cli review")
        ap.add_argument("--log", default=None)
        return run_review(ap.parse_args(argv[1:]))

    ap = argparse.ArgumentParser(prog="agent.cli")
    ap.add_argument("--image", required=True, help="待分类图片路径")
    ap.add_argument("--true", default=None, help="真实类别：英文类名或中文名，如 cat / 猫")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--ckpt", default=None,
                    help="权重路径，默认 ../pytorch-image-classification/outputs/resnet18/best.pth")
    return run_predict(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
