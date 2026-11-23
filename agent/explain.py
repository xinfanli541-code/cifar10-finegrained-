"""Agent 核心：加载已有 ResNet18 权重、预测 top3、Grad-CAM、CAM 统计、一句话中文解释。

复用已有代码，不重写：
- 权重加载、Grad-CAM（layer4）、denorm、三联图：src.analyze
- 类名、数据 transform：src.dataset
- 模型结构：src.model（经由 src.analyze.load_model）

诚实底线：Grad-CAM 只能告诉我们热力图集中在图片的哪个位置，
不能告诉我们模型“看到了脸/耳朵/毛”这类语义细节。所以 explain()
只说位置、置信度差距、是不是已知难混淆对，绝不编造语义断言。
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# 仓库根 = agent/ 的上一级；权重默认在仓库外的上一个项目里。
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = REPO_ROOT / ".." / "pytorch-image-classification" / "outputs" / "resnet18" / "best.pth"

CN_NAMES = {
    "airplane": "飞机", "automobile": "汽车", "bird": "鸟", "cat": "猫",
    "deer": "鹿", "dog": "狗", "frog": "青蛙", "horse": "马",
    "ship": "船", "truck": "卡车",
}
# 英文名 -> 中文名 的反查（含中文输入本身）
NAME_TO_EN = {**{v: k for k, v in CN_NAMES.items()},
              **{k: k for k in CN_NAMES}}

# 已验证的难混淆对（true -> pred），来自 outputs/comparison_summary.json
# aug-base：猫->狗 166、狗->猫 58、猫->鸟 45、猫->鹿 44
KNOWN_CONFUSIONS = {
    ("cat", "dog"), ("dog", "cat"), ("cat", "bird"), ("cat", "deer"),
    ("bird", "deer"), ("deer", "bird"),
}

# 3x3 格子中文名：行=上/中/下，列=左/中/右
_GRID_ROWS = ["上", "中", "下"]
_GRID_COLS = ["左", "中", "右"]


def resolve_class(name):
    """接受英文类名或中文类名，返回英文类名；不认识返回 None。"""
    if name is None:
        return None
    return NAME_TO_EN.get(str(name).strip())


def load_agent_model(ckpt_path=None, device=None):
    """加载已有权重。返回 (model, ckpt_info)。不训练，只推理。"""
    # 延迟 import：src.analyze 在模块顶层 import matplotlib 等，较重。
    from src.analyze import load_model

    ckpt_path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
    device = device or torch.device("cpu")
    model, ckpt = load_model(str(ckpt_path), device)
    return model, ckpt


def preprocess_pil(img):
    """PIL 图 -> (1,3,32,32) 归一化张量；同时返回 32x32 原图 uint8 数组。"""
    from src.analyze import denorm
    from src.dataset import test_transform

    img = img.convert("RGB").resize((32, 32))
    tensor = test_transform(img).unsqueeze(0)
    orig = denorm(tensor[0])
    return tensor, orig


def predict_top3(model, img_tensor, device=None):
    """返回 [(英文类名, 概率), ...]，按概率从高到低取前 3。"""
    from src.dataset import CLASSES

    device = device or torch.device("cpu")
    with torch.no_grad():
        probs = F.softmax(model(img_tensor.to(device))[0], dim=0).cpu()
    top = torch.topk(probs, 3)
    return [(CLASSES[i], float(p)) for i, p in zip(top.indices.tolist(), top.values.tolist())]


def gradcam_for(model, img_tensor, target, device=None):
    """复用 src.analyze.gradcam。target 可为英文类名或类别序号。

    返回 (cam 32x32 float32 0~1, logits)。"""
    from src.analyze import gradcam
    from src.dataset import CLASSES

    if isinstance(target, str):
        target = CLASSES.index(target)
    device = device or torch.device("cpu")
    return gradcam(model, img_tensor, target, device)


def cam_summary(cam):
    """只从 CAM 数值本身算统计量，不做语义推断。

    返回 dict：centroid_x/centroid_y（0~31，x=列，y=行）、
    hottest_cell（3x3 最热格子，如“中上”）、hottest_share（该格热度占比）、
    center_share（中心 16x16 区域热度占比）、edge_share、peak（最大值）。
    """
    cam = np.asarray(cam, dtype=np.float64)
    total = float(cam.sum()) + 1e-8
    h, w = cam.shape
    ys, xs = np.mgrid[0:h, 0:w]
    cy = float((cam * ys).sum() / total)   # 行方向质心
    cx = float((cam * xs).sum() / total)   # 列方向质心

    # 3x3 格子
    grid = np.zeros((3, 3))
    for r in range(3):
        for c in range(3):
            grid[r, c] = cam[r * h // 3:(r + 1) * h // 3,
                             c * w // 3:(c + 1) * w // 3].sum()
    r, c = np.unravel_index(int(np.argmax(grid)), grid.shape)
    hottest_cell = f"{_GRID_ROWS[r]}{_GRID_COLS[c]}"
    hottest_share = float(grid[r, c] / total)

    # 中心 16x16（行 8~23，列 8~23）vs 边缘
    center_share = float(cam[8:24, 8:24].sum() / total)
    return {
        "centroid_x": round(cx, 1),
        "centroid_y": round(cy, 1),
        "hottest_cell": hottest_cell,
        "hottest_share": round(hottest_share, 3),
        "center_share": round(center_share, 3),
        "edge_share": round(1.0 - center_share, 3),
        "peak": round(float(cam.max()), 3),
    }


def explain(pred, top3, cam_stats, true_label=None):
    """生成一句话中文解释。只依据：预测、top3 概率、CAM 统计、已知难混淆对。

    pred: 英文类名；top3: [(类名, 概率), ...]；cam_stats: cam_summary() 的结果；
    true_label: 英文类名或 None。
    """
    pred_cn = CN_NAMES[pred]
    p1 = top3[0][1]
    p2 = top3[1][1] if len(top3) > 1 else 0.0
    gap = p1 - p2
    second_cn = CN_NAMES[top3[1][0]] if len(top3) > 1 else ""

    # 置信度措辞：gap 小 = 留余地，gap 大 = 较确定
    if gap < 0.05:
        conf = f"但它对{second_cn}也有 {p2:.0%} 的信心，两者非常接近，这个判断不太稳"
    elif gap < 0.15:
        conf = f"第二名{second_cn}有 {p2:.0%}，差距不大，结论只能算比较确定"
    else:
        conf = f"第二名{second_cn}只有 {p2:.0%}，差距比较大，判断相对确定"

    loc = (f"热力图最热在图片{cam_stats['hottest_cell']}部"
           f"（占 {cam_stats['hottest_share']:.0%}），"
           f"中心区域占 {cam_stats['center_share']:.0%}")

    true_en = resolve_class(true_label)
    if true_en is not None:
        true_cn = CN_NAMES[true_en]
        if true_en == pred:
            return (f"判对了：实际是{true_cn}，模型也判成{pred_cn}（{p1:.0%}），"
                    f"{conf}；{loc}。")
        known = "，这是已知的难混淆对" if (true_en, pred) in KNOWN_CONFUSIONS else ""
        return (f"判错了：实际是{true_cn}，模型判成{pred_cn}（{p1:.0%}）"
                f"{known}；{conf}；{loc}。")
    return f"模型判成{pred_cn}（{p1:.0%}），{conf}；{loc}。"
