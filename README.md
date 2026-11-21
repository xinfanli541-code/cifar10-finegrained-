# CIFAR-10 难类细粒度分类 + 可解释性

## 目标
以 ResNet18（之前对比实验最优组，测试准确率 87.02%）为起点，聚焦猫、鸟
等难类的细粒度误判：对比数据增强策略对难类的影响，并用 Grad-CAM
可视化典型误判，说明模型在看哪里、为什么错。

## 基线（复用之前项目，不重新训练）
- 之前项目：`~/workspace/pytorch-image-classification/`
- 数据：CIFAR-10 parquet，训练 45,000 / 验证 5,000 / 测试 10,000，种子 42
- ResNet18（CIFAR 适配版，从头训练，Adam lr 1e-3，8 轮，基线增强）：
  测试准确率 **87.02%**，最优验证 87.72%（第 8 轮）
- 逐类：猫 65.6%（最难）、鸟 83.7%、狗 84.8%，其余均 ≥86.5%

## 第一步已完成：混淆分析（旧最优权重复跑测试集）
结果见 `analysis/resnet18-prior/confusion.json`，
混淆矩阵见 `analysis/resnet18-prior/confusion_matrix.png`。
- 复跑测试准确率 87.02%，与报告一致。
- 最大混淆：**猫→狗 166/1000**，其次狗→猫 58、猫→鸟 45、猫→鹿 44。
  猫是混淆中心：猫认错的 344 张里，166 张认成狗。
- Grad-CAM（`analysis/resnet18-prior/gradcam_*.png`，12 张）：
  猫→狗 ×2、狗→猫 ×2、猫→鸟 ×2、猫→鹿 ×2、鸟→青蛙 ×2、鹿→鸟 ×2。
  已目视检查一张猫→狗：热力图集中在猫的身体躯干而非脸/耳朵，
  说明模型更多依据体型姿态而非猫的面部特征做判断。

## 对照实验设计（单变量：只改数据增强）
模型均为 ResNet18，Adam lr 1e-3，weight decay 5e-4，余弦退火，batch 128，
8 轮，种子 42。

| 组 | 增强 | 状态 |
|---|---|---|
| aug-base | RandomCrop(32, padding=4) + 水平翻转 | 复用之前跑出的 87.02%（猫 65.6%、鸟 83.7%），不重训 |
| aug-none | 无增强 | 待训练 `outputs/aug-none/` |
| aug-strong | 基线 + ColorJitter(0.4) + 随机旋转 15° | 待训练 `outputs/aug-strong/` |

## 训练与断点续跑
- `src/train.py`：每 100 个 batch 存一次 `checkpoint.pth`（含 batch 序号
  与本轮累计 loss/正确数），轮末也存；中断后自动从 batch 中间续跑。
  （之前小验证被系统重启打断、轮末断点丢了一整轮，因此改成 batch 级断点。）
- 每轮约 35–45 分钟 CPU；两组新训练约 9–10 小时，按 aug-none → aug-strong
  顺序跑。
- 训练完每组后用 `src/analyze.py` 复跑测试集（混淆矩阵 + Grad-CAM），
  再汇总写报告。

## 产物
- `outputs/<组>/best.pth`、`history.json`、`training_curves.png`
- `analysis/<组>/confusion.json`、`confusion_matrix.png`、`gradcam_*.png`
- `对比实验报告.md`（待写）：准确率表、逐类表、结论、简历一句话
