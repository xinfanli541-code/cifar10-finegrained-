# 会解释的图像分类 Agent

## 问题

上一阶段的细粒度项目只回答了“准不准”：ResNet18 在 CIFAR-10 测试集上
87.02%，猫最难（65.6%），猫→狗错了 166 张。但它回答不了两个更实际的问题：

- 分类对了，不知道为什么对——是真的认出了猫，还是蒙对的？
- 分类错了，不知道错在哪——是两个类本来就长得像，还是模型在看错的地方？

这个 Agent 就是补上这一环：**分类 → Grad-CAM → 一句话解释 → 错判日志**，
形成一个可以一直攒错判样本的闭环。

## 做法

```
图片 -> ResNet18（已有权重，不重训）-> Top3 预测
     -> Grad-CAM（layer4，复用 src/analyze.py）-> CAM 统计
     -> 一句话中文解释 -> 判错且给了真实标签时，记入错判日志
```

- `explain.py`（核心）：`load_agent_model()` 加载已有权重；
  `predict_top3()`；`gradcam_for()` 复用 `src.analyze.gradcam`；
  `cam_summary()` 只从 CAM 数值算质心、3×3 最热格子、中心/边缘热度占比；
  `explain()` 生成一句话中文解释。
- `cli.py`：`python -m agent.cli --image <图> [--true cat] [--out-dir agent/runs/<时间戳>]`，
  打印 Top3、一句话解释，保存原图/热力图/叠加图三联图；判错且给了 `--true`
  时追加到 `agent/logs/misclassified.jsonl`。`python -m agent.cli review`
 （或 `python -m agent.review`）汇总错判对数 top5。
- `demo_test.py`：从测试集取 120 张（每类 12 张，seed 42），输出
  `agent/runs/demo_test/summary.json` + 6 张典型错判三联图。

解释的诚实底线：Grad-CAM 只能说热力图集中在图片的哪个位置，
不能说模型“看到了猫耳朵/脸/毛”。所以解释只说位置、Top1/Top2 有多接近、
是不是已知难混淆对（猫↔狗、猫→鸟、猫→鹿等），绝不编造语义细节。
Top2 接近时（差距 < 0.15），解释会明确说“不太稳 / 只能算比较确定”，留余地。

## 工具

- PyTorch 2.15.0+cpu，ResNet18（CIFAR 适配版，11M 参数），CPU 推理
- Grad-CAM（ResNet18 layer4）
- 只用 torch / PIL / numpy / matplotlib，不装 Gradio，没有 Web 界面，
  Agent 核心就是命令行 + 日志

## 怎么跑（在仓库根目录）

```bash
# 用项目 venv
../pytorch-image-classification/venv/bin/python -m agent.cli --image /tmp/test_cat.png --true cat
../pytorch-image-classification/venv/bin/python -m agent.cli review
../pytorch-image-classification/venv/bin/python -m agent.demo_test
```

权重默认在 `../pytorch-image-classification/outputs/resnet18/best.pth`
（仓库外，43M），可用 `--ckpt` 覆盖。权重和数据都不进 GitHub，
见下面“坑”。

## 实测结果（2026-11-23，CPU，如实记录）

CLI 单张（测试集原图）：

| 图 | 预测 | 解释 |
|---|---|---|
| 猫（test idx 0） | 猫 96.25%，狗 3.73% | 判对了；热力图最热在中中（24%），中心占 45% |
| 狗（test idx 12） | 狗 98.96%，猫 0.62% | 判对了；热力图最热在中中（19%），中心占 37% |
| 猫→狗（test idx 8580） | 狗 48.38%，猫 41.06% | 判错了，已知难混淆对；两者只差 7 个百分点，判断不太稳；已记入错判日志 |

demo_test（120 张，每类 12 张，seed 42）：

- 准确率 **92.5%（111/120）**，错判 9 张
- 逐类：飞机 100%、汽车 100%、鸟 83.3%、猫 83.3%、鹿 100%、狗 100%、
  青蛙 83.3%、马 91.7%、船 83.3%、卡车 100%
- Top 混淆：鸟→猫 2 次，其余（马→狗、青蛙→猫、青蛙→狗、船→卡车、
  船→青蛙、猫→鹿、猫→狗）各 1 次
- 6 张典型错判三联图在 `agent/runs/demo_test/`：
  `misclassified_cat_as_dog_8580.png`、`misclassified_cat_as_deer_8557.png`、
  `misclassified_bird_as_cat_2321.png`、`misclassified_bird_as_cat_9082.png`、
  `misclassified_horse_as_dog_567.png`、`misclassified_frog_as_cat_1572.png`

## 坑

- **权重 43M 不进 GitHub**：`best.pth` 只在本地
  `../pytorch-image-classification/outputs/resnet18/best.pth`，仓库里没有，
  别人 clone 下来跑不了，要自己放权重（见“怎么跑”）。
- **Grad-CAM 只能说位置，不能说语义**：热力图在“中中”不等于模型在看脸，
  在“下中”也不等于在看腿。所有解释都只写位置和概率，不写语义断言。
- **Top2 接近时解释要留余地**：比如上面猫→狗那张，狗 48% 对猫 41%，
  只差 7 个百分点，解释明确说“不太稳”，不能写成“模型确信是狗”。
- **120 张小样本的数字不能和全集比**：demo_test 的 92.5% 高于全测试集的
  87.02%，猫 83.3% 也高于全集的 65.6%，这是小样本运气，不是模型变好了。
  看效果要看逐类 + 混淆对 + 错判日志 top5，不要只报总准确率。

## 效果怎么衡量

不要只报总准确率，要同时看三样：

1. `demo_test` 的准确率、逐类准确率、top 混淆对（`summary.json`）
2. 错判日志的 top5（`python -m agent.cli review`）：现在只有 1 条，
   猫→狗 1 次；日志攒多了才能看出稳定的错判模式
3. 6 张错判三联图里，解释和热力图位置是否对得上（比如猫→狗那张，
   Top2 只差 7 个百分点，解释有没有留余地）

## 文件

```
agent/
  explain.py      核心：加载、预测、Grad-CAM、CAM 统计、一句话解释
  cli.py          命令行：单张预测 + review 汇总
  review.py       review 的独立入口（等价于 python -m agent.cli review）
  demo_test.py    120 张实测
  README.md       本文件
  logs/misclassified.jsonl   错判日志（只有判错且给了真实标签才记）
  runs/           每次运行的三联图 + result.json / summary.json
```
