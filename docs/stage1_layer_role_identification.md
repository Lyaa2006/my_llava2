# Stage 1：无专家的层级边界识别

本实验用于在**不引入任何专家 LoRA、role memory 或 routing 机制**的前提下，对 `base LLaVA` backbone 的三段式结构进行分析，并为后续 `Hi-DESC` 的层级协同设计提供依据。

实验实现严格基于原始 `base LLaVA` 前向，只保留：

- `vision tower`
- `mm_projector`
- `LLM backbone`

不使用：

- `HiDESC` 中的 `text tower`
- `role-aware routing`
- `anchor update`
- `expert collaboration`

因此，Stage 1 的目的不是评估某种专家策略，而是先回答一个更基础的问题：**在最朴素的 backbone 前向中，不同深度的层更偏向承担什么功能。**

## 实验目标

Stage 1 旨在识别两个边界点，并将 32 层 LLaVA backbone 划分为三段：

- 深层：通用多模态推理协同
- 中层：任务级语义特化
- 浅层：输出语言风格控制

在当前版本中，我们最终采用如下工作划分：

- 深层：`1-10`
- 中层：`11-27`
- 浅层：`28-32`

对应边界为：

- 深中边界：`10`
- 中浅边界：`28`

## 实验流程

最新版 Stage 1 使用统一脚本一次性完成两类分析：

```bash
python3 scripts/analyze_llava_hierarchical_boundaries.py \
  --model-path /mnt/lyaa/my_llava/llava-v1.5-7b \
  --vision-tower-path /mnt/lyaa/my_llava/clip-vit-large-patch14-336 \
  --ucit-root /mnt/lyaa/my_llava/UCIT \
  --output-dir docs/hierarchical_boundary_outputs_fixed_10_28_1k \
  --stage1-samples-per-task 1000 \
  --style-samples-per-task 1000 \
  --min-search-layer 20 \
  --fixed-deep-boundary 10 \
  --fixed-mid-shallow-boundary 28 \
  --device cuda
```

其中：

- `stage1-samples-per-task = 1000`：用于深中边界分析
- `style-samples-per-task = 1000`：用于中浅边界分析

这一轮实验使用了更大规模的 UCIT 样本，避免仅用几十个样本带来的稳定性争议。

## 信号设计

统一实验同时输出五条逐层曲线：

- `Reasoning Signal`
- `Task Separation`
- `Style Hidden Signal`
- `Style Logit Signal`
- `Style Combined Signal`

它们分别承担不同角色。

### 1. Reasoning Signal

该信号用于描述：模型在推理型任务中，对真实图像信息的依赖程度随层深如何变化。

实现方式是：

- 在 `ArxivQA`、`IconQA`、`CLEVR` 上
- 保持文本输入不变
- 比较“真实图像输入”和“空白图像输入”所引起的表征偏移

如果某层对图像内容的真实推理依赖更强，那么这一层的 `Reasoning Signal` 更高。

### 2. Task Separation

该信号用于描述：不同任务在某一层的表征空间中是否已经被明显分开。

实现方式是：

- 在全部六个 UCIT 任务上
- 计算类 Fisher 的任务间 / 任务内分离度

如果某层更偏向任务级语义特化，那么不同任务会在这一层更容易被分离。

### 3. Style Hidden Signal

该信号用于描述：当输出格式要求变化时，最后一个 prompt token 附近的内部表示变化有多强。

它主要反映模型对“即将以什么形式作答”的内部风格敏感性。

### 4. Style Logit Signal

这是最新版实验中最关键的改进指标。

它不再只测 hidden state 的变化，而是直接测量：

- 同语义问题
- 不同输出格式要求
- 在每一层通过 `logit lens` 投影得到的**首个答案 token 分布分歧**

这使得风格信号更贴近“模型最终打算输出什么形式的答案”，而不是仅仅测到“模型是否已经理解了格式要求”。

### 5. Style Combined Signal

最终用于中浅边界分析的风格曲线是：

- `0.35 * Style Hidden Signal`
- `0.65 * Style Logit Signal`

这样做的原因是：

- `Style Hidden Signal` 能补充内部表示信息
- `Style Logit Signal` 更接近生成端决策

因此组合后可以更稳定地刻画浅层的输出风格控制能力。

## 两个边界点的选择依据

### 深中边界为什么取 10

在早期自动搜索实验中，深中边界曾出现过更靠前的候选值，例如 `7`。  
但进一步分析后，我们没有直接采用这一更早的值，而是固定为 `10`，原因如下：

1. `7` 更像是“任务可分性开始提前出现”的位置，而不够稳定地代表“通用推理主导区间已经结束”。
2. 在 proposal 的方法叙事中，深层需要承担跨 role 的通用推理协同。如果深层只保留到 `7`，该区间过薄，解释力较弱。
3. 从统一实验曲线来看，前 `10` 层仍可视作一个相对完整的前段：在这一段内，模型尚未完全转入以任务语义分化为主的区间。
4. `10` 是一个更保守、也更稳定的工作边界，便于后续方法设计与实验复现。

因此，我们将深中边界固定为 `10`，并把它解释为：

**从第 1 层到第 10 层，模型更适合承担跨任务共享的通用推理协同；从第 11 层开始，任务特异语义逐渐成为主导。**

### 中浅边界为什么取 28

中浅边界的选择经历了多轮修正。

早期版本使用较粗糙的 style 指标时，得到过明显偏早的结果，例如 `10` 或 `17`。  
这些值之所以不理想，是因为当时的实验更容易测到：

- prompt 模板差异
- 指令理解差异
- 任务形式差异

而不是“最终答案表面风格真正由哪几层控制”。

在最新版中，我们将风格指标改为以**首个答案 token 的 logit 分布差异**为主，因此中浅边界的含义变得更明确：

- 不是“模型何时开始察觉格式要求”
- 而是“模型何时真正开始决定答案该输出成字母、短语还是完整句子”

在这套指标下，风格相关信号的高点明显后移，并在模型尾部形成稳定的高值区域。最终取 `28` 的依据是：

1. `Style Combined Signal` 在最后几层达到峰值，说明风格控制主要集中在输出端附近。
2. `28` 将浅层限定在 `28-32`，这一段与“最终语言风格控制”这一功能更一致。
3. 该划分与已有关于“最后若干层更贴近表面表达与解码风格”的先验知识相一致。

因此，我们将中浅边界固定为 `28`，并把它解释为：

**从第 28 层开始，模型进入更接近最终答案表面形式控制的浅层区间。**

## 图像解释

最新版实验的主要图像为：

- `docs/hierarchical_boundary_outputs_fixed_10_28_1k/hierarchical_boundary_curves.png`

这张图同时画出了五条逐层曲线，并用阴影标出三段结构。

### 图中每条曲线的含义

- 红线 `Reasoning Signal`
  表示图像驱动推理依赖的强弱。

- 蓝线 `Task Separation`
  表示任务级语义特化的强弱。

- 紫线 `Style Hidden Signal`
  表示最后一个 prompt token 附近的隐藏表征对输出格式的敏感性。

- 绿线 `Style Logit Signal`
  表示首个答案 token 的分布对输出格式的敏感性。

- 黄线 `Style Combined Signal`
  是最终用于中浅边界判断的综合风格曲线。

### 图中阴影区域的含义

- 左侧阴影 `1-10`
  对应深层区间，表示这里主要承担通用推理协同。

- 中间阴影 `11-27`
  对应中层区间，表示这里主要承担任务级语义特化。

- 右侧阴影 `28-32`
  对应浅层区间，表示这里主要承担最终输出风格控制。

### 图应如何解读

这张图的核心不是要求五条曲线在每一层都完全分离，而是观察三种功能在不同区间的**相对主导关系**：

1. 前段主要由推理与早期任务分化信号主导
2. 中段主要体现任务特化的持续作用
3. 尾段风格控制信号，特别是 `Style Logit Signal` 和 `Style Combined Signal`，在接近输出端时达到高值

因此，该图支持如下分层解释：

- 前 10 层更适合做跨 role 的通用推理协同
- 11 到 27 层更适合做同 role 内的任务语义精化
- 28 到 32 层更适合做答案格式与语言风格的最终收敛

## 输出文件

统一实验会生成：

- `docs/hierarchical_boundary_outputs_fixed_10_28_1k/hierarchical_boundary_report.json`
- `docs/hierarchical_boundary_outputs_fixed_10_28_1k/hierarchical_boundary_curves.png`

其中：

- `report.json` 保存最终边界与各条曲线
- `curves.png` 用于展示三段结构的可解释性

## 小结

基于大规模 UCIT 样本、纯 `base LLaVA` 前向以及生成端风格控制指标，Stage 1 最终采用如下层级划分：

- 深层：`1-10`
- 中层：`11-27`
- 浅层：`28-32`

这一划分的意义在于：

- 它不是直接凭经验指定
- 也不是从已训练好的专家中反推得到
- 而是先从 backbone 本身的层功能出发，给出一个可解释的层级结构

这为后续 `Hi-DESC` 的 `Role-aware Progressive Collaboration` 提供了结构先验：  
先确定哪一段更适合做通用推理、哪一段更适合做任务特化、哪一段更适合做输出风格控制，再据此设计专家协同方式。
