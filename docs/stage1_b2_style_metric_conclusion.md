# Stage 1：基于 Style 指标确定 `b2` 工作区间

## 1. 实验目标

本轮实验的目标不是证明所有数据集都共享同一个精确的 `b2` 单点，而是寻找一个可以用于训练干预和层路由的 late-style 工作区间：

```text
b2_window = [b2_low, b2_high]
```

当前重点是判断后层是否稳定承担输出格式和语言风格控制，并据此给出一个约三层、接近第 30 层的工程工作区间。

本轮暂时不使用 `dcl` 决定 `b2`，因为 `dcl` 的答案空间过窄，更适合作为诊断数据，而不是后层 style 边界的主证据来源。

## 2. 原始指标存在的问题

上一版 style 指标使用过窄的 token 集合，主要包括：

- `A/B/C/D`
- 数字
- `yes/no`
- `true/false`

这会导致两个问题：

1. 输出风格被压缩成少量答案 token，无法充分表示完整句、短语和标签式回答之间的差异。
2. `style` probe 的 `paired_group_id` 曾经被变体名称改写，导致同一语义样本的多个 style 变体没有进入同一个 pair group，最终 `style_signal` 大量为零。

此外，旧的顶层汇总会扫描历史遗留的 `boundary_report.json`，使旧结果污染新统计。

## 3. 修正后的 Style 指标

### 3.1 Probe 配对

同一张图像、同一个语义问题构造多个 style 变体：

- `style_label_only`
- `style_short_phrase`
- `style_full_sentence`

这些变体保留同一个 `paired_group_id`，只有 `sample_id` 和 prompt instruction 不同。

因此，每个 style group 内部比较的是：

```text
same image
same semantic question
different answer format instruction
```

### 3.2 Expanded Style Vocabulary

新的指标名称为：

```text
expanded_style_vocab_js
```

它在每一层的 hidden state 上经过 LM head 投影，得到扩展 style-control 词表上的 logits。词表包含：

- 选项和标签 token：`A/B/C/D/E/F`
- 数字和真假词：`0-9`, `yes/no`, `true/false`
- 句法和格式 token：`. , : ; -`
- 常见句子起始词：`The`, `There`, `It`, `This`, `An`
- 常见谓词和连接形式：`is`, `are`, `shows`, `contains`, `appears`

对同一个 `paired_group_id` 内的 style 变体，计算两两 Jensen-Shannon divergence：

```text
style_signal(l)
  = mean(
      JS(
        P_l(style_variant_i),
        P_l(style_variant_j)
      )
    )
```

其中 `l` 是模型层索引，`P_l` 是该层 hidden state 经 LM head 投影后，在 expanded style vocabulary 上得到的概率分布。

这个定义比窄 token 集合更接近“模型在输出层面如何响应格式和语言风格 instruction”。

## 4. 实验流程

### 4.1 数据和协议

实验使用 canonical probe pool，并跑：

- `leave_one_dataset_out`
- `leave_one_template_out`

本轮用于确定 `b2` 主工作区间的数据集为：

- `acl`
- `ucit`

`dcl` 保留用于诊断，但不参与 `b2` 主范围估计。

### 4.2 Layer-wise 分析

对每个 probe 样本缓存：

- 每层 hidden state
- style-control vocabulary logits

然后在每个 split 内：

1. 按 `paired_group_id` 聚合 style 变体。
2. 计算每一层的 pairwise JS divergence。
3. 得到完整的 `style_signal(l)` 曲线。
4. 查找跨数据集重叠的后层高峰区。

### 4.3 不把峰值和过渡边界混为一谈

需要区分两个概念：

1. `middle -> late` 统计过渡边界
2. late-style 信号最稳定的工作核心

当前 `b2=29-31` 指的是：

```text
late-style working core
```

它适合作为训练干预和后层路由的工程范围，但不能写成所有数据集共享的精确统计边界。

## 5. 结果

正式结果文件：

- [style_metric_redef_v2 canonical summary](./stage1_protocol_results/llava/style_metric_redef_v2/canonical/stage1_boundary_summary.json)

### 5.1 ACL

ACL 的 style signal 高峰层为：

```text
32, 31, 30, 29, 28
```

说明第 28 到 32 层构成连续的后层 style 高响应区。

### 5.2 UCIT

UCIT 的 style signal 高响应层包括：

```text
32, 31, 30, 29, 28
```

虽然 UCIT 还存在部分中层局部峰值，但其最稳定、最靠后的共同高响应区仍集中在第 28 到 32 层。

### 5.3 跨数据集重叠

ACL 和 UCIT 的后层重叠结果：

```text
top-5 overlap: 29, 31, 32
top-8 overlap: 29, 30, 31, 32
top-10 overlap: 28, 29, 30, 31, 32
```

因此，最接近第 30 层、同时保持三层宽度的工作区间为：

```text
b2_window = [29, 31]
```

中心工作点为：

```text
b2_center = 30
```

## 6. 当前结论

当前可以采用下面的保守结论：

> 在排除答案空间过窄的 `dcl` 后，ACL 和 UCIT 在 expanded style vocabulary JS 指标上共同显示出第 28 到 32 层的 late-style 高响应区。为保持约三层宽度并以第 30 层为中心，当前训练策略使用 `b2_window=[29,31]`，`b2_center=30`。

这条结论的证据等级为：

```text
跨两个数据集、同一模型、同一协议下的工作范围证据
```

还不能写成：

```text
跨模型、跨 seed、跨所有数据集都确认的唯一 middle-late 边界
```

## 7. 训练阶段的使用方式

在当前实验阶段，可以先把层区间定义为：

```text
early  = 1..b1
middle = b1+1..28
late   = 29..32
```

其中：

- `29..31` 是 `b2` 的三层工作核心。
- 第 32 层保留在 late region 中，作为后层尾部。
- 不建议把 `b2=30` 当成唯一不可变的单层边界。

下一步训练干预应围绕以下设置验证这个工作区间：

- `freeze_late`: 冻结第 29 到 32 层，观察 style-following 损失。
- `train_only_late`: 只训练第 29 到 32 层，观察 style-following 收益。
- `freeze_middle`: 与 late 区间做对照。
- `train_only_middle`: 与 late 区间做对照。

只有当 `freeze_late` 明显损害 style-following，且 `train_only_late` 在多个 seed 下稳定带来收益时，`29..31` 才能升级为强训练策略依据。

## 8. 复现入口

指标实现：

```text
scripts/stage1_analyze_boundaries.py
```

Style 配对实现：

```text
scripts/stage1_build_canonical_pool.py
```

当前 dataset-level 验证结果：

```text
docs/stage1_protocol_results/llava/style_metric_redef_v2/canonical/
```

## 9. InternVL 对照实验

为检验 `b2` 工作区间是否依赖模型架构，`InternVL` 采用与 `LLaVA` 当前稳定版完全一致的口径：

- `expanded_style_vocab_js`
- `cluster_balanced` dataset-level aggregation
- canonical probe pool
- `leave_one_dataset_out`
- `ACL + UCIT` 作为 `b2` 主证据
- `DCL` 仅保留为诊断，不参与最终 `b2` 估计

本节以 2026 年 8 月 11 日的两条正式复跑结果为准：

- [seed17 canonical summary](/mnt/lyaa/MCITlib/docs/stage1_b1_windowed_dataset_only_cluster_balanced/internvl/seed17/canonical/stage1_boundary_summary.json)
- [seed27 canonical summary](/mnt/lyaa/MCITlib/docs/stage1_b1_windowed_dataset_only_cluster_balanced/internvl/seed27/canonical/stage1_boundary_summary.json)

此前文档中的 `internvl_style_metric_v1` 仅可视为早期探索；当前 `InternVL b2` 结论统一以这轮同口径复跑为准。

### 9.1 InternVL 的 late-style overlap

按与 `LLaVA` 相同的 late-style peak overlap 口径，只读取后层峰带：

#### seed17

- `ACL late top-5 = 29,28,27,30,32`
- `UCIT late top-5 = 28,27,29,30,31`
- `overlap = 27,28,29,30`

#### seed27

- `ACL late top-5 = 32,27,28,31,29`
- `UCIT late top-5 = 27,28,29,30,31`
- `overlap = 27,28,29,31`

两条 seed 的稳定交集核心为：

```text
27..29
```

其中 `30` 和 `31` 分别只在单个 seed 中进入后层重叠，因此更适合作为外层支持，而不作为最稳核心。

### 9.2 当前 InternVL 的 `b2` 工作区间

因此，当前 `InternVL` 的正式 `b2` 结论写为：

```text
b2_window = [27, 29]
b2_center = 28
```

如果需要保守描述外层支持带，可以放宽为：

```text
27..31
```

但真正跨 seed 稳定、适合训练和路由的三层工作核心仍然是 `27..29`。

### 9.3 为什么不直接使用联合分割的 `recommended_b2`

这轮 `cluster_balanced` 复跑中，联合分割给出的 `recommended_b2` 仍明显早于 late-style 峰值。例如：

```text
seed17: ACL=17, UCIT=20
seed27: ACL=18, UCIT=24
```

这说明 `InternVL` 的联合分割点仍然会受到中层 objective 信号牵引，因此可以作为诊断信息保留，但不应替代 style-only 的 late-style 工作区间。

### 9.4 与 LLaVA 的比较

两种模型在同口径下的 `b2` 工作核心为：

| model | b2 工作区间 | 中心层 |
|---|---:|---:|
| LLaVA | `29..31` | 30 |
| InternVL | `27..29` | 28 |

因此，InternVL 的 late-style 核心相对 LLaVA 约提前 2 层，但两者都稳定落在模型后部。当前更稳妥的跨模型表述是：

```text
late-style region:
LLaVA    = 29..31
InternVL = 27..29

conservative cross-model candidate:
27..31
```

`27..31` 是跨模型的候选后层范围，不等于每个模型都应该训练同样的层。后续训练干预应分别使用模型内工作区间，并将 `27..31` 作为统一敏感性分析范围。
