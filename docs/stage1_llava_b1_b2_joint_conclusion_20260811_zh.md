# Stage 1：LLaVA 的 `b1 / b2` 联合结论文档

日期：2026 年 8 月 11 日

## 1. 结论先行

当前 `LLaVA` 的 Stage-1 边界更适合写成两个区间，而不是两个单点：

- `b1_window = [15, 18]`
- `b1_center = 16`
- `b2_window = [29, 31]`
- `b2_center = 30`

如果需要一个更宽但更保守的外层描述，可以写成：

- `b1` 的主共识带：`15..18`
- `b2` 的后层支持带：`28..32`

据此，当前最稳妥的工程分段是：

```text
early  = 1..16
middle = 17..28
late   = 29..32
```

其中：

- `16` 是 `b1` 的最高支持层
- `29..31` 是 `b2` 的三层工作核心
- 第 `32` 层保留在 late region 中，作为后层尾部

## 2. 为什么 `b1` 和 `b2` 不能用同一种判定法

`b1` 和 `b2` 对应的是两个不同问题：

1. `b1` 要回答的是：

   `reasoning -> objective` 的阶段转移大约发生在哪里

2. `b2` 要回答的是：

   输出风格和格式控制最稳定的后层工作核心在哪里

因此，这两个边界不应该被同一个“联合最优点”强行绑定。

当前采用的原则是：

- `b1` 用 crossover band 判定
- `b2` 用 late-style peak overlap 判定

也就是说：

- `b1` 看 reasoning 和 objective 的局部交叉带
- `b2` 看 style 信号在后层的稳定重叠区

这也是为什么当前文档把 `b1` 和 `b2` 分开论证，最后再组合成统一层区间。

## 3. `b1` 是怎么确定的

### 3.1 目标

`b1` 不再被写成“唯一最优层”，而是被写成一个较窄的阶段转移区间。

当前目标是找到：

```text
b1_window = [b1_low, b1_high]
```

它表示模型从 reasoning-dominant 过渡到 objective-dominant 的稳定候选带。

### 3.2 使用的三条信号

实现位于 [scripts/stage1_analyze_boundaries.py](/mnt/lyaa/MCITlib/scripts/stage1_analyze_boundaries.py)。

对每个 split，分析器都会计算三条逐层曲线：

1. `reasoning_signal`

   对 reasoning-family 样本，按标签做 Fisher separation。

2. `objective_signal`

   对 objective-family 样本，按标签做 Fisher separation。

3. `style_signal`

   对 style-family 样本，计算 paired group 内 style logits 的两两 Jensen-Shannon divergence 均值。

三条曲线都会做：

- min-max 归一化
- 窗口为 `3` 的平滑

记处理后的曲线为：

- `R_l`
- `O_l`
- `S_l`

### 3.3 `b1` 的打分策略

`b1` 当前使用的策略是：

```text
boundary_strategy = b1_only_crossover
```

主搜索窗固定为：

- `b1_search_low = 10`
- `b1_search_high = 20`
- `transition_min_run = 3`
- `min_transition_support = 0.01`
- `min_pre_reasoning_margin = -0.20`
- `b1_band_score_tolerance = 0.03`
- `b1_band_min_width = 3`
- `b1_band_max_width = 5`

对每个候选层 `b1`，定义：

```text
pre_margin(b1)  = mean(R_l - O_l)   on the 3 layers before b1
post_margin(b1) = mean(O_l - R_l)   on the 3 layers after  b1
boundary_contrast(b1) = pre_margin(b1) + post_margin(b1)
```

局部 crossover 分数为：

```text
local_score(b1)
  = min(pre_margin, post_margin)
    + boundary_contrast
    + late_preference * (b1 - low)
```

其中：

- `late_preference = 0.01`

### 3.4 为什么输出的是区间而不是单点

当前实现不是直接取最优单点，而是：

1. 找到所有 near-best 的候选 `b1`
2. 在这些候选点上选择一个 `3-5` 层宽的连续 band
3. 输出：

```text
boundary_windows.b1 = [low, high]
```

因此：

- `10..20` 是搜索窗
- `15..18` 是结果

也就是说，`15..18` 不是先验写死的范围，而是数据和指标共同算出来的主共识带。

## 4. `b1` 的实验配置与结果

### 4.1 数据与 split

`b1` 的大规模稳定性实验使用：

- 模型：`LLaVA`
- benchmark：`ucit acl dcl`
- split：`leave_one_dataset_out`
- `max_samples_per_task = 128`

正式多 seed 运行入口是：

- [scripts/run_stage1_b1_multiseed.sh](/mnt/lyaa/MCITlib/scripts/run_stage1_b1_multiseed.sh)

结果汇总文档见：

- [docs/stage1_b1_interval_report_20260811_zh.md](/mnt/lyaa/MCITlib/docs/stage1_b1_interval_report_20260811_zh.md)

### 4.2 使用的 seed

当前 `LLaVA b1` 大规模实验共使用 6 个 seed：

- `7`
- `13`
- `21`
- `29`
- `35`
- `42`

对应结果见：

- [seed7](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/llava/seed7/canonical/stage1_boundary_summary.json)
- [seed13](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/llava/seed13/canonical/stage1_boundary_summary.json)
- [seed21](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/llava/seed21/canonical/stage1_boundary_summary.json)
- [seed29](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/llava/seed29/canonical/stage1_boundary_summary.json)
- [seed35](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/llava/seed35/canonical/stage1_boundary_summary.json)
- [seed42](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/llava/seed42/canonical/stage1_boundary_summary.json)

### 4.3 结果概括

当前 `b1` 的主结论是：

- 推荐的主共识区间：`15..18`
- 更尖锐的中心带：`16..18`
- 所有 valid split 中的最高投票层：`16`

按 seed 的 valid band 来看：

| Seed | Valid `b1` band |
| --- | --- |
| 7 | `15..19` |
| 13 | `13..17` |
| 21 | `10..14` |
| 29 | `16..20` |
| 35 | `14..18` |
| 42 | `12..16` |

虽然存在偏早 seed，但有效证据的大多数质量集中在 `15..18`。

### 4.4 如何解释这些结果

这组结果支持下面的判断：

1. `b1` 已经稳定到区间，但还没稳定到唯一单层。

2. `b1` 的主峰不再塌在搜索窗下界附近。

   这一点说明当前 crossover 指标已经明显好于早期那类“贴下界崩塌”的失败版本。

3. `acl` 和 `dcl` 对后移 band 的支持更强，`ucit` 仍然是不稳定来源。

因此，`b1` 当前可以写成：

```text
b1_window = [15, 18]
b1_center = 16
```

但不该写成：

```text
b1 = 某个唯一层
```

## 5. `b2` 是怎么确定的

### 5.1 目标

`b2` 的目标不是寻找唯一精确统计边界，而是寻找一个 late-style 工作核心：

```text
b2_window = [b2_low, b2_high]
```

它描述的是：

- 输出格式
- 语言风格
- 风格跟随

最稳定地集中在哪个后层区间。

### 5.2 使用的 style 指标

`b2` 使用的 style 指标仍然是：

```text
expanded_style_vocab_js
```

原始说明见：

- [docs/stage1_b2_style_metric_conclusion.md](/mnt/lyaa/MCITlib/docs/stage1_b2_style_metric_conclusion.md)

style family 通过以下变体构造：

- `style_label_only`
- `style_short_phrase`
- `style_full_sentence`

在同一个 `paired_group_id` 内，对每层的 style logits 计算 pairwise JS divergence，得到：

```text
style_signal(l)
```

### 5.3 `b2` 为什么不直接读联合分割的 `recommended_b2`

这一点非常重要。

`b1` 运行脚本产出的 `recommended_boundaries.b2` 是一个联合分割点，它同时受到：

- reasoning
- objective
- style

三条曲线的影响。

但 `b2` 当前研究的问题不是“联合分割最优点”，而是：

```text
late-style working core
```

因此，当前 `b2` 结论主要看的是：

- `curves.style_signal_norm`
- `ACL/UCIT` 跨数据集的后层峰值重叠

而不是直接把 `recommended_b2` 当成最终 `b2`。

### 5.4 最新稳定版为什么引入 `cluster_balanced`

在大样本 `ACL` 上，如果直接对所有 style group 做 raw 平均，容易被以下 cluster 过度主导：

- `APP`
- `OCR`
- `counting`
- `mcq_visual_qa`

这会让 `ACL` 的中层 style 峰被放大，从而削弱跨数据集的 late-style overlap。

因此，这一轮 `LLaVA b2` 稳定化没有改 `expanded_style_vocab_js` 定义，而是只改了 dataset-level 聚合方式：

```text
style_aggregation = cluster_balanced
```

具体做法是：

1. 先在每个 `prompt_cluster` 内聚合 style group
2. 再对 cluster 均值做等权平均

这个修正同时作用于 `ACL` 和 `UCIT`，不是对某个 benchmark 的单独补丁。

实现位于：

- [scripts/stage1_analyze_boundaries.py](/mnt/lyaa/MCITlib/scripts/stage1_analyze_boundaries.py)

## 6. `b2` 的实验配置与结果

### 6.1 数据与口径

当前稳定版 `b2` 证据使用：

- 模型：`LLaVA`
- canonical probe pool
- split：`leave_one_dataset_out`
- 主数据集：`ACL + UCIT`
- `DCL` 只保留为诊断

需要注意：

- 这里使用的是 style 曲线的重叠分析
- 不是把联合分割产出的 `recommended_b2` 当成 `b2`

### 6.2 使用的 seed

最新稳定性验证使用了以下结果：

- [seed7](/mnt/lyaa/MCITlib/docs/stage1_b1_windowed_dataset_only_cluster_balanced/llava/seed7/canonical/stage1_boundary_summary.json)
- [seed13](/mnt/lyaa/MCITlib/docs/stage1_b1_windowed_dataset_only_cluster_balanced/llava/seed13/canonical/stage1_boundary_summary.json)
- [seed21](/mnt/lyaa/MCITlib/docs/stage1_b1_windowed_dataset_only_cluster_balanced/llava/seed21/canonical/stage1_boundary_summary.json)
- [validonly_tiny seed7](/mnt/lyaa/MCITlib/docs/stage1_b1_validonly_tiny_cluster_balanced/llava/seed7/canonical/stage1_boundary_summary.json)

这里前三个 seed 是主证据，`validonly_tiny` 主要起到“原本跑偏场景是否被纠正”的补充验证作用。

### 6.3 各 seed 的重叠结果

在 `cluster_balanced` 口径下：

#### seed7

- `ACL top-5 = 32,31,30,29,28`
- `UCIT top-5 = 30,29,31,32,28`
- `top-5 overlap = 28,29,30,31,32`

#### seed13

- `ACL top-5 = 32,31,30,29,28`
- `UCIT top-5 = 32,31,30,29,28`
- `top-5 overlap = 28,29,30,31,32`

#### seed21

- `ACL top-5 = 30,32,31,29,28`
- `UCIT top-5 = 32,31,30,29,28`
- `top-5 overlap = 28,29,30,31,32`

#### validonly_tiny seed7

- 原先跑偏的 `ACL` 已被拉回后层
- `top-5 overlap = 28,29,30,31,32`

### 6.4 如何解释这些结果

这轮结果支持下面的判断：

1. `LLaVA` 的 late-style 区间已经稳定到后层。

2. 最稳的跨 seed 重叠带是：

```text
28..32
```

3. 若要保持三层宽度并围绕第 30 层取工程核心，则：

```text
b2_window = [29, 31]
b2_center = 30
```

因此：

- `28..32` 是后层支持带
- `29..31` 是训练和路由更方便的三层工作核心

## 7. `b1 / b2` 联合解读

当前 `LLaVA` 的 `b1` 和 `b2` 可以这样统一理解：

### 7.1 `b1`

`b1` 表示从 reasoning 主导逐步过渡到 objective 主导的区间。

它的结论来自：

- reasoning/objective crossover band
- 多 seed
- valid split 投票聚合

当前最强共识是：

```text
15..18
```

### 7.2 `b2`

`b2` 表示 late-style 工作核心，而不是联合分割点。

它的结论来自：

- style-only late peak
- `ACL + UCIT` 的跨数据集后层重叠
- 多 seed 稳定性验证

当前最强共识是：

```text
29..31
```

### 7.3 为什么这两个区间是兼容的

两者之间不存在冲突，因为它们描述的是不同阶段：

- `15..18`：middle stage 的起始过渡带
- `29..31`：late stage 的 style 工作核心

合起来正好形成当前可用的三段式结构：

```text
early  = 1..16
middle = 17..28
late   = 29..32
```

## 8. 当前推荐写法

如果要写进论文或方法文档，当前最稳妥的表述是：

> 对于 LLaVA，Stage-1 边界不应被固定为两个精确单层，而应分别写成一个中层过渡区间和一个后层 style 工作区间。基于多 seed 的 crossover-band 分析，`b1` 的主共识位于 `15-18` 层附近，峰值支持集中在第 `16` 层。基于 expanded style vocabulary JS 指标和 cluster-balanced 的 dataset-level 聚合，`b2` 的稳定后层核心位于 `29-31` 层，外层支持带约为 `28-32`。据此，当前 LLaVA 的推荐分段为：`early=1..16`，`middle=17..28`，`late=29..32`。

## 9. 不该怎么写

当前还不建议写成：

```text
b1 = 唯一某一层
b2 = 唯一某一层
```

也不建议写成：

```text
b1 / b2 都由同一种联合边界指标直接给出
```

更不建议把 `b1` 的联合 `recommended_b2` 当成最终 `b2`。

## 10. 复现入口

`b1` 大规模多 seed：

- [scripts/run_stage1_b1_multiseed.sh](/mnt/lyaa/MCITlib/scripts/run_stage1_b1_multiseed.sh)

`b1` 主 runner：

- [scripts/run_stage1_b1_results.sh](/mnt/lyaa/MCITlib/scripts/run_stage1_b1_results.sh)

`b2 / style` 分析核心：

- [scripts/stage1_analyze_boundaries.py](/mnt/lyaa/MCITlib/scripts/stage1_analyze_boundaries.py)

canonical pool 构造：

- [scripts/stage1_build_canonical_pool.py](/mnt/lyaa/MCITlib/scripts/stage1_build_canonical_pool.py)

原始 `b2` 小规模文档：

- [docs/stage1_b2_style_metric_conclusion.md](/mnt/lyaa/MCITlib/docs/stage1_b2_style_metric_conclusion.md)

最新 `b1` 区间文档：

- [docs/stage1_b1_interval_report_20260811_zh.md](/mnt/lyaa/MCITlib/docs/stage1_b1_interval_report_20260811_zh.md)

## 11. InternVL 对照结果：当前 `b1 / b2` 区间

这一节补充 `InternVL` 在与 `LLaVA` 相同分析口径下的当前 `b1 / b2` 结果，不改变前文针对 `LLaVA` 的主结论。

### 11.1 为什么可以与 `LLaVA` 对照

`InternVL` 这里使用的是和 `LLaVA` 相同的 `b1` 实验框架：

- 同一个多 seed 启动器
- 同一个 `b1_only_crossover` 指标
- 同一个搜索窗 `10..20`
- 同一组目标 seed：`7 13 21 29 35 42`
- 同样的 `leave_one_dataset_out`
- 同样的 `max_samples_per_task = 128`

也就是说，这里做的是“同方法、不同模型”的对照，而不是另一套不可比流程。

### 11.2 当前完成情况

截至 2026 年 8 月 11 日，这轮 `InternVL` 大规模 `b1` 实验已完成 `5` 个 seed：

- `7`
- `13`
- `21`
- `29`
- `35`

对应结果见：

- [seed7](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/internvl/seed7/canonical/stage1_boundary_summary.json)
- [seed13](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/internvl/seed13/canonical/stage1_boundary_summary.json)
- [seed21](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/internvl/seed21/canonical/stage1_boundary_summary.json)
- [seed29](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/internvl/seed29/canonical/stage1_boundary_summary.json)
- [seed35](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/internvl/seed35/canonical/stage1_boundary_summary.json)

`seed42` 未完成。中断原因不是指标失效，而是磁盘写满，日志见：

- [seed42.log](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/internvl/logs/seed42.log)

因此，下面的 `InternVL b1` 区间结论是基于当前已经完成的 `5` 个 seed，而不是完整 `6` 个 seed。

`InternVL b2` 则已在同一天按与 `LLaVA` 一致的 `cluster_balanced + leave_one_dataset_out + ACL/UCIT late-style overlap` 口径完成两条正式复跑：

- [seed17 b2 canonical summary](/mnt/lyaa/MCITlib/docs/stage1_b1_windowed_dataset_only_cluster_balanced/internvl/seed17/canonical/stage1_boundary_summary.json)
- [seed27 b2 canonical summary](/mnt/lyaa/MCITlib/docs/stage1_b1_windowed_dataset_only_cluster_balanced/internvl/seed27/canonical/stage1_boundary_summary.json)

因此，当前文档中关于 `InternVL b2` 的结论统一以这两条正式复跑为准，不再沿用此前那些临时的、未完全收敛的占位说法。

### 11.3 当前结果汇总

按当前已完成的 `5` 个 seed 汇总：

| Seed | Valid `b1` band |
| --- | --- |
| 7 | `11..15` |
| 13 | `14..18` |
| 21 | `10..15` |
| 29 | `10..14` |
| 35 | `10..14` |

把所有 valid split 的 band 做逐层投票后，当前票数主峰为：

- `14` 层：`9` 票
- `12` 层：`8` 票
- `13` 层：`8` 票
- `11` 层：`7` 票
- `10` 层：`6` 票

由此得到当前 `InternVL` 的区间汇总：

- 推荐窄区间：`12..14`
- 峰值代表层：`14`
- 若要给更保守的宽区间：`10..14`

### 11.4 按 dataset 看当前稳定性

在当前 `5` 个已完成 seed 上：

- `acl`：`4 / 5` valid
- `dcl`：`3 / 5` valid
- `ucit`：`2 / 5` valid

这说明：

- `InternVL` 的 `acl` 仍然是最稳定的证据来源
- `dcl` 会在 `10..14` 和 `14..18` 之间摆动
- `ucit` 相比 `LLaVA` 不再完全失效，但当前支持仍主要落在更靠前的 `10..14`

### 11.5 `InternVL b2` 的当前稳定区间

按与 `LLaVA` 一致的 late-style peak overlap 口径，只读取 `ACL/UCIT` 的后层主峰：

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

因此，当前 `InternVL` 的 `b2` 推荐写法为：

```text
b2_window = [27, 29]
b2_center = 28
```

如果需要外层支持带，可以写成 `27..31`；但真正跨 seed 稳定的三层工作核心是 `27..29`。

需要注意，`InternVL` 的联合分割 `recommended_b2` 仍然更早，例如这轮复跑中 `seed17` 的 `ACL/UCIT` 为 `17/20`，`seed27` 为 `18/24`。这再次说明最终 `b2` 不能直接由联合分割点替代，而应以 late-style overlap 为准。

### 11.6 与 `LLaVA` 的直接对比

在相同方法下，当前两模型的 `b1 / b2` 位置均存在模型差异：

- `LLaVA`：`b1_window = [15, 18]`，`b1_center = 16`；`b2_window = [29, 31]`，`b2_center = 30`
- `InternVL`：`b1_window = [12, 14]`，峰值层 `14`；`b2_window = [27, 29]`，`b2_center = 28`

因此，当前证据支持下面这个模型级结论：

1. `b1` 确实和模型有关。

2. 在同一指标和同一实验协议下，`InternVL` 的 stage-1 过渡位置整体早于 `LLaVA`。

3. `b2` 也表现出明确的模型依赖性，但两种模型都在后层形成稳定的 late-style 工作区。

4. 这也解释了为什么把边界理解为“更偏模型内部属性，而不是纯数据属性”是合理的，但仍然需要跨 seed 去验证其稳定区间。

### 11.7 当前建议写法

如果要把 `InternVL` 一并写进当前结论，最稳妥的表达是：

> 在相同的边界分析框架下，不同模型的 `b1 / b2` 区间并不相同。对 `LLaVA` 而言，`b1` 的主共识集中在 `15-18` 层附近，`b2` 的 late-style 工作核心位于 `29-31` 层；对 `InternVL` 而言，当前已完成 5 个 seed 的 `b1` 主共识更靠前，集中在 `12-14` 层附近，而按同口径 late-style overlap 复跑得到的 `b2` 工作核心位于 `27-29` 层。这说明阶段边界更接近模型内部属性，而不是跨模型固定不变的数据常数。

在当前版本里，更推荐把两者并列写成：

```text
LLaVA   : b1_window = [15, 18], b1_center = 16, b2_window = [29, 31], b2_center = 30
InternVL: b1_window = [12, 14], b1_center ≈ 14, b2_window = [27, 29], b2_center = 28
```

但也应同时注明：

- `InternVL` 目前是 `5` 个已完成 seed 的暂定结论
- `seed42` 尚未补完
- 因此 `InternVL b1` 已足够作为模型对照结论，但仍可在补完 `seed42` 后进一步收尾
