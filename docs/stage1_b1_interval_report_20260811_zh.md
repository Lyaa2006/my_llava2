# Stage 1 `b1` 区间实验报告

日期：2026 年 8 月 11 日

## 结论先行

以当前证据强度来看，`b1` 更适合被表述为一个区间，而不是单一精确层号。

- 推荐的主共识区间：`15-18`
- 更尖锐的中心带：`16-18`
- 在所有 valid split 上的最高投票层：`16`

这个证据已经足够支撑论文里“稳定区间但仍有少量 seed 敏感性”的说法，但还不足以支撑“唯一精确层”或“完全与数据无关”的强表述。

## 在正式大跑前做了什么修正

这一轮只针对 `b1`，没有改动 `b2` 的既有流程。

- `b2` 工作流保持原样
- `b1` 使用独立 runner：[scripts/run_stage1_b1_results.sh](/mnt/lyaa/MCITlib/scripts/run_stage1_b1_results.sh)
- 多 seed 启动器为：[scripts/run_stage1_b1_multiseed.sh](/mnt/lyaa/MCITlib/scripts/run_stage1_b1_multiseed.sh)

最关键的变化是：`b1` 不再被当作“单个最优点”来判定，而是改成围绕最佳 crossover 区域返回一个 `3-5` 层宽的候选带。

## 实验是怎么开展的

### 数据划分协议

我们对以下三个 benchmark 做 leave-one-dataset-out 分析：

- `acl`
- `dcl`
- `ucit`

对每个 seed，分析器都会输出三个 held-out dataset 的报告。

### Seed 与规模

最终的大规模实验共用了 6 个 seed：

- `7`
- `13`
- `21`
- `29`
- `35`
- `42`

对应结果在：

- [seed7](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/llava/seed7/canonical/stage1_boundary_summary.json)
- [seed13](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/llava/seed13/canonical/stage1_boundary_summary.json)
- [seed21](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/llava/seed21/canonical/stage1_boundary_summary.json)
- [seed29](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/llava/seed29/canonical/stage1_boundary_summary.json)
- [seed35](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/llava/seed35/canonical/stage1_boundary_summary.json)
- [seed42](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/llava/seed42/canonical/stage1_boundary_summary.json)

大规模启动器会尽量复用历史 `b1` 实验产生的 sample cache。对于同 seed 的重分析，这种复用非常有效；对于新 seed，由于采样样本本身会变化，因此 cache 复用只能部分生效。

### 为什么要看跨 Seed 稳定性

跨 seed 的目的，是检验结论对采样扰动是否稳定。

如果某个 `b1` 候选带只在单个 seed 下成立，它很可能只是抽样偶然；如果它在多个 seed 下都持续出现，就更可能反映模型内部的真实阶段转移。

## 指标公式

当前实现位于 [scripts/stage1_analyze_boundaries.py](/mnt/lyaa/MCITlib/scripts/stage1_analyze_boundaries.py)。

### 三条阶段信号曲线

对每个 split，我们计算三条逐层曲线：

1. `reasoning_signal`

   对 reasoning-family 样本，按照标签分组，在每一层上计算 Fisher separation：

   `Fisher(l) = BetweenClassVar(l) / max(WithinClassVar(l), eps)`

2. `objective_signal`

   对 objective-family 样本，同样计算逐层 Fisher separation。

3. `style_signal`

   对 style-family 样本，在每个 paired group 内计算 style logits 的两两 Jensen-Shannon divergence 均值：

   `JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m), m = 0.5 * (p + q)`

### 归一化与平滑

三条曲线都会做：

- min-max 归一化到 `[0, 1]`
- 窗口为 `3` 的移动平均平滑

记处理后的三条曲线为：

- `R_l`：reasoning
- `O_l`：objective
- `S_l`：style

### `b1` 的 crossover 打分

当前 `b1` 策略是 `b1_only_crossover`，搜索窗为：

- `b1_search_low = 10`
- `b1_search_high = 20`
- `transition_min_run = 3`
- `min_transition_support = 0.01`
- `min_pre_reasoning_margin = -0.20`
- `b1_band_score_tolerance = 0.03`
- `b1_band_min_width = 3`
- `b1_band_max_width = 5`

对每个候选层 `b1`，定义：

- `pre_margin(b1) = mean(R_l - O_l)`，取 `b1` 前面连续 `3` 层
- `post_margin(b1) = mean(O_l - R_l)`，取 `b1` 后面连续 `3` 层
- `boundary_contrast(b1) = pre_margin(b1) + post_margin(b1)`

局部 crossover 分数定义为：

`local_score(b1) = min(pre_margin(b1), post_margin(b1)) + boundary_contrast(b1) + late_preference * (b1 - low)`

其中 `late_preference = 0.01`。

### 什么样的候选点算 valid

一个候选 `b1` 会进入 viable pool，当且仅当：

- `post_margin(b1) >= 0.01`
- `pre_margin(b1) >= -0.20`

这里特意没有要求 `pre_margin` 严格大于 0。原因是之前实验已经显示，若把 early reasoning 条件设得过硬，会把那些更晚、也更合理的 `b1` 带整体误杀掉。

### 如何从单点扩成区间

对所有 viable candidates 打分后：

- 保留所有与最佳 `local_score` 相差不超过 `0.03` 的候选点
- 在这些近最佳点上，选择一个包含最佳点的连续区间
- 区间宽度限制为 `3-5` 层
- 优先选择覆盖近最佳候选点最多的区间
- 若支持数相同，则优先靠近最佳点

最终输出：

`boundary_windows.b1 = [low, high]`

因此，这个实验并没有把 `b1` 硬限制在 `15-18`。真正固定的只是搜索窗 `10-20`；`15-18` 是指标在数据上算出来的结果，而不是事先写死的目标区间。

### 这轮分析中 `b2` 是怎么选的

虽然这份报告的重点是 `b1`，脚本仍会顺带给出一个伴随的 `b2`：

- middle score：在 `[b1, b2)` 上计算 `mean(O_l - max(R_l, S_l))`
- late score：在 `[b2, end)` 上计算 `mean(S_l - max(R_l, O_l))`
- 选择使 `middle score + late score` 最大的 `b2`

这一部分没有改动此前已经建立的 `b2` 工作流。

## 实验结果

### 各 Seed 汇总

| Seed | Valid Splits | Valid `b1` 区间 | 备注 |
| --- | --- | --- | --- |
| 7 | 2 / 3 | `15-19` | `acl` 和 `dcl` 都支持偏后的 band |
| 13 | 1 / 3 | `13-17` | 只有 `acl` valid |
| 21 | 1 / 3 | `10-14` | 明显偏早的 outlier seed |
| 29 | 2 / 3 | `16-20` | 对偏后 band 的支持最强 |
| 35 | 2 / 3 | `14-18` | 偏后 band 依然稳定 |
| 42 | 2 / 3 | `10-15` | 混合型 seed，部分仍偏早 |

### 按 Dataset 看稳定性

在 6 个 seed 上统计：

- `acl`：`6 / 6` valid
- `dcl`：`4 / 6` valid
- `ucit`：`0 / 6` valid

解释如下：

- `acl` 对 `b1` band 的支持最稳定
- `dcl` 大多数时候也支持同一段偏后的 band，但偶尔会跳早
- `ucit` 是目前最主要的不稳定来源，在当前指标下始终没能给出可接受的偏后 crossover band

### Valid Band 投票聚合

把所有 valid split 的 `b1` 区间叠加后，逐层投票结果如下：

| 层号 | 票数 |
| --- | --- |
| 10 | 2 |
| 11 | 2 |
| 12 | 3 |
| 13 | 4 |
| 14 | 5 |
| 15 | 6 |
| 16 | 8 |
| 17 | 7 |
| 18 | 6 |
| 19 | 5 |
| 20 | 2 |

这说明：

- 最高票层是 `16`
- `15-18` 上的支持最密集
- 随着 seed 增多，共识质量比之前明显更集中到中后层

## 为什么这些结果支持当前结论

这轮证据之所以足以支持 `b1` 区间结论，主要有三点。

1. 主峰已经不再贴着下界。

   更早失败的指标经常塌到 `10-12`，这非常像 lower-bound artifact。现在的主共识已经明显移动到中后段，并以 `16` 为峰值。

2. 支持偏后 band 的 seed 现在占了主导。

   `seed7`、`seed29`、`seed35` 都稳定支持 `15-20` 一带的 later band，并构成了当前最强的 valid 证据主体。

3. 结果在“区间层面”已经稳定，即便在“单点层面”还不稳定。

   精确层号仍然会随 split 和 seed 变化，但票数质量已经集中到了一个窄带。这也正是为什么 `b1` 应该被写成区间，而不是单层点估计。

## 建议写进论文的表述

当前最稳妥的说法是：

> 在不同随机 seed 下，stage-1 边界 `b1` 并不会收缩为唯一单层，而是集中在一个较窄的区间内。当前最强共识位于 `15-18` 层附近，峰值支持集中在第 `16` 层。这表明 `b1` 主要反映模型内部的阶段转移，但仍保留一定的 seed 级敏感性，尤其是在 `ucit` 上更明显。

更直白地说：

- 可以接受的 claim：`b1` 集中在 `15-18`
- 可以接受的更强摘要：峰值在 `16` 附近
- 目前还不该写：`b1` 就是唯一某一层，或者 `b1` 完全与数据无关

## 如何复现实验

多 seed 大规模启动命令：

```bash
bash scripts/run_stage1_b1_multiseed.sh
```

当前默认 seeds：

```bash
7 13 21 29 35 42
```

如果后面想扩更多 seed：

```bash
SEEDS='7 13 21 29 35 42 49 56' bash scripts/run_stage1_b1_multiseed.sh
```

## 当前决策建议

是的，当前仓库里的证据已经足够支持一个“暂定 `b1` 区间”。

推荐操作性结论是：

- 主区间：`15-18`
- 中心强调：`16-18`
- 如果一定要取单点代表值：`16`

但这个结论仍应当被视为“有支持的区间估计”，而不是“最终唯一精确点估计”。
