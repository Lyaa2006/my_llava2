# HiDESC Eval Band-Schedule 方案说明

日期：2026 年 8 月 12 日

## 1. 目标

本文档定义 HiDESC 在 `eval` 阶段如何与 Stage 1 的 `b1/b2 valid band` 结果结合，并明确要求：

1. 采用“第二档”方案：`band` 位置与 `center/core` 由 Stage 1 证据给出，`band` 内采用低自由度的“核心平台 + 肩部过渡”日程，而不是从 noisy 的逐层强度直接回归每层权重。
2. eval 视觉侧只接 `fft/spectral prototype`，不再把 `role_image_prototypes` 作为 active 路由信号。
3. band 外的层策略保持固定，只有 band 内做相邻阶段的平滑过渡。
4. 不显著增加 eval 时的在线计算量。

本文档既说明为什么这样做，也说明 coding agent 应该如何修改代码。

---

## 2. 为什么要这样做

### 2.1 不再把 `b1/b2` 当成单点边界

Stage 1 当前最稳妥的结论不是“唯一精确层”，而是“稳定窄带”：

- LLaVA `b1_window = [15, 18]`
- LLaVA `b2_window = [29, 31]`

参考：

- `docs/stage1_boundary_paper_style_summary_20260811_zh.md`
- `docs/stage1_b1_interval_report_20260811_zh.md`
- `docs/stage1_b2_style_metric_conclusion.md`

这意味着 eval 层路由也不应该继续使用“硬切换单层边界”的想法。更合适的做法是：

- `band` 外：策略固定
- `band` 内：只在相邻两种阶段策略之间连续过渡

这比单点切换更符合 Stage 1 的证据形态。

### 2.2 为什么选“第二档”而不是纯手调

三种可能方案分别是：

1. 纯手动超参数
2. 半数据驱动
3. 完全样本自适应

这里选择第 2 档，因为它同时满足三个要求：

- 比手写 `0.2/0.5/0.8/1.0` 更有证据基础
- 比完全自适应更稳定、更容易 debug
- 基本不增加在线推理开销

第 2 档的核心思想是：

- `band` 的位置来自 Stage 1
- `center/core` 的位置来自 Stage 1
- `band` 内不拟合高自由度逐层曲线，而是围绕 `center/core` 构造低自由度的固定结构
- eval 时只查表，不在线估计

这样做的理由是：Stage 1 当前真正稳定下来的，是 `window` 和 `peak/core`，而不是每一层的精细强弱比例。

### 2.3 为什么视觉侧只接 `fft prototype`

HiDESC 当前的 spectral 路由主干已经建立，视觉描述子来自 patch-grid FFT，而不是 global image embedding。

因此 eval 路由最好保持语义一致：

- task-level visual memory：`spectral_image_anchors`
- role-level visual prototype：`role_spectral_prototypes`
- 不再依赖 `role_image_prototypes`

这样可避免混用“频域视觉记忆”和“旧全局图像 prototype”两套不完全一致的视觉空间。

---

## 3. Stage 1 证据如何映射到 eval

### 3.1 当前 LLaVA 推荐分段

按 `docs/stage1_boundary_paper_style_summary_20260811_zh.md`，当前推荐：

```text
early  = 1..16
middle = 17..28
late   = 29..32
```

但在 eval 路由里，不建议直接把它实现成三块硬切。

更适合改写为：

```text
early core         = 1..14
b1 transition band = 15..18
middle core        = 19..28
b2 transition band = 29..31
late tail/core     = 32
```

解释：

- `15..18` 是 `early -> middle` 的过渡带
- `29..31` 是 `middle -> late` 的过渡带
- `band` 外不再漂移

这样做能保留 Stage 1 的区间结论，而不是把区间再压回单点。

### 3.2 `b1` 的 band 内处理

`b1` 当前稳定的是：

- `b1_window = [15, 18]`
- `b1_center = 16`

不稳定的是：

- `15..18` 内每一层到底应该给多少精确混合权重

因此不建议再从逐层投票反推一条高自由度 `alpha_b1(l)` 曲线。更稳妥的做法是直接构造：

```text
entry shoulder   -> transition core -> exit shoulder
```

对当前 `b1=[15,18], center=16`，建议定义为：

```text
entry shoulder   = 15
transition core  = 16..17
exit shoulder    = 18
```

语义是：

- 15：仍以 `early` 为主，只少量混入 `middle`
- 16..17：视为 `early/middle` 的混合核心
- 18：已偏向 `middle`，但仍保留少量 `early`

这里的重点是：`b1_center=16` 用来锚定核心开始位置，而不是要求 `16` 成为唯一切换点。

### 3.3 `b2` 的 band 内处理

`b2` 当前稳定的是：

- `b2_window = [29, 31]`
- `b2_center = 30`

这更像 late-style working core，而不是 crossover 统计边界。

因此同样不建议从逐层 `style_signal` 强行拟合一条精细曲线，而是直接构造：

```text
entry shoulder   -> transition core -> exit shoulder
```

对当前 `b2=[29,31], center=30`，建议定义为：

```text
entry shoulder   = 29
transition core  = 30
exit shoulder    = 31
```

语义是：

- 29：仍以 `middle` 为主，只少量混入 `late`
- 30：作为 `middle/late` 的混合中心
- 31：已偏向 `late`，但仍保留少量 `middle`

### 3.4 为什么采用“核心平台 + 肩部过渡”

这是本方案与“逐层强度拟合”最关键的区别。

原因有三点：

1. Stage 1 当前更稳定的是 `window + center/core`，不是逐层 slope。
2. `band` 本身就是不确定区，因此更适合低自由度处理，而不是精细回归。
3. coding 上更稳、更容易迁移到别的模型，不会把 Stage 1 的局部噪声重新带回 eval。

---

## 4. Eval 路由的目标形式

### 4.1 不混输出 logits，只混 route weights

不要对模型最终输出 logits 做 band interpolation。

正确的混合对象应是每层送入 LoRA experts 的 route weights。原因：

1. 这与 HiDESC 当前 progressive route plan 的结构一致
2. 这样可以复用现有 `early / middle / late` 路由头
3. 计算量更低
4. 更容易解释每层到底在走什么路由策略

### 4.2 每层只在相邻阶段之间过渡

设：

- `W_early`：固定 early 路由方案
- `W_middle`：固定 middle 路由方案
- `W_late`：固定 late 路由方案

则每层实际使用：

```text
layer in early core:
    W(l) = W_early

layer in b1 band:
    W(l) = (1 - alpha_b1(l)) * W_early + alpha_b1(l) * W_middle

layer in middle core:
    W(l) = W_middle

layer in b2 band:
    W(l) = (1 - alpha_b2(l)) * W_middle + alpha_b2(l) * W_late

layer in late tail:
    W(l) = W_late
```

要求：

- 只允许 `early <-> middle` 在 `b1` 里混
- 只允许 `middle <-> late` 在 `b2` 里混
- 不允许三路同时混

这能保证 band 的语义清楚，也最接近 Stage 1 的解释框架。

### 4.3 `alpha(l)` 不从 noisy 曲线拟合，而从 band 结构直接生成

这里的 `alpha_b1(l)` / `alpha_b2(l)` 不应来自逐层 vote 或逐层 signal 的直接拟合。

推荐做法是：由 `window + center/core` 直接生成一个低自由度 schedule。

例如对一个 band：

```text
band = [L, R]
core = [C_low, C_high]
```

则：

- `L .. C_low-1`：entry shoulder
- `C_low .. C_high`：transition core
- `C_high+1 .. R`：exit shoulder

第一版建议使用离散查表，而不是连续函数拟合。

一个简单且稳定的规则是：

- entry shoulder：`alpha = 0.25`
- transition core：`alpha = 0.50`
- exit shoulder：`alpha = 0.75`

注意这三个值不是最终不可变常数，而是“固定结构模板”的默认实现。它们的来源不是逐层实验噪声，而是过渡语义：

- 0.25：前一阶段主导
- 0.50：两阶段平衡
- 0.75：后一阶段主导

如果某个 band 较宽，可以允许更细的模板；但第一版不应超过这类低自由度结构。

### 4.4 为什么这不会明显增加计算量

低开销版本必须满足：

1. `W_early / W_middle / W_late` 各算一次
2. 每层只做一次查表和一次线性混合
3. 不能为每层分别重算三套完整 scorer

因此 coding agent 必须把实现做成：

```text
compute basis route plans once
-> lookup precomputed alpha(l)
-> form per-layer route weights by interpolation
```

而不能做成：

```text
for each layer:
    recompute early scorer
    recompute middle scorer
    recompute late scorer
```

后者会显著放大路由开销，不符合本方案目标。

---

## 5. 视觉侧如何接上 FFT prototype

### 5.1 active visual memory 只保留两类

eval 视觉路由只使用：

1. `spectral_image_anchors`
   - 用于 task-level score
2. `role_spectral_prototypes`
   - 用于 role-level prototype score

`role_image_prototypes` 可以保留在 checkpoint 中做兼容，但不再参与 active score。

### 5.2 当前代码里已经接近目标，但 checkpoint 可能不完整

在 `llava_arch.py` 当前实现中：

- task 评分 `_compute_shared_task_scores()` 已经走 `spectral_image_anchors`
- role 评分 `_score_roles()` 已经支持 `role_spectral_prototypes`
- `role_image_prototypes` 并没有出现在 active role score 中

这部分是好消息：主干设计已经比较接近我们要的状态。

真正的问题在于 checkpoint 完整性：

- 一些转换/补写流程没有把 `role_spectral_prototypes` 写回最终 eval checkpoint
- 这会导致 `use_spectral_role_prototype=True` 时虽然代码支持，但权重并不完整

因此需要先补一条 offline repair 流程。

### 5.3 必须新增 spectral role prototype repair

建议新增一个离线脚本，例如：

```text
scripts/MCITlib/rebuild_hidesc_spectral_role_memory.py
```

输入：

- `spectral_image_anchors`
- `task_role_membership`
- `role_task_count`
- checkpoint root

输出：

- 回写 `role_spectral_prototypes.{i}`

重建公式建议为：

```text
role_spectral_prototype[r]
  = normalize(
      sum_t membership[t, r] * normalize(spectral_image_anchor[t])
    )
```

如果 `role_task_count[r] == 0`，则保持零向量。

注意：

- 这是离线 checkpoint repair，不应在每个 eval batch 里动态重建
- 第一版不需要碰 `role_image_prototypes`

---

## 6. 建议的配置产物

### 6.1 新增 eval band schedule JSON

建议新增一个显式 schedule 文件，例如：

```text
configs/routing_configs/HiDESC/llava_stage1_band_eval_schedule.json
```

第一版内容建议包含：

```json
{
  "model_family": "llava",
  "version": "stage1_band_eval_v1",
  "early_core": [1, 14],
  "b1_band": [15, 18],
  "middle_core": [19, 28],
  "b2_band": [29, 31],
  "late_core": [32, 32],
  "b1_center": 16,
  "b2_center": 30,
  "b1_core": [16, 17],
  "b2_core": [30, 30],
  "alpha_by_layer": {
    "15": 0.25,
    "16": 0.50,
    "17": 0.50,
    "18": 0.75,
    "29": 0.25,
    "30": 0.50,
    "31": 0.75
  },
  "alpha_source": {
    "b1": "window plus center-derived shoulder/core template",
    "b2": "window plus center-derived shoulder/core template"
  }
}
```

这里的 `alpha_by_layer` 仍应由离线脚本生成，不要在模型代码里硬编码。  
但生成依据应是 `window + center/core template`，而不是逐层 noisy signal。

### 6.2 relation config 中新增开关

建议在 `relation_routing_config` 中新增：

```text
use_stage1_band_schedule_eval
stage1_band_schedule_path
eval_use_role_spectral_prototype
eval_disable_role_image_prototype
```

推荐默认：

```text
use_stage1_band_schedule_eval = true
eval_use_role_spectral_prototype = true
eval_disable_role_image_prototype = true
```

注意：

- `use_spectral_role_prototype` 也应被显式置为 `true`
- 不要依赖当前默认值，因为当前默认值仍可能是 `false`

---

## 7. 代码实现方案

本节按 coding agent 的实现顺序来写。

### 7.1 第一步：修复 eval 配置注入链路

#### 问题

当前标准 UCIT eval 入口大致是：

```text
eval_*.sh
-> llava.eval.CoIN.model_others
-> llava.model.builder.load_pretrained_model
```

而 checkpoint 自带 `config.json` 中并不稳定包含完整的 `relation_routing_config`。

所以 band schedule 和 `use_spectral_role_prototype` 不能指望自动从 checkpoint 恢复。

#### 要做什么

把外部 eval config 里的 routing schedule 明确传进模型加载链。

#### 建议修改点

1. `LLaVA/HiDESC/scripts/MCITlib/Eval_UCIT/eval_*.sh`
   - 从 eval config 读取：
     - `routing_config_path`
     - `stage1_band_schedule_path`
   - 传给 Python 入口

2. `LLaVA/HiDESC/llava/eval/CoIN/model_others.py`
   - 新增命令行参数：
     - `--routing-config-path`
     - `--stage1-band-schedule-path`

3. `LLaVA/HiDESC/llava/model/builder.py`
   - `load_pretrained_model()` 增加可选参数：
     - `routing_config_path=None`
     - `stage1_band_schedule_path=None`
   - 加载 base model + non_lora weights 之后，显式调用：
     - `model.configure_relation_routing(...)`
     - `model.set_stage1_band_schedule(...)`

#### 原则

eval 所需的 routing policy 不应隐式藏在 checkpoint 里，应该允许由外部 config 显式控制。

### 7.2 第二步：新增 band schedule 读取和查表

#### 建议新增函数

在 `LLaVA/HiDESC/llava/model/llava_arch.py` 中新增：

```text
load_stage1_band_schedule(...)
set_stage1_band_schedule(...)
_get_stage1_band_schedule()
_get_stage1_band_alpha(layer_idx)
_get_stage1_band_region(layer_idx)
_get_stage1_band_core(layer_idx)
```

要求：

- schedule 读入后缓存到 model 上
- `layer_idx` 与 schedule 的层号体系保持一致
- 明确统一使用 1-based 还是 0-based 层号

建议做法：

- schedule 文件用 1-based，便于和 Stage 1 文档一致
- 代码内部在查询时把 `layer_idx` 转成 `layer_no = layer_idx + 1`

### 7.3 第三步：保留现有三套 basis 路由头，只改组合方式

当前已经有：

- `_build_early_route_weights()`
- `_build_middle_route_weights()`
- late 权重逻辑，在 `_build_progressive_route_plan()` 中构造

这些 basis 路由头不要推翻重写。

#### 要改什么

把 `_build_progressive_route_plan()` 从“返回三套阶段权重 + 由固定 layer counts 决定每层 stage”改成：

```text
1. 先算 basis:
   early_basis
   middle_basis
   late_basis

2. 再按 schedule 生成每层的最终 route weights:
   route_plan["per_layer"][layer_idx] = ...
```

建议输出结构：

```python
{
    "early_basis": ...,
    "middle_basis": ...,
    "late_basis": ...,
    "per_layer": [tensor_for_layer0, tensor_for_layer1, ...],
    "candidate_experts": [...]
}
```

### 7.4 第四步：修改 `_apply_relation_weights_to_experts()`

当前 `_apply_relation_weights_to_experts()` 是通过 `_get_layer_stage()` 决定每层属于 `early/middle/late`。

这对 band-aware eval 不够。

应改成：

1. 如果 `route_plan` 里有 `per_layer`
   - 直接取 `route_plan["per_layer"][layer_idx]`
2. 否则
   - fallback 到旧的 `early/middle/late` 逻辑

这样可以做到：

- 新 band schedule 开启时走新逻辑
- 旧 checkpoint / baseline 还能继续跑

### 7.5 第五步：确保 role score 只接 spectral prototype

`_score_roles()` 当前逻辑已经基本正确：

- text prototype score 来自 `role_text_prototypes`
- spectral prototype score 来自 `role_spectral_prototypes`

coding agent 需要做的不是再引入 `role_image_prototypes`，而是：

1. 明确保证 `use_spectral_role_prototype=True`
2. 明确不要再把 `role_image_prototypes` 接入 active role score
3. 若 checkpoint 缺少 `role_spectral_prototypes`，提前报错或触发 repair 提示

建议新增一个 sanity check：

```text
_spectral_role_bank_available(active_roles)
```

若打开了 `use_spectral_role_prototype` 但 bank 不可用，应抛出清晰错误，而不是静默退化。

### 7.6 第六步：新增 offline repair script

建议新增：

```text
LLaVA/HiDESC/scripts/MCITlib/rebuild_hidesc_spectral_role_memory.py
```

脚本职责：

1. 读取 `non_lora_trainables.bin`
2. 提取：
   - `spectral_image_anchors.*`
   - `task_role_membership`
   - `role_task_count`
3. 重建：
   - `role_spectral_prototypes.*`
4. 回写原 checkpoint 或写入新目录

建议支持：

- `--inplace`
- `--output-root`
- `--fail-if-present`

这个脚本是整个方案的前置保障之一。

---

## 8. 离线 schedule 生成方案

### 8.1 建议新增 schedule builder

建议新增一个非常小的脚本，例如：

```text
scripts/build_hidesc_eval_band_schedule.py
```

作用：

1. 从 Stage 1 摘要 JSON 读取：
   - `b1_window`
   - `b1_center`
   - `b2_window`
   - `b2_center`
2. 生成 `core + shoulder` 模板
3. 写出 `alpha_by_layer`
4. 写出 schedule 文件

### 8.2 `b1` 的生成规则

输入：

- `b1_band = [15, 18]`
- `b1_center = 16`

规则：

```text
if band width == 4 and center == left-middle:
    entry shoulder  = first layer
    core            = middle two layers
    exit shoulder   = last layer
```

并映射成：

```text
entry shoulder -> alpha = 0.25
core           -> alpha = 0.50
exit shoulder  -> alpha = 0.75
```

### 8.3 `b2` 的生成规则

输入：

- `b2_band = [29, 31]`
- `b2_center = 30`

规则：

```text
entry shoulder  = 29
core            = 30
exit shoulder   = 31
```

映射成：

```text
29 -> 0.25
30 -> 0.50
31 -> 0.75
```

### 8.4 为什么不是 0 / 1 硬切

因为 band 的意义就是“不确定但受限的过渡区”。

如果在 band 内继续使用 `0` 或 `1`，那就等于把 band 又退化回硬边界。

### 8.5 为什么要离线生成

因为这样可以保证：

- eval 无额外统计开销
- 同一模型同一版本 schedule 可复现
- coding agent 不需要把 Stage 1 分析逻辑塞进 eval 前向

---

## 9. 测试要求

coding agent 至少需要补三类测试。

### 9.1 schedule 级单元测试

建议新增测试覆盖：

1. `alpha_by_layer` 在 band 内单调不减
2. `core` 层的 alpha 等于预期混合强度
3. shoulder 层的 alpha 小于 exit shoulder
4. band 外不会返回插值 alpha

### 9.2 route-plan 级测试

建议在 `LLaVA/HiDESC/tests/` 增加：

1. `b1` 外层使用固定 basis
2. `b1` 的 core 层确实使用 `early/middle` 平衡混合
3. `b2` 的 core 层确实使用 `middle/late` 平衡混合
4. shoulder 层确实偏向对应一侧
5. `per_layer` 权重归一化后总和为 1

### 9.3 checkpoint repair 测试

对一个缺少 `role_spectral_prototypes` 的 mock state dict：

1. repair 前应检测到缺失
2. repair 后应成功生成 `role_spectral_prototypes.*`
3. repair 后 `_score_roles()` 可正常启用 spectral role prototype 分支

---

## 10. 不要这样实现

coding agent 应避免以下错误实现。

### 10.1 不要在每层重算三套 scorer

禁止：

```text
for each layer:
    recompute early scorer
    recompute middle scorer
    recompute late scorer
```

这会把路由头开销放大到不必要的程度。

### 10.2 不要把 `b1/b2` 再写回单点边界

禁止：

```text
if layer <= 16: early
elif layer <= 30: middle
else: late
```

这等于抹掉了 valid band 的意义。

### 10.3 不要重新启用 `role_image_prototypes`

本方案的重点就是统一到 FFT/spectral 空间。

因此禁止在新的 eval active path 中再引入：

```text
role_image_prototypes
```

它可以继续保存，但不该成为新的主视觉 prototype。

### 10.4 不要把 schedule 隐式硬编码在 Python 常量里

允许短期 demo 用常量，但正式实现必须有：

- 外部 JSON schedule
- 可重复生成来源
- 配置可见性

否则后续换模型或换 Stage 1 结论时会非常难维护。

---

## 11. 推荐的实施顺序

建议 coding agent 按下面顺序实现。

### 阶段 A：先打通不改行为的基础设施

1. 新增 schedule JSON
2. eval shell -> Python -> builder 的参数透传
3. model 上新增 schedule 读取和缓存

完成后应保证：

- 即使 schedule 还未启用，现有 eval 也不坏

### 阶段 B：补齐 spectral role prototype

1. 新增 repair script
2. 对目标 checkpoint 执行 repair
3. 增加 spectral role bank sanity check

完成后应保证：

- eval 确实拥有可用的 `role_spectral_prototypes`

### 阶段 C：切换到 band-aware eval

1. 改 `_build_progressive_route_plan()`
2. 改 `_apply_relation_weights_to_experts()`
3. 打开：
   - `use_stage1_band_schedule_eval`
   - `use_spectral_role_prototype`

完成后应保证：

- band 外固定
- band 内渐变
- 不显著增加计算量

### 阶段 D：补测试和 smoke eval

1. 单元测试
2. 一个 checkpoint 的 smoke eval
3. 对比 band-aware 开关前后的运行稳定性

---

## 12. 最终想要达到的状态

当本方案落地后，HiDESC eval 应满足：

1. 使用 Stage 1 证据驱动的 per-layer 路由日程
2. `b1/b2` 以区间形式进入 eval，而不是被压成单点
3. 视觉 prototype 统一到 FFT/spectral 空间
4. band-aware 插值几乎不增加在线成本
5. 路由策略通过外部配置可复现、可替换、可审计

简化成一句话就是：

> HiDESC eval 不再依赖“硬切层号 + noisy 逐层拟合 + 旧 image prototype”，而是使用 Stage 1 的 `window + center/core` 证据驱动的、低开销的、FFT prototype 优先的层内渐变路由。

---

## 13. 给 coding agent 的一句执行摘要

如果只给 coding agent 一句话任务描述，可以写成：

> 给 HiDESC 的 eval 链路加入一个外部可配置的 Stage 1 band schedule：使用 `b1/b2` 的 `window + center/core` 生成低自由度的 shoulder/core 模板，在 `llava_arch.py` 中只对 route weights 做相邻阶段插值，band 外保持固定策略；视觉 role prototype 只使用 `role_spectral_prototypes`，并先补一个 offline repair 脚本保证 checkpoint 中这组权重完整可用。
