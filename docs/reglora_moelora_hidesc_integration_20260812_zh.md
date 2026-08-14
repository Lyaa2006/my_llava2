# RegLoRA 接入 HiDESC 训练机制与 MoELoRA 接入 HiDESC Eval 机制实施文档

日期：2026 年 8 月 12 日

## 1. 文档目标

本文档回答两个具体工程问题：

1. 如何把 HiDESC 当前已经实现的 `train` 侧方法迁移到 `RegLoRA`，形成一个“强训练正则 baseline + 我们的训练机制”的对照版本。
2. 如何把 HiDESC 当前已经实现的 `eval` 侧分层激活机制迁移到 `MoELoRA`，形成一个“MoE-LoRA 结构 + 我们的分层 eval”对照版本。

这里的核心原则不是“把 HiDESC 全量复制过去”，而是：

- `RegLoRA` 侧尽量只加训练机制，不强行引入 HiDESC 的整套推理路由。
- `MoELoRA` 侧尽量只加 eval 分层激活，不强行引入 HiDESC 的 description continual-learning loss。

这样做的目的，是让训练创新和推理创新的证据链彼此独立，避免实验归因混杂。

## 2. 先说结论

建议把工程拆成两条线。

### 2.1 RegLoRA 线

目标版本定义为：

`RegLoRA + HiDESC-train-loss`

它保留：

- RegLoRA 原有 LoRA 正则化训练方式
- RegLoRA 原有模型结构
- RegLoRA 原有 eval 方式

它新增：

- base-anchored description cache
- B1 band align loss
- B2 band struct loss
- focus / energy 两个 description regularizer
- band 内自适应权重分配

### 2.2 MoELoRA 线

目标版本定义为：

`MoELoRA + HiDESC-eval-routing`

它保留：

- MoE-LoRA 本身的训练方式
- MoE-LoRA 本身的专家参数结构
- MoE-LoRA 本身的 checkpoint

它新增：

- early / middle / late 三阶段 route basis
- Stage 1 的 `band schedule`
- `b1` 和 `b2` 区间内的相邻阶段插值
- 可选的 spectral / role prototype 评分信号

在当前仓库中，你说的 `MoELoRA` 最接近的是 [LLaVA/MR-LoRA](/mnt/lyaa/MCITlib/LLaVA/MR-LoRA) 这一套实现；其底层 MoE-LoRA 线性层在 [clitmoelora.py](/mnt/lyaa/MCITlib/LLaVA/MR-LoRA/CoIN/peft/tuners/clitmoelora.py)。

## 3. 当前仓库中可直接复用的 HiDESC 资产

### 3.1 训练侧可复用资产

HiDESC 的训练侧已经基本成型，核心在：

- 训练参数定义：[LLaVA/HiDESC/llava/train/train_MOE.py](/mnt/lyaa/MCITlib/LLaVA/HiDESC/llava/train/train_MOE.py)
- trainer loss 实现：[LLaVA/HiDESC/llava/train/llava_trainer.py](/mnt/lyaa/MCITlib/LLaVA/HiDESC/llava/train/llava_trainer.py)
- description token 选择：[LLaVA/HiDESC/llava/train/description_utils.py](/mnt/lyaa/MCITlib/LLaVA/HiDESC/llava/train/description_utils.py)

已经实现的关键能力包括：

- `enable_description_cl`
- `extract_description_cache_only`
- `description_cache_dir`
- `description_focus_weight`
- `description_energy_weight`
- `align_loss_weight`
- `b1_low_layer / b1_high_layer`
- `b2_low_layer / b2_high_layer`
- `align_band_eta / struct_band_eta`

trainer 中已经支持：

- 标准 CE 与额外 loss 的联合优化
- B1 区间的 align loss 聚合
- B2 区间的 struct loss 聚合
- band 内 `EMA + detach + 位置先验` 的自适应加权

### 3.2 Eval 侧可复用资产

HiDESC 的 eval 分层激活主干在：

- 分层路由主体：[LLaVA/HiDESC/llava/model/llava_arch.py](/mnt/lyaa/MCITlib/LLaVA/HiDESC/llava/model/llava_arch.py)
- 路由配置默认值：[LLaVA/HiDESC/llava/model/language_model/llava_llama.py](/mnt/lyaa/MCITlib/LLaVA/HiDESC/llava/model/language_model/llava_llama.py)
- Stage 1 日程表：[configs/routing_configs/HiDESC/llava_stage1_band_eval_schedule.json](/mnt/lyaa/MCITlib/configs/routing_configs/HiDESC/llava_stage1_band_eval_schedule.json)

已经实现的关键能力包括：

- `early / middle / late` 三类 basis route weights
- `b1 / b2` band 内的线性插值
- `use_stage1_band_schedule_eval`
- `stage1_band_schedule_path`
- `eval_use_role_spectral_prototype`
- `eval_disable_role_image_prototype`

## 4. Part A: RegLoRA 如何接入我们的训练手段

## 4.1 目标定义

这里我们不是要把 RegLoRA 改成 HiDESC，而是要做：

`RegLoRA + HiDESC loss`

因此迁移边界必须清楚：

- 保留 RegLoRA 自身 regularization loss
- 保留 RegLoRA 的 backbone / LoRA / forward 主路径
- 只把 HiDESC 的 description supervision 机制挂到 trainer 层

这意味着最合适的迁移策略是：

`迁移 HiDESC 的数据管线 + trainer loss，不迁移 HiDESC 的 eval routing`

## 4.2 RegLoRA 当前的真实 loss 注入位置

RegLoRA 当前不是在 `Trainer.compute_loss()` 里显式加正则，而是在模型 forward 里做的。

关键位置：

- [LLaVA/RegLoRA/llava/model/language_model/llava_llama.py](/mnt/lyaa/MCITlib/LLaVA/RegLoRA/llava/model/language_model/llava_llama.py)

现状是：

1. `self.model(...)` 返回 `loss_reg, outputs`
2. 然后在 CausalLM forward 里做：

```text
loss = 2.5e3 * loss_reg + CE
```

这意味着如果你直接把 HiDESC 的 trainer 拿过来，而不处理 RegLoRA 的 `loss_reg`，很容易出现重复加权或权重失衡。

因此正确做法是：

- 继续让 RegLoRA 模型内部负责 `loss_reg`
- 让 trainer 只把 `model(...).loss` 视为 “RegLoRA 的标准主损失”
- 再在这个基础上追加 HiDESC 的 `align/struct` 损失

换句话说，RegLoRA 接入后，总损失应写成：

```text
L_total
= w_reglora_main * L_reglora_main
+ L_struct^B2
+ w_align * L_align^B1
```

其中：

- `L_reglora_main = CE + lambda_reg * RegLoRA_internal_regularization`
- `L_struct^B2` 和 `L_align^B1` 来自 HiDESC trainer

最安全的第一版是：

```text
w_reglora_main = 1.0
```

不要一开始就动 RegLoRA 内部 `2.5e3` 的缩放，先在外面把 HiDESC 新增项调平。

## 4.3 推荐迁移路线

建议分三步。

### Step 1: 把 HiDESC 的 description 数据管线迁到 RegLoRA

需要迁移的内容：

- `DataArguments` 中与 description 相关的字段
- `TrainingArguments` 中与 description CL 和 band loss 相关的字段
- dataset 中构造 `description_input_ids`
- collator 中打包：
  - `description_input_ids`
  - `description_attention_mask`
  - `description_key_mask`
  - `reference_description_states`
  - `reference_description_mask`
  - `reference_description_available`

建议迁移来源：

- [LLaVA/HiDESC/llava/train/train_MOE.py](/mnt/lyaa/MCITlib/LLaVA/HiDESC/llava/train/train_MOE.py)

建议迁移目标：

- [LLaVA/RegLoRA/llava/train/train.py](/mnt/lyaa/MCITlib/LLaVA/RegLoRA/llava/train/train.py)

最小迁移单元：

1. `DataArguments`
2. `TrainingArguments`
3. `select_expanded_description_tokens()` 所依赖的辅助函数
4. `LazySupervisedDataset.__getitem__`
5. `DataCollatorForSupervisedDataset`

强烈建议把 [description_utils.py](/mnt/lyaa/MCITlib/LLaVA/HiDESC/llava/train/description_utils.py) 原样复制到：

- `LLaVA/RegLoRA/llava/train/description_utils.py`

这样 trainer 层逻辑就能低成本复用。

### Step 2: 把 HiDESC 的 trainer loss 迁到 RegLoRA trainer

当前 RegLoRA trainer 在：

- [LLaVA/RegLoRA/llava/train/llava_trainer.py](/mnt/lyaa/MCITlib/LLaVA/RegLoRA/llava/train/llava_trainer.py)

它现在基本只有：

- sampler
- optimizer
- checkpoint save

没有自定义 `compute_loss()`。

因此推荐的最稳妥方案是：

- 以 HiDESC 的 trainer 为模板
- 在 RegLoRA 的 `LLaVATrainer` 中新增一套 description-CL compute_loss
- 但把 `standard_loss` 的来源改成 RegLoRA model forward 的 `outputs.loss`

建议直接迁移或改写以下方法：

- `_temporary_anchor_update`
- `_unwrap_model`
- `_get_transformer_layers`
- `_resolve_loss_band`
- `_prepare_answer_multimodal_inputs`
- `_prepare_description_multimodal_inputs`
- `_compute_align_loss_for_hidden`
- `_compute_band_weights`
- `_compute_align_band_loss`
- `_extract_description_states`
- `_compute_focus_loss`
- `_compute_energy_loss`
- `_compute_struct_band_loss`
- `compute_loss`

来源文件：

- [LLaVA/HiDESC/llava/train/llava_trainer.py](/mnt/lyaa/MCITlib/LLaVA/HiDESC/llava/train/llava_trainer.py)

目标文件：

- [LLaVA/RegLoRA/llava/train/llava_trainer.py](/mnt/lyaa/MCITlib/LLaVA/RegLoRA/llava/train/llava_trainer.py)

### Step 3: 增加 description cache 抽取模式

如果没有 cache，HiDESC 的训练机制就不完整。

RegLoRA 需要新增：

- `extract_description_cache_only`
- `description_cache_model_source`
- `description_cache_max_new_entries`

最简单做法不是自己重写一遍，而是直接把 HiDESC 中 cache 抽取逻辑迁进去。

推荐迁移来源：

- [LLaVA/HiDESC/llava/train/train_MOE.py](/mnt/lyaa/MCITlib/LLaVA/HiDESC/llava/train/train_MOE.py)

重点函数包括：

- `count_cached_description_entries`
- `get_description_cache_path`
- `pad_description_sequences`
- `build_description_key_mask`
- `extract_description_cache`

## 4.4 RegLoRA 具体改哪些文件

推荐改动文件列表如下。

### 必改

- `LLaVA/RegLoRA/llava/train/train.py`
- `LLaVA/RegLoRA/llava/train/llava_trainer.py`

### 新增

- `LLaVA/RegLoRA/llava/train/description_utils.py`

### 可选

- `configs/train_configs/RegLoRA/...`
- `LLaVA/RegLoRA/scripts/MCITlib/Train/Taskn.sh`

## 4.5 RegLoRA 参数层面建议新增的开关

建议在 RegLoRA 的 `TrainingArguments` 中新增以下字段：

```text
enable_description_cl: bool = False
extract_description_cache_only: bool = False
description_cache_model_source: str = "base"
description_cache_max_new_entries: int = -1
description_max_tokens: int = 32
description_focus_weight: float = 0.2
description_focus_alpha: float = 0.5
description_energy_weight: float = 1e-4
description_energy_margin: float = 30.0
b1_low_layer: int = 15
b1_high_layer: int = 18
b2_low_layer: int = 29
b2_high_layer: int = 31
align_band_eta: float = 0.5
struct_band_eta: float = 0.35
struct_band_energy_rho: float = 1.0
loss_band_ema_gamma: float = 0.9
loss_band_position_eps: float = 0.05
align_loss_weight: float = 0.01
standard_ce_weight: float = 1.0
```

这里的 `standard_ce_weight` 在 RegLoRA 里建议理解为：

- “对 `model(...).loss` 的整体缩放”
- 而不是只对裸 CE 缩放

因为 RegLoRA 的 `model(...).loss` 已经内含其正则项。

## 4.6 RegLoRA 版总损失建议

第一版推荐：

```text
L_total
= L_reglora_model
+ L_struct^B2
+ w_align * L_align^B1
```

其中：

- `L_reglora_model = outputs.loss`
- `L_struct^B2` 内部已经带 `focus_weight / energy_weight`
- `w_align = align_loss_weight`

也就是在 `compute_loss()` 里最好写成：

```text
standard_loss = standard_outputs.loss
total_loss = standard_loss + struct_loss + align_weight * boundary_align_loss
```

不要再把 RegLoRA 的 `standard_loss` 拆开重组。

## 4.7 RegLoRA 最小可行版本

如果你想先做一个最小版本验证思路，推荐顺序是：

1. 只加 `description cache`
2. 只加 `align loss`
3. 再加 `focus/energy`
4. 最后再开 band 自适应加权

也就是：

- `RegLoRA + align-only`
- `RegLoRA + struct-only`
- `RegLoRA + full`

这样一旦效果不稳，你能快速判断是：

- cache 有问题
- description token 选择有问题
- 还是 RegLoRA 内部正则与新 loss 冲突

## 4.8 RegLoRA 需要特别注意的风险

### 风险 1：loss 尺度冲突

RegLoRA 内部已经有一个较强正则缩放项，如果外部再直接加 `struct/align`，可能出现：

- 新 loss 太弱，基本不起作用
- 新 loss 太强，覆盖 RegLoRA 原本的训练目标

建议做法：

- 第一轮先固定 RegLoRA 原始缩放不动
- 只调：
  - `description_focus_weight`
  - `description_energy_weight`
  - `align_loss_weight`

### 风险 2：hidden state 接口不一致

HiDESC 的 trainer 依赖 `output_hidden_states=True`。

因此必须确认 RegLoRA 的 multimodal forward 链路在训练时返回：

- `outputs.hidden_states`

当前从 [llava_llama.py](/mnt/lyaa/MCITlib/LLaVA/RegLoRA/llava/model/language_model/llava_llama.py) 看，这一条件是满足的。

### 风险 3：description path 多次 forward 带来的显存上升

HiDESC trainer 会在一个 step 内至少做：

- answer forward
- description forward

必要时还要：

- 关闭 gradient checkpointing

HiDESC 原实现已经在 train 入口中这样处理，RegLoRA 迁移时建议保持一致。

## 4.9 RegLoRA 验收标准

建议最少检查以下日志指标：

- `loss/ce`
- `loss/align`
- `loss/struct`
- `loss/description_focus`
- `loss/description_energy`
- `align/layer_*_weight`
- `struct/layer_*_weight`
- `description/key_mass`
- `description/focus_ratio`

如果你看到：

- `align/layer_15~18_weight` 在波动
- `struct/layer_29~31_weight` 在波动
- `loss/align` 和 `loss/struct` 非零且可下降

说明训练侧融合至少在数值上已经打通。

## 5. Part B: MoELoRA 如何融合我们的 eval 手段

## 5.1 先澄清目标

这里的目标不是把 MoELoRA 训练过程改成 HiDESC，而是：

`MoELoRA checkpoint + HiDESC eval routing`

因此只动推理时的激活与路由，不动：

- MoELoRA 已训练好的专家权重
- MoELoRA 的训练损失
- MoELoRA 的数据

## 5.2 当前 MR-LoRA / MoE-LoRA 的真实结构特点

当前仓库里的 MR-LoRA 不是 HiDE 式的“跨任务 expert list + 每层统一 route plan”，而更像是：

- 每个 LoRA 线性层内部有多个 expert
- 每层内部自己做 token-level softmax gating

核心实现：

- [LLaVA/MR-LoRA/CoIN/peft/tuners/clitmoelora.py](/mnt/lyaa/MCITlib/LLaVA/MR-LoRA/CoIN/peft/tuners/clitmoelora.py)

关键点在：

- `self.lora_router`
- `router = softmax(linear(x))`
- 每个 token 在每层 LoRA 内部选择专家混合

这和 HiDESC eval 的“按层指定 task expert 权重”不是同一个接口层级。

所以 MoELoRA 融合 HiDESC eval 有两条路线。

## 5.3 路线 A：最小侵入版

### 核心思想

不去改 MoE-LoRA 层内部 `router(x)` 的公式，而是在其外层增加一个“阶段先验门控”。

即：

```text
final_router(layer, token)
= stage_prior(layer) * native_moe_router(token)
```

更具体地说：

1. 先基于样本得到三套 basis prior：
   - `W_early`
   - `W_middle`
   - `W_late`
2. 再按 `b1/b2` 和 layer index 生成：
   - `W(layer)`
3. 最后把 `W(layer)` 作为每层 router logits 的 bias 或 multiplicative prior

也就是把原本：

```text
router_probs = softmax(router_logits)
```

改成：

```text
router_probs = softmax(router_logits + log W(layer))
```

这是最推荐的第一版，因为：

- 不需要重写 MoE-LoRA 的前向主逻辑
- 不需要强行把 HiDESC 的 task-route 结构塞进 token-route 结构
- 数学上也更自然，属于“先验偏置”

### 需要改的核心文件

- [LLaVA/MR-LoRA/CoIN/peft/tuners/clitmoelora.py](/mnt/lyaa/MCITlib/LLaVA/MR-LoRA/CoIN/peft/tuners/clitmoelora.py)

建议新增能力：

1. 每个 MoE-LoRA 线性层支持一个外部注入的 `router_prior`
2. forward 时做：

```text
router_logits = native_router_logits + prior_logits
```

其中 `prior_logits` 由上层模型在进入当前 decoder layer 前设置。

### 还需要模型上层做什么

你仍需要一个和 HiDESC 类似的“阶段日程生成器”，也就是：

- 根据当前图像 / 文本 guide 信号，得到 `early/middle/late` 三套 basis prior
- 根据 [llava_stage1_band_eval_schedule.json](/mnt/lyaa/MCITlib/configs/routing_configs/HiDESC/llava_stage1_band_eval_schedule.json) 生成每层 prior

这部分逻辑建议新加在：

- `LLaVA/MR-LoRA/llava/model/llava_arch.py`

可以直接仿照 HiDESC 的：

- `_get_stage1_band_region`
- `_get_stage1_band_alpha`
- `_build_progressive_route_plan`

但要注意，这里输出的不是“task expert weights”，而是“MoE expert prior”。

## 5.4 路线 B：一致性最强版

### 核心思想

直接把 HiDESC 的 per-layer route plan 做成 MoELoRA 的主路由控制器。

也就是：

- HiDESC 提供每层该偏向哪个阶段
- 每个阶段再决定当前层应该激活哪些 MoE expert

这要求你给 MoELoRA 明确建立三套 phase basis：

- `early_basis`
- `middle_basis`
- `late_basis`

但问题是 MR-LoRA 当前的 MoE expert 并没有天然的“task语义身份”。

所以这条路线需要额外做一件事：

- 给每个 MoE expert 建立 role / task / semantic prototype

否则你无法回答：

- “第 2 个 MoE expert 为何属于 early 相关专家？”

因此这条路线工程量大得多，适合后续版本，不适合作为第一版消融。

## 5.5 对你当前论文实验，推荐用哪条路线

推荐：

`路线 A：最小侵入版`

因为你的论文目的不是证明“MoELoRA 彻底变成 HiDESC”，而是证明：

- HiDESC 的分层 eval 先验，对另一类 expert/gating 结构也有帮助

只要能做到：

- 同一 MoELoRA checkpoint
- native eval vs band-schedule eval

就足以构成有说服力的附加证据。

## 5.6 MoELoRA 版 eval 路由建议如何落地

### 方案定义

对每层 `l`，构造一个 prior：

```text
P_l
=
early_prior, if layer in early core
(1-alpha_b1(l)) * early_prior + alpha_b1(l) * middle_prior, if layer in b1
middle_prior, if layer in middle core
(1-alpha_b2(l)) * middle_prior + alpha_b2(l) * late_prior, if layer in b2
late_prior, if layer in late core
```

然后在 MoE-LoRA 层内部做：

```text
router_probs = softmax(native_router_logits + tau * log(P_l))
```

其中：

- `tau` 是 prior 强度系数
- 建议初值 `0.5 ~ 1.0`

这样你就把 HiDESC 的分层激活思想转换成了：

- “对 native MoE gating 的层依赖先验偏置”

### 你需要新增的配置项

建议在 MR-LoRA 模型配置中新增：

```text
use_stage1_band_schedule_eval
stage1_band_schedule_path
eval_stage_prior_strength
eval_stage_prior_mode
```

建议默认：

```text
eval_stage_prior_strength = 0.75
eval_stage_prior_mode = "logit_bias"
```

## 5.7 MoELoRA 具体改哪些文件

### 必改

- `LLaVA/MR-LoRA/CoIN/peft/tuners/clitmoelora.py`
- `LLaVA/MR-LoRA/llava/model/llava_arch.py`

### 可能需要改

- `LLaVA/MR-LoRA/llava/model/language_model/llava_llama.py`
- `LLaVA/MR-LoRA/llava/model/builder.py`

### 配置建议新增

- `configs/train_configs/MR-LoRA/.../eval*.json`
- 可新建：
  - `configs/routing_configs/MR-LoRA/llava_stage1_band_eval_schedule.json`

但更推荐直接复用：

- [configs/routing_configs/HiDESC/llava_stage1_band_eval_schedule.json](/mnt/lyaa/MCITlib/configs/routing_configs/HiDESC/llava_stage1_band_eval_schedule.json)

## 5.8 MoELoRA 的 basis prior 从哪里来

这是实现中的关键点。

最现实的第一版有两个来源。

### 来源 A：直接从 native router 统计中构造

做法：

1. 在验证集上离线跑一遍 native MoELoRA
2. 统计每层平均 router distribution
3. 再把层分到：
   - early core
   - middle core
   - late core
4. 分别求平均，得到：
   - `early_prior`
   - `middle_prior`
   - `late_prior`

这版的优点是：

- 完全不需要新增 prototype
- 不需要改训练

缺点是：

- 不够“语义化”
- 更像结构先验，不像任务语义先验

### 来源 B：从样本级 guide feature 动态构造

做法：

1. 提取样本的 image/text guide feature
2. 用这些 guide feature 对 MoE experts 做相似度评分
3. 得到三套 phase prior

这版更接近 HiDESC，但工作量明显更大。

对于当前论文实验，建议先做来源 A。

## 5.9 MoELoRA 版最小可行实验

建议先只做这个矩阵：

1. `MR-LoRA native eval`
2. `MR-LoRA + hard-split prior eval`
3. `MR-LoRA + band-schedule prior eval`

其中：

- `hard-split prior eval` 证明“分层先验本身”是否有帮助
- `band-schedule prior eval` 证明“平滑 band 过渡”是否优于硬切层

这已经足够支撑你的 eval-side 论证。

## 5.10 MoELoRA 风险点

### 风险 1：HiDESC 的 route weights 语义和 MoE internal experts 语义不对齐

这是最大风险。

所以第一版不建议直接把 HiDESC 的 `task expert weights` 生搬硬套到 MR-LoRA 的 MoE expert 上，而是应该把它降级成：

- layer-dependent prior bias

### 风险 2：prior 太强，覆盖 native router

如果 `tau` 太大，MoELoRA 会失去自身 gate 的意义。

建议：

- 先扫 `tau in {0.25, 0.5, 0.75, 1.0}`

### 风险 3：每层 prior 注入接口不方便

如果你不想在每个 LoRA 层对象里塞 `router_prior`，也可以退一步：

- 在 forward 前把 prior 存到 model 全局上下文
- 由每层读取当前 `layer_idx` 对应 prior

这会比逐层显式传参更容易接进现有代码。

## 6. 推荐的开发顺序

建议严格按下面顺序做。

### Phase A: RegLoRA train

1. 复制 `description_utils.py`
2. 扩展 `train.py` 参数
3. 扩展 dataset/collator
4. 迁移 HiDESC trainer loss
5. 增加 cache extraction 模式
6. 跑 smoke：
   - `align-only`
   - `struct-only`
   - `full`

### Phase B: MR-LoRA eval

1. 先实现 stage1 band schedule 读取
2. 实现 per-layer prior 生成
3. 在 `clitmoelora.py` 加 router prior bias
4. 先做 offline/固定 prior 版本
5. 跑：
   - native eval
   - hard-split eval
   - band-schedule eval

## 7. 建议的实验命名

为了避免后期结果混乱，建议直接采用下面命名。

### RegLoRA 线

- `RegLoRA-native`
- `RegLoRA-align`
- `RegLoRA-struct`
- `RegLoRA-full`

### MoELoRA 线

- `MRLoRA-native-eval`
- `MRLoRA-hardsplit-eval`
- `MRLoRA-bandschedule-eval`

## 8. 最后的建议

从论文写作角度看，最重要的不是“两个 baseline 都彻底 HiDESC 化”，而是：

1. `RegLoRA` 证明：
   - 即使在强训练正则 baseline 上，HiDESC 的 description-aligned loss 仍然有独立增益。

2. `MoELoRA` 证明：
   - 即使在另一类 MoE gating 结构上，HiDESC 的分层 eval 先验仍然有迁移价值。

所以工程上请优先做：

- `RegLoRA + HiDESC-train-loss`
- `MR-LoRA + HiDESC-band-eval`

而不要第一轮就追求：

- `RegLoRA + full HiDESC`
- `MR-LoRA + full HiDESC`

前者更容易形成干净、可解释、可写进论文的证据链。
