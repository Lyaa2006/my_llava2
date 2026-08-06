# HiDESC Image-Spectral Routing Implementation Plan

本文档定义 HiDESC 下一步的 image-side spectral routing 实现路线。

本版本采用以下明确约束：

1. image 侧只使用 CLIP image patch token 的二维 FFT 谱描述。
2. text 侧暂时保持现有 pooled CLIP text feature 和 `text_anchor` 路径不变。
3. 原始 `image_anchor` 不参与默认 routing，只保留用于旧 checkpoint 兼容、baseline 和 fallback。
4. 不把 global image score、low-frequency score、high-frequency score 作为三条并行路径直接相加。
5. 先完成 task-level spectral routing，再决定是否加入 role-level spectral prototype。
6. LoRA expert forward 和现有 progressive route plan 尽量不改变，最终仍输出长度为 `T` 的 task weights。

## 1. 目标数据流

默认主路径：

```text
CLIP image patch tokens [B, N, D_patch]
        ->
2D FFT over patch grid
        ->
low/high radial spectral descriptor
        ->
unified image spectral descriptor
        ->
spectral task score

CLIP text pooled feature [B, D_clip]
        ->
existing text anchor similarity
        ->
text task score

spectral image score + text score + optional history prior
        ->
task_scores
        ->
existing role/progressive route plan
        ->
HiDeMOE LoRA expert weights
```

默认情况下，不再使用：

```text
global image feature -> image_anchor similarity
```

旧的 global image anchor 仍然保存在 checkpoint 中，但不是新方法的 active route signal。

## 2. 当前代码中的相关对象

主要代码位置：

- [llava_arch.py](/mnt/lyaa/MCITlib/LLaVA/HiDESC/llava/model/llava_arch.py)
- [clip_encoder.py](/mnt/lyaa/MCITlib/LLaVA/HiDESC/llava/model/multimodal_encoder/clip_encoder.py)
- [llava_llama.py](/mnt/lyaa/MCITlib/LLaVA/HiDESC/llava/model/language_model/llava_llama.py)
- [train_MOE.py](/mnt/lyaa/MCITlib/LLaVA/HiDESC/llava/train/train_MOE.py)
- [clitmoelora.py](/mnt/lyaa/MCITlib/LLaVA/HiDESC/HiDESC/peft/tuners/clitmoelora.py)

当前 image/text guide feature：

```text
clip_image_features:
    [B, 768]

patch features:
    [B, 576, 1024]

text_guide_features:
    [B, 768]
```

当前实现中：

- `image_anchors[t]` 是 global image prototype；
- `text_anchors[t]` 是 global text prototype；
- `role_image_prototypes[r]` 是 global image role prototype；
- `role_text_prototypes[r]` 是 global text role prototype。

本路线中：

- `text_anchors` 继续作为 active text memory；
- 新增 spectral image task prototype；
- `image_anchors` 和 `role_image_prototypes` 默认不再参与 active score；
- `role_text_prototypes` 可以继续保留，但 role-level spectral prototype 要在 task-level routing 验证后再加入。

## 3. 维度和 vision feature 来源

当前默认视觉配置为 CLIP ViT-L/14@336：

```text
image resolution = 336
patch size       = 14
patch grid       = 24 x 24
N                = 576
D_patch          = 1024
D_clip           = 768
```

### 3.1 不要对 global image embedding 做 FFT

禁止：

```text
FFT2(global_image_feature[768])
```

`[768]` embedding 的 channel 顺序没有二维空间含义。

FFT 必须作用于 patch token 的二维网格：

```text
P_raw:
    [B, 576, 1024]

P_raw_grid:
    [B, 24, 24, 1024]
```

### 3.2 patch token 的来源要求

当前 [clip_encoder.py](/mnt/lyaa/MCITlib/LLaVA/HiDESC/llava/model/multimodal_encoder/clip_encoder.py:30) 的 `feature_select()` 默认返回配置指定层的 patch token，而 `clip_image_features` 来自 CLIP 最终 image projection。

不要因为维度相同，就直接把任意中间层 patch token 送入最终层 `visual_projection`。默认实现应满足以下二选一：

1. 使用 CLIP 最终 vision layer 的 patch hidden states，再使用 CLIP `visual_projection`；
2. 如果必须使用中间层 patch token，新增独立的 frozen/learned `spectral_patch_projection`，不要假设最终层 `visual_projection` 已经对齐中间层语义。

第一版优先选择方案 1，减少额外假设。

`clip_encoder.py` 应能同时提供：

```text
global image embedding:
    [B, 768]       # legacy compatibility only

final patch hidden states:
    [B, 576, 1024] # spectral routing input

projected patch states:
    [B, 576, 768]  # FFT input after projection
```

## 4. Image-side 频谱描述

### 4.1 Patch projection

将 final patch hidden states reshape 为：

```text
P_grid:
    [B, 24, 24, 1024]
```

使用与 CLIP final vision representation 对齐的 projection：

```text
P_clip:
    [B, 24, 24, 768]
```

FFT 前转换为 float32：

```text
P_clip_fp32 = P_clip.float()
```

### 4.2 2D FFT

只在空间维度上变换：

```text
F = FFT2(P_clip_fp32, dim=(1, 2))
F_shift = fftshift(F, dim=(1, 2))
```

输出：

```text
F_shift:
    [B, 24, 24, 768] complex
```

使用幅度谱：

```text
S = log1p(abs(F_shift))
```

第一版不使用 phase，避免复杂数 running mean 和位置变化导致的 phase cancellation。

### 4.3 Low/high radial pooling

定义中心化频率坐标和归一化半径 `r(u, v)`。

第一版支持：

```text
spectral_cutoff = 0.33
```

定义：

```text
low mask:
    r <= spectral_cutoff

high mask:
    r > spectral_cutoff
```

low/high mask 必须：

- 覆盖全部频率；
- 互不重叠；
- 与 `fftshift` 后的坐标保持一致。

每个频带划分 radial bins：

```text
spectral_low_bins  = 4
spectral_high_bins = 4
```

得到：

```text
Z_low:
    [B, 768, 4]

Z_high:
    [B, 768, 4]
```

展平：

```text
z_low:
    [B, 3072]

z_high:
    [B, 3072]
```

对 low/high 分别做 normalization，再构造一个统一的 image spectral descriptor：

```text
z_low_norm  = normalize(flatten(Z_low), dim=-1)
z_high_norm = normalize(flatten(Z_high), dim=-1)

z_image_spec =
    normalize([
        spectral_low_scale  * z_low_norm,
        spectral_high_scale * z_high_norm
    ])
```

推荐初始值：

```text
spectral_low_scale  = 0.7
spectral_high_scale = 0.3
```

这里 low/high 只是构造一个 image spectral descriptor 的两个子带，不再分别产生三条并行 task route。

## 5. Spectral image task prototypes

### 5.1 Prototype 形状

如果 low/high 都为 `3072`，统一 image spectral descriptor 维度为：

```text
D_spec = 6144
```

新增：

```text
spectral_image_anchors:
    logical shape [T, 6144]
```

也可以内部保留结构化形式：

```text
spectral_low_anchors:
    [T, 768, 4]

spectral_high_anchors:
    [T, 768, 4]
```

但 active routing 时必须先按同样的 scale 和 normalization 组合成：

```text
spectral_image_anchor[t]:
    [6144]
```

建议代码命名优先使用 `spectral_image_anchors`，避免和旧的 `image_anchors` 混淆。

### 5.2 Training-time running mean

当前 task 为 `t` 时，对 batch 内每个样本计算 `z_image_spec`，并更新：

```text
A_spec[t] =
    (n_t * A_spec[t] + sum(z_image_spec_batch))
    / (n_t + B)
```

要求：

- running sum/count 使用 float32；
- 更新逻辑放在现有 global anchor update 附近；
- `disable_anchor_update=True` 时同时关闭 global 和 spectral prototype 更新；
- batch size 大于 1 时逐样本计算 descriptor，再对 batch 求和；
- 不允许用 batch 内 max 代替 prototype update。

### 5.3 Text anchor 保持原逻辑

text 侧不做 FFT，不改变当前 pooled CLIP text feature：

```text
text_guide_features:
    [B, 768]

text_anchors[t]:
    [768]
```

继续使用现有 running mean 更新：

```text
T[t] =
    (n_t * T[t] + sum(text_guide_features))
    / (n_t + B)
```

## 6. Eval 阶段 task score

### 6.1 Image spectral score

当前样本得到：

```text
z_image_spec:
    [B, 6144]
```

task prototype bank：

```text
spectral_image_bank:
    [T, 6144]
```

计算：

```text
s_spec[b, t] =
    cosine(z_image_spec[b], spectral_image_anchor[t])
```

这里的 `s_spec` 是默认 image task signal。

不要再计算或加入：

```text
cosine(global_image_feature, image_anchors[t])
```

### 6.2 Text score

继续使用当前 text anchor：

```text
s_text[b, t] =
    cosine(text_guide_features[b], text_anchor[t])
```

### 6.3 Final task logits

先分别对 image spectral score 和 text score 做 batch-wise、task-wise 的稳定 normalization，避免不同分支的数值范围不一致。

默认：

```text
task_logits[b, t] =
      spectral_image_weight * normalized(s_spec[b, t])
    + text_weight          * normalized(s_text[b, t])
    + history_weight       * history_score[t]
```

推荐初始配置：

```text
spectral_image_weight = 0.5
text_weight            = 0.5
history_weight         = 0.15
```

如果只做 task routing，不需要改变后续接口：

```text
task_scores:
    [T]       # batch size = 1
task_scores:
    [B, T]    # batch size > 1
```

### 6.4 Batch 处理要求

当前代码在 [llava_arch.py](/mnt/lyaa/MCITlib/LLaVA/HiDESC/llava/model/llava_arch.py:343) 使用 `max(dim=0)` 合并 batch 内 image/text score。

spectral routing 接入时必须改为逐样本 route：

```text
for sample b:
    task_scores[b] -> route_plan[b]
```

如果当前 LoRA expert weight 只能接受一个 route plan，则第一版明确限制 `batch_size=1`，并在代码中 assert，而不是静默使用 batch max。

## 7. Role routing 的接入顺序

### Phase 1: 只替换 task-level image signal

第一阶段只修改 task score：

```text
spectral image task score
        +
original text task score
        ->
existing task_scores interface
```

role/progressive route plan 暂时保持接口不变。

但 active role score 中不要再使用旧的 global image prototype。过渡期可以使用：

```text
role_score[r] =
      text_role_weight * text_role_score[r]
    + member_weight     * member_task_score[r]
    + prior_weight      * role_prior[r]
```

### Phase 2: Spectral role prototype

只有当 Phase 1 确认 spectral task score 有稳定收益后，再新增：

```text
role_spectral_anchors:
    [R, 6144]
```

role spectral prototype 由成员 task 的 spectral image anchors 聚合得到：

```text
role_spectral_anchor[r] =
    weighted mean(
        spectral_image_anchor[t]
        for t in members(role r)
    )
```

role score：

```text
role_score[r] =
      spectral_role_weight * cosine(z_image_spec, role_spectral_anchor[r])
    + text_role_weight     * cosine(text_feature, role_text_prototype[r])
    + member_weight        * member_task_score[r]
    + prior_weight         * role_prior[r]
    - size_penalty         * log1p(role_size[r])
```

不要同时加入旧的 `role_image_prototypes` similarity。

### Phase 3: Existing progressive route plan

最终仍然调用：

```text
_build_progressive_route_plan()
_build_early_route_weights()
_build_middle_route_weights()
_build_sparse_relation_weights()
```

这些函数的输入仍然是 task/role scores，不需要让 `clitmoelora.py` 了解 FFT。

## 8. Training / Eval 阶段顺序

### Phase 0: Offline analysis

不改变 active routing，只提取并统计：

```text
global-only baseline
spectral-image-only score
spectral-image + original-text score
low-only ablation
high-only ablation
low+high unified descriptor
```

至少报告：

- task routing top-1 accuracy；
- top-1 score margin；
- route entropy；
- benchmark accuracy；
- 不同 task 的收益和退化；
- image augmentation 下的 route stability；
- text paraphrase 下的 route stability；
- low/high descriptor 的 finite ratio；
- high-frequency 能量比例 `q_high`。

Phase 0 的重点不是证明 high-frequency 一定有效，而是确认 image spectral descriptor 是否能提供独立的 task discrimination。

### Phase 1: Offline spectral task prototypes

使用 frozen CLIP 提取训练数据：

```text
spectral_image_anchors:
    [T, 6144]
```

保存为独立文件，例如：

```text
spectral_prototypes.pt
```

metadata 至少包括：

```text
vision tower
vision select layer
image resolution
patch grid
projection version
FFT normalization version
spectral cutoff
low/high radial bins
low/high scales
```

第一版只在 eval 中加载，不修改已有训练 checkpoint。

### Phase 2: Active task routing

接入：

```text
spectral image score + original text score
```

关闭 active global image score。

保持后续 route plan 接口不变。

### Phase 3: Training-time spectral update

将 spectral task prototype 更新接入训练：

```text
spectral_image_anchors
spectral_image_boundary
```

并保存到 `non_lora_trainables.bin`。

### Phase 4: Spectral role prototype

确认 task-level route 稳定后，再增加：

```text
role_spectral_prototypes
```

role assignment、role update 和 role score 使用统一的 spectral image space。

### Phase 5: Optional high-frequency gate

默认不启用动态 high-frequency gate。

后续可以使用：

```text
q_high
low/high score disagreement
task score margin
```

调整 `spectral_high_scale` 或 routing temperature。

## 9. 预期代码改动

### 9.1 `clip_encoder.py`

需要让 vision tower 暴露 spectral routing 所需的 final patch states：

```text
final_patch_features:
    [B, 576, 1024]
```

建议不要在 `encode_images()` 中重复运行 vision tower。一次 vision forward 同时返回：

```text
clip_image_features
final_patch_features
projected_llava_features
```

其中：

- `clip_image_features` 只用于 legacy compatibility；
- `final_patch_features` 用于 FFT；
- `projected_llava_features` 继续用于 multimodal input。

### 9.2 `llava_arch.py`

新增或拆分以下函数：

```text
_extract_image_spectral_descriptor()
_build_radial_frequency_masks()
_pool_spectral_bands()
_get_spectral_image_anchor()
_update_spectral_image_anchor()
_compute_spectral_task_scores()
```

修改以下入口：

```text
encode_images()
prepare_inputs_labels_for_multimodal()
_compute_shared_task_scores()
_score_roles()
当前 anchor update 逻辑
```

`_compute_shared_task_scores()` 的默认逻辑应变成：

```text
spectral_image_scores
    +
text_scores
    +
optional history prior
```

禁止默认读取 `image_anchors` 计算 image score。

### 9.3 `llava_llama.py`

新增 prototype/boundary 容器：

```text
spectral_image_anchors
spectral_image_boundary
```

新增 routing config：

```text
use_spectral_image_routing = true
use_legacy_global_image_routing = false
use_text_anchor_routing = true
use_spectral_role_prototype = false

spectral_cutoff = 0.33
spectral_low_bins = 4
spectral_high_bins = 4
spectral_low_scale = 0.7
spectral_high_scale = 0.3

spectral_image_weight = 0.5
text_weight = 0.5
history_weight = 0.15
```

### 9.4 `train_MOE.py`

将以下 state 加入 persistent non-LoRA state：

```text
spectral_image_anchors
spectral_image_boundary
```

如果后续启用 role spectral prototype，再加入：

```text
role_spectral_prototypes
```

### 9.5 `clitmoelora.py`

如果最终仍输出：

```text
task_weights: [T]
```

则 LoRA expert forward 不需要修改。

## 10. Checkpoint 兼容性

旧对象继续保留：

```text
image_anchors
text_anchors
image_boundary
text_boundary
role_image_prototypes
role_text_prototypes
```

其中：

- `text_anchors` 继续 active；
- `image_anchors` 默认 inactive；
- `role_image_prototypes` 默认 inactive；
- 旧 checkpoint 没有 spectral state 时，可以显式关闭 spectral routing，回退到旧 global image/text baseline。

新对象：

```text
spectral_image_anchors
spectral_image_boundary
```

旧 checkpoint 不应尝试从 `[T, 768]` 的 `image_anchors` 伪造 spectral anchor。

## 11. 必须验证的约束

每个样本进入 routing 前：

```text
text feature:
    [B, 768]

final patch feature:
    [B, 24, 24, 1024]

projected patch:
    [B, 24, 24, 768]

low descriptor:
    [B, 768, 4]

high descriptor:
    [B, 768, 4]

unified image spectral descriptor:
    [B, 6144]

spectral image task scores:
    [B, T]

text task scores:
    [B, T]

final task scores:
    [B, T]
```

必须检查：

- FFT 使用 float32；
- FFT mask 与 `fftshift` 坐标一致；
- low/high mask 覆盖全部频率且无重叠；
- sample/prototype 的 scale 和 normalization 完全一致；
- 所有 score 和 prototype 都是 finite；
- batch size 大于 1 时不使用 batch max 混合样本；
- generation 后续 token forward 继续复用第一次 multimodal route；
- 默认配置下不会调用旧 `image_anchors` image similarity；
- text anchor 路径行为与 baseline 一致。

## 12. 第一版默认配置

```text
use_spectral_image_routing = true
use_legacy_global_image_routing = false
use_text_anchor_routing = true
use_spectral_role_prototype = false

spectral_cutoff = 0.33
spectral_low_bins = 4
spectral_high_bins = 4

spectral_low_scale = 0.7
spectral_high_scale = 0.3

spectral_image_weight = 0.5
text_weight = 0.5
history_weight = 0.15
```

第一版不启用：

```text
phase-aware spectral feature
adaptive high-frequency gate
global image anchor active score
spectral role prototype
```

## 13. Coding Agent 执行清单

按以下顺序实现，不要一次性修改全部 routing：

1. 修改 `clip_encoder.py`，在一次 vision forward 中暴露 final patch hidden states。
2. 在 `llava_arch.py` 实现 FFT、`fftshift`、radial mask 和 low/high pooling。
3. 写一个纯离线单元测试，验证 descriptor shape、finite、mask coverage 和 batch 独立性。
4. 实现 `spectral_image_anchors` 的 offline prototype 提取和加载。
5. 实现 `spectral image score + original text score` 的 task routing。
6. 默认关闭 `image_anchors` 的 active image similarity。
7. 保持 `_build_progressive_route_plan()` 和 LoRA expert forward 接口不变。
8. 先运行 global-only、spectral-only、spectral+text 三组 ablation。
9. 只有 task-level route 有稳定收益后，才实现 training-time spectral update。
10. 只有 task-level spectral update 稳定后，才考虑 `role_spectral_prototypes`。

最小验收标准：

```text
global-only:
    仍可运行，作为 baseline/fallback

spectral-only:
    可独立计算 task score

spectral+text:
    默认 active route

old image anchor:
    不参与默认 image score

text anchor:
    行为保持不变

LoRA forward:
    不需要理解 FFT
```
