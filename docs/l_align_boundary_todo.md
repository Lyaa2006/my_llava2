# HiDESC 训练阶段 `L_align` 实现 To-Do

本文档用于指导后续 coding agent 在 `HiDESC / LLaVA` 的**训练阶段**加入一个低显存、低额外计算开销的 boundary `L_align`，并统一当前训练损失的命名口径。

注意：

- 本文档描述的是**下一步应实现的设计**，不是当前代码已经具备的能力。
- 当前讨论对象是 `HiDESC` 的训练阶段，不是 `HiDe`，也不是推理阶段 routing。
- 本轮需要把现有的 `L_focus` 和 `L_energy` 在概念上合并为 `L_struct`。
- 本轮 `L_align` 只作用在**单个 deep/middle boundary layer** 上，不扩展成逐层对齐。
- 边界层的最终取值暂时**不在本轮固定**；先保留一个默认值，通过参数控制，后续可直接改配置复用同一实现。

## 1. 本轮目标

在当前 `HiDESC` 训练目标中：

- 保留标准监督损失 `L_ce`
- 保留现有的 `L_focus`
- 保留现有的 `L_energy`
- 新增一个 boundary `L_align`

但在训练口径和日志口径上统一成：

```text
L_struct = w_focus * L_focus + w_energy * L_energy
L_total = L_ce + L_struct + w_align * L_align
```

其中：

- `L_struct` 是 `L_focus` 与 `L_energy` 的聚合名，不再把二者和 `L_align` 并列描述成三个互不相关的辅助项
- `L_align` 是额外增加的单点跨模态一致性约束

本轮 `L_align` 的作用不是做全网络最强对齐，而是：

- 在多模态基础语义已经形成之后
- 在更强的任务特化继续展开之前
- 用最低额外成本约束 image/text 表征不要过早明显漂移

## 2. 总体原则

后续 agent 必须遵守以下原则：

1. 不要把 `L_align` 扩展成多层平均或全深层逐层约束
2. 不要实现 token-level image-text 全对全匹配
3. 不要把 `L_align` 移到 `mm_projector` 输出后立即计算
4. 不要为了 `L_align` 长期开启全层 hidden state 缓存
5. 不要把 `L_align` 设计成独立的新训练框架，必须贴合现有 `HiDESC` trainer
6. `L_align` 必须只做一次 image/text 分离、一次 pooling、一次相似度约束

本轮设计的关键目标是：

- 让 `HiDESC` 的训练目标对外只有三大块：`CE`、`L_struct`、`L_align`
- 让 `L_align` 成为一个轻量 regularizer，而不是新的主任务

## 3. 推荐损失定义

推荐先实现最轻量版本：

```text
L_align = mean(1 - cosine_similarity(h_img_boundary, h_txt_boundary))
```

其中：

- `h_img_boundary`：边界层 image tokens 的 masked mean pooled feature
- `h_txt_boundary`：边界层 text tokens 的 masked mean pooled feature

如果 batch 内有多个样本，则对 batch 求平均。

第一版不要引入：

- contrastive negatives
- triplet loss
- token-to-token matching
- attention pooling
- asymmetric stop-gradient

## 4. 推荐修改位置

### 4.1 主要文件

- `LLaVA/HiDESC/llava/model/llava_arch.py`
- `LLaVA/HiDESC/llava/model/language_model/llava_llama.py`
- `LLaVA/HiDESC/llava/train/llava_trainer.py`
- `LLaVA/HiDESC/llava/train/train_MOE.py`

### 4.2 职责划分

推荐职责如下：

- `llava_arch.py`
  - 在 multimodal 拼接阶段构造 `image_token_mask` / `text_token_mask`
  - 保证 mask 与最终送入 LLM 的 `inputs_embeds` 完全对齐

- `llava_llama.py`
  - 只负责让训练路径能够拿到计算 `L_align` 所需的中间信息
  - 不建议把 `L_align` 的主计算逻辑硬塞进 model forward，除非现有接口无法承载

- `llava_trainer.py`
  - 作为 `L_struct` 与 `L_align` 的主要实现位置
  - 负责 boundary hidden state 抓取、pooling、loss 计算与总 loss 汇总
  - 负责日志命名统一

- `train_MOE.py`
  - 负责新增训练参数
  - 负责默认值、开关控制和配置暴露

## 5. 具体 To-Do

### 5.1 统一训练损失命名

后续 agent 需要先统一 `HiDESC` 的训练命名口径：

- `L_focus`：保留现有定义
- `L_energy`：保留现有定义
- `L_struct`：定义为 `w_focus * L_focus + w_energy * L_energy`
- `L_align`：新增 boundary image/text 对齐项

要求：

1. 不改变当前 `L_focus` / `L_energy` 的数学定义
2. 只改变“总损失组织方式”和“日志命名口径”
3. 日志里既要能看到 `loss_struct`，也建议保留 `loss_focus` / `loss_energy` 便于排查

推荐最终日志最少包含：

- `loss_ce`
- `loss_struct`
- `loss_align`
- `loss_total`

建议同时保留：

- `loss_focus`
- `loss_energy`

### 5.2 在多模态拼接阶段生成 mask

后续 agent 需要在 `prepare_inputs_labels_for_multimodal(...)` 中补齐与 `new_input_embeds` 同步的两个 mask：

- `image_token_mask`: `[B, T]`
- `text_token_mask`: `[B, T]`

要求：

1. `image_token_mask` 标记由 `image_features` 插入得到的 patch token 位置
2. `text_token_mask` 标记原始文本 token 对应的位置
3. padding 区域必须是 `False`
4. 二者必须与最终 `new_input_embeds_padded` 的长度、padding 方向完全一致

实现要点：

- 当前 `prepare_inputs_labels_for_multimodal(...)` 已经在逐样本构造 `cur_new_input_embeds`
- 同时也在构造 `cur_new_labels`
- 后续 agent 应在同一个循环里同步构造：
  - `cur_image_mask_parts`
  - `cur_text_mask_parts`
- image patch 插入时：
  - image 部分 append 全 `True` 的 image mask
  - 对应 text mask append 全 `False`
- 文本片段插入时：
  - text 部分 append 全 `True` 的 text mask
  - 对应 image mask append 全 `False`

注意：

- 不要在后处理阶段靠 `labels == IGNORE_INDEX` 去反推 image/text 边界
- 必须在拼接当下显式记录

### 5.3 让训练路径拿到 boundary hidden state

后续 agent 需要在 `HiDESC` 的训练路径中支持 boundary hidden state 提取。

推荐做法：

1. 增加配置项：
   - `enable_boundary_align`
   - `align_boundary_layer`
   - `align_loss_weight`

2. 训练时仅在 `enable_boundary_align=True` 时启用相关逻辑

3. 第一版允许用最直接的方式抓取目标 boundary layer hidden state

4. 但实现目标应是：
   - 只消费一个目标层
   - 不把 `L_align` 变成长期全层缓存依赖

实现建议：

- 如果现有 trainer helper 方便复用，优先在 `llava_trainer.py` 中抓单层 hidden state
- 若第一版需要短期使用 `output_hidden_states=True`，也必须只消费一个目标 boundary layer，并在文档/代码中标明这只是过渡方案
- 若后续显存仍偏高，再考虑 hook 方式或只抓目标层输出的更轻实现

### 5.4 在 boundary layer 上做单次 pooling

后续 agent 需要在取到 boundary hidden state 后做两次 masked mean pooling：

```text
h_img = masked_mean(hidden_state, image_token_mask)
h_txt = masked_mean(hidden_state, text_token_mask)
```

要求：

1. pooling 必须逐样本计算
2. 必须避免除以 0
3. mask 需要 cast 到与 hidden state 兼容的 dtype
4. pooling 后张量形状建议为 `[B, H]`

推荐实现一个小工具函数，例如：

```text
masked_mean(hidden_states, mask) -> pooled_states
```

不要把 pooling 分散写在多个地方。

### 5.5 计算 `L_align`

推荐第一版：

```text
L_align = mean(1 - cosine_similarity(h_img, h_txt))
```

要求：

1. 先对 `h_img` / `h_txt` 做 `F.normalize`
2. 再做点积或 cosine
3. batch 内求平均
4. `L_align` 作为单独标量返回或记录

### 5.6 把 `L_align` 并入总 loss

推荐最终形式：

```text
L_struct = w_focus * L_focus + w_energy * L_energy
L_total = L_ce + L_struct + w_align * L_align
```

其中：

- `L_ce` 是当前标准监督损失
- `L_struct` 是当前 `HiDESC` 训练阶段的结构约束总名
- `L_align` 只在单个 boundary layer 额外加一次

后续 agent 必须保证：

- 当 `enable_boundary_align=False` 时，训练行为与当前版本尽量保持一致
- 当 `enable_boundary_align=True` 时，新增日志中能看到：
  - `loss_ce`
  - `loss_struct`
  - `loss_align`
  - `loss_total`

## 6. 推荐实现路径

建议后续 agent 按下面顺序做，不要一次把所有优化混在一起：

1. 先在文档和日志口径上把 `L_focus + L_energy` 统一命名为 `L_struct`
2. 在 `prepare_inputs_labels_for_multimodal(...)` 中把 `image_token_mask` / `text_token_mask` 构造对
3. 确保这些 mask 能沿训练 forward 路径传到计算 `L_align` 的位置
4. 用最简单的单层抓取方式跑通 boundary `L_align`
5. 完成语法检查和单 batch smoke test
6. 再观察显存占用是否可接受
7. 只有在显存仍明显超标时，再考虑只抓目标层输出的优化实现

## 7. 推荐默认配置

本轮先给出一个**临时默认值**，不把它解释成最终实验结论：

- `enable_boundary_align = False`
- `align_boundary_layer = 15`
- `align_loss_weight = 0.01`

说明：

- `align_boundary_layer` 只是第一版默认值
- 最终 deep/middle 边界取值以后续实验或配置为准
- 后续 agent 必须在代码注释或文档里明确说明该参数使用的是 `0-based` 还是 `1-based` 编号

## 8. 不要做的事

后续 agent 不要做下面这些修改：

- 不要把 `L_align` 放回 `mm_projector` 输出后立即计算
- 不要把 `L_align` 扩展成 deep layers 的逐层平均
- 不要靠手写硬编码 token 下标猜测 image/text 边界
- 不要在 eval 路径里默认启用 `L_align`
- 不要把 `L_align` 实现成脱离 `HiDESC` trainer 的单独训练框架
- 不要因为边界层最终值还没定，就把实现写死在某个固定层编号上

## 9. 验收标准

后续 agent 完成实现后，至少应满足以下检查项：

1. `image_token_mask` / `text_token_mask` 与最终 multimodal sequence 严格对齐
2. `L_align` 只在单个 boundary layer 计算一次
3. 总 loss 对外只有三大组成部分：
   - `CE`
   - `L_struct`
   - `L_align`
4. `L_struct` 内部仍能追踪 `L_focus` 与 `L_energy`
5. 默认关闭 `L_align` 时，训练路径与当前版本兼容
6. 开启后可以在单 batch 上稳定前向、反向
7. 至少完成一次基础语法检查或 smoke test

## 10. 给后续 agent 的一句话约束

本轮 `L_align` 的实现目标不是“做最强的跨模态对齐”，而是：

**在 `HiDESC` 训练阶段，用最低额外成本，在一个可配置的 boundary layer 上增加单点的 image/text 语义一致性约束，并将现有 `focus + energy` 统一收口为 `L_struct`。**
