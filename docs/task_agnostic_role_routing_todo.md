# Task-Agnostic Role Routing 修改说明

本文档列出当前 HiDeRA_RPC / Adaptive Role-Aware Progressive Collaboration 代码还需要修改的点，供后续 coding agent 继续实现。

## 1. 核心问题

当前实现仍然在若干位置隐含使用 `cur_task` 作为当前 sample 的真实任务身份。这在训练阶段是可接受的，因为训练第 `T` 个任务时确实知道 `cur_task`；但在 continual learning benchmark 的 task-agnostic eval 阶段，测试 sample 来自哪个历史 task 是未知的。

因此 eval 阶段不能使用：

- `cur_task` 作为 self expert；
- `task_role_membership[cur_task]` 作为当前 sample 的 role membership；
- `routing_self_weight` 强行给 `cur_task` expert 加权；
- “当前任务 LoRA” 作为最后一层保底 expert。

正确方向是：

```text
sample guide features -> sample-level expert score -> sample-level role score
                    -> early / middle / late layer fuse
```

所有 eval routing 都必须由当前 sample 的 image/text guide features 推断得到，不能读取真实 task id。

## 2. HiDe 原始行为澄清

原始 HiDe 并不是前向到最后一层才知道 sample 与各 expert 的关系。

它在进入 transformer layers 之前，已经根据当前 sample 的 image/text guide features 与历史 task anchors 计算 expert weight。

但原始 HiDe 主要只把 sample-level expert weight 写入最后一层 LoRA projection。非最后层仍使用较粗的历史 expert 融合逻辑。

因此新方法可以继承这个思想：

- 在 transformer 前统一计算 sample-level routing weights；
- 不只服务最后一层，而是用于指导浅层、中层、深层不同 fuse 策略；
- eval 阶段全程 task-agnostic。

## 3. Transformer 前需要统一计算的量

在 `prepare_inputs_labels_for_multimodal` 中，在进入 transformer 前先计算：

### 3.1 Sample-level expert score

对每个历史 expert / task anchor：

```text
expert_score[e] =
    alpha * sim(sample_image, image_anchor[e])
  + beta  * sim(sample_text,  text_anchor[e])
  + gamma * expert_usage_prior[e]
```

其中：

- `sample_image` 来自当前 batch 的 image guide feature summary；
- `sample_text` 来自当前 batch 的 text guide feature summary；
- eval 时只使用已有历史 anchors；
- 训练时可以包含当前 task expert，但 eval 时不能把 `cur_task` 当作真实 sample label。

输出：

```text
expert_logits
expert_weight = softmax(expert_logits / tau_expert)
```

### 3.2 Sample-level role score

role score 可以复用 HiDeRA relation score 的思想，但必须改成 sample-to-role relation，而不是 current-task-to-role relation。

建议：

```text
prototype_score[r] =
    alpha * sim(sample_image, role_image_prototype[r])
  + beta  * sim(sample_text,  role_text_prototype[r])

member_score[r] = pool(expert_score[e] for e in members(role r))

role_score[r] =
    lambda_proto  * prototype_score[r]
  + lambda_member * member_score[r]
  + lambda_prior  * role_usage_prior[r]
```

`member_score` 的 pooling 建议使用：

- `top-m mean`，优先；
- 或 `logsumexp pooling`；
- 不建议简单 `mean`，会惩罚成员多的 role；
- 不建议纯 `max`，会被单个偶然高分 expert 支配。

输出：

```text
role_weight = softmax(topM_mask(role_score) / tau_role)
```

## 4. 浅层 fuse 修改

浅层应该做 broad role-level collaboration。

目标：

- 多个 role 可以共同参与；
- 不直接跨所有 experts 做扁平竞争；
- 用 role 作为粗粒度协同单位。

建议逻辑：

```text
for each candidate role r:
    role_repr[r] = mean_or_weighted_sum(expert_outputs[e] for e in members(role r))

hidden = hidden + sum_r role_weight[r] * role_repr[r]
```

实现简化版可以不真的显式构造 `role_repr`，而是把每个 role 的权重均分或按 expert weight 分配到 role 内 experts：

```text
early_expert_weight[e in role r] =
    role_weight[r] / num_members(r)
```

或：

```text
early_expert_weight[e in role r] =
    role_weight[r] * normalized_expert_weight_within_role[e]
```

浅层不要使用 `cur_task` self boost。

## 5. 中层 fuse 修改

中层应该做 role-aware intra-role expert selection。

目标：

- expert 只在同一个 role 内比较；
- role 之间仍由 `role_weight` 决定上层优先级；
- 不同 role 的 expert score 不直接扁平化比较。

建议：

```text
for each candidate role r:
    intra_weight[e | r] = softmax(expert_score[e] within role r / tau_intra)
    role_repr[r] = sum_e intra_weight[e | r] * expert_output[e]

hidden = hidden + sum_r role_weight[r]^gamma_mid * role_repr[r]
```

映射成 LoRA expert weight：

```text
middle_expert_weight[e in role r] =
    role_weight[r]^gamma_mid * intra_weight[e | r]
```

最后在所有 selected experts 上归一化。

`gamma_mid > 1`，用于让高相关 role 更突出。

## 6. 深层 fuse 修改

深层应该收缩到 sample 推断出的主导 role，而不是跨 role 重新打平比较 expert。

推荐默认逻辑：

```text
r_star = argmax(role_weight)
late_candidates = members(r_star)
late_expert_weight = top-k softmax(expert_score within r_star)
```

如果 role 判断不确定，可以允许 top-2 roles：

```text
if role_entropy > threshold or role_weight[top1] - role_weight[top2] < margin:
    use top-2 roles
else:
    use top-1 role
```

但即使使用 top-2 roles，也要保留层级结构：

```text
late_expert_weight[e in role r] =
    role_weight[r]^gamma_late * intra_weight[e | r]
```

不要再做：

```text
candidate_logits = _score_tasks(history_candidates, ...)
late_weights = global_top_k(candidate_logits)
```

因为这会让低相关 role 中的高分 expert 压过高相关 role 中的低分 expert，破坏 role 的上层语义。

## 7. 训练阶段与 eval 阶段的区别

### 7.1 训练阶段

训练时可以知道 `cur_task`，但它的用途应限制在：

- 更新当前 task anchor；
- 更新 role memory；
- 训练当前 task expert；
- bootstrap 阶段只启用当前 task expert。

bootstrap 结束后，建议尽量也走 sample-inferred routing，以减少 train/eval mismatch。

### 7.2 Eval 阶段

eval 阶段必须严格 task-agnostic：

- 不能用真实 task id；
- 不能用 `task_role_membership[cur_task]`；
- 不能给 `cur_task` expert 加 self weight；
- 不能把 `cur_task` 当作 top role 或 top expert 的默认值。

eval 阶段应完全依赖：

- sample image guide feature；
- sample text guide feature；
- historical task anchors；
- role prototypes；
- role/task usage priors。

## 8. 当前代码中需要重点修改的位置

### 8.1 `LLaVA/HiDeRA/llava/model/llava_arch.py`

需要重构以下逻辑：

- `_build_progressive_route_plan`
  - 当前逻辑会从每个 role 只取一个 best expert，再跨 role 全局重打分；
  - 应改成 `role_weight -> intra-role weights -> layer-specific expert weights`。

- `_blend_with_current_task`
  - 当前逻辑会给 `cur_task` 加 `routing_self_weight`；
  - eval 阶段必须禁用；
  - 更好是拆成 train-only self boost 和 eval-only pseudo expert routing。

- eval branch in `prepare_inputs_labels_for_multimodal`
  - 当前使用 `task_role_membership[self.cur_task]`；
  - 应改为根据当前 sample 重新计算 role membership；
  - role membership 不应来自真实 task id。

- train branch after bootstrap
  - 可以保留当前 task anchor 更新；
  - 但 progressive route plan 最好也由 sample-level expert/role score 驱动。

建议新增或替换为以下函数：

```text
_compute_sample_expert_scores(image_summary, text_summary, active_experts)
_compute_sample_role_scores(image_summary, text_summary, expert_scores, active_roles)
_build_early_weights(role_weights, expert_scores)
_build_middle_weights(role_weights, expert_scores)
_build_late_weights(role_weights, expert_scores)
_build_task_agnostic_route_plan(...)
```

### 8.2 `LLaVA/HiDeRA/HiDeRA/peft/tuners/clitmoelora.py`

当前 LoRA 层已经支持按 `progressive_stage` 使用不同 expert weights。

后续需要确保：

- early/middle/late 三种 stage 对应的权重来自 task-agnostic route plan；
- eval 时 `active_experts` 使用历史 expert 数，而不是 `cur_task + 1`；
- 不再把最后一层特殊化为唯一 sample-aware 层。

### 8.3 `LLaVA/HiDeRA/llava/model/language_model/llava_llama.py`

保留 role memory 状态：

- `role_image_prototypes`
- `role_text_prototypes`
- `role_task_count`
- `role_usage_prior`
- `task_role_membership`
- `active_role_count`

但 eval routing 不应直接查 `task_role_membership[cur_task]`。

`task_role_membership` 只应用于：

- role 的成员定义；
- 找某个 role 下有哪些 historical experts；
- 训练后更新 memory。

### 8.4 `LLaVA/HiDeRA/llava/train/train_MOE.py`

当前新增的 routing config 可以保留，但建议补充：

```text
role_member_pooling = top_m_mean
role_member_top_m = 2
middle_role_gamma = 1.5
late_role_gamma = 2.0
late_role_entropy_threshold
late_role_margin
eval_disable_self_boost = true
```

## 9. 推荐的最终层级策略

最终实现建议如下：

```text
Before transformer:
    compute expert_score[e]
    compute role_score[r]
    compute role_weight[r]

Early layers:
    multi-role broad fuse
    role gate temperature high
    experts within role are averaged or softly weighted

Middle layers:
    multi-role but sharper
    intra-role expert softmax
    role_weight^gamma_mid controls role priority

Late layers:
    top-1 role by default
    top-2 roles only if role uncertainty is high
    expert selection only within selected role(s)
    no global cross-role expert re-ranking
```

## 10. Definition of Done

修改完成后应满足：

- eval 阶段不依赖真实 task id；
- `cur_task` 不参与 eval route weight 构造；
- early/middle/late 的 expert weights 都来自 sample-level expert/role score；
- deep layer 不再跨 role 扁平化比较 expert；
- role weight 使用 sample-to-role relation score；
- 静态语法检查通过：

```bash
python3 -m py_compile LLaVA/HiDeRA/llava/model/llava_arch.py
python3 -m py_compile LLaVA/HiDeRA/llava/model/language_model/llava_llama.py
python3 -m py_compile LLaVA/HiDeRA/llava/train/train_MOE.py
python3 -m py_compile LLaVA/HiDeRA/HiDeRA/peft/tuners/clitmoelora.py
```

