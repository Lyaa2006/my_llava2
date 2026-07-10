# HiDESC 下一步修改说明

本文档用于指导后续 coding agent 继续修改 `HiDESC` 的分层 fuse 策略。

注意：本文件描述的是**下一步目标设计**，不是当前代码已经完成的状态。后续 agent 必须以本文档为准，而不是机械延续最近一次实验性改动。

## 1. 目标

将 `HiDESC` 的 eval 路由整理成下面这套更可解释的三级策略：

- `early`：保留原来的 role-aware 粗融合策略
- `middle`：保留原来的 role-aware 细化融合策略
- `late`：接受当前已经改好的 `image_anchor` / `text_anchor` 驱动的 HiDe-style 稀疏 task fuse

必须满足两个持续学习约束：

- `one-pass`：训练阶段不能回看旧数据；允许继续使用历史保存的 anchors / role memory
- `task-agnostic eval`：eval 阶段不能使用显式 task id

## 2. 当前共识

### 2.1 late 只放最后 3 层

推荐默认层次划分：

- `early`: 第 `0-15` 层，共 16 层
- `middle`: 第 `16-28` 层，共 13 层
- `late`: 第 `29-31` 层，共 3 层

理由：

- 原始 `HiDe` 只在最后 `1` 层做 anchor 路由
- `HiDESC` 若想比 `HiDe` 稍强一些，可将 sparse task routing 扩展到最后 `2-4` 层
- 最后 `3` 层是一个合理默认值：足够接近 logits，解释性清晰，又不会过早扰动共享表征

### 2.2 late 的当前结论

`late` 的当前设计视为**已确定**，后续 agent 暂时不要继续修改它。当前共识是：

- 只使用历史 `image_anchor` / `text_anchor`
- 使用当前输入的 guide features 计算共享 `task score`
- 所有 `late` 层共享同一套 sparse route
- 默认只作用于最后 `3` 层

### 2.3 late 的权重只算一次

如果 late 路由只由当前输入与历史 task anchors 的相似度决定，那么：

- 同一个 batch / 同一次 forward
- 所有 `late` 层

应共享同一套 `task score` 与 `late expert weight`。

不要为每个 late layer 重新计算一遍 route。

## 3. 三级策略的职责解释

后续实现必须保持下面的可解释性。

### 3.1 early

职责：`role recall`

含义：

- 从历史 role 记忆中做宽范围召回
- 保留跨任务共享知识
- 不做过强的 task 判别

实现要求：

- 保持之前设计的 role-aware 粗融合思想
- 不要改成 task-anchor 直接路由

### 3.2 middle

职责：`role disambiguation`

含义：

- 在 role 内部继续筛选更相关的 experts
- 比 early 更细，但仍然保持 role 结构

实现要求：

- 保持之前设计的 role-aware 细化融合
- 不要改成全局 task 扁平竞争

### 3.3 late

职责：`task decision`

含义：

- 使用 task anchors 做最后阶段的 task-level 判别
- 只在最后几层进行 sparse fuse

实现要求：

- 当前实现已经满足这三个要求，后续不作为主要修改对象

## 4. 后续重点：early 和 middle

本轮后续修改的重点不是 late，而是让 `early` / `middle` 真正体现层级化 role-aware fuse。

当前 late 已经有较清晰的解释：

- `late` 负责最终 task 判别
- `late` 只依赖 task anchors
- `late` 只在最后 `3` 层生效

因此后续 agent 需要把精力放在：

- `early`：如何做稳健的 role-level broad collaboration
- `middle`：如何做 role 内的细化 expert selection
- `early/middle` 与当前 late 的接口如何对齐

### 4.1 early 的修改目标

目标：

- 多个高分 role 可以共同参与
- 不直接扁平化成全局 task 竞争
- 尽量保留共享知识，不要过早注入强 task 偏置

推荐解释：

- `early` 回答的是“当前样本更像哪些历史 role 经验池”
- 它不应该回答“最终应该信哪个 task”

推荐实现方向：

1. 先得到 `role_weight`
2. 对每个被选中的 role，把 role 内 experts 聚合成 role-level response
3. 用 `role_weight` 去组合这些 role-level response

如果不想显式构造 role-level hidden，也可以先映射成 expert weight：

```text
early_expert_weight[e in role r] =
    role_weight[r] / num_members(r)
```

或者：

```text
early_expert_weight[e in role r] =
    role_weight[r] * normalized_expert_weight_within_role[e]
```

其中第二种通常更灵活，但第一种更稳、更容易解释。

### 4.2 middle 的修改目标

目标：

- 保持 role 结构
- 在 role 内进一步筛选更相关的 experts
- 不做跨 role 的全局扁平重排序

推荐解释：

- `middle` 回答的是“在这些相关 role 里，哪些成员 expert 更该起作用”

推荐实现方向：

1. 先保留 `role_weight`
2. 对每个 role 内部单独计算 `intra_role expert weight`
3. 最终的 middle 权重由：

```text
middle_expert_weight[e in role r] =
    role_weight[r]^gamma_mid * intra_weight[e | r]
```

其中：

- `intra_weight[e | r]` 只在 role 内归一化
- `gamma_mid > 1` 可让高相关 role 更突出

### 4.3 early / middle 与 late 的职责边界

后续 agent 必须保持以下边界清晰：

- `early` 不负责最终 task 判别
- `middle` 不负责最终 task 判别
- `late` 才负责最终 task 判别

也就是说：

- `early/middle` 可以继续依赖 role memory
- `late` 只依赖 task anchors

不要把 `late` 的 task-anchor sparse route 再反向扩散到 `early/middle`。

## 5. 后续 agent 需要做的代码修改

### 5.1 重点文件

- `LLaVA/HiDESC/llava/model/llava_arch.py`
- `LLaVA/HiDESC/llava/model/language_model/llava_llama.py`
- 如有必要：
  `LLaVA/HiDESC/HiDESC/peft/tuners/clitmoelora.py`

### 5.2 修改原则

1. 接受当前 `late` 的实现，不作为本轮重点修改对象
2. 重点重构 `early/middle` 的权重生成逻辑
3. 保持 `early/middle/late` 的职责边界清晰
4. eval 路径中不要因为实现方便而重新把 `cur_task` 当作 sample 的真实 task
5. 如果需要新增配置，优先新增：
   - `routing_early_mode`
   - `routing_middle_mode`
   - `routing_middle_gamma`
   - `routing_early_role_pool`

## 6. 明确不要做的事

后续 agent 不要做下面这些错误修改：

- 不要把 `early` 和 `middle` 也改成只看 task anchors 的全局 task fuse
- 不要让 `late` 依赖真实 `cur_task`
- 不要去推翻当前 late 的共享 sparse route 设计
- 不要把 `middle` 退化成“所有 experts 全局 softmax”
- 不要把 `early` 做成过强的 task 选择器

## 7. 验收标准

后续 agent 完成修改后，至少要检查：

1. `early/middle/late` 的层数划分符合：
   - `16 / 13 / 3`
2. `late` 路由继续只依赖：
   - 当前输入 guide features
   - 历史 `image/text anchors`
3. `late` 权重继续只计算一次，并复用于所有 late 层
4. `early` 体现 role-level broad collaboration
5. `middle` 体现 intra-role expert selection
6. eval 阶段不使用显式 task id
7. 代码通过最基本的语法检查
8. 最好补一轮小规模 eval，对比：
   - 原始 `HiDeCL`
   - 当前 `HiDESC`
   - 新 `HiDESC early/middle-refined`

## 8. 推荐执行顺序

建议后续 agent 按下面顺序推进：

1. 先确认当前 late 逻辑不再继续修改
2. 单独重构 `early` 的 role-level fuse
3. 再重构 `middle` 的 intra-role fuse
4. 确保 `early/middle` 与 `late` 的接口兼容
5. 跑语法检查
6. 跑 `task2-task5` 的小规模 eval

## 9. 简短结论

后续修改的核心不是“再改 late”，而是：

- 接受当前 late 的共享 sparse task router
- 把 `early/middle` 真正做成清晰、稳定、可解释的 role-aware 协同

这才符合当前已达成的设计共识。
