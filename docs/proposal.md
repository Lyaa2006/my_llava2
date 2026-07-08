# Adaptive Role-Aware Progressive Collaboration for HiDe

## 1. 目标

本文档描述一个直接建立在 HiDe 系列方法之上的新实现方案，用于替换当前历史 task experts 在前层的过早等权融合方式。

方法目标是：

- 对未知新任务先建立初始任务表征；
- 在线判断该任务应接入哪些历史协同模式；
- 再沿网络深度逐步将协同从粗粒度角色级收缩到稀疏任务级 expert 选择。

本文档只说明如何实现该方法，供后续 coding agent 直接落地，不讨论方法优越性。

方法名可暂定为：

`Adaptive Role-Aware Progressive Collaboration`

## 2. 方法总览

整个方法由三个阶段组成：

1. `Task Bootstrap`
2. `Online Role Induction`
3. `Role-Aware Progressive Collaboration`

这三个阶段分别解决三个问题：

- 新任务在完全未知时，如何先得到初始任务表示；
- 如何把当前任务接入已有历史协同结构；
- 如何让协同从浅层到深层逐步收缩，而不是一开始就全 expert 等权融合。

## 3. 设计原则

实现时遵循以下约束：

- 不允许使用 benchmark-specific 的固定 task-role 划分；
- role 必须由历史任务表示在线形成；
- 新任务到来时，允许复用已有 role，也允许创建新 role；
- 初版只做 `role birth` 与 `role update`，不做 `role merge`；
- 历史 experts 默认冻结；
- 现有 HiDe 的 task anchor 机制保留；
- 新增模块尽量只围绕 routing / gating / memory 展开，不重写主干模型。

## 4. 需要维护的状态

### 4.1 任务级状态

每个历史任务 `t` 需要继续维护：

- `image_anchor[t]`
- `text_anchor[t]`
- `task_sample_count[t]`

如果已有实现中还维护了历史 relation usage，也可保留：

- `task_relation_scores[t, :]`
- `expert_usage_prior[:]`

### 4.2 角色级状态

新增一个动态 role memory bank。对每个 role `r`，维护：

- `role_image_prototype[r]`
- `role_text_prototype[r]`
- `role_member_tasks[r]`
- `role_task_count[r]`
- `role_usage_prior[r]`

初版不需要给 role 人工语义标签。role 只是历史任务自动形成的 latent collaboration unit。

### 4.3 当前任务的临时状态

在训练第 `T` 个任务时，额外维护：

- `current_task_bootstrap_image_anchor`
- `current_task_bootstrap_text_anchor`
- `current_task_role_membership`
- `current_candidate_roles`
- `current_candidate_experts`

这些状态只在当前任务训练期内持续更新。

## 5. 三阶段主流程

### 5.1 Stage 1: Task Bootstrap

#### 5.1.1 目标

在不知道当前新任务属于哪类历史协同模式时，先用少量训练步得到该任务的初始任务表示。

#### 5.1.2 输入

输入为当前任务的前若干个 batch。

#### 5.1.3 执行过程

在 bootstrap 阶段：

- 不启用复杂 role-aware progressive collaboration；
- 只使用当前任务 expert 正常前向训练；
- 历史 experts 可完全关闭，或只保留最简单的 HiDe/HiDeRA 默认融合；
- 同时累计当前任务的 image / text guide features；
- 对这些特征做 running average，形成当前任务的 bootstrap anchors。

#### 5.1.4 输出

bootstrap 结束后得到：

- `current_task_bootstrap_image_anchor`
- `current_task_bootstrap_text_anchor`

这两个向量作为当前任务进入第二阶段的初始任务表示。

#### 5.1.5 初版建议

初版可以采用固定 warm-up 长度：

- 前 `N_bootstrap_steps` 只做 bootstrap

不建议第一版就做复杂的自适应结束条件。

### 5.2 Stage 2: Online Role Induction

#### 5.2.1 目标

根据当前任务 bootstrap 后得到的 anchors，决定它应连接到哪些已有 role，或者是否需要创建新 role。

#### 5.2.2 输入

输入包括：

- 当前任务 bootstrap anchors
- 历史 role memory bank
- 历史 task anchors

#### 5.2.3 role 匹配分数

对每个已有 role `r`，计算当前任务与该 role prototype 的匹配分数。

匹配分数可由以下几部分组成：

- 当前任务 image anchor 与 `role_image_prototype[r]` 的相似度
- 当前任务 text anchor 与 `role_text_prototype[r]` 的相似度
- 可选的 role usage prior

初版建议只使用：

- image 相似度
- text 相似度

即：

`role_score[r] = alpha * sim_image + beta * sim_text`

#### 5.2.4 role assignment 方式

实现时有两种可选方式：

- `hard assignment`
- `soft assignment`

初版建议采用：

- `soft assignment + top-M role selection`

即：

- 对所有 role 分数做 softmax，得到 role membership；
- 只保留前 `M` 个 role 作为候选 role。

#### 5.2.5 role birth

如果当前任务对所有已有 role 的最大匹配分数都低于阈值 `role_birth_threshold`，则创建一个新 role。

新 role 的初始化方式：

- `role_image_prototype[new] = current_task_bootstrap_image_anchor`
- `role_text_prototype[new] = current_task_bootstrap_text_anchor`
- `role_member_tasks[new] = {current_task}`
- `role_task_count[new] = 1`

#### 5.2.6 role update

如果当前任务匹配到已有 role，则在当前任务训练完成后更新该 role prototype。

更新方式初版建议使用 running average：

- 按 role 内任务数做均值更新；
- 不回看历史原始样本。

#### 5.2.7 输出

第二阶段输出：

- `current_task_role_membership`
- `current_candidate_roles`

这两个结果会被第三阶段直接使用。

### 5.3 Stage 3: Role-Aware Progressive Collaboration

#### 5.3.1 目标

正式进入分层协同，将历史 experts 的参与方式从浅层到深层逐步收缩：

- 浅层：role-level collaboration
- 中层：intra-role expert selection
- 深层：sparse task-level expert fusion

#### 5.3.2 层分段

将参与协同的 transformer layers 分成三段：

- `early collaborative layers`
- `middle collaborative layers`
- `late collaborative layers`

初版建议按层索引固定切分：

- 前 `K1` 层为浅层
- 中间 `K2` 层为中层
- 后 `K3` 层为深层

### 5.3.3 浅层：Role-Level Collaboration

浅层不直接对所有历史 task experts 融合，而是先对 role 进行门控。

执行步骤：

1. 收集历史 experts 在当前层的输出
2. 按当前 `candidate roles` 将这些 experts 分组
3. 在每个 role 内对 expert 输出做 role-level aggregation
4. 根据当前样本特征与 role prototypes 的匹配结果计算 role gate
5. 用 role gate 对各 role 表示加权融合

role 内聚合方式初版建议采用：

- 均值聚合

即先把每个 role 压缩成一个 role representation，再做 role 间融合。

浅层输出形式保持 residual：

- `hidden = hidden + role_fused_output`

### 5.3.4 中层：Intra-Role Expert Selection

中层开始在已激活 role 内部选择具体的历史 task experts。

执行步骤：

1. 从浅层 role gate 中保留前 `M` 个高分 role
2. 对每个候选 role 内的历史 task experts 单独打分
3. role 内打分只在该 role 成员之间归一化
4. 得到每个 role 内的 selected expert representation
5. 再由 role gate 做 role 间融合

role 内 expert 打分可以依赖：

- 当前层特征与 task image anchor 的相似度
- 当前层特征与 task text anchor 的相似度
- 可选的历史 usage prior

初版建议：

- 沿用现有 image/text anchor 相似度逻辑；
- 先不引入额外小网络；
- 只做相似度打分 + softmax。

### 5.3.5 深层：Sparse Task-Level Collaboration

深层不再保留 role 级宽协同，而是收缩到少量 task experts。

执行步骤：

1. 汇总中层各 role 给出的高分 task experts
2. 形成当前任务的 candidate expert pool
3. 在 candidate pool 上重新全局打分
4. 取 `top-k experts`
5. 用 `top-k` experts 做最终稀疏融合

深层融合输出同样保持 residual：

- `hidden = hidden + sparse_expert_fused_output`

初版建议使用：

- `top-k masking + softmax`

不建议第一版就引入额外稀疏正则器替代 hard top-k。

## 6. 需要新增的模块

### 6.1 Role Memory Manager

负责管理 role memory bank。

需要实现的接口：

- `init_role_from_task(task_id, image_anchor, text_anchor)`
- `score_roles(image_anchor, text_anchor)`
- `assign_roles(image_anchor, text_anchor)`
- `create_role(image_anchor, text_anchor, task_id)`
- `update_role(role_id, image_anchor, text_anchor, task_id)`

### 6.2 Bootstrap Manager

负责当前任务 warm-up 期间的临时 anchor 统计。

需要实现的接口：

- `reset_bootstrap_state(task_id)`
- `update_bootstrap_state(image_features, text_features)`
- `get_bootstrap_anchors()`
- `is_bootstrap_finished(global_step_or_seen_batches)`

### 6.3 Progressive Collaboration Scheduler

负责定义哪些层属于浅层、中层、深层。

需要实现的接口：

- `get_stage_for_layer(layer_idx, total_layers)`

返回值为：

- `early`
- `middle`
- `late`

### 6.4 Role Gate Scorer

负责计算当前样本对 candidate roles 的 gate 分数。

输入：

- 当前层特征
- role prototypes

输出：

- role-level weights

初版可直接使用相似度，不一定需要新增 MLP。

### 6.5 Intra-Role Expert Scorer

负责在候选 role 内部给具体 task experts 打分。

输入：

- 当前层特征
- 该 role 内所有 task anchors

输出：

- role 内各 task expert 的局部权重

### 6.6 Final Sparse Expert Selector

负责从中层留下的 candidate experts 中选 `top-k`。

输入：

- candidate expert features
- candidate task anchors

输出：

- final expert weights
- final selected indices

## 7. 与现有 HiDe 系列结构的挂接点

### 7.1 HiDe 当前关键入口

在 HiDe 中，当前核心逻辑是：

- 训练时更新 `image_anchors` 和 `text_anchors`
- 推理时用 image/text 相似度直接得到 `expert_weight`
- 再把该权重写入最后一层或少数层的 expert fuse 位置

因此，新方法需要保留的现有组件包括：

- task anchor 统计逻辑
- expert weight 写回 LoRA expert 的接口

### 7.2 HiDeRA 当前关键入口

HiDeRA 在 HiDe 的基础上，已经额外提供了：

- task-level relation weight 计算
- sample-level relation weight 计算
- top-k sparse masking
- relation cache / prior 的持久化结构
- trainer 中的 auxiliary loss 挂钩

因此，如果基于 HiDeRA 实现，本方法不需要从零搭建以下内容：

- 稀疏路由接口
- relation prior 缓存接口
- trainer 中的 routing / transfer aux loss 接口

### 7.3 新方法需要替换的核心逻辑

不论基于 HiDe 还是 HiDeRA，实现上的真正改动点都在：

1. 单一 task-level relation scoring
   改成
   `bootstrap -> role induction -> progressive collaboration`

2. 扁平 expert 权重
   改成
   `role-level weights + intra-role weights + final sparse weights`

3. 仅按 task 做历史统计
   改成
   `task memory + role memory`

## 8. 最小实现版本

为了降低第一轮实现复杂度，建议 coding agent 先做最小可运行版本。

### 8.1 第一轮必须实现

- 当前任务 bootstrap anchors
- 动态 role memory bank
- role birth / role update
- 浅层 role-level collaboration
- 中层 intra-role expert selection
- 深层 top-k expert selection

### 8.2 第一轮明确不做

- role merge
- layer-wise role prototypes
- learned role semantic labels
- 复杂的 role scorer MLP
- 复杂的 uncertainty-based bootstrap stopping
- 多种 auxiliary loss 的完整调参

### 8.3 第一轮推荐的简化

- role prototype 只维护全局 image/text anchor
- role assignment 采用 `softmax + top-M`
- final sparse selection 采用 `top-k masking`
- 历史 experts 冻结
- backbone 冻结
- 只训练当前任务 expert 与新协同模块

## 9. 训练流程

训练第 `T` 个任务时，执行顺序如下：

1. 加载历史 experts
2. 加载 task anchors
3. 加载 role memory bank
4. 初始化当前任务 bootstrap 状态
5. 前 `N_bootstrap_steps` 只做 Task Bootstrap
6. bootstrap 完成后，根据当前任务 bootstrap anchors 做 Online Role Induction
7. 确定 candidate roles
8. 后续训练步全部进入 Role-Aware Progressive Collaboration
9. 在每个 batch 内根据层位置分别执行浅层、中层、深层协同
10. 用当前任务监督信号计算主任务 loss
11. 若实现了 aux loss，则追加 routing / sparsity 相关项
12. 训练结束后写回当前任务 task anchors
13. 更新 role memory bank

## 10. 推理流程

推理时不再需要 bootstrap 学习，但仍需要角色诱导和渐进协同。

单个测试样本的执行过程如下：

1. 用当前样本提取 guide features
2. 根据当前任务已有 anchor 或样本级 summary，估计 candidate roles
3. 浅层执行 role-level collaboration
4. 中层执行 intra-role expert selection
5. 深层执行 sparse task-level fusion
6. 输出最终预测结果

如果推理阶段仍按 task eval，而不是完全 task-agnostic eval，则可直接复用该任务在训练结束时写回的 role membership 作为先验。

## 11. 数据持久化

为了支持多任务顺序训练，需要新增 role memory 的持久化文件。

建议在每个 task checkpoint 下保存：

- `task_anchor_cache.pt`
- `role_memory_bank.pt`
- 可选的 `routing_cache.pt`

其中 `role_memory_bank.pt` 至少包含：

- role prototypes
- role member task ids
- role usage prior
- role count

## 12. 推荐的实现顺序

coding agent 按以下顺序落地最稳：

1. 抽出 `RoleMemoryManager`
2. 在现有 anchor 更新逻辑外层加入 `BootstrapManager`
3. 实现 `Online Role Induction`
4. 实现 layer scheduler
5. 在现有 expert fuse 前插入浅层 role aggregation
6. 再加入中层 intra-role selection
7. 最后加入深层 top-k sparse selection
8. 最后再决定是否接 auxiliary loss

## 13. 第一轮更推荐基于 HiDe 还是 HiDeRA

结论：

**第一轮更推荐基于 HiDeRA 修改，而不是直接基于原始 HiDe 从头实现。**

原因不是因为 HiDeRA 的最终思路和本方法完全一致，而是因为它已经提供了本方法最难复用的一组工程骨架：

- relation routing config
- 稀疏 top-k 权重构造
- 历史 relation prior 缓存
- trainer 中的 auxiliary loss 接口
- 对所有目标投影层统一写入 expert 权重的逻辑

从实现成本看，本方法虽然要放弃 HiDeRA 当前“task relation 直接加权”的核心定义，但仍然可以复用 HiDeRA 的以下底座：

- 任务 anchor 存储方式
- relation / routing 配置项注入方式
- sample-level 与 task-level 权重注入接口
- relation cache 持久化结构

因此，更合理的策略是：

- 以 HiDeRA 为工程底座；
- 删弱它当前扁平 task relation scoring 的主逻辑；
- 把这部分替换成 `role memory + online role induction + progressive collaboration`。

## 14. 为什么这一轮不建议直接从 HiDe 重写

如果直接从 HiDe 出发，虽然概念上更干净，但 coding agent 需要额外补齐很多基础设施：

- 多层统一的 routing 权重注入
- top-k 稀疏路由
- routing 配置管理
- 辅助 loss 接口
- relation / usage 统计缓存

这样第一轮很容易把大量时间花在“补工程骨架”上，而不是实现你真正关心的新协同机制。

因此当前更推荐的实施路径是：

- `base branch = HiDeRA`
- `replace flat relation routing with adaptive role-aware progressive collaboration`

## 15. 推荐的分支策略

为了避免把 HiDeRA 当前逻辑彻底污染，建议单独开一个新分支或新目录，例如：

- `HiDeRPC`
- `HiDeRA_RPC`

代码层面建议：

- 先复制 HiDeRA 目录为新方法目录
- 保留训练与评测脚本骨架
- 逐步替换 `llava_arch.py` 中的 relation routing 实现
- 最后再更新 train config 和 eval config

这样最有利于并行开发和回退。
