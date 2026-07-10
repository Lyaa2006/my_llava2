# 自适应角色感知渐进协同

## 1. 总览

对于每个输入样本，模型首先提取图像与文本引导特征，并在进入 transformer layers 之前计算样本级 expert score 与 role score。这些 routing 信号用于控制不同深度中的协同行为。

## 2. 样本级路由

给定样本级图像与文本摘要，模型计算：

- 样本与历史任务 anchors 之间的 expert scores；
- 样本与 role prototypes 及 role 内成员 experts 之间的 role scores。

随后，将 expert scores 归一化为 expert weights，将 role scores 归一化为 role weights。后续所有协同过程均由这些样本级权重驱动，而不依赖已知任务身份。

## 3. 渐进式协同

### 浅层

浅层执行宽范围的 role-level collaboration。多个高分 role 可以同时被激活，每个 role 内的 experts 先聚合为一个 role-level response，再由 role weights 对这些 role response 进行加权融合。

### 中层

中层执行 role-aware expert selection。experts 仅在同一 role 内进行比较，而不同 role 之间仍由 role weights 控制。由此形成层级化融合：role-level relevance 决定上层优先级，role 内 expert weights 决定细粒度迁移。

### 深层

深层进一步收缩协同空间。模型选择主导 role，或极少数高置信 role，并仅在这些选中 role 内执行 expert fusion。在这一阶段，不再进行全局的跨 role expert 重排序。

## 4. Role 结构

历史任务被组织为若干 latent roles。每个 role 由多模态 role prototypes 和一组成员 experts 表示。在 routing 过程中，role weights 用于刻画当前样本更相关的协同模式，而 expert weights 用于识别这些模式内部最有用的 experts。

## 5. 训练阶段：HiDeCL 损失

训练阶段采用 HiDeCL 式目标，对当前任务监督、描述对齐和描述效用进行联合优化。

### 标准监督损失

主损失仍为标准自回归交叉熵损失，用于优化当前任务输出。

### 描述对齐损失

模型为每个样本构造 description branch，并提取指定 hidden layer 的 description states。当前 description states 与离线缓存的 reference description states 在共享 token 位置上进行对齐，形成 description alignment loss。

### 描述效用损失

模型基于 description branch 构造 description utility objective，使 description 表示参与答案生成目标。该项作为额外训练信号与标准监督损失共同优化。

### 联合目标

训练总损失由以下三项加权组成：

- standard CE loss
- description alignment loss
- description utility loss

训练阶段同时更新当前 task anchors、role memory，以及 HiDeCL 相关描述缓存所对应的表示。

## 6. 推理阶段

推理阶段完全由样本驱动：模型直接根据输入样本估计 role weights 和 expert weights，并在不依赖任务标签的情况下应用相同的浅层、中层和深层融合策略。
