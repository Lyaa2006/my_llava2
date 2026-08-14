# Stage 1 边界结论简述（论文风格）

日期：2026 年 8 月 11 日

## 摘要

在相同 Stage 1 分析框架下，`LLaVA` 的主结论仍为：`b1_window=[15,18]`、`b1_center=16`、`b2_window=[29,31]`、`b2_center=30`。对 `InternVL` 而言，当前最稳的 `b1` 结论是 `b1_window=[12,14]`、`b1_center≈14`；`b2_window=[27,29]` 目前更适合作为工程上的工作推荐值，而不是与 `LLaVA` 同等强度的最终锁定结论。

若把 `HiDESC` 在 `LLaVA` 上的“task anchor -> role prototype -> role routing”思路直接迁移到 `InternVL/UCIT`，则基于 `Task6` checkpoint 中已保存的 anchor 做离线归纳，当前 `UCIT` 会被分成 `2` 个 role：`ArxivQA` 单独成类，其余 `5` 个任务共享另一类。这说明 `InternVL` 上的 `HiDESC` 路由结构同样具有可离线归纳、可解释分析的性质。

## 方法概述

`b1` 仍沿用与 `LLaVA` 一致的 crossover-band 分析口径：在固定搜索窗 `10..20` 中寻找 reasoning 与 objective 的局部交叉带，并输出候选区间而非单点。`b2` 继续沿用 late-style overlap 的工程口径，但当前 `InternVL` 侧证据还不如 `LLaVA` 那样整齐，因此对 `27-29` 的表述应保持克制。

对 `HiDESC` 路由结构的补充分析，则不重新训练 expert，而是读取 `Task6` 之后 checkpoint 中已经保存的 `image_anchors / text_anchors`，再用与 `HiDESC` 兼容的离线 role 构造规则恢复 `role prototype`、`task-role membership` 与 `active_role_count`。本次使用的核心超参数与 `HiDESC` 注入脚本保持一致，包括 `routing_image_weight=0.5`、`routing_text_weight=0.5`、`routing_history_weight=0.15`、`routing_temperature=0.1`、`role_birth_threshold=0.70`。

## InternVL 结果

`InternVL` 的 `b1` 使用与 `LLaVA` 相同的大规模 crossover-band 框架，当前已完成 `seed=7,13,21,29,35`。基于现有 valid split 投票，`b1` 的主峰位于第 `14` 层，最稳的窄区间为 `12-14`。

`InternVL` 的 `b2` 当前参考以下两条 `cluster_balanced` 复跑：

- [seed17](/mnt/lyaa/MCITlib/docs/stage1_b1_windowed_dataset_only_cluster_balanced/internvl/seed17/canonical/stage1_boundary_summary.json)
- [seed27](/mnt/lyaa/MCITlib/docs/stage1_b1_windowed_dataset_only_cluster_balanced/internvl/seed27/canonical/stage1_boundary_summary.json)

从现有 summary 看，`b2_window=[27,29]` 更适合作为当前工程工作推荐值。

在 `HiDESC` 设定下，我们复用了 `checkpoints/UCIT/InternVL/HiDe/Task6_internvl_lora/non_lora_trainables.bin` 中已保存的 task anchor 做离线 role 归纳。结果显示：`UCIT` 六个任务最终分成 `2` 个 role，其中 `ArxivQA` 独立成类；`ImageNet-R`、`VizWiz`、`IconQA`、`CLEVR-Math`、`Flickr30k` 共同组成另一个 role。

## 结论

1. `b1` 与模型有关，应作为模型相关的阶段过渡区间来报告。
2. `LLaVA` 当前推荐：`b1_window=[15,18]`、`b2_window=[29,31]`。
3. `InternVL` 当前推荐：`b1_window=[12,14]`；`b2_window=[27,29]` 作为工程工作值使用。
4. 若迁移 `HiDESC` 的 role 归纳逻辑到 `InternVL/UCIT`，则 `Task6` 后的离线结果为 `2` 个 role，且 `ArxivQA` 单独成类。
