# Experiment 2: `L_focus` 前置实验设计

## 1. 这个实验现在要证明什么

在 [docs/proposal_patch.md](/mnt/lyaa/MCITlib/docs/proposal_patch.md) 的新方案里，显式的 `deep / middle / shallow` 架构已经被取消，但 `L_focus` 被保留了，而且仍然只作用在固定的 `description layer` 上。

因此，这个前置实验不应该再去证明“某一层更像中层，所以需要 `L_focus`”，而应该直接证明下面这条更本质的理论依据：

> 对新任务的有效适应，应该尽量围绕 description 中承载视觉语义的 content tokens 展开，而不是让 prompt template token 发生无意义扩散漂移。

换句话说，`L_focus` 的作用不是制造层级，而是维持 **description 空间的语义更新集中性与跨模板稳定性**。

## 2. 为什么不能只看 key token drift 更大

旧版思路把重点放在：

1. `key token drift > template token drift`
2. `focus ratio = template / key < 1`

但这条叙事本身不够稳。现有 `ce-only` 预实验结果里，若只看 drift 均值，很多 task 的 `key_mean` 和 `template_mean` 已经非常接近，例如 [docs/experiment2_prelim_ce_only/description_drift_report.json](/mnt/lyaa/MCITlib/docs/experiment2_prelim_ce_only/description_drift_report.json) 中 `Task 1` 的 `focus_ratio` 约为 `1.012`，`Task 2` 约为 `1.010`。

这说明仅凭“谁漂得更大”很难构成一个稳健论据。更合适的问题是：

> 当 template token 漂移变大时，它会不会破坏相同 content words 在不同 prompt 模板下的语义对齐稳定性？

如果答案是会，那么 `L_focus` 就有明确必要性，因为它抑制的不是“所有变化”，而是“会污染语义坐标系的模板漂移”。

## 3. 推荐的论证结构

推荐把前置实验拆成两步：

1. 现象验证：证明 `template drift` 确实和 `content-word cross-template instability` 相关。
2. 轻量因果验证：证明加入 `L_focus` 后，这种不稳定性会下降。

这样最终就能支撑两级结论：

1. `L_focus` 的优化目标是有意义的。
2. `L_focus` 本身对这个目标确实有贡献。

## 4. Phase A：观测型前置实验

### 4.1 核心假设

对同一图像，如果我们只改变 description prompt 的外层模板，而保持视觉语义词不变，那么一个稳定的 description space 应该满足：

1. 相同 content words 的 hidden states 在不同模板下仍然彼此接近。
2. template token 的训练前后漂移越大，上述 content-word 对齐越容易被破坏。

这一步完全不依赖显式层级，只依赖固定的 `description layer`，与当前 patch 方案一致。

### 4.2 实验设置

对每个 UCIT task，比较训练前后两个 checkpoint：

1. `Task 1`：`base LLaVA` vs `Task1`
2. `Task t (t >= 2)`：`Task(t-1)` vs `Taskt`

对同一样本构造一组 prompt：

1. `canonical`
2. `template variant A`
3. `template variant B`

这些 prompt 的区别只体现在 instruction wording，上层模板不同，但 content words 固定。例如你们当前脚本里已经使用的：

1. `objects`
2. `attributes`
3. `shapes`
4. `colors`
5. `textures`
6. `scene context`
7. `visible text`
8. `spatial relations`

固定提取：

1. 同一个 `description_hidden_layer`
2. 同一个 `description_max_tokens`
3. 每个 content word 对应的 token span hidden states
4. canonical prompt 中 template token 的 hidden-state drift

### 4.3 指标设计

不要把主结论压在单一 drift 均值上，推荐用三类指标。

第一类：语义稳定性指标

1. `content alignment`
   `A_content = cos(h_canonical(word), h_variant(word))`
2. `alignment drop`
   `Delta_align = A_before - A_after`
3. `content retrieval hit@k / margin`
   用 canonical 中某个 content word 去检索 variant 中对应词的表示，观察训练后是否更难找回

第二类：模板扰动指标

1. `template drift`
   `D_template = mean ||h_after - h_before||`
2. `template drift concentration`
   看漂移是否集中在少数模板词，还是整个模板段一起移动

第三类：耦合指标

1. `corr(D_template, Delta_align)`
2. `corr(D_template, retrieval_drop)`

如果模板漂移越大，content alignment 越差、retrieval 越差，那么就直接说明 template drift 会污染 description 空间。

### 4.4 判据

Phase A 支持 `L_focus` 的条件不是“key drift 一定更大”，而是下面三条至少稳定满足两条：

1. `Delta_align > 0`，即训练后跨模板 content 对齐下降。
2. `corr(D_template, Delta_align) > 0`。
3. `corr(D_template, retrieval_drop) > 0`。

这三条一旦成立，就能说明：

> 模板部分的漂移不是无害噪声，而是会真实破坏语义词在不同 prompt 表达下的可比性。

这正是 `L_focus` 要抑制的对象。

### 4.5 现有脚本基础

你们已经有两套现成脚本可复用：

1. drift 分析：
   [analyze_description_drift.py](/mnt/lyaa/MCITlib/LLaVA/HiDe/scripts/MCITlib/Analysis/analyze_description_drift.py)
2. cross-template 对齐分析：
   [analyze_template_alignment.py](/mnt/lyaa/MCITlib/LLaVA/HiDe/scripts/MCITlib/Analysis/analyze_template_alignment.py)

已有并行入口：

1. [experiment2_prelim_parallel.sh](/mnt/lyaa/MCITlib/LLaVA/HiDe/scripts/MCITlib/Analysis/experiment2_prelim_parallel.sh)
2. [experiment2_template_alignment_parallel.sh](/mnt/lyaa/MCITlib/LLaVA/HiDe/scripts/MCITlib/Analysis/experiment2_template_alignment_parallel.sh)

所以 Phase A 基本不需要新训练，主要是把现有两个分析口径统一成同一套结论。

## 5. Phase B：最小因果验证

### 5.1 为什么还需要这一步

Phase A 只能说明“这种现象存在”，但别人仍然可能追问：

> 即使 template drift 会伤害语义稳定性，为什么一定需要 `L_focus`，而不是别的正则也能做到？

因此建议补一个很轻的 ablation，只跑小规模 task 子集，不要求完整 6-task 主实验。

### 5.2 最小 ablation 设计

保持当前 patch 方案不变：

1. 仍然使用固定 `description layer`
2. 不恢复显式层级
3. `L_energy`、`L_align`、CE 和 gating 形式都保持一致

只改一个变量：

1. `w_focus = 0`
2. `w_focus > 0`

推荐先跑两种规模：

1. `Task1 -> Task2`
2. `Task1 -> Task2 -> Task3`

这样成本足够低，但已经能观察持续学习中的 early interference。

### 5.3 需要比较的结果

比较 `with L_focus` 和 `without L_focus` 两组模型在相同样本上的：

1. `alignment_drop_mean`
2. `retrieval_drop_mean`
3. `corr(D_template, Delta_align)`
4. 当前任务 accuracy
5. 上一任务保持率

最理想的结果模式是：

1. 加 `L_focus` 后，`alignment_drop_mean` 更小
2. 加 `L_focus` 后，`retrieval_drop_mean` 更小
3. 加 `L_focus` 后，`template drift` 与语义退化的相关性更弱
4. 当前任务性能不掉，或只付出很小代价

如果这四点基本成立，就能把论证从“有意义”推进到“有效”。

## 6. 与当前隐式层级方案的关系

这个实验和旧版显式分层的关系应该被明确切开：

1. 它不试图证明某一段层更适合 semantic / task / format。
2. 它只验证固定 `description layer` 上的监督目标是否合理。
3. 它证明的是 description supervision 的性质，而不是 backbone 的层级划分。

因此在新方案叙事里，`L_focus` 的位置可以写成：

> Although the full architecture no longer relies on explicit hierarchical boundaries, `L_focus` is retained because description-space adaptation should remain concentrated on semantic content words and avoid template-driven drift. We verify this through a template-robustness preliminary experiment at the fixed description layer.

## 7. Proposal 里建议怎么写

建议把前置实验的 claim 写得克制一点，不要写成“证明 key token 一定比 template token 漂得更大”，而写成下面这种更稳的版本：

> Preliminary evidence suggests that the harmful part of description drift is not update magnitude per se, but template-driven drift that destabilizes cross-template semantic alignment. This motivates `L_focus`, which biases adaptation toward content-bearing tokens and suppresses semantically uninformative template drift.

## 8. 一句话版本

如果你要把这个实验压缩成一句话放进 proposal，可以直接写：

> 我们先通过跨模板对齐实验验证：description 中真正需要被保留的是 content-word 的语义坐标一致性，而模板 token 的漂移会显著破坏这种一致性；因此 `L_focus` 的目标不是单纯缩小更新，而是把更新限制在有语义价值的位置上。
