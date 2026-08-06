# 前置实验

## 实验一：层级边界识别

### 1. 实验目的

实验一的目标是先在**不引入专家 LoRA、role memory 和 routing 机制**的 base LLaVA 上判断不同层更偏向承担什么功能。这样后续 Hi-DESC 的分层训练和分层推理不是凭经验划分，而是由 backbone 本身的逐层行为给出结构先验。

最终采用的三段划分为：

1. 深层：`1-13`
2. 中层：`14-28`
3. 浅层：`29-32`

对应两个边界：

1. 深中边界：`13`
2. 中浅边界：`29`

实验图见：`docs/hierarchical_boundary_outputs_1k_v3/hierarchical_boundary_curves.png`

### 2. 三条指标曲线

图中三条曲线分别对应三种层功能信号。

1. `Reasoning Signal`

该曲线衡量每一层对真实图像信息的推理依赖程度。实现时，在 `ArxivQA`、`IconQA`、`CLEVR` 上保持文本 prompt 不变，分别输入真实图像和空白图像，提取每一层最后一个 prompt token 的 hidden state，并计算二者的 cosine distance。差异越大，说明该层越依赖图像内容参与推理。

2. `Task Separation`

该曲线衡量每一层是否已经出现任务相关的粗粒度分流信号。实现时，在六个 UCIT 任务上提取每一层最后一个 prompt token 的 hidden state，然后计算类 Fisher 分离度，即任务间距离与任务内距离的比值。数值越高，说明不同任务在该层表征空间中越容易被分开。

需要注意的是，`Task Separation` 不能直接等价为稳定的任务特异推理能力。早期的可分性可能来自图像分布、prompt 模板、答案形式或数据集统计差异。因此它更适合解释为“任务线索已经出现”，而不是“可以立刻固定到某个 task expert”。

3. `Style Logit Signal`

该曲线衡量每一层对最终输出形式的控制程度。实现时，对同一个语义问题构造不同回答格式要求，例如只输出选项、输出选项文本、输出完整句子；然后把每一层 hidden state 经过 final norm 和 lm head 做 logit lens 投影，比较首个答案 token 分布的 Jensen-Shannon divergence。差异越大，说明该层越接近“决定答案以什么形式输出”。

### 3. 边界如何确定

深中边界由 `Reasoning Signal` 和 `Task Separation` 的交接关系确定。具体来说，在前半段搜索 reasoning 上升、task separation 下降且两条曲线接近的位置。实验结果将该交接点定位在第 `13` 层。

这一结论的含义不是“`1-13` 没有任务区分”，而是：这一阶段已经开始出现任务线索，但区分仍然偏粗，不宜直接把样本按照 image/text token 固定到单一 task expert。更合理的做法是先保留多个相关 role/expert 的参与，让后续层继续完成任务语义收敛。

中浅边界由 `Style Logit Signal` 决定。该指标在后段显著升高，并在第 `29` 层附近达到峰值，说明从第 `29` 层开始模型更接近最终 token 生成决策。因此 `29-32` 被划为浅层，主要承担答案格式和语言风格的收敛。

### 4. 对训练策略的推导

实验一说明三段层的功能不同，因此训练时不应把所有层当作同一种语义空间处理。

1. 深层 `1-13` 处在共享多模态推理和早期任务分流的过渡阶段。这里已经能看到任务线索，但这些线索还不够稳定，因此应保持跨任务可复用性，避免过早把表征固定到某个具体任务。

2. 深中交接层 `13` 是从粗粒度任务分流进入任务语义特化的关键位置，因此在这里加入 `L_align`。它只做单点 image/text 语义一致性约束，用来保证图文表征在进入中层任务特化前不会过早漂移或被固定死。

3. 中层 `14-28` 主要承担任务语义特化。训练目标中的 `L_struct` 在这里提供结构约束：其中 `description focus loss` 让更新集中在 key tokens，`description energy loss` 限制相对 base reference 的整体漂移，避免当前任务专家破坏跨任务可比性。

4. 浅层 `29-32` 更接近输出形式控制。这里不适合施加强图文对齐约束，否则会压制任务答案格式和语言风格的差异化；主要由 `standard CE` 保证当前任务输出能力。

因此训练目标可以概括为：

```text
L_total = L_ce + L_struct + w_align * L_align
L_struct = w_focus * L_focus + w_energy * L_energy
```

### 5. 对推理策略的推导

实验一同样决定了 eval 阶段不同层级的专家协同方式。

1. 深层 `1-13`：采用跨 role 的 broad collaboration。原因是这一段虽然已有任务线索，但仍处在粗粒度分流阶段，应让多个相关 role 的专家共同参与，而不是过早进行强 task 选择。

2. 中层 `14-28`：采用 role 内 expert selection。原因是经过前段宽松协同后，这一段更适合逐步收敛到更相关的任务语义；因此应先保留 role 结构，再在同一 role 内筛选更相关的专家，避免把所有专家扁平化竞争。

3. 浅层 `29-32`：采用更稀疏的 top-k expert fuse。原因是这一段已经接近最终输出决策，应收敛到与当前样本最相关的专家，以匹配答案格式和语言风格。

训练阶段可以使用当前任务信息更新 task anchor、role memory 和当前专家；eval 阶段必须 task-agnostic，不能使用真实 task id，只能根据当前样本的 image/text guide features 与历史 anchors、role prototypes 计算路由权重。

### 6. 小结

实验一的核心结论是：LLaVA 的层功能存在可解释的递进关系。前段是共享推理和粗粒度任务分流并存的过渡区，中段更适合任务语义特化，尾段更适合输出风格收敛。Hi-DESC 因此在训练时把 `L_align` 放在深中交接层，并用 `L_struct` 约束任务专家的结构化更新；在 eval 时则采用深层跨 role 协同、中层 role 内筛选、浅层 top-k 收敛的分层专家策略。

## 实验二：`L_focus` 的前置验证

### 1. 实验方向

在当前的隐式自适应 HiDESC 方案中，显式 `deep / middle / shallow` 划分已经取消，但 `L_focus` 被保留，并继续只作用在固定 `description layer` 上。

因此，实验二不再服务于“层级边界划分”，而是直接验证 `L_focus` 的理论依据：

1. description 中真正应该被稳定保留的是 content-word 的语义坐标；
2. prompt template token 的无意义漂移会破坏这种跨模板语义稳定性；
3. 所以训练更新应该尽量集中在有语义价值的位置，而不是扩散到模板词上。

### 2. 核心设置

对同一张图像构造一个 canonical description prompt 和多个 template variants。它们只改变外层 instruction wording，但保持相同的视觉语义词，例如 `objects`、`attributes`、`shapes`、`colors`、`textures`、`scene context`、`visible text` 和 `spatial relations`。

然后在训练前后模型上提取固定 `description layer` 的 hidden states，比较三类量：

1. content words 在 canonical 与 variants 之间的对齐是否下降；
2. canonical prompt 中 template tokens 的漂移是否变大；
3. template drift 是否与 content-word alignment / retrieval 的下降相关。

### 3. 预期结论

实验二希望支持的不是“key token drift 一定更大”，而是更稳健的结论：

1. 如果 template token 漂移过大，那么相同语义词在不同模板下的 hidden representation 会更不稳定；
2. 这种不稳定可以表现为 content-word cosine alignment 下降、retrieval hit rate 下降或 retrieval margin 变差；
3. 因而 `L_focus` 的目标应当是抑制 template-driven drift，而不是简单缩小全部更新。

### 4. 对方法的指导

实验二主要用于说明为什么在取消显式分层后，`L_focus` 仍然有必要保留。

`L_focus` 的作用不是定义层级，而是让 description 空间的适应保持“语义集中”：

1. 少让模板词主导 hidden-state 变化；
2. 多让更新围绕真正承载视觉语义的 content words 展开；
3. 从而让不同任务专家在共享 description 锚点下仍然保持可比。

详细设计见：[docs/experiment2_l_focus_description_drift.md](/mnt/lyaa/MCITlib/docs/experiment2_l_focus_description_drift.md)

## 实验三：next-task description 生成失败

### 1. 实验目的

实验三用于验证一个关键问题：在持续学习过程中，能否直接用完成 `task n-1` 训练后的 checkpoint 去生成 `task n` 的 visual descriptions。

如果这种方式可靠，那么 description cache 可以随着任务推进由当前模型在线生成；如果不可靠，就必须使用更稳定的 reference 来源。

### 2. 实验设置

实验沿 UCIT 任务顺序进行。在每个 transition 中，使用刚训练完上一个任务的 checkpoint，直接为下一个任务样本生成 description：

1. `Task1` checkpoint 生成 `Task2 / ArxivQA` descriptions
2. `Task2` checkpoint 生成 `Task3 / VizWiz` descriptions
3. `Task3` checkpoint 生成 `Task4 / IconQA` descriptions
4. `Task4` checkpoint 生成 `Task5 / CLEVR` descriptions
5. `Task5` checkpoint 生成 `Task6 / Flickr30k` descriptions

详细样例见：`docs/experiment3_next_task_description_failures.md`

### 3. 主要观察

实验中出现了两类稳定失败模式。

1. `hallucinated / off-task description`

模型生成的 description 与真实图像或新任务语义明显不匹配。例如在科学图表任务上生成“cherries”“spider web”等与图像无关的自然图像描述。

2. `answer-mode collapse`

在后续 transition 中，模型不再生成 visual description，而是直接输出答案式短 token，例如 `3`、`4`、`10`、`8`。这说明模型已经被前序任务的回答模式污染，无法稳定保持 description 生成角色。

### 4. 结论与设计影响

实验三说明：**当前任务训练后的 checkpoint 不能被可靠地用作下一任务 description generator**。description 一旦由持续更新的模型滚动生成，就会把上一任务的答案模式、任务偏置和幻觉一起带入下一任务，导致 description cache 本身变成不稳定监督源。

因此 Hi-DESC 采用固定的 base model description cache 作为跨任务共享语义锚点，而不是使用 `task n-1` checkpoint 为 `task n` 动态生成 reference。这样可以避免 reference 随任务推进持续漂移，使不同任务专家始终围绕同一个稳定语义基准进行更新。
